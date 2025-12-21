# Third party modules
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

"""
TransFIRA sigmoid calibration script.

Performs sigmoid calibration (Platt scaling) on CCS and NNCCS scores to restore
discriminative variation when raw cosine similarities saturate near 1.0. This is
primarily needed for body recognition (Section IV-E, Table IV, Figure 4 in paper),
where raw CCS/NNCCS collapse into a narrow range (mean ~0.97).

Per Section IV-E and Figure 4:
- Raw body recognition similarities: mean 0.97, variance 3.69×10⁻⁵
- After sigmoid calibration: mean 0.50, variance 0.09, Brier score 0.1564
- Calibration restores discriminative power for CCAS as a separability measure

The calibration process:
1. Treats CCS values as positive class (matches), NNCCS as negative class (non-matches)
2. Fits logistic regression with sigmoid method to map raw scores to calibrated probabilities
3. Applies calibration to produce cal_ccs and cal_nnccs columns spread across [0,1]
4. Computes calibrated CCAS as cal_ccs - cal_nnccs (Equation 6)

Outputs:
- Calibrated CSV with cal_ccs, cal_nnccs, and cal_ccas columns
- Calibration model saved as .pkl file for reuse on validation/test data
- Diagnostic plots: raw/calibrated distributions and reliability curves
- Brier score for quantitative calibration quality assessment

Usage:
    python calibrate.py --csv path/to/labels.csv --outdir path/to/output/
"""


def calibrate_scores(csv_path, outdir, save_plots=True):
    """Perform sigmoid calibration on CCS and NNCCS scores.

    Implements the calibration procedure described in Section IV-E (Body Recognition)
    and Appendix B of the paper. When raw cosine similarities saturate near 1.0,
    sigmoid calibration spreads them across [0,1] to restore discriminative variation.

    Args:
        csv_path: Path to CSV file with 'ccs' and 'nnccs' columns
        outdir: Directory to save calibrated CSV, model, and plots
        save_plots: Whether to save calibration plots (raw/calibrated distributions,
                    reliability curves)

    Returns:
        df: DataFrame with calibrated scores added (cal_ccs, cal_nnccs, cal_ccas)
        cal_clf: Fitted CalibratedClassifierCV model (sklearn)
        brier: Brier score on calibrated predictions (lower is better, paper reports 0.1564)
    """
    # Load data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    if "ccs" not in df.columns or "nnccs" not in df.columns:
        raise ValueError("Input CSV must include columns 'ccs' and 'nnccs'.")

    # Prepare data for calibration
    ccs = df["ccs"].values
    nnccs = df["nnccs"].values

    # Labels: 1 = CCS (matches), 0 = NNCCS (non-matches)
    y = np.concatenate([np.ones(len(ccs)), np.zeros(len(nnccs))])
    X = np.concatenate([ccs.reshape(-1, 1), nnccs.reshape(-1, 1)])

    print("Fitting sigmoid calibration model...")
    # Fit calibration (Platt scaling / sigmoid method)
    # Per Section IV-E and Figure 4: sigmoid calibration maps CCS/NNCCS to [0,1]
    # with mean 0.5, restoring discriminative variation when raw scores saturate
    # Uses default cv (5-fold) to match the notebook implementation
    base_clf = LogisticRegression(max_iter=1000)
    cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid")
    cal_clf.fit(X, y)

    # Apply calibration to get probabilities
    print("Applying calibration to CCS and NNCCS...")
    df["cal_ccs"] = cal_clf.predict_proba(ccs.reshape(-1, 1))[:, 1]
    df["cal_nnccs"] = cal_clf.predict_proba(nnccs.reshape(-1, 1))[:, 1]
    df["cal_ccas"] = df["cal_ccs"] - df["cal_nnccs"]

    # For reference, also compute raw CCAS if not present
    if "ccas" not in df.columns:
        df["ccas"] = df["ccs"] - df["nnccs"]

    # Save calibrated CSV
    os.makedirs(outdir, exist_ok=True)
    output_csv = os.path.join(outdir, "calibrated_labels.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved calibrated labels to {output_csv}")

    # Save calibration model
    model_path = os.path.join(outdir, "calibration_model.pkl")
    joblib.dump(cal_clf, model_path)
    print(f"Saved calibration model to {model_path}")

    # Compute Brier score
    y_pred = np.concatenate([df["cal_ccs"].values, df["cal_nnccs"].values])
    brier = brier_score_loss(y, y_pred)
    print(f"Brier Score (lower is better): {brier:.4f}")

    # Save plots if requested
    if save_plots:
        print("Generating calibration plots...")
        _save_calibration_plots(df, y, y_pred, brier, outdir)

    return df, cal_clf, brier


def _save_calibration_plots(df, y, y_pred, brier, outdir):
    """Generate and save calibration diagnostic plots.

    Produces three diagnostic plots as shown in Figure 4 of the paper:
    1. Raw CCS/NNCCS distributions (before calibration)
    2. Calibrated probability distributions (after sigmoid calibration)
    3. Reliability diagram (calibration curve) with Brier score
    """

    # Plot 1: Distribution of raw scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["ccs"], bins=30, alpha=0.6, label="CCS (Matches)", color="blue")
    ax.hist(df["nnccs"], bins=30, alpha=0.6, label="NNCCS (Non-matches)", color="red")
    ax.set_xlabel("Raw Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Raw Cosine Similarities (Before Calibration)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "raw_distributions.png"), dpi=150)
    plt.close()

    # Plot 2: Distribution of calibrated probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        df["cal_ccs"],
        bins=30,
        alpha=0.6,
        label="Calibrated CCS (Matches)",
        color="blue",
    )
    ax.hist(
        df["cal_nnccs"],
        bins=30,
        alpha=0.6,
        label="Calibrated NNCCS (Non-matches)",
        color="red",
    )
    ax.set_xlabel("Calibrated Probability")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Calibrated Probabilities (After Calibration)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibrated_distributions.png"), dpi=150)
    plt.close()

    # Plot 3: Reliability curve (calibration curve)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y, y_pred, n_bins=10
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        linewidth=2,
        label="Sigmoid Calibrated",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Reliability Diagram (Brier Score: {brier:.4f})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reliability_curve.png"), dpi=150)
    plt.close()

    print(f"Saved plots to {outdir}")


def apply_calibration(csv_path, model_path, outdir, save_plots=True):
    """Apply a saved calibration model to new data (validation/test sets).

    Uses a calibration model trained on the training set to transform validation
    or test CCS/NNCCS scores. This ensures consistent calibration across splits.

    Args:
        csv_path: Path to CSV file with 'ccs' and 'nnccs' columns
        model_path: Path to saved calibration model (.pkl from calibrate_scores)
        outdir: Directory to save calibrated CSV and plots
        save_plots: Whether to save calibration diagnostic plots

    Returns:
        df: DataFrame with calibrated scores added (cal_ccs, cal_nnccs, cal_ccas)
        brier: Brier score on calibrated predictions for evaluation
    """
    # Load data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    if "ccs" not in df.columns or "nnccs" not in df.columns:
        raise ValueError("Input CSV must include columns 'ccs' and 'nnccs'.")

    # Load calibration model
    print(f"Loading calibration model from {model_path}")
    cal_clf = joblib.load(model_path)

    # Apply calibration
    print("Applying calibration to CCS and NNCCS...")
    ccs = df["ccs"].values
    nnccs = df["nnccs"].values

    df["cal_ccs"] = cal_clf.predict_proba(ccs.reshape(-1, 1))[:, 1]
    df["cal_nnccs"] = cal_clf.predict_proba(nnccs.reshape(-1, 1))[:, 1]
    df["cal_ccas"] = df["cal_ccs"] - df["cal_nnccs"]

    # For reference, also compute raw CCAS if not present
    if "ccas" not in df.columns:
        df["ccas"] = df["ccs"] - df["nnccs"]

    # Save calibrated CSV
    os.makedirs(outdir, exist_ok=True)
    output_csv = os.path.join(outdir, "calibrated_labels.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved calibrated labels to {output_csv}")

    # Compute Brier score for evaluation
    y = np.concatenate([np.ones(len(ccs)), np.zeros(len(nnccs))])
    y_pred = np.concatenate([df["cal_ccs"].values, df["cal_nnccs"].values])
    brier = brier_score_loss(y, y_pred)
    print(f"Brier Score (lower is better): {brier:.4f}")

    # Save plots if requested
    if save_plots:
        print("Generating calibration plots...")
        _save_calibration_plots(df, y, y_pred, brier, outdir)

    return df, brier


def main():
    """Main calibration script."""
    parser = argparse.ArgumentParser(
        description="Sigmoid calibration for TransFIRA CCS and NNCCS scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train calibration model on training data
  python calibrate.py --csv train_labels.csv --outdir ./calibration_output/

  # Apply saved calibration model to test data
  python calibrate.py --csv test_labels.csv --outdir ./test_calibrated/ \\
                      --model ./calibration_output/calibration_model.pkl

  # Train without saving plots
  python calibrate.py --csv train_labels.csv --outdir ./calibration_output/ --no-plots
        """,
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to CSV file with ccs and nnccs columns"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Output directory for calibrated files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to saved calibration model (if applying existing model)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating calibration plots",
    )

    args = parser.parse_args()
    save_plots = not args.no_plots

    if args.model is None:
        # Train new calibration model
        print("=" * 80)
        print("TRAINING NEW CALIBRATION MODEL")
        print("=" * 80)
        df, cal_clf, brier = calibrate_scores(args.csv, args.outdir, save_plots)
    else:
        # Apply existing calibration model
        print("=" * 80)
        print("APPLYING SAVED CALIBRATION MODEL")
        print("=" * 80)
        df, brier = apply_calibration(args.csv, args.model, args.outdir, save_plots)

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Output directory: {args.outdir}")
    print(f"Brier Score: {brier:.4f}")


if __name__ == "__main__":
    main()
