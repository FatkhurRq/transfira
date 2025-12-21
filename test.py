# Third party modules
import os
import toml
import torch
import argparse
import numpy as np
import pandas as pd
from box import Box
from tqdm import tqdm
from torch.utils.data import DataLoader

# Local modules
from train_utils.torch_utils import ImageSet, image_pipeline

from models import get_backbone
from models.ScorePredictor import RecognizabilityPredictionNetwork

"""
TransFIRA testing and aggregation script with filtering and weighting.

Implements recognizability-informed template aggregation (Section III-C):
1. Generate features using pretrained backbone
2. Predict CCS and CCAS scores using trained predictor
3. Aggregate templates with optional filtering (CCAS > 0) and weighting (CCS)

The CCAS > 0 filter provides the first principled, parameter-free cutoff for FIQA,
removing samples on the wrong side of the decision boundary. CCS-based weighting
(Equation 10) emphasizes reliable embeddings within each template.

Usage:
    python test.py --config config.toml --generate_features \\
                   --predict_scores --aggregate_features \\
                   --use_filter --use_weight
"""


def generate_features(model, dataloader, cfg, feature_dim):
    """Generate and save features using the backbone."""
    model.eval()
    num_samples = len(dataloader.dataset)
    features = np.empty((num_samples, feature_dim), dtype=np.float32)

    idx = 0
    for inputs in tqdm(dataloader, desc="Generating features", leave=False):
        inputs = inputs.cuda()
        batch_size = inputs.shape[0]
        with torch.no_grad():
            predictions = model(inputs).cpu().numpy()
        features[idx : idx + batch_size] = predictions
        idx += batch_size

    assert idx == num_samples, f"Expected {num_samples} samples, processed {idx}"

    os.makedirs(cfg["outdir"], exist_ok=True)
    features_path = os.path.join(cfg["outdir"], "test_features.npy")
    np.save(features_path, features)
    print(f"Saved {len(features)} features to {features_path}")

    return features


def predict_scores(model, dataloader, cfg):
    """Predict and save CCS and CCAS scores using the score predictor."""
    model.eval()
    num_samples = len(dataloader.dataset)

    ccs_scores = np.empty(num_samples, dtype=np.float32)
    ccas_scores = np.empty(num_samples, dtype=np.float32)

    idx = 0
    for inputs in tqdm(dataloader, desc="Predicting scores", leave=False):
        inputs = inputs.cuda()
        batch_size = inputs.shape[0]
        with torch.no_grad():
            ccs, ccas = model(inputs)
        ccs_scores[idx : idx + batch_size] = ccs.cpu().numpy().flatten()
        ccas_scores[idx : idx + batch_size] = ccas.cpu().numpy().flatten()
        idx += batch_size

    assert idx == num_samples, f"Expected {num_samples} samples, processed {idx}"

    os.makedirs(cfg["outdir"], exist_ok=True)
    ccs_path = os.path.join(cfg["outdir"], "test_ccs.npy")
    ccas_path = os.path.join(cfg["outdir"], "test_ccas.npy")
    np.save(ccs_path, ccs_scores)
    np.save(ccas_path, ccas_scores)
    print(f"Saved {len(ccs_scores)} CCS and CCAS scores to {cfg['outdir']}")

    return ccs_scores, ccas_scores


def aggregate_features(
    cfg, features, ccs_scores, ccas_scores, use_weight=False, use_filter=False
):
    """Construct and save templates for each subject using TransFIRA aggregation."""
    print("Aggregating features...")
    print(f"Processing {len(features)} features")
    print(f"Use weighting: {use_weight}, Use filtering: {use_filter}")

    df = pd.read_csv(cfg["data"]["annotations"])
    assert len(df) == len(
        features
    ), f"CSV has {len(df)} rows but features has {len(features)} samples"

    template_ids = df[cfg["data"]["id_col_name"]].values
    unique_template_ids = np.unique(template_ids)

    print(f"Found {len(unique_template_ids)} unique templates")
    if len(unique_template_ids) == 0:
        raise ValueError("No templates found in annotations")

    templates = {}

    for template_id in tqdm(unique_template_ids, desc="Aggregating templates"):
        template_mask = template_ids == template_id
        template_features = features[template_mask]
        template_ccs = ccs_scores[template_mask]
        template_ccas = ccas_scores[template_mask]

        if use_filter:
            filter_mask = template_ccas > 0
            if np.any(filter_mask):
                template_features = template_features[filter_mask]
                template_ccs = template_ccs[filter_mask]
                template_ccas = template_ccas[filter_mask]
            else:
                print(
                    f"Warning: Template {template_id} has no positive CCAS scores, keeping all samples"
                )

        if use_weight:
            weights = template_ccs[:, np.newaxis]
            template = np.sum(template_features * weights, axis=0) / np.sum(weights)
        else:
            template = np.mean(template_features, axis=0)

        templates[template_id] = template

    method_suffix = (
        "_filter_weight"
        if use_filter and use_weight
        else "_filter" if use_filter else "_weight" if use_weight else "_baseline"
    )

    templates_path = os.path.join(cfg["outdir"], f"templates{method_suffix}.npy")
    np.save(templates_path, templates)
    print(f"Saved {len(templates)} templates to {templates_path}")

    return templates


def make_dataloader(cfg):
    """Create dataloader from config."""
    test_set = ImageSet(
        cfg["data"]["data_dir"],
        cfg["data"]["annotations"],
        transform=cfg["xform"],
        path_col_name=cfg["data"]["path_col_name"],
    )
    return DataLoader(
        test_set, batch_size=cfg["data"]["batch_size"], num_workers=8, pin_memory=True
    )


def get_model(cfg, load_score_predictor=False):
    """Load model from config, optionally with score predictor."""
    model, outdim, cfg = get_backbone(cfg)

    if load_score_predictor:
        model = RecognizabilityPredictionNetwork(cfg["local_rank"], model, outdim).to(
            cfg["local_rank"]
        )

        checkpoint = torch.load(
            cfg["checkpoint_path"],
            map_location=lambda storage, loc: storage.cuda(cfg["local_rank"]),
        )
        model.load_state_dict(checkpoint)
        print(f"Loaded score predictor from {cfg['checkpoint_path']}")

    return model, outdim, cfg


def main():
    """Main Testing Code"""
    parser = argparse.ArgumentParser(
        description="TransFIRA Testing: Generate features, predict scores, and aggregate templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Generate features
  python test.py --config config.toml --generate_features

  # Step 2: Predict recognizability scores
  python test.py --config config.toml --predict_scores

  # Step 3: Aggregate with filtering and weighting (Section III-C in paper)
  python test.py --config config.toml --aggregate_features --use_filter --use_weight

  # Run all steps at once
  python test.py --config config.toml --generate_features --predict_scores --aggregate_features --use_filter --use_weight

  # Aggregate using previously saved features/scores from a different location
  python test.py --config config.toml --aggregate_features --use_filter --use_weight \\
    --features_path /path/to/features.npy \\
    --ccs_path /path/to/ccs.npy \\
    --ccas_path /path/to/ccas.npy
        """,
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the .toml config file"
    )
    parser.add_argument(
        "--generate_features",
        action="store_true",
        help="Generate and save features using the backbone",
    )
    parser.add_argument(
        "--predict_scores",
        action="store_true",
        help="Predict and save CCS and CCAS scores using the score predictor",
    )
    parser.add_argument(
        "--aggregate_features",
        action="store_true",
        help="Construct and save templates for each subject using saved features and scores",
    )
    parser.add_argument(
        "--use_weight",
        action="store_true",
        help="Use CCS-based weighting when aggregating (Equation 10 in paper)",
    )
    parser.add_argument(
        "--use_filter",
        action="store_true",
        help="Use CCAS > 0 filtering when aggregating (Section III-C in paper)",
    )

    # Paths for loading previously saved features/scores
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Path to previously saved features file (default: <outdir>/test_features.npy)",
    )
    parser.add_argument(
        "--ccs_path",
        type=str,
        default=None,
        help="Path to previously saved CCS scores file (default: <outdir>/test_ccs.npy)",
    )
    parser.add_argument(
        "--ccas_path",
        type=str,
        default=None,
        help="Path to previously saved CCAS scores file (default: <outdir>/test_ccas.npy)",
    )

    args = parser.parse_args()
    cfg = Box(toml.load(args.config))
    cfg["local_rank"] = 0

    features, ccs_scores, ccas_scores = None, None, None

    # Step 1: Generate features
    if args.generate_features:
        print("\n" + "=" * 80)
        print("STEP 1: Generating Features")
        print("=" * 80)
        model, outdim, cfg = get_model(cfg, load_score_predictor=False)
        dataloader = make_dataloader(cfg)
        features = generate_features(model, dataloader, cfg, outdim)
        del model
        torch.cuda.empty_cache()

    # Step 2: Predict scores
    if args.predict_scores:
        print("\n" + "=" * 80)
        print("STEP 2: Predicting Recognizability Scores")
        print("=" * 80)
        model, _, cfg = get_model(cfg, load_score_predictor=True)
        dataloader = make_dataloader(cfg)
        ccs_scores, ccas_scores = predict_scores(model, dataloader, cfg)
        del model
        torch.cuda.empty_cache()

    # Step 3: Aggregate features
    if args.aggregate_features:
        print("\n" + "=" * 80)
        print("STEP 3: Aggregating Features into Templates")
        print("=" * 80)

        # Load features and scores if not already in memory
        if features is None:
            features_path = args.features_path or os.path.join(
                cfg["outdir"], "test_features.npy"
            )
            print(f"Loading features from {features_path}")
            if not os.path.exists(features_path):
                raise FileNotFoundError(
                    f"Features not found at {features_path}. Run with --generate_features first."
                )
            features = np.load(features_path)

        if ccs_scores is None or ccas_scores is None:
            ccs_path = args.ccs_path or os.path.join(cfg["outdir"], "test_ccs.npy")
            ccas_path = args.ccas_path or os.path.join(cfg["outdir"], "test_ccas.npy")
            print(f"Loading CCS from {ccs_path}")
            print(f"Loading CCAS from {ccas_path}")
            if not os.path.exists(ccs_path) or not os.path.exists(ccas_path):
                raise FileNotFoundError(
                    f"Scores not found. Run with --predict_scores first."
                )
            ccs_scores = np.load(ccs_path)
            ccas_scores = np.load(ccas_path)

        templates = aggregate_features(
            cfg,
            features,
            ccs_scores,
            ccas_scores,
            use_weight=args.use_weight,
            use_filter=args.use_filter,
        )

    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
