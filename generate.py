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
from sklearn.metrics.pairwise import cosine_similarity

# Local modules
from train_utils.torch_utils import ImageSet
from models import get_backbone

"""
TransFIRA feature generation and recognizability label computation script.

Extracts backbone features and computes geometry-derived recognizability labels
directly from embedding space (no human annotations required):

- CCS (Class Center Similarity): Equation 4 - Cosine similarity to correct class center
- NNCCS (Nearest Nonmatch Class Center Similarity): Equation 5 - Max similarity to impostor centers
- CCAS (Class Center Angular Separation): Equation 6 - Decision boundary margin (CCS - NNCCS)

These labels serve as training targets for the recognizability prediction network.

Usage:
    python generate.py --config path/to/config.toml
"""


def generate_features(model, dataset, name, cfg, feature_dim, save_features=True):
    """Generate features for a single dataset"""
    model.eval()
    num_samples = len(dataset.dataset)
    features = np.empty((num_samples, feature_dim), dtype=np.float32)

    idx = 0
    train_tqdm = tqdm(dataset, desc=f"{name}", leave=False, total=len(dataset))
    for inputs in train_tqdm:
        inputs = inputs.cuda()
        batch_size = inputs.shape[0]
        with torch.no_grad():
            predictions = model(inputs).cpu().numpy()
        features[idx : idx + batch_size] = predictions
        idx += batch_size

    assert idx == num_samples, f"Expected {num_samples} samples, processed {idx}"

    if save_features:
        os.makedirs(cfg["outdir"], exist_ok=True)
        predictions_path = os.path.join(cfg["outdir"], f"{name}_features.npy")
        np.save(predictions_path, features)

    return features


def generate_centers(
    annotations_file,
    features,
    outdir,
    name,
    subject_col_name="subject_id",
):
    """Compute class centers for each subject_id.

    For gallery-probe datasets: Equation 2 (centers from gallery only)
    For non-gallery datasets: Equation 3 (centers from all samples per identity)
    """
    df = pd.read_csv(annotations_file)
    assert len(df) == len(
        features
    ), f"CSV has {len(df)} rows but features has {len(features)} samples"

    subject_ids = df[subject_col_name].values
    centers = {}

    for subject_id in np.unique(subject_ids):
        mask = subject_ids == subject_id
        subject_features = features[mask]
        centers[subject_id] = np.mean(subject_features, axis=0)

    os.makedirs(outdir, exist_ok=True)
    centers_path = os.path.join(outdir, f"{name}_centers.npy")
    np.save(centers_path, centers)

    return centers


def generate_labels(
    subject_ids,
    features,
    centers,
    outdir,
    name,
):
    """Compute geometry-derived recognizability labels directly from embedding space.

    - CCS (Class Center Similarity): Equation 4 - Cosine similarity to correct class center
    - NNCCS (Nearest Nonmatch Class Center Similarity): Equation 5 - Max similarity to impostor centers

    These labels serve as training targets for the recognizability prediction network.
    """
    assert len(subject_ids) == len(
        features
    ), f"subject_ids has {len(subject_ids)} entries but features has {len(features)} samples"

    num_samples = len(features)
    ccs = np.zeros(num_samples, dtype=np.float32)
    nnccs = np.zeros(num_samples, dtype=np.float32)

    for idx in range(num_samples):
        feature = features[idx : idx + 1]
        subject_id = subject_ids[idx]

        matching_center = centers[subject_id].reshape(1, -1)
        ccs[idx] = cosine_similarity(feature, matching_center)[0, 0]

        max_nonmatching_sim = -1.0
        for other_id, other_center in centers.items():
            if other_id != subject_id:
                other_center_reshaped = other_center.reshape(1, -1)
                sim = cosine_similarity(feature, other_center_reshaped)[0, 0]
                max_nonmatching_sim = max(max_nonmatching_sim, sim)

        nnccs[idx] = max_nonmatching_sim

    os.makedirs(outdir, exist_ok=True)
    ccs_path = os.path.join(outdir, f"{name}_ccs.npy")
    nnccs_path = os.path.join(outdir, f"{name}_nnccs.npy")
    np.save(ccs_path, ccs)
    np.save(nnccs_path, nnccs)

    return ccs, nnccs


def get_model(cfg):
    """
    Load the feature extraction backbone model.

    Returns the backbone model, feature dimension, and updated config with transforms.
    """
    model, outdim, cfg = get_backbone(cfg)
    model.to(cfg["local_rank"])
    return model, outdim, cfg


def make_dataloader(cfg):
    """
    Create dataloaders for all configured datasets.

    Processes gallery, probe, and standalone datasets defined in config.
    """
    dataloader = {}
    for dataset_name in [
        "train",
        "val",
    ]:
        if cfg[dataset_name]["annotations"] != "":
            dataset = ImageSet(
                cfg[dataset_name]["data_dir"],
                cfg[dataset_name]["annotations"],
                transform=cfg["xform"],
                path_col_name=cfg[dataset_name]["path_col_name"],
            )
            dataloader["train" if dataset_name == "data" else dataset_name] = (
                DataLoader(
                    dataset,
                    batch_size=cfg["batch_size"],
                    sampler=None,
                    num_workers=8,
                    pin_memory=True,
                )
            )
    return dataloader


def main():
    """Generate features and compute CCS/NNCCS/CCAS labels for datasets."""
    parser = argparse.ArgumentParser(
        description="TransFIRA Label Generation: Extract features and compute recognizability labels"
    )
    parser.add_argument("--config", type=str, help="Path to the .toml config file")

    args = parser.parse_args()
    cfg = Box(toml.load(args.config))
    cfg["local_rank"] = 0

    if "save_features" not in cfg:
        cfg["save_features"] = False

    print("Initializing model")
    model, feature_dim, cfg = get_model(cfg)

    print("Generating dataloader")
    dataloader = make_dataloader(cfg)

    all_features = {}
    all_centers = {}

    for name, dataset in dataloader.items():
        print(f"\n=== Processing dataset: {name} ===")

        dataset_key = name.replace("train", "data") if name == "train" else name
        annotations_file = cfg[dataset_key]["annotations"]

        print(f"1. Generating features for {name}")
        features = generate_features(
            model, dataset, name, cfg, feature_dim, save_features=cfg["save_features"]
        )
        all_features[name] = features

        if "_gallery" in name:
            centers_path = cfg[dataset_key].get("centers_path", None)

            if centers_path is not None and centers_path != "":
                print(f"2. Loading pre-computed centers from {centers_path}")
                all_centers[name] = np.load(centers_path, allow_pickle=True).item()
            else:
                print(f"2. Computing centers for {name}")
                all_centers[name] = generate_centers(
                    annotations_file=annotations_file,
                    features=features,
                    outdir=cfg["outdir"],
                    name=name,
                )

        elif "_probes" in name:
            gallery_name = name.replace("_probes", "_gallery")
            centers_path = cfg[dataset_key].get("centers_path", None)

            if centers_path is not None and centers_path != "":
                print(f"2. Loading pre-computed centers from {centers_path}")
                all_centers[name] = np.load(centers_path, allow_pickle=True).item()
            elif os.path.exists(
                os.path.join(cfg["outdir"], f"{gallery_name}_centers.npy")
            ):
                saved_centers_path = os.path.join(
                    cfg["outdir"], f"{gallery_name}_centers.npy"
                )
                print(f"2. Loading saved centers from {saved_centers_path}")
                all_centers[name] = np.load(
                    saved_centers_path, allow_pickle=True
                ).item()
            elif gallery_name in all_centers:
                print(f"2. Using centers from {gallery_name}")
                all_centers[name] = all_centers[gallery_name]
            else:
                raise ValueError(
                    f"No centers available for {name}. Expected gallery '{gallery_name}' to be processed first or centers_path to be specified."
                )

            print(f"3. Computing CCS (Eq. 4) and NNCCS (Eq. 5) for {name}")
            df = pd.read_csv(annotations_file)
            subject_ids = df["subject_id"].values
            ccs, nnccs = generate_labels(
                subject_ids=subject_ids,
                features=features,
                centers=all_centers[name],
                outdir=cfg["outdir"],
                name=name,
            )

            print(f"4. Saving updated CSV for {name}")
            df["ccs"] = ccs
            df["nnccs"] = nnccs
            df["ccas"] = (
                ccs - nnccs
            )  # CCAS (Class Center Angular Separation) - Equation 6
            os.makedirs(cfg["outdir"], exist_ok=True)
            output_csv_path = os.path.join(cfg["outdir"], f"{name}_labels.csv")
            df.to_csv(output_csv_path, index=False)
            print(f"   Saved to {output_csv_path}")

        else:
            centers_path = cfg[dataset_key].get("centers_path", None)

            if centers_path is not None and centers_path != "":
                print(f"2. Loading pre-computed centers from {centers_path}")
                all_centers[name] = np.load(centers_path, allow_pickle=True).item()
            else:
                print(f"2. Computing centers for {name}")
                all_centers[name] = generate_centers(
                    annotations_file=annotations_file,
                    features=features,
                    outdir=cfg["outdir"],
                    name=name,
                )

            print(f"3. Computing CCS (Eq. 4) and NNCCS (Eq. 5) for {name}")
            df = pd.read_csv(annotations_file)
            subject_ids = df["subject_id"].values
            ccs, nnccs = generate_labels(
                subject_ids=subject_ids,
                features=features,
                centers=all_centers[name],
                outdir=cfg["outdir"],
                name=name,
            )

            print(f"4. Saving updated CSV for {name}")
            df["ccs"] = ccs
            df["nnccs"] = nnccs
            df["ccas"] = (
                ccs - nnccs
            )  # CCAS (Class Center Angular Separation) - Equation 6
            os.makedirs(cfg["outdir"], exist_ok=True)
            output_csv_path = os.path.join(cfg["outdir"], f"{name}_labels.csv")
            df.to_csv(output_csv_path, index=False)
            print(f"   Saved to {output_csv_path}")


if __name__ == "__main__":
    main()
