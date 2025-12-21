# Third party modules
import os
import toml
import json
import torch
import argparse
import numpy as np
from box import Box
import torch.distributed
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from datetime import datetime

# Local modules
from train_utils.dist_utils import init_distributed
from train_utils.data_utils import convert_posixpath_to_string
from train_utils.torch_utils import ImageRecognizabilitySet

from models import get_backbone
from models.ScorePredictor import RecognizabilityPredictionNetwork

"""
TransFIRA dual-head recognizability prediction training script.

Trains a lightweight MLP head to jointly predict CCS and CCAS from images.
Uses MSE loss (Equation 9) for both CCS and CCAS predictions.
Both backbone and prediction heads are trained end-to-end.

Usage:
    torchrun --nproc_per_node=<num_gpus> train.py --config path/to/config.toml

Config Requirements:
    - ccs_col_name: CCS column (e.g., 'ccs')
    - ccas_col_name: CCAS column (e.g., 'ccas')
"""


def train(
    model,
    dataloader,
    loss_function,
    optimizer,
    scheduler,
    writer,
    cfg,
):
    """Main Training Code"""

    for epoch in tqdm(range(cfg["opt"]["num_epochs"]), desc="Training Epoch"):
        torch.distributed.barrier()

        dataloader["train"].sampler.set_epoch(epoch)
        batch_losses = []
        running_loss = 0
        img_ctr = 0

        model.train()
        optimizer.zero_grad()

        print(f"Starting epoch {epoch}")
        train_tqdm = tqdm(
            dataloader["train"],
            desc="Train Step",
            leave=False,
            total=len(dataloader["train"]),
        )
        for inputs, labels, ccas in train_tqdm:
            with torch.set_grad_enabled(True):
                inputs = inputs.cuda()
                labels = labels[:, None].float().cuda()
                ccas = ccas[:, None].float().cuda()
                predictions, ccas_predictions = model(inputs)
            predictions_loss = loss_function(predictions, labels)
            ccas_loss = loss_function(ccas_predictions, ccas)
            loss = predictions_loss + ccas_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_tqdm.set_postfix(loss=loss.item())
            running_loss += loss.item()
            batch_losses.append(loss.item())
            img_ctr += inputs.size(0)

        epoch_lr = scheduler.get_last_lr()[0]
        epoch_losses = np.mean(np.array(batch_losses), axis=0)
        writer.add_scalar("epoch_lr", epoch_lr, epoch)
        writer.add_scalar("epoch_loss", epoch_losses, epoch)

        scheduler.step()

        torch.distributed.barrier()
        if epoch % cfg["save_per_epoch"] == 0:
            do_validation(dataloader, model, cfg, writer, epoch)
            filename = f'{Path(cfg["model_path"]).stem}_{epoch:04}.pt'
            out_path = cfg["checkpoint_outdir"] / filename
            torch.save(model.module.state_dict(), out_path)
            print(f"Saved model to {out_path}")

    return


def do_validation(dataloader, model, cfg, writer, epoch):
    model.eval()

    num_samples = len(dataloader["val"].dataset)
    validation_predictions = np.empty(num_samples, dtype=np.float32)
    validation_ccas_predictions = np.empty(num_samples, dtype=np.float32)
    validation_labels = np.empty(num_samples, dtype=np.float32)
    validation_ccas = np.empty(num_samples, dtype=np.float32)

    idx = 0
    with torch.no_grad():
        for inputs, labels, ccas in tqdm(
            dataloader["val"], desc="Val Step", leave=False
        ):
            inputs = inputs.cuda()
            labels = labels[:, None].float().cuda()
            ccas = ccas[:, None].float().cuda()
            batch_size = inputs.shape[0]

            predictions, ccas_predictions = model(inputs)

            validation_predictions[idx : idx + batch_size] = (
                predictions.cpu().numpy().flatten()
            )
            validation_ccas_predictions[idx : idx + batch_size] = (
                ccas_predictions.cpu().numpy().flatten()
            )
            validation_labels[idx : idx + batch_size] = labels.cpu().numpy().flatten()
            validation_ccas[idx : idx + batch_size] = ccas.cpu().numpy().flatten()
            idx += batch_size

    # Mean Absolute Error (MAE)
    mae_ccs = np.mean(np.abs(validation_predictions - validation_labels))
    mae_ccas = np.mean(np.abs(validation_ccas_predictions - validation_ccas))

    # Pearson and Spearman Correlation Coefficients
    pearson_corr, _ = pearsonr(validation_predictions, validation_labels)
    spearman_corr, _ = spearmanr(validation_predictions, validation_labels)
    pearson_corr_ccas, _ = pearsonr(validation_ccas_predictions, validation_ccas)
    spearman_corr_ccas, _ = spearmanr(validation_ccas_predictions, validation_ccas)

    writer.add_scalar("Validation CCS L1", mae_ccs, epoch)
    writer.add_scalar("Validation CCAS L1", mae_ccas, epoch)
    writer.add_scalar("Validation CCS Pearson Corr", pearson_corr, epoch)
    writer.add_scalar("Validation CCS Spearman Corr", spearman_corr, epoch)
    writer.add_scalar("Validation CCAS Pearson Corr", pearson_corr_ccas, epoch)
    writer.add_scalar("Validation CCAS Spearman Corr", spearman_corr_ccas, epoch)
    print(f"Validation CCS L1: {mae_ccs}")
    print(f"Validation CCAS L1: {mae_ccas}")
    print(f"Validation CCS Pearson Corr: {pearson_corr}")
    print(f"Validation CCS Spearman Corr: {spearman_corr}")
    print(f"Validation CCAS Pearson Corr: {pearson_corr_ccas}")
    print(f"Validation CCAS Spearman Corr: {spearman_corr_ccas}")

    return


def prepare_environment(cfg):
    """
    Prepare the environment and pre-compute objects necessary for training
    """
    init_distributed(dist_url=cfg["env"]["dist_url"])
    cfg["local_rank"] = int(os.environ["LOCAL_RANK"])
    print(f"Local rank: {cfg['local_rank']}")

    time = datetime.now().strftime("%m%d%H%M%S")
    cfg["run_name"] = f"{time}"

    checkpoint_dir = Path(cfg["checkpoint_outdir"])
    cfg["log_dir"] = checkpoint_dir.joinpath("runs", cfg["run_name"])
    checkpoint_dir = checkpoint_dir.joinpath(cfg["run_name"])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    cfg["checkpoint_outdir"] = checkpoint_dir
    with open(checkpoint_dir / "run_params.json", "w", encoding="utf-8") as json_file:
        json.dump(convert_posixpath_to_string(cfg), json_file, indent=4)
    return cfg


def get_model(cfg):
    """
    Returns a model and transform consistent with the config options
    """
    model, outdim, cfg = get_backbone(cfg)

    model = RecognizabilityPredictionNetwork(cfg["local_rank"], model, outdim).to(
        cfg["local_rank"]
    )

    model = DistributedDataParallel(
        model,
        device_ids=[cfg["local_rank"]],
        output_device=cfg["local_rank"],
        find_unused_parameters=True,
    )

    return model, cfg


def make_dataloader(cfg):
    """
    Returns a dataloader consistent with the config options and annotation file
    """
    train_set = ImageRecognizabilitySet(
        cfg["data"]["data_dir"],
        cfg["data"]["annotations"],
        transform=cfg["xform"],
        path_col_name=cfg["data"]["path_col_name"],
        ccs_col_name=cfg["data"]["ccs_col_name"],
        ccas_col_name=cfg["data"]["ccas_col_name"],
    )
    val_set = ImageRecognizabilitySet(
        cfg["val"]["data_dir"],
        cfg["val"]["annotations"],
        transform=cfg["xform"],
        path_col_name=cfg["val"]["path_col_name"],
        ccs_col_name=cfg["val"]["ccs_col_name"],
        ccas_col_name=cfg["val"]["ccas_col_name"],
    )

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    dataloader = {
        "train": DataLoader(
            train_set,
            batch_size=cfg["data"]["batch_size"],
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_set,
            batch_size=cfg["val"]["batch_size"],
            sampler=None,
            num_workers=8,
            pin_memory=True,
        ),
    }
    return dataloader


def get_optimizer(cfg, model):
    """
    Returns an optimizer, scheduler, and loss function consistent with the config options.

    Uses MSE loss for both CCS and CCAS predictions.
    """
    if cfg["opt"]["optimizer"] == "SGD":
        optimizer_class = torch.optim.SGD
    elif cfg["opt"]["optimizer"] == "Adam":
        optimizer_class = torch.optim.Adam
    elif cfg["opt"]["optimizer"] == "AdamW":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError("Invalid optimizer type")
    optimizer = optimizer_class(
        list(model.parameters()),
        lr=cfg["opt"]["lr"],
        weight_decay=cfg["opt"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg["opt"]["milestones"], gamma=cfg["opt"]["gamma"]
    )

    if cfg["opt"]["loss_func"] == "L1":
        loss_func = nn.L1Loss()
    if cfg["opt"]["loss_func"] == "MSE":
        loss_func = nn.MSELoss()
    else:
        raise ValueError("Invalid loss function type")

    return optimizer, scheduler, loss_func


def main():
    """Main Training Code"""
    parser = argparse.ArgumentParser(description="PyTorch Model Training")
    parser.add_argument("--config", type=str, help="Path to the .toml config file")

    args = parser.parse_args()
    cfg = Box(toml.load(args.config))

    print("Launching Distributed Training Environment")
    cfg = prepare_environment(cfg)

    print("Initializing model")
    model, cfg = get_model(cfg)

    print("Generating dataloader")
    dataloader = make_dataloader(cfg)

    print("Starting training")
    optimizer, scheduler, loss_func = get_optimizer(cfg, model)

    # Set up TensorBoard
    writer = SummaryWriter(log_dir=cfg["log_dir"])

    train(
        model,
        dataloader,
        loss_func,
        optimizer,
        scheduler,
        writer,
        cfg,
    )

    writer.close()


if __name__ == "__main__":
    main()
