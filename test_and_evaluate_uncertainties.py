"""Evaluate robustness to image uncertainties using the maintained pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

import model_
from Preprocessing import Create_Testing_Datasets_uncertainties
from test_and_evaluate import VIEW_PAIRS, evaluate_retrieval, extract_feature
from utils import get_best_weight, get_device, get_id, get_yaml_value


def _load_embedding_model(params, checkpoint_path, device):
    if not checkpoint_path:
        raise FileNotFoundError("No best checkpoint was found for the requested query and height")
    model = model_.model_dict[params["model"]](
        params["classes"], params["drop_rate"], pretrained=False
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.classifier.classifier = nn.Identity()
    return model.eval().to(device)


def eval_and_test(cfg_path, types, heights, csv_dir_path):
    params = get_yaml_value(cfg_path)
    device = get_device(params.get("device"))
    csv_root = Path(csv_dir_path) / params["model"]
    csv_root.mkdir(parents=True, exist_ok=True)
    use_amp = bool(params.get("eval_amp", False)) and device.type == "cuda"

    for uncertainty_type in types:
        rows = {}
        for height in heights:
            test_path = Path(params["dataset_path"]) / "Testing" / str(height)
            _, loaders = Create_Testing_Datasets_uncertainties(
                test_data_path=str(test_path),
                batch_size=params["batch_size"],
                image_size=params["image_size"],
                gap=params.get("uncertainty_gap", 50),
                type=uncertainty_type,
                num_workers=params.get("num_workers", 4),
                pin_memory=params.get("pin_memory", device.type == "cuda"),
                prefetch_factor=params.get("prefetch_factor", 2),
            )
            metrics = []
            for query_name, gallery_name in VIEW_PAIRS:
                checkpoint_path = get_best_weight(
                    query_name, params["model"], height, params["weight_save_path"]
                )
                model = _load_embedding_model(params, checkpoint_path, device)
                started = time.time()
                query_features = extract_feature(
                    model,
                    loaders[query_name],
                    which_view=1 if "satellite" in query_name else 2,
                    device=device,
                    use_amp=use_amp,
                ).to(device)
                gallery_features = extract_feature(
                    model,
                    loaders[gallery_name],
                    which_view=1 if "satellite" in gallery_name else 2,
                    device=device,
                    use_amp=use_amp,
                ).to(device)
                query_labels, _ = get_id(loaders[query_name].dataset.imgs)
                gallery_labels, _ = get_id(loaders[gallery_name].dataset.imgs)
                cmc, ap, _ = evaluate_retrieval(
                    query_features,
                    query_labels,
                    gallery_features,
                    gallery_labels,
                    dist="Cos",
                    chunk_size=params.get("eval_chunk_size", 256),
                    device=device,
                )
                elapsed = time.time() - started
                top1p = min(round(len(gallery_labels) * 0.01), len(gallery_labels) - 1)
                metrics.extend([float(cmc[0] * 100), float(ap * 100)])
                print(
                    f"height={height} type={uncertainty_type} "
                    f"{query_name}->{gallery_name} "
                    f"Recall@1={cmc[0] * 100:.2f} "
                    f"Recall@1%={cmc[top1p] * 100:.2f} AP={ap * 100:.2f} "
                    f"Time={elapsed:.2f}s"
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows[height] = metrics

        table = pd.DataFrame(
            rows,
            index=[
                f"Drecall@1_{uncertainty_type}",
                f"DAP_{uncertainty_type}",
                f"Srecall@1_{uncertainty_type}",
                f"SAP_{uncertainty_type}",
            ],
        )
        table.index.name = "index"
        table.to_csv(csv_root / f"{params['model']}_{uncertainty_type}.csv")


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="settings.yaml")
    parser.add_argument("--types", nargs="+", default=["rain"])
    parser.add_argument("--heights", nargs="+", type=int, default=[150])
    parser.add_argument("--csv_save_path", type=str, default="./result")
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    options = parse_opt(True)
    eval_and_test(options.cfg, options.types, options.heights, options.csv_save_path)
