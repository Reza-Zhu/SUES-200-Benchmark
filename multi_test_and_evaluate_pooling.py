"""Multi-query pooling evaluation for the SUES-200 drone query."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import model_
from Preprocessing import Create_Testing_Datasets
from test_and_evaluate import evaluate_retrieval, extract_feature
from utils import get_best_weight, get_device, get_id, get_yaml_value


def _load_embedding_model(params, checkpoint_path, device):
    if not checkpoint_path:
        raise FileNotFoundError("No best checkpoint was found for the requested model and height")
    model = model_.model_dict[params["model"]](
        params["classes"], params["drop_rate"], pretrained=False
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.classifier.classifier = nn.Identity()
    return model.eval().to(device)


def _pool_queries(features, labels, multi_coff, pool_type):
    labels = np.asarray(labels).reshape(-1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) == 0 or len(set(counts.tolist())) != 1:
        raise ValueError("Multi-query pooling requires the same number of images per identity")
    if counts[0] % multi_coff != 0:
        raise ValueError("multi must evenly divide the images per identity")
    per_query = counts[0] // multi_coff
    grouped = features.view(len(unique_labels), multi_coff, per_query, -1)
    if pool_type == "max":
        pooled = grouped.max(dim=2).values
    elif pool_type in ("ave", "avg", "mean"):
        pooled = grouped.mean(dim=2)
    else:
        raise ValueError("type must be max or ave")
    return pooled.reshape(-1, features.shape[-1]), np.repeat(unique_labels, multi_coff)


def eval_and_test(multi_coff, config_file, pool_type, save_path):
    params = get_yaml_value(config_file)
    device = get_device(params.get("device"))
    height = params["height"]
    data_path = Path(params["dataset_path"]) / "Testing" / str(height)
    loaders_data, loaders = Create_Testing_Datasets(
        str(data_path),
        params["batch_size"],
        params["image_size"],
        num_workers=params.get("num_workers", 4),
        pin_memory=params.get("pin_memory", device.type == "cuda"),
        prefetch_factor=params.get("prefetch_factor", 2),
    )
    query_name, gallery_name = "query_drone", "gallery_satellite"
    checkpoint_path = get_best_weight(
        query_name, params["model"], height, params["weight_save_path"]
    )
    model = _load_embedding_model(params, checkpoint_path, device)
    started = time.time()
    query_features = extract_feature(
        model,
        loaders[query_name],
        view_index=2,
        device=device,
        use_amp=bool(params.get("eval_amp", False)) and device.type == "cuda",
    )
    gallery_features = extract_feature(
        model,
        loaders[gallery_name],
        view_index=1,
        device=device,
        use_amp=bool(params.get("eval_amp", False)) and device.type == "cuda",
    )
    query_labels, _ = get_id(loaders_data[query_name].imgs)
    gallery_labels, _ = get_id(loaders_data[gallery_name].imgs)
    query_features, query_labels = _pool_queries(
        query_features, query_labels, multi_coff, pool_type
    )
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
    result = {
        "recall@1": float(cmc[0] * 100),
        "recall@5": float(cmc[min(4, len(gallery_labels) - 1)] * 100),
        "recall@10": float(cmc[min(9, len(gallery_labels) - 1)] * 100),
        "recall@1p": float(cmc[top1p] * 100),
        "AP": float(ap * 100),
        "time": elapsed,
    }
    print(f"multi={multi_coff} type={pool_type} {result}")
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path = output_path / (
        f"{params['model']}_{height}_multi_query_{pool_type}.csv"
    )
    pd.DataFrame(result, index=[f"multi_query_{multi_coff}_{height}"]).to_csv(table_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="settings.yaml")
    parser.add_argument("--multi", type=int, default=1)
    parser.add_argument("--type", type=str, default="ave")
    parser.add_argument("--csv_save_path", type=str, default="./result")
    options = parser.parse_args()
    eval_and_test(options.multi, options.cfg, options.type, options.csv_save_path)
