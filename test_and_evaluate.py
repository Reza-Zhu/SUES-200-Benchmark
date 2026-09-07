"""Feature extraction and cross-view retrieval evaluation for SUES-200."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from Preprocessing import Create_Testing_Datasets
from utils import (
    fliplr,
    get_device,
    get_id,
    get_yaml_value,
    load_network_from_path,
    which_view,
)


METRIC_INDEX = ("recall@1", "recall@5", "recall@10", "recall@1p", "AP", "time")
VIEW_PAIRS = (
    ("query_drone", "gallery_satellite"),
    ("query_satellite", "gallery_drone"),
)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rank(query_features: torch.Tensor, gallery_features: torch.Tensor, dist: str) -> np.ndarray:
    """Return gallery indices sorted from best to worst for one query."""
    if dist == "Cos":
        scores = torch.mm(gallery_features, query_features.view(-1, 1)).squeeze(1)
        return np.argsort(scores.detach().cpu().numpy())[::-1]
    if dist in ("Eu", "Man"):
        p = 2 if dist == "Eu" else 1
        scores = torch.cdist(query_features.view(1, -1), gallery_features, p=p).squeeze(0)
        return np.argsort(scores.detach().cpu().numpy())
    raise ValueError(f"Unsupported distance: {dist}. Choose Cos, Eu, or Man.")


def compute_mAP(index, good_index, junk_index):
    """Compute AP and the CMC curve for one ranked query.

    The return shape and CMC convention are kept compatible with the original
    implementation so older analysis scripts can continue to use this helper.
    """
    index = np.asarray(index).reshape(-1)
    good_index = np.asarray(good_index).reshape(-1)
    junk_index = np.asarray(junk_index).reshape(-1)
    cmc = torch.zeros(len(index), dtype=torch.int32)
    if good_index.size == 0:
        cmc[0] = -1
        return 0.0, cmc

    index = index[~np.isin(index, junk_index)]
    rows_good = np.flatnonzero(np.isin(index, good_index))
    if len(rows_good) != len(good_index):
        # A malformed gallery should not produce an inflated AP.
        good_index = good_index[np.isin(good_index, index)]
        rows_good = np.flatnonzero(np.isin(index, good_index))
    if len(rows_good) == 0:
        cmc[0] = -1
        return 0.0, cmc

    cmc[rows_good[0]:] = 1
    ngood = len(rows_good)
    ap = 0.0
    for rank, row in enumerate(rows_good):
        recall_step = 1.0 / ngood
        precision = (rank + 1) / (row + 1)
        old_precision = rank / row if row != 0 else 1.0
        ap += recall_step * (old_precision + precision) / 2
    return ap, cmc


def evaluate(qf, ql, gf, gl, dist):
    """Rank a single query and return its AP/CMC pair."""
    gallery_labels = np.asarray(gl).reshape(-1)
    good_index = np.flatnonzero(gallery_labels == ql)
    junk_index = np.flatnonzero(gallery_labels == -1)
    index = _rank(qf, gf, dist)
    return compute_mAP(index, good_index, junk_index)


def extract_feature(model, dataloaders, view_index=1, device=None, use_amp=False):
    """Extract flip-augmented, L2-normalized embeddings into host memory."""
    device = device or next(model.parameters()).device
    feature_batches = []
    with torch.inference_mode():
        for images, _ in dataloaders:
            images = images.to(device, non_blocking=True)
            embedding = None
            for batch_images in (images, fliplr(images)):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if view_index == 1:
                        outputs, _ = model(batch_images, None)
                    elif view_index == 2:
                        _, outputs = model(None, batch_images)
                    else:
                        raise ValueError(f"Unknown view index: {view_index}")
                outputs = outputs.float()
                embedding = outputs if embedding is None else embedding + outputs
            feature_batches.append(torch.nn.functional.normalize(embedding, p=2, dim=1).cpu())

    if not feature_batches:
        return torch.empty((0, 512))
    return torch.cat(feature_batches, dim=0)


def evaluate_retrieval(
    query_features,
    query_labels,
    gallery_features,
    gallery_labels,
    dist="Cos",
    chunk_size=256,
    device=None,
):
    """Evaluate all queries with chunked batched ranking.

    Similarity/distance computation is batched, while AP/CMC accumulation stays
    on CPU. This avoids one GPU kernel launch and host transfer per query.
    """
    if device is None:
        device = get_device()
    query_features = query_features.to(device, non_blocking=True)
    gallery_features = gallery_features.to(device, non_blocking=True)
    query_labels = np.asarray(query_labels).reshape(-1)
    gallery_labels = np.asarray(gallery_labels).reshape(-1)
    valid_gallery = gallery_labels != -1
    gallery_features = gallery_features[torch.as_tensor(valid_gallery, device=device)]
    gallery_labels = gallery_labels[valid_gallery]
    if len(query_labels) == 0 or len(gallery_labels) == 0:
        raise ValueError("Query and gallery must both contain at least one valid image")

    cmc = np.zeros(len(gallery_labels), dtype=np.int64)
    ap_total = 0.0
    valid_queries = 0
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(query_labels), chunk_size):
        stop = min(start + chunk_size, len(query_labels))
        batch = query_features[start:stop]
        if dist == "Cos":
            scores = torch.mm(batch, gallery_features.t())
            ranking = np.argsort(scores.detach().cpu().numpy(), axis=1)[:, ::-1].copy()
        elif dist in ("Eu", "Man"):
            p = 2 if dist == "Eu" else 1
            distances = torch.cdist(batch, gallery_features, p=p)
            ranking = np.argsort(distances.detach().cpu().numpy(), axis=1)
        else:
            raise ValueError(f"Unsupported distance: {dist}. Choose Cos, Eu, or Man.")

        for offset, ranked_indices in enumerate(ranking):
            good_index = np.flatnonzero(gallery_labels == query_labels[start + offset])
            ap, query_cmc = compute_mAP(ranked_indices, good_index, np.empty(0, dtype=int))
            if query_cmc[0] == -1:
                continue
            cmc += query_cmc.numpy()
            ap_total += ap
            valid_queries += 1

    if valid_queries == 0:
        raise ValueError("No query has a matching gallery identity")
    return cmc / valid_queries, ap_total / valid_queries, valid_queries


def _format_result(cmc, ap, elapsed, gallery_size):
    top1p_index = min(round(gallery_size * 0.01), gallery_size - 1)
    values = (
        cmc[0] * 100,
        cmc[min(4, gallery_size - 1)] * 100,
        cmc[min(9, gallery_size - 1)] * 100,
        cmc[top1p_index] * 100,
        ap * 100,
        elapsed,
    )
    return values, (
        "Recall@1:{:.2f} Recall@5:{:.2f} Recall@10:{:.2f} "
        "Recall@top1:{:.2f} AP:{:.2f} Time:{:.2f}"
    ).format(*values)


def _load_results_table(path: Path) -> pd.DataFrame:
    if path.exists():
        table = pd.read_csv(path)
        if "index" in table.columns:
            table.index = table["index"]
        table = table.drop(
            columns=[column for column in table.columns if column.startswith("index")],
            errors="ignore",
        )
        table = table.drop(columns=["drone_max", "satellite_max"], errors="ignore")
        return table
    return pd.DataFrame(index=METRIC_INDEX)


def eval_and_test(cfg_path, name, seqs, dist, checkpoint=None):
    print("Testing Start >>>>>>>>")
    params = get_yaml_value(cfg_path)
    if seqs < 1:
        raise ValueError("seq must be at least 1")
    device = get_device(params.get("device"))
    dataset_path = Path(params["dataset_path"]) / "Testing" / str(params["height"])
    weight_root = Path(params["weight_save_path"])
    name = name or params.get("name", "")
    checkpoint = checkpoint or params.get("checkpoint_path")
    if not name and not checkpoint:
        raise ValueError("Specify --name or add name to the config")
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        if not name:
            name = checkpoint_path.stem
        checkpoint_paths = [checkpoint_path]
    else:
        model_dir = weight_root / name
        checkpoint_paths = sorted(model_dir.glob("*.pth"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No .pth checkpoint found in {model_dir}")
    weight_root.mkdir(parents=True, exist_ok=True)

    _, data_loader = Create_Testing_Datasets(
        test_data_path=str(dataset_path),
        batch_size=params["batch_size"],
        image_size=params["image_size"],
        num_workers=params.get("num_workers", 4),
        pin_memory=params.get("pin_memory", device.type == "cuda"),
        prefetch_factor=params.get("prefetch_factor", 2),
    )
    result_table_path = weight_root / f"{name}.csv"
    result_table = _load_results_table(result_table_path)
    use_amp = bool(params.get("eval_amp", False)) and device.type == "cuda"
    feature_chunk_size = params.get("eval_chunk_size", 256)

    selected_checkpoints = checkpoint_paths if checkpoint else checkpoint_paths[-seqs:]
    for checkpoint_path in selected_checkpoints:
        model, net_name = load_network_from_path(
            params["model"], checkpoint_path, params["classes"], params["drop_rate"], "cpu"
        )
        model.classifier.classifier = nn.Identity()
        model = model.eval().to(device)
        print(net_name)

        features = {}
        feature_times = {}
        for view_name in ("query_drone", "query_satellite", "gallery_drone", "gallery_satellite"):
            view_index = which_view(view_name)
            started = time.time()
            features[view_name] = extract_feature(
                model,
                data_loader[view_name],
                view_index=view_index,
                device=device,
                use_amp=use_amp,
            )
            features[view_name] = features[view_name].to(device, non_blocking=True)
            _synchronize(device)
            feature_times[view_name] = time.time() - started
            print(f"{view_name}: {len(features[view_name])} features in {feature_times[view_name]:.2f}s")

        for query_name, gallery_name in VIEW_PAIRS:
            query_labels, _ = get_id(data_loader[query_name].dataset.imgs)
            gallery_labels, _ = get_id(data_loader[gallery_name].dataset.imgs)
            _synchronize(device)
            started = time.time()
            cmc, ap, valid_queries = evaluate_retrieval(
                features[query_name],
                query_labels,
                features[gallery_name],
                gallery_labels,
                dist=dist,
                chunk_size=feature_chunk_size,
                device=device,
            )
            _synchronize(device)
            retrieval_time = time.time() - started
            elapsed = feature_times[query_name] + feature_times[gallery_name] + retrieval_time
            values, result_text = _format_result(cmc, ap, elapsed, len(gallery_labels))
            result_table[f"{query_name}_{net_name}"] = values
            result_dir = weight_root / name
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file = result_dir / (
                f"{query_name[6:]}_to_{gallery_name[8:]}_{net_name[:7]}_"
                f"{values[0]:.2f}_{values[4]:.2f}.txt"
            )
            result_file.write_text(result_text, encoding="utf-8")
            print(f"{query_name} -> {gallery_name}: {result_text} valid_queries={valid_queries}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    network_columns = list(result_table.columns)
    drone_columns = [column for column in network_columns if column.startswith("query_drone_")]
    satellite_columns = [column for column in network_columns if column.startswith("query_satellite_")]
    if drone_columns:
        result_table["drone_max"] = result_table[drone_columns].max(axis=1)
    if satellite_columns:
        result_table["satellite_max"] = result_table[satellite_columns].max(axis=1)
    result_table.columns.name = "net"
    result_table.index.name = "index"
    result_table.to_csv(result_table_path)
    return result_table


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="settings.yaml", help="config file")
    parser.add_argument("--name", type=str, default="", help="checkpoint directory name")
    parser.add_argument("--checkpoint", type=str, default="", help="direct checkpoint file path")
    parser.add_argument("--seq", type=int, default=1, help="number of checkpoints from the end")
    parser.add_argument("--dist", type=str, default="Cos", choices=("Cos", "Eu", "Man"))
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    options = parse_opt(True)
    eval_and_test(options.cfg, options.name, options.seq, options.dist, options.checkpoint)
