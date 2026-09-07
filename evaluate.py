"""Evaluate a legacy ``pytorch_result.mat`` feature file.

New experiments should call ``test_and_evaluate.py`` directly. This module is
kept for old feature-dump workflows and reuses the maintained metric code.
"""

from pathlib import Path
import argparse

import scipy.io
import torch

from test_and_evaluate import evaluate_retrieval
from utils import get_device, get_yaml_value


def evaluate_feature_file(result_path="pytorch_result.mat", config_path="settings.yaml"):
    params = get_yaml_value(config_path)
    query_name = params.get("query", "satellite")
    if query_name not in ("satellite", "drone"):
        raise ValueError("query must be 'satellite' or 'drone'")
    gallery_name = "drone" if query_name == "satellite" else "satellite"
    result = scipy.io.loadmat(result_path)
    query_features = torch.as_tensor(result["query_f"], dtype=torch.float32)
    gallery_features = torch.as_tensor(result["gallery_f"], dtype=torch.float32)
    query_labels = result["query_label"].reshape(-1)
    gallery_labels = result["gallery_label"].reshape(-1)
    device = get_device(params.get("device"))
    query_features = query_features.to(device)
    gallery_features = gallery_features.to(device)

    cmc, ap, _ = evaluate_retrieval(
        query_features,
        query_labels,
        gallery_features,
        gallery_labels,
        dist="Cos",
        chunk_size=params.get("eval_chunk_size", 256),
        device=device,
    )
    top1p_index = min(round(len(gallery_labels) * 0.01), len(gallery_labels) - 1)
    output = (
        f"Recall@1:{cmc[0] * 100:.2f} "
        f"Recall@5:{cmc[min(4, len(gallery_labels) - 1)] * 100:.2f} "
        f"Recall@10:{cmc[min(9, len(gallery_labels) - 1)] * 100:.2f} "
        f"Recall@top1:{cmc[top1p_index] * 100:.2f} "
        f"AP:{ap * 100:.2f}"
    )
    save_dir = Path(params.get("weight_save_path", ".")) / params.get("name", "")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f"{query_name}_to_{gallery_name}_result.txt"
    save_file.write_text(output, encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default="pytorch_result.mat")
    parser.add_argument("--cfg", default="settings.yaml")
    options = parser.parse_args()
    evaluate_feature_file(options.result, options.cfg)
