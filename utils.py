import glob
import os
import re
from pathlib import Path

import pandas as pd
import torch
import yaml

import model_



def get_yaml_value(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return values


def get_device(device_name=None):
    """Resolve a configured device while falling back safely when CUDA is absent."""
    if device_name and str(device_name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))


def save_network(network, dir_model_name, epoch_label, weight_save_path=None):
    """Save a checkpoint without relying on the process working directory."""
    if weight_save_path is None:
        weight_save_path = get_yaml_value("settings.yaml")["weight_save_path"]
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = Path(weight_save_path) / dir_model_name
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), save_path / save_filename)


def fliplr(img):
    """Flip a BCHW tensor horizontally."""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1, device=img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'drone' in name:
        return 2
    else:
        raise ValueError(f"Unknown view name: {name}")


def get_model_list(dirname, key, seq):
    if not os.path.isdir(dirname):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {dirname}")

    def checkpoint_key(path):
        match = re.search(r"(\d+)(?=\.pth$)", path.name)
        return (0, int(match.group(1))) if match else (1, path.name)

    gen_models = sorted(
        (path for path in Path(dirname).glob(f"*{key}*.pth") if path.is_file()),
        key=checkpoint_key,
    )
    if not gen_models:
        raise FileNotFoundError(f"No checkpoint matching '*{key}*.pth' in {dirname}")
    try:
        return str(gen_models[seq])
    except IndexError as exc:
        raise IndexError(
            f"Checkpoint index {seq} is out of range; found {len(gen_models)} files in {dirname}"
        ) from exc


def load_network_from_path(model_name, checkpoint_path, classes, drop_rate, device=None):
    model_factory = model_.model_dict[model_name]
    try:
        model = model_factory(classes, drop_rate, pretrained=False)
    except TypeError:
        # Keep compatibility with older/custom model constructors.
        model = model_factory(classes, drop_rate)
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    return model, os.path.basename(checkpoint_path)


def load_network(model_name, name, weight_save_path, classes, drop_rate, seq, device=None):
    dirname = os.path.join(weight_save_path, name)
    checkpoint_path = get_model_list(dirname, "net", seq)
    print(checkpoint_path + " " + "seq: " + str(seq))
    return load_network_from_path(model_name, checkpoint_path, classes, drop_rate, device)


def get_id(img_path):
    labels = []
    paths = []
    for path, _ in img_path:
        folder_name = Path(path).parent.name
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths


def create_dir(path):
    os.makedirs(path, exist_ok=True)

def _height_csvs(model_name, csv_path, height):
    return [
        Path(path)
        for path in glob.glob(os.path.join(csv_path, f"{model_name}*.csv"))
        if f"_{height}_" in Path(path).stem or Path(path).stem.endswith(f"_{height}")
    ]


def _best_csv(csv_paths, score_column):
    if not csv_paths:
        return None

    def score(path):
        table = pd.read_csv(path, index_col=0)
        if score_column in table.columns:
            return float(table.at["recall@1", score_column])
        prefix = "query_drone_" if score_column == "drone_max" else "query_satellite_"
        columns = [column for column in table.columns if column.startswith(prefix)]
        return max(float(table.at["recall@1", column]) for column in columns)

    return max(csv_paths, key=score)


def select_best_weight(model_name, csv_path):
    """Return the best result CSV for each supported height and query view."""
    drone_list = []
    satellite_list = []
    for height in (150, 200, 250, 300):
        csv_paths = _height_csvs(model_name, csv_path, height)
        drone_list.append(str(_best_csv(csv_paths, "drone_max")) if csv_paths else None)
        satellite_list.append(str(_best_csv(csv_paths, "satellite_max")) if csv_paths else None)
    return drone_list, satellite_list


def get_best_weight(query_name, model_name, height, csv_path):
    """Resolve the checkpoint with the best Recall@1 for one query direction."""
    score_column = "drone_max" if "drone" in query_name else "satellite_max"
    best_csv = _best_csv(_height_csvs(model_name, csv_path, height), score_column)
    if best_csv is None:
        raise FileNotFoundError(
            f"No evaluation CSV for model={model_name}, height={height} under {csv_path}"
        )

    table = pd.read_csv(best_csv, index_col=0)
    view = "drone" if "drone" in query_name else "satellite"
    prefix = f"query_{view}_"
    candidates = [column for column in table.columns if column.startswith(prefix)]
    if not candidates:
        raise ValueError(f"No {view} query columns found in {best_csv}")
    checkpoint_name = max(
        candidates, key=lambda column: float(table.at["recall@1", column])
    )[len(prefix):]
    checkpoint_path = best_csv.with_suffix("") / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint_path}")
    return str(checkpoint_path)

def parameter(index_name, index_number):
    with open("settings.yaml", "r", encoding="utf-8") as f:
        setting_dict = yaml.load(f, Loader=yaml.FullLoader)
        setting_dict[index_name] = index_number
        # print(setting_dict)
        f.close()
        with open("settings.yaml", "w", encoding="utf-8") as f:
            yaml.dump(setting_dict, f)
            f.close()



if __name__ == '__main__':
    # param = get_yaml_value("settings.yaml")
    # print(param['height'])
    # for height in [150, 200, 250, 300]:
    #     print("----")
    #     parameter("height", height)
    #     param = get_yaml_value("settings.yaml")
    #     print(param['height'])

    for height in [150, 200, 250, 300]:
        print(height)
        for i in ["satellite", "drone"]:
            print(i)
            result = get_best_weight(i, "vit", str(height), "/home/sues/media/disk2/save_model_weight")
            print(result)
