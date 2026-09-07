"""Create the ImageFolder layout used by the SUES-200 training scripts.

The raw SUES-200 archive stores one directory per scene. This script creates
the train/query/gallery view of that archive using directory symlinks by
default, so preparing all four heights does not duplicate the image files.
Use ``--mode copy`` only when the destination must be self-contained.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


DEFAULT_HEIGHTS = ("150", "200", "250", "300")
SCENE_NAMES = tuple(f"{index:04d}" for index in range(1, 201))


def load_train_scenes(index_file: Path) -> set[str]:
    with index_file.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    train_scenes = values.get("index")
    if not isinstance(train_scenes, list) or not train_scenes:
        raise ValueError(f"Expected a non-empty 'index' list in {index_file}")
    train_scenes = {str(scene).zfill(4) for scene in train_scenes}
    unknown = train_scenes.difference(SCENE_NAMES)
    if unknown:
        raise ValueError(f"Unknown scene names in {index_file}: {sorted(unknown)}")
    return train_scenes


def find_source_roots(dataset_root: Path) -> tuple[Path, Path]:
    satellite_candidates = ("satellite-view", "satellite_view_512", "satellite-view_512")
    drone_candidates = ("drone_view_512", "drone-view", "drone-view_512")
    satellite_root = next(
        (dataset_root / name for name in satellite_candidates if (dataset_root / name).is_dir()),
        None,
    )
    drone_root = next(
        (dataset_root / name for name in drone_candidates if (dataset_root / name).is_dir()),
        None,
    )
    missing = []
    if satellite_root is None:
        missing.append("satellite-view (or satellite_view_512/satellite-view_512)")
    if drone_root is None:
        missing.append("drone_view_512 (or drone-view/drone-view_512)")
    if missing:
        raise FileNotFoundError(
            f"Missing raw SUES-200 source directory under {dataset_root}: {', '.join(missing)}"
        )
    return satellite_root, drone_root


def install_view(source: Path, destination: Path, mode: str) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Existing symlink points elsewhere: {destination}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}. Remove only this generated split or choose another root."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source, target_is_directory=True)
    else:
        shutil.copytree(source, destination)


def prepare_split(
    dataset_root: Path,
    index_file: Path,
    heights: tuple[str, ...],
    mode: str,
) -> None:
    satellite_root, drone_root = find_source_roots(dataset_root)
    train_scenes = load_train_scenes(index_file)
    test_scenes = set(SCENE_NAMES).difference(train_scenes)
    print(f"dataset={dataset_root}")
    print(f"satellite_root={satellite_root}")
    print(f"drone_root={drone_root}")
    print(f"train_scenes={len(train_scenes)} test_scenes={len(test_scenes)} mode={mode}")

    jobs = (
        ("Training", "drone", train_scenes, drone_root, True),
        ("Training", "satellite", train_scenes, satellite_root, False),
        ("Testing", "query_drone", test_scenes, drone_root, True),
        ("Testing", "query_satellite", test_scenes, satellite_root, False),
        ("Testing", "gallery_drone", set(SCENE_NAMES), drone_root, True),
        ("Testing", "gallery_satellite", set(SCENE_NAMES), satellite_root, False),
    )
    prepared = 0
    for height in heights:
        before = prepared
        for split, view, scenes, source_root, height_dependent in jobs:
            for scene in sorted(scenes):
                source = source_root / scene
                if height_dependent:
                    source /= height
                if not source.is_dir():
                    raise FileNotFoundError(f"Missing source scene/height directory: {source}")
                destination = dataset_root / split / height / view / scene
                install_view(source, destination, mode)
                prepared += 1
        print(f"height={height}: prepared {prepared - before} scene links/copies")
    print(f"prepared={prepared}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="raw SUES-200 dataset directory")
    parser.add_argument(
        "--index-file",
        type=Path,
        default=Path(__file__).with_name("indexs.yaml"),
        help="YAML file containing the training scene list",
    )
    parser.add_argument(
        "--heights", nargs="+", default=list(DEFAULT_HEIGHTS),
        help="heights to prepare (default: 150 200 250 300)",
    )
    parser.add_argument(
        "--mode", choices=("symlink", "copy"), default="symlink",
        help="link raw directories without duplication, or copy them",
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    prepare_split(
        dataset_root=options.path.expanduser().resolve(),
        index_file=options.index_file.expanduser().resolve(),
        heights=tuple(str(height) for height in options.heights),
        mode=options.mode,
    )
