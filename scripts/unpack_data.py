from typing import Optional
import argparse
import os
import re
import tarfile
from pathlib import Path


FRAME_RE = re.compile(r"^scene_(\d+)_(\d+)\.(png|npy)$")
CAM_RE = re.compile(r"^scene_(\d+)_(\d+)_cam\.txt$")
CAM_TXT_RE = re.compile(r"^scene_(\d+)_(\d+)\.txt$")


def get_env_or_raise(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Missing required env var: {key}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unpack SimpleProc HF WebDataset shards (shard-*.tar) into scene folders "
            "(scene_<id>/images|depths|cams)."
        )
    )
    parser.add_argument(
        "--hf-data-root",
        type=Path,
        default=None,
        help="Folder containing downloaded HF shard tar files (default: env hf_data_root).",
    )
    parser.add_argument(
        "--dst-folder",
        type=Path,
        default=None,
        help="Destination dataset folder (default: env dst_folder).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files in dst_folder.",
    )
    return parser.parse_args()


def find_shards(hf_data_root: Path) -> list[Path]:
    shards = sorted(hf_data_root.rglob("shard-*.tar"))
    if not shards:
        raise FileNotFoundError(f"No shard tar files found under: {hf_data_root}")
    return shards


def destination_for_member(name: str, dst_folder: Path) -> Optional[Path]:
    base = Path(name).name

    m = FRAME_RE.match(base)
    if m:
        scene_id, frame_id, ext = m.groups()
        scene_folder = dst_folder / f"scene_{int(scene_id)}"
        if ext == "png":
            return scene_folder / "images" / f"{int(frame_id):08d}.png"
        if ext == "npy":
            return scene_folder / "depths" / f"{int(frame_id):08d}.npy"

    m = CAM_RE.match(base)
    if m:
        scene_id, frame_id = m.groups()
        scene_folder = dst_folder / f"scene_{int(scene_id)}"
        return scene_folder / "cams" / f"{int(frame_id):08d}_cam.txt"

    m = CAM_TXT_RE.match(base)
    if m:
        scene_id, frame_id = m.groups()
        scene_folder = dst_folder / f"scene_{int(scene_id)}"
        return scene_folder / "cams" / f"{int(frame_id):08d}_cam.txt"

    return None


def unpack_shard(shard_path: Path, dst_folder: Path, overwrite: bool) -> tuple[int, int]:
    extracted = 0
    skipped = 0

    with tarfile.open(shard_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue

            target = destination_for_member(member.name, dst_folder)
            if target is None:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                skipped += 1
                continue

            source = tf.extractfile(member)
            if source is None:
                continue

            with source, open(target, "wb") as out:
                out.write(source.read())
            extracted += 1

    return extracted, skipped


def main() -> None:
    args = parse_args()
    hf_data_root = args.hf_data_root
    if hf_data_root is None:
        hf_data_root = Path(get_env_or_raise("hf_data_root"))

    dst_folder = args.dst_folder
    if dst_folder is None:
        dst_folder = Path(get_env_or_raise("dst_folder"))

    hf_data_root = hf_data_root.expanduser().resolve()
    dst_folder = dst_folder.expanduser().resolve()

    if not hf_data_root.exists():
        raise FileNotFoundError(f"hf_data_root does not exist: {hf_data_root}")

    dst_folder.mkdir(parents=True, exist_ok=True)
    shards = find_shards(hf_data_root)

    total_extracted = 0
    total_skipped = 0
    for i, shard in enumerate(shards, start=1):
        print(f"[{i}/{len(shards)}] Unpacking {shard}")
        extracted, skipped = unpack_shard(shard, dst_folder, args.overwrite)
        total_extracted += extracted
        total_skipped += skipped

    print(
        f"Done. Extracted {total_extracted} files, skipped {total_skipped} existing files into: {dst_folder}"
    )


if __name__ == "__main__":
    main()
