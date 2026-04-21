from pathlib import Path

import pandas as pd


# 1) basic path
BASIC_PATH = Path("/mnt/hdd/huggingface/hub")

# 2) folder path map (requested 8 folders)
FOLDER_PATHS = {
    "dex1_diversemanip": "UnifoLM_G1_Dex1_DiverseManip_Dataset",
    "wbt": "UnifoLM_WBT_Dataset",
    "vla0": "UnifoLM-VLA-0",
    "wma0": "UnifoLM-WMA-0",
    "dex1": "UnifoLM_G1_Dex1_Dataset",
    "brainco": "UnifoLM_G1_Brainco_Dataset",
    "z1": "UnifoLM_Z1_Arm_Dataset",
    "dex3": "UnifoLM_G1_Dex3_Dataset",
}

# 3) For now, use one example based on G1_Dex1 dataset.
ACTIVE_FOLDER_PATH = FOLDER_PATHS["dex1"]
DATASET_PATH = "datasets--unitreerobotics--G1_Dex1_Clean_Table"


def find_snapshot_base_path(
    basic_path: Path,
    folder_path: str,
    dataset_path: str,
) -> Path:
    """Return snapshots/<hash>/data/chunk-000 for the given dataset folder."""
    dataset_root = basic_path / folder_path / dataset_path
    snapshots_root = dataset_root / "snapshots"

    if not snapshots_root.is_dir():
        raise FileNotFoundError(
            f"Snapshots directory not found: {snapshots_root}"
        )

    # Pick the newest snapshot directory by modification time.
    candidates = [
        p / "data" / "chunk-000"
        for p in snapshots_root.iterdir()
        if p.is_dir()
    ]
    candidates = [p for p in candidates if p.is_dir()]

    if not candidates:
        raise FileNotFoundError(
            f"No chunk directory found under: {snapshots_root}"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def inspect_sample_episode(snapshot_base_path: Path) -> None:
    sample_episode = snapshot_base_path / "episode_000020.parquet"
    if not sample_episode.is_file():
        candidates = sorted(snapshot_base_path.glob("episode_*.parquet"))
        if not candidates:
            print("No episode parquet file found. Check dataset contents.")
            return
        sample_episode = candidates[0]

    print(f"sample_episode: {sample_episode}")

    df = pd.read_parquet(sample_episode, engine="pyarrow")
    print(f"rows: {len(df)}")
    print(f"columns ({len(df.columns)}):")
    for col in df.columns:
        print(f" - {col}")

    print("\nhead:")
    print(df.head())


def main() -> None:
    # 4) base_path style: snapshots/<hash>/data/chunk-000
    snapshot_base_path = find_snapshot_base_path(
        basic_path=BASIC_PATH,
        folder_path=ACTIVE_FOLDER_PATH,
        dataset_path=DATASET_PATH,
    )

    print(f"basic_path: {BASIC_PATH}")
    print(f"folder_path: {ACTIVE_FOLDER_PATH}")
    print(f"dataset_path: {DATASET_PATH}")
    print(
        "base_path: "
        f"{snapshot_base_path.relative_to(BASIC_PATH / ACTIVE_FOLDER_PATH / DATASET_PATH)}"
    )
    print(f"absolute_base_path: {snapshot_base_path}\n")

    inspect_sample_episode(snapshot_base_path)


if __name__ == "__main__":
    main()