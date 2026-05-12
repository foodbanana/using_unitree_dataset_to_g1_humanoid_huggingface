from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# -----------------------------
# Hardcoded dataset location
# -----------------------------
BASIC_PATH = "/mnt/hdd/huggingface/hub"
FOLDER_NAME = "UnifoLM_G1_Brainco_Dataset"
DATASET_NAME = "datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset"

# Default initial stationary window size
INITIAL_WINDOW_SIZE = 30


# Expected joint order for G1 + Brainco hand dataset (26 joints)
DATASET_JOINT_ORDER = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
    "kLeftHandThumb",
    "kLeftHandThumbAux",
    "kLeftHandIndex",
    "kLeftHandMiddle",
    "kLeftHandRing",
    "kLeftHandPinky",
    "kRightHandThumb",
    "kRightHandThumbAux",
    "kRightHandIndex",
    "kRightHandMiddle",
    "kRightHandRing",
    "kRightHandPinky",
]


def find_parquet_file(dataset_root: Path) -> Path:
    """Find a parquet file under snapshots/*/data/**."""
    candidates = sorted(dataset_root.glob("snapshots/*/data/**/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet file found under: {dataset_root}")
    return candidates[-1]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute initial static-window STD per episode and print mean STD across episodes"
        )
    )
    parser.add_argument(
        "input_parquet",
        nargs="?",
        help="Optional parquet file path. If omitted, use hardcoded dataset root fallback.",
    )
    parser.add_argument(
        "--initial-window-size",
        type=int,
        default=INITIAL_WINDOW_SIZE,
        help=f"Number of initial frames per episode for baseline STD (default: {INITIAL_WINDOW_SIZE})",
    )
    return parser.parse_args()


def resolve_parquet_input(input_parquet: str | None) -> Path:
    """Resolve parquet input path from CLI arg first, then fallback to default dataset root."""
    if input_parquet:
        parquet_path = Path(input_parquet).expanduser().resolve()
        if not parquet_path.exists():
            raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")
        if parquet_path.suffix.lower() != ".parquet":
            raise ValueError(f"Input file must be a .parquet file: {parquet_path}")
        return parquet_path

    dataset_root = Path(BASIC_PATH) / FOLDER_NAME / DATASET_NAME
    return find_parquet_file(dataset_root)


def infer_joint_names(df: pd.DataFrame, state_column: str) -> list[str]:
    """Infer joint names from state length; fallback to generic names if needed."""
    first_state = np.asarray(df[state_column].iloc[0], dtype=float)
    n_joints = int(first_state.shape[0])

    if n_joints == len(DATASET_JOINT_ORDER):
        return DATASET_JOINT_ORDER

    return [f"joint_{i:02d}" for i in range(n_joints)]


def compute_initial_std_per_episode(
    df: pd.DataFrame,
    initial_window_size: int,
    state_column: str = "observation.state",
    episode_column: str = "episode_index",
    frame_column: str = "frame_index",
    joint_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-episode STD using the first N frames of each episode.

    Returns:
        DataFrame(index=episode_index, columns=joint_names)
    """
    required_columns = {state_column, episode_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    if initial_window_size <= 0:
        raise ValueError("initial_window_size must be > 0")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    if joint_names is None:
        joint_names = infer_joint_names(df, state_column)

    per_episode_stds: list[pd.Series] = []

    for episode_id, episode_df in df.groupby(episode_column, sort=True):
        if frame_column in episode_df.columns:
            episode_df = episode_df.sort_values(frame_column)

        static_window_df = episode_df.head(initial_window_size)
        if len(static_window_df) == 0:
            continue

        state_matrix = np.vstack(static_window_df[state_column].apply(np.asarray).values).astype(float)
        std_values = np.std(state_matrix, axis=0, ddof=0)

        if len(std_values) != len(joint_names):
            raise ValueError(
                f"Joint length mismatch at episode {episode_id}: "
                f"expected {len(joint_names)}, got {len(std_values)}"
            )

        per_episode_stds.append(pd.Series(std_values, index=joint_names, name=episode_id))

    if not per_episode_stds:
        return pd.DataFrame(columns=joint_names)

    out_df = pd.DataFrame(per_episode_stds)
    out_df.index.name = episode_column
    return out_df


def main() -> None:
    args = parse_cli_args()

    parquet_file_path = resolve_parquet_input(args.input_parquet)
    print(f"[INFO] parquet file: {parquet_file_path}")

    df = pd.read_parquet(parquet_file_path)
    print()
    print(f"[INFO] loaded rows={len(df)}, cols={len(df.columns)}")

    initial_std_df = compute_initial_std_per_episode(
        df=df,
        initial_window_size=args.initial_window_size,
        state_column="observation.state",
        episode_column="episode_index",
        frame_column="frame_index",
    )

    if initial_std_df.empty:
        print("[INFO] No episodes found to compute initial STD.")
        return

    mean_std_per_joint = initial_std_df.mean(axis=0)
    grand_mean_std = float(mean_std_per_joint.mean())

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.8f}")

    print()
    print(f"[INFO] initial window size: {args.initial_window_size}")
    print()
    print("[INFO] Mean initial STD across episodes (per joint):")
    print(mean_std_per_joint.to_string())

    print()
    print(f"[INFO] Grand mean of per-joint mean STD: {grand_mean_std:.8f}")


if __name__ == "__main__":
    main()

