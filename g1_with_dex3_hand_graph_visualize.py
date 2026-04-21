from pathlib import Path
from datetime import datetime
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# 1) Hardcoded runtime settings
# -----------------------------
BASIC_PATH = "/mnt/hdd/huggingface/hub"
FOLDER_NAME = "UnifoLM_G1_Dex3_Dataset"

# IMPORTANT: this should include the "datasets--" prefix.
DATASET_NAME = "datasets--unitreerobotics--G1_Dex3_BlockStacking_Dataset"

# Reference MJCF used for joint interpretation.
TARGET_MJCF_PATH = Path(
    "/home/taeung/g1_datasets_huggingface/mujoco_menagerie/unitree_g1/g1_with_hands.xml"
)

# Plot first N episodes only.
MAX_EPISODES_TO_PLOT = 7

# Output settings
DATASET_TAG = "g1_with_dex3_hand"
ALGORITHM_TAG = "joint_angle_timeseries_2x2"
ANNOTATED_CSV_BASE_DIR = Path(
    "/home/taeung/g1_datasets_huggingface/joint_angle_graphs/g1_with_dex3_hand/csv_files"
)
GRAPH_IMAGE_BASE_DIR = Path(
    "/home/taeung/g1_datasets_huggingface/joint_angle_graphs/g1_with_dex3_hand/graph_images"
)

# ----------------------------------
# 2) Dataset joint order (fixed spec in huggingface)
# Lerobot form of g1_with_dex3_hand dataset has 28 joints:
# 14 arm joints + 14 hand joints (Dex3)
# ----------------------------------
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
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
]


# ---------------------------------
# 3) Joint groups for 2x2 plotting
# ---------------------------------
JOINT_GROUPS = {
    "left_arm": [
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
    ],
    "right_arm": [
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
    ],
    "left_dex3_hand": [
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
    ],
    "right_dex3_hand": [
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
}

ROBOT_MOVE_COL = "robot_move_annotation"
DOMINANT_JOINT_COL = "dominant_joint_by_std"


def find_parquet_file(dataset_root: Path) -> Path:
    """Find a parquet file under snapshots/*/data/**."""
    candidates = sorted(dataset_root.glob("snapshots/*/data/**/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet file found under: {dataset_root}")
    return candidates[0]


def find_readme_file(dataset_root: Path) -> Path | None:
    """Find README.md under snapshots/* (if present)."""
    candidates = sorted(dataset_root.glob("snapshots/*/README.md"))
    return candidates[0] if candidates else None


def extract_joint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand observation.state (28-dim) into named joint columns."""
    if "observation.state" not in df.columns:
        raise KeyError("Missing required column: observation.state")

    state_matrix = np.vstack(df["observation.state"].apply(np.asarray).values).astype(float)
    if state_matrix.shape[1] != len(DATASET_JOINT_ORDER):
        raise ValueError(
            "Mismatch between observation.state width and DATASET_JOINT_ORDER length: "
            f"{state_matrix.shape[1]} vs {len(DATASET_JOINT_ORDER)}"
        )

    out_df = df.copy()
    out_df[DATASET_JOINT_ORDER] = state_matrix
    return out_df


def _build_csv_output_path(output_dir: Path) -> Path:
    """Build a unique CSV filename with date + daily run index + algorithm tag."""
    date_str = datetime.now().strftime("%Y%m%d")
    candidates = list(output_dir.glob(f"{DATASET_TAG}_{date_str}_run*_{ALGORITHM_TAG}.csv"))

    run_indices: list[int] = []
    regex = re.compile(
        rf"^{re.escape(DATASET_TAG)}_{date_str}_run(\d+)_{re.escape(ALGORITHM_TAG)}\.csv$"
    )
    for path in candidates:
        match = regex.match(path.name)
        if match:
            run_indices.append(int(match.group(1)))

    next_run_idx = (max(run_indices) + 1) if run_indices else 1
    file_name = f"{DATASET_TAG}_{date_str}_run{next_run_idx:02d}_{ALGORITHM_TAG}.csv"
    return output_dir / file_name


def _dataset_output_dir_name(dataset_name: str) -> str:
    """Extract a clean dataset folder name from huggingface-style dataset path."""
    return dataset_name.split("--")[-1] if "--" in dataset_name else dataset_name


def save_joint_dataframe_csv(df_joint: pd.DataFrame, output_dir: Path) -> Path:
    """Save a DataFrame to CSV and return saved path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _build_csv_output_path(output_dir)
    df_joint.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def build_motion_annotations(df_joint: pd.DataFrame) -> pd.DataFrame:
    """Create two annotation columns using joint trajectories per episode."""
    if "episode_index" not in df_joint.columns:
        raise KeyError("Missing required column: episode_index")

    out = pd.DataFrame(index=df_joint.index)
    out[ROBOT_MOVE_COL] = "stable"
    out[DOMINANT_JOINT_COL] = "unknown"

    grouped_episodes = df_joint.groupby("episode_index", sort=True)
    for _, episode_data in grouped_episodes:
        joint_values = episode_data[DATASET_JOINT_ORDER].to_numpy(dtype=float)

        std_per_joint = np.std(joint_values, axis=0)
        dominant_joint = DATASET_JOINT_ORDER[int(np.argmax(std_per_joint))]

        # Per-frame motion score from consecutive joint-state differences.
        deltas = np.linalg.norm(
            np.diff(joint_values, axis=0, prepend=joint_values[[0], :]), axis=1
        )
        threshold = np.percentile(deltas, 60)
        move_labels = np.where(deltas > threshold, "moving", "stable")

        out.loc[episode_data.index, ROBOT_MOVE_COL] = move_labels
        out.loc[episode_data.index, DOMINANT_JOINT_COL] = dominant_joint

    return out


def _choose_x_axis(episode_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Choose frame_index first, then timestamp, else synthetic index."""
    if "frame_index" in episode_df.columns:
        return episode_df["frame_index"].to_numpy(), "Frame Index"
    if "timestamp" in episode_df.columns:
        return episode_df["timestamp"].to_numpy(), "Timestamp"
    return np.arange(len(episode_df), dtype=int), "Timestep"


def save_joint_group_episode_previews_2x2(
    df: pd.DataFrame,
    output_dir: Path,
    max_episodes_to_plot: int = MAX_EPISODES_TO_PLOT,
) -> list[Path]:
    """Save per-episode 2x2 joint-angle plots (left/right arm + left/right Dex3 hand)."""
    grouped_episodes = df.groupby("episode_index", sort=True)
    episode_ids = sorted(grouped_episodes.groups.keys())[:max_episodes_to_plot]
    if not episode_ids:
        raise ValueError("No episodes found for visualization.")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    group_order = [
        ("left_arm", "Left Arm (7 joints)"),
        ("right_arm", "Right Arm (7 joints)"),
        ("left_dex3_hand", "Left Dex3 Hand (7 joints)"),
        ("right_dex3_hand", "Right Dex3 Hand (7 joints)"),
    ]

    for episode_id in episode_ids:
        episode_data = grouped_episodes.get_group(episode_id)
        x, x_label = _choose_x_axis(episode_data)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), squeeze=False)
        axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

        for ax, (group_key, group_title) in zip(axes_flat, group_order):
            joint_names = JOINT_GROUPS[group_key]
            for joint_name in joint_names:
                ax.plot(
                    x,
                    episode_data[joint_name].to_numpy(),
                    linewidth=1.0,
                    alpha=0.9,
                    label=joint_name,
                )

            ax.set_title(group_title, fontsize=11, fontweight="bold")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Joint Angle")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=7)

        fig.suptitle(
            f"Episode {episode_id} G1 Dex3 Joint Angles (2x2)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout(rect=(0, 0, 1, 0.97))

        file_path = output_dir / f"episode_{int(episode_id):04d}_joint_2x2.png"
        fig.savefig(file_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(file_path)

    return saved_paths


def main() -> None:
    # Load dataset
    dataset_root = Path(BASIC_PATH) / FOLDER_NAME / DATASET_NAME
    parquet_file_path = find_parquet_file(dataset_root)
    readme_path = find_readme_file(dataset_root)

    print(f"[INFO] parquet file: {parquet_file_path}")
    if readme_path is not None:
        print(f"[INFO] dataset metadata(README): {readme_path}")
    else:
        print("[INFO] dataset metadata(README): not found")
    print(f"[INFO] target MJCF: {TARGET_MJCF_PATH}")
    print()

    # Read parquet
    df = pd.read_parquet(parquet_file_path)
    print(f"[INFO] Downloaded dataset --> loaded rows={len(df)}, cols={len(df.columns)}")
    print()

    # Check required columns
    required_columns = {"observation.state", "episode_index"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Extract 28 joint columns from observation.state
    df_joint = extract_joint_columns(df)

    print(f"[INFO] after joint extraction --> rows={len(df_joint)}, cols={len(df_joint.columns)}")
    print(f"[INFO] extracted joint columns for internal use: {len(DATASET_JOINT_ORDER)}")
    print()

    annotation_df = build_motion_annotations(df_joint)
    df_annotated = df.copy()
    df_annotated[ROBOT_MOVE_COL] = annotation_df[ROBOT_MOVE_COL]
    df_annotated[DOMINANT_JOINT_COL] = annotation_df[DOMINANT_JOINT_COL]

    print()
    print(f"[INFO] after annotation --> rows={len(df_annotated)}, cols={len(df_annotated.columns)}")
    print()
    print(f"[INFO] added column_1 name: {df_annotated.columns[-2]}")
    print(f"[INFO] added column_2 name: {df_annotated.columns[-1]}")
    print()
    print(
        "[INFO] robot_move_annotation sample(top 3): "
        f"{df_annotated[ROBOT_MOVE_COL].head(3).tolist()}"
    )
    print(
        "[INFO] dominant_joint_by_std sample(top 3): "
        f"{df_annotated[DOMINANT_JOINT_COL].head(3).tolist()}"
    )
    print()

    dataset_output_name = _dataset_output_dir_name(DATASET_NAME)
    csv_output_dir = ANNOTATED_CSV_BASE_DIR / dataset_output_name
    graph_output_dir = GRAPH_IMAGE_BASE_DIR / dataset_output_name

    csv_output_path = save_joint_dataframe_csv(df_annotated, csv_output_dir)
    print(f"[INFO] saved annotated dataframe csv: {csv_output_path}")
    print(f"[INFO] saved csv --> rows={len(df_annotated)}, cols={len(df_annotated.columns)}")

    preview_paths = save_joint_group_episode_previews_2x2(
        df_joint,
        output_dir=graph_output_dir,
        max_episodes_to_plot=MAX_EPISODES_TO_PLOT,
    )
    for p in preview_paths:
        print(f"[INFO] saved 2x2 episode image: {p}")


if __name__ == "__main__":
    main()
