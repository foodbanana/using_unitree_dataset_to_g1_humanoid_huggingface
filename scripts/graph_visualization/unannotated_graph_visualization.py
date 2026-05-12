from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter  # 줄바꿈(Enter)을 유지하기 위해 필요합니다.
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


# ----------------------------
# Runtime settings (see docs/hf_dataset_installed.md)
# -----------------------------
BASIC_PATH = "/mnt/hdd/huggingface/hub"
FOLDER_NAME = "UnifoLM_G1_Brainco_Dataset"

# IMPORTANT: this must include the "datasets--" prefix.
DATASET_NAME = "datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset"

# Plot first N episodes only.
MAX_EPISODES_TO_PLOT = 5

# Supported options based on docs/hf_dataset_installed.md
FOLDER_TO_DATASETS = {
    "UnifoLM_G1_Brainco_Dataset": [
        "datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset",
        "datasets--unitreerobotics--G1_Brainco_GraspRubiksCube_Dataset",
        "datasets--unitreerobotics--G1_Brainco_PickApple_Dataset",
        "datasets--unitreerobotics--G1_Brainco_PickCharger_Dataset",
        "datasets--unitreerobotics--G1_Brainco_PickDoll_Dataset",
        "datasets--unitreerobotics--G1_Brainco_PickDrink_Dataset",
        "datasets--unitreerobotics--G1_Brainco_PickTissues_Dataset",
        "datasets--unitreerobotics--G1_Brainco_PickToothpaste_Dataset",
    ],
    "UnifoLM_G1_Dex1_Dataset": [
        "datasets--unitreerobotics--G1_Dex1_Bag_Insert",
    ],
    "UnifoLM_G1_Dex3_Dataset": [
        "datasets--unitreerobotics--G1_Dex3_BlockStacking_Dataset",
        "datasets--unitreerobotics--G1_Dex3_CameraPackaging_Dataset",
        "datasets--unitreerobotics--G1_Dex3_GraspSquare_Dataset",
        "datasets--unitreerobotics--G1_Dex3_ObjectPlacement_Dataset",
        "datasets--unitreerobotics--G1_Dex3_PickApple_Dataset",
        "datasets--unitreerobotics--G1_Dex3_PickBottle_Dataset",
        "datasets--unitreerobotics--G1_Dex3_PickCharger_Dataset",
        "datasets--unitreerobotics--G1_Dex3_PickDoll_Dataset",
        "datasets--unitreerobotics--G1_Dex3_PickGum_Dataset",
        "datasets--unitreerobotics--G1_Dex3_PickSnack_Dataset",
        "datasets--unitreerobotics--G1_Dex3_PickTissue_Dataset",
        "datasets--unitreerobotics--G1_Dex3_Pouring_Dataset",
        "datasets--unitreerobotics--G1_Dex3_ToastedBread_Dataset",
    ],
    "UnifoLM_WBT_Dataset": [
        "datasets--unitreerobotics--G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
        "datasets--unitreerobotics--G1_WBT_Brainco_Make_The_Bed",
        "datasets--unitreerobotics--G1_WBT_Brainco_Pickup_Pillow",
        "datasets--unitreerobotics--G1_WBT_Dex1_Put_Clothes_into_Washing_Machine",
        "datasets--unitreerobotics--G1_WBT_Inspire_Collect_Clothes_MainCamOnly",
        "datasets--unitreerobotics--G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
        "datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_Into_Basket",
        "datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_into_Washing_Machine",
        "datasets--unitreerobotics--G1_WBT_Inspire_Put_Clothes_into_Washing_Machine_MainCamOnly",
        "datasets--unitreerobotics--G1_WBT_Inspire_Put_Drinks_Into_Fridge",
        "datasets--unitreerobotics--G1_WBT_Inspire_Put_Vegetables_Into_Basket",
    ],
}

ALL_FOLDER_NAMES = sorted(FOLDER_TO_DATASETS.keys())
ALL_DATASET_NAMES = sorted(
    {name for datasets in FOLDER_TO_DATASETS.values() for name in datasets}
)

# Output root
GRAPH_ROOT_DIR = Path(
    "/home/taeung/g1_datasets_huggingface/graphs/unannotated_graphs_sample"
)

FPS = 30.0
X_TICK_SECONDS = 3.0

matplotlib.use("Agg")


UPPER_BODY_JOINT_ORDER_BRAINCO = [
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

UPPER_BODY_JOINT_ORDER_DEX3 = [
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

WBT_HAND_ORDER_BRAINCO = [
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

WBT_HAND_ORDER_INSPIRE = [
    "kLeftHandIndex",
    "kLeftHandMiddle",
    "kLeftHandRing",
    "kLeftHandPinky",
    "kLeftHandThumb",
    "kLeftHandThumbAux",
    "kRightHandIndex",
    "kRightHandMiddle",
    "kRightHandRing",
    "kRightHandPinky",
    "kRightHandThumb",
    "kRightHandThumbAux",
]

WBT_BODY_JOINT_ORDER = [
    "kLeftHipPitch",
    "kLeftHipRoll",
    "kLeftHipYaw",
    "kLeftKnee",
    "kLeftAnklePitch",
    "kLeftAnkleRoll",
    "kRightHipPitch",
    "kRightHipRoll",
    "kRightHipYaw",
    "kRightKnee",
    "kRightAnklePitch",
    "kRightAnkleRoll",
    "kWaistYaw",
    "kWaistRoll",
    "kWaistPitch",
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
]


UPPER_BODY_GROUPS = {
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
    "left_hand": [
        "kLeftHandThumb",
        "kLeftHandThumbAux",
        "kLeftHandIndex",
        "kLeftHandMiddle",
        "kLeftHandRing",
        "kLeftHandPinky",
    ],
    "right_hand": [
        "kRightHandThumb",
        "kRightHandThumbAux",
        "kRightHandIndex",
        "kRightHandMiddle",
        "kRightHandRing",
        "kRightHandPinky",
    ],
}

DEX3_HAND_GROUPS = {
    "left_hand": [
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
    ],
    "right_hand": [
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
}

WBT_GROUPS = {
    "left_arm": UPPER_BODY_GROUPS["left_arm"],
    "right_arm": UPPER_BODY_GROUPS["right_arm"],
    "left_hand": UPPER_BODY_GROUPS["left_hand"],
    "right_hand": UPPER_BODY_GROUPS["right_hand"],
    "waist": ["kWaistYaw", "kWaistRoll", "kWaistPitch"],
    "left_leg": [
        "kLeftHipPitch",
        "kLeftHipRoll",
        "kLeftHipYaw",
        "kLeftKnee",
        "kLeftAnklePitch",
        "kLeftAnkleRoll",
    ],
    "right_leg": [
        "kRightHipPitch",
        "kRightHipRoll",
        "kRightHipYaw",
        "kRightKnee",
        "kRightAnklePitch",
        "kRightAnkleRoll",
    ],
}


def classify_control_type(folder_name: str, dataset_name: str) -> str:
    combined = f"{folder_name} {dataset_name}".lower()
    return "WBT" if "wbt" in combined else "Upper_body_control"


def classify_hand_type(folder_name: str, dataset_name: str) -> str:
    combined = f"{folder_name} {dataset_name}".lower()
    if "brainco" in combined:
        return "g1_with_brainco"
    if "dex3" in combined:
        return "g1_with_dex3"
    if "inspire" in combined:
        return "g1_with_inspire"
    raise ValueError("Hand type could not be inferred from folder_name/dataset_name.")


def validate_configuration(control_type: str, hand_type: str) -> None:
    if hand_type == "g1_with_inspire" and control_type != "WBT":
        raise ValueError("Inspire hand exists only in WBT datasets.")


def validate_dataset_selection(folder_name: str, dataset_name: str) -> None:
    allowed_datasets = FOLDER_TO_DATASETS.get(folder_name)
    if allowed_datasets is None:
        raise ValueError(f"Unsupported folder_name: {folder_name}")
    if dataset_name not in allowed_datasets:
        allowed_text = "\n  - ".join(allowed_datasets)
        raise ValueError(
            f"dataset_name '{dataset_name}' is not in folder '{folder_name}'.\n"
            f"Allowed datasets:\n  - {allowed_text}"
        )


def parse_dataset_name(value: str) -> str:
    if value not in ALL_DATASET_NAMES:
        raise argparse.ArgumentTypeError(
            "Unsupported --dataset-name. Use --list-supported-datasets to see candidates."
        )
    return value


def parse_folder_name(value: str) -> str:
    if value not in ALL_FOLDER_NAMES:
        raise argparse.ArgumentTypeError(
            "Unsupported --folder-name. Use --list-supported-datasets to see candidates."
        )
    return value


def resolve_snapshot_dir(base_path: str, folder_name: str, dataset_name: str) -> Path:
    snapshots_dir = Path(base_path) / folder_name / dataset_name / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Snapshots dir not found: {snapshots_dir}")
    snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under: {snapshots_dir}")
    return snapshots[-1]


def find_first_parquet_file(snapshot_dir: Path) -> Path:
    data_dir = snapshot_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")
    return parquet_files[0]


def get_wbt_hand_order(hand_type: str) -> list[str]:
    if hand_type == "g1_with_brainco":
        return WBT_HAND_ORDER_BRAINCO
    if hand_type == "g1_with_inspire":
        return WBT_HAND_ORDER_INSPIRE
    raise ValueError(f"Unsupported WBT hand type: {hand_type}")


def get_joint_order(control_type: str, hand_type: str) -> list[str]:
    if control_type == "WBT":
        hand_order = get_wbt_hand_order(hand_type)
        return WBT_BODY_JOINT_ORDER + hand_order

    if hand_type == "g1_with_dex3":
        return UPPER_BODY_JOINT_ORDER_DEX3
    if hand_type == "g1_with_inspire":
        raise ValueError("Upper-body Inspire joint order is not defined.")
    return UPPER_BODY_JOINT_ORDER_BRAINCO


def get_hand_groups_from_order(hand_order: list[str]) -> dict[str, list[str]]:
    half = len(hand_order) // 2
    return {"left_hand": hand_order[:half], "right_hand": hand_order[half:]}


def get_joint_groups(control_type: str, hand_type: str) -> dict[str, list[str]]:
    if control_type == "WBT":
        hand_order = get_wbt_hand_order(hand_type)
        hand_groups = get_hand_groups_from_order(hand_order)
        return {
            "left_arm": UPPER_BODY_GROUPS["left_arm"],
            "right_arm": UPPER_BODY_GROUPS["right_arm"],
            "left_hand": hand_groups["left_hand"],
            "right_hand": hand_groups["right_hand"],
            "waist": WBT_GROUPS["waist"],
            "left_leg": WBT_GROUPS["left_leg"],
            "right_leg": WBT_GROUPS["right_leg"],
        }

    if hand_type == "g1_with_dex3":
        return {
            "left_arm": UPPER_BODY_GROUPS["left_arm"],
            "right_arm": UPPER_BODY_GROUPS["right_arm"],
            "left_hand": DEX3_HAND_GROUPS["left_hand"],
            "right_hand": DEX3_HAND_GROUPS["right_hand"],
        }
    if hand_type == "g1_with_inspire":
        raise ValueError("Upper-body Inspire joint groups are not defined.")
    return UPPER_BODY_GROUPS


def build_output_dir(control_type: str, hand_type: str, dataset_name: str) -> Path:
    return GRAPH_ROOT_DIR / control_type / hand_type / dataset_name


def load_episode_dataframe(parquet_path: Path, control_type: str) -> pd.DataFrame:
    if control_type == "WBT":
        columns = [
            "episode_index",
            "frame_index",
            "observation.state.robot_q_current",
            "observation.state.hand_state",
        ]
    else:
        columns = ["episode_index", "frame_index", "observation.state"]
    df = pd.read_parquet(parquet_path, columns=columns)
    print(f"DataFrame columns: {list(df.columns)}")
    if len(df) > 0:
        print(f"First row frame_index/state sample: {df.iloc[0].to_dict()}" )
    return df


def prepare_state_matrix(
    episode_df: pd.DataFrame,
    control_type: str,
    hand_type: str,
    joint_order: list[str],
) -> np.ndarray:
    if control_type == "WBT":
        robot_q = np.stack(
            episode_df["observation.state.robot_q_current"].to_numpy()
        )
        hand_state = np.stack(episode_df["observation.state.hand_state"].to_numpy())
        print(f"robot_q.shape={robot_q.shape}, hand_state.shape={hand_state.shape}")
        body_start = 7
        body_end = body_start + len(WBT_BODY_JOINT_ORDER)
        if robot_q.shape[1] < body_end:
            raise ValueError(
                "robot_q_current does not contain enough joint values."
            )
        body_q = robot_q[:, body_start:body_end]
        print(f"body_q.shape={body_q.shape}, expected body cols={len(WBT_BODY_JOINT_ORDER)}")
        state_matrix = np.hstack([body_q, hand_state])
    else:
        state_series = episode_df["observation.state"]
        state_matrix = np.stack(state_series.to_numpy())
        print(f"state_matrix.shape={state_matrix.shape} (upper-body)")

    if state_matrix.shape[1] != len(joint_order):
        raise ValueError(
            f"State length mismatch: got {state_matrix.shape[1]}, "
            f"expected {len(joint_order)} for {control_type}/{hand_type}."
        )
    return state_matrix


def plot_episode(
    episode_df: pd.DataFrame,
    episode_index: int,
    joint_order: list[str],
    joint_groups: dict[str, list[str]],
    output_dir: Path,
    control_type: str,
    hand_type: str,
) -> None:
    episode_df = episode_df.sort_values("frame_index")
    frame_seconds = episode_df["frame_index"].to_numpy(dtype=float) / FPS
    state_matrix = prepare_state_matrix(
        episode_df,
        control_type,
        hand_type,
        joint_order,
    )

    joint_to_index = {name: idx for idx, name in enumerate(joint_order)}
    group_items = list(joint_groups.items())

    fig_height = max(3.0, 2.5 * len(group_items))
    fig, axes = plt.subplots(
        nrows=len(group_items),
        ncols=1,
        figsize=(16, fig_height),
        sharex=True,
    )
    if len(group_items) == 1:
        axes = [axes]

    for ax, (group_name, joints) in zip(axes, group_items):
        plotted = False
        plotted_joints = []
        missing_joints = []
        for joint_name in joints:
            if joint_name not in joint_to_index:
                missing_joints.append(joint_name)
                continue
            joint_idx = joint_to_index[joint_name]
            ax.plot(frame_seconds, state_matrix[:, joint_idx], label=joint_name)
            plotted = True
            plotted_joints.append(joint_name)

        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No matching joints",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        if plotted_joints:
            print(f"Group '{group_name}': plotted {len(plotted_joints)} joints: {plotted_joints}")
        if missing_joints:
            print(f"Group '{group_name}': missing joints: {missing_joints}")
        ax.set_ylabel("state")
        ax.set_title(group_name)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.xaxis.set_major_locator(MultipleLocator(X_TICK_SECONDS))

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Episode {episode_index}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode_{episode_index}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    class HelpFormatter(RawTextHelpFormatter):
        def __init__(self, prog: str) -> None:
            super().__init__(prog, max_help_position=36, width=110)

    description = (
        "Plot joint-group graphs for Hugging Face G1 robot datasets.\n"
        "\n"
        "Reads parquet files from the local HuggingFace hub cache and saves\n"
        "one PNG per episode under graphs/unannotated_graphs_sample/.\n"
        "\n"
        "Control type and hand type are inferred automatically from the\n"
        "--folder-name / --dataset-name pair:\n"
        "  Control types : Upper_body_control | WBT\n"
        "  Hand types    : g1_with_brainco | g1_with_dex3 | g1_with_inspire"
    )

    epilog = (
        "----------------------------------------------------------------\n"
        "Examples\n"
        "----------------------------------------------------------------\n"
        "  # Plot first 3 episodes of a Brainco upper-body dataset\n"
        "  python unannotated_graph_visualization.py \\\n"
        "    --folder-name  UnifoLM_G1_Brainco_Dataset \\\n"
        "    --dataset-name datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset \\\n"
        "    --max-episodes 3\n"
        "\n"
        "  # Print all supported folder / dataset combinations\n"
        "  python unannotated_graph_visualization.py --list-supported-datasets\n"
        "----------------------------------------------------------------\n"
        "Full dataset list: docs/hf_dataset_installed.md\n"
    )

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=HelpFormatter,
        epilog=epilog,
    )

    # ── Dataset selection ────────────────────────────────────────────────
    ds = parser.add_argument_group("Dataset selection")
    ds.add_argument(
        "--folder-name",
        default=FOLDER_NAME,
        type=parse_folder_name,
        metavar="FOLDER_NAME",
        help=(
            "Folder name under the HuggingFace hub cache.\n"
            f"  default : {FOLDER_NAME}\n"
            "  options : UnifoLM_G1_Brainco_Dataset\n"
            "            UnifoLM_G1_Dex1_Dataset\n"
            "            UnifoLM_G1_Dex3_Dataset\n"
            "            UnifoLM_WBT_Dataset\n"
        ),
    )
    ds.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        type=parse_dataset_name,
        metavar="DATASET_NAME",
        help=(
            "Dataset name (must include the 'datasets--' prefix).\n"
            f"  default : {DATASET_NAME}\n"
            "  example : datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset\n"
            "            datasets--unitreerobotics--G1_Dex3_PickBottle_Dataset\n"
            "            datasets--unitreerobotics--G1_WBT_Inspire_Put_Drinks_Into_Fridge\n"
        ),
    )

    # ── Visualization options ────────────────────────────────────────────
    viz = parser.add_argument_group("Visualization options")
    viz.add_argument(
        "--max-episodes",
        type=int,
        default=MAX_EPISODES_TO_PLOT,
        metavar="N",
        help=(
            "Number of episodes to plot, starting from episode 0.\n"
            f"  default : {MAX_EPISODES_TO_PLOT}\n"
        ),
    )

    # ── Utility ──────────────────────────────────────────────────────────
    util = parser.add_argument_group("Utility")
    util.add_argument(
        "--list-supported-datasets",
        action="store_true",
        help="Print all supported folder / dataset combinations and exit.",
    )

    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()
        if args.list_supported_datasets:
            print("Supported folder/dataset combinations:")
            for folder_name in ALL_FOLDER_NAMES:
                print(f"- {folder_name}")
                for dataset_name in FOLDER_TO_DATASETS[folder_name]:
                    print(f"    * {dataset_name}")
            return

        validate_dataset_selection(args.folder_name, args.dataset_name)
        control_type = classify_control_type(args.folder_name, args.dataset_name)
        hand_type = classify_hand_type(args.folder_name, args.dataset_name)
        validate_configuration(control_type, hand_type)
        print(f"Folder name: {args.folder_name}")
        print(f"Dataset name: {args.dataset_name}")
        print(f"Control type: {control_type}")
        print(f"Hand type: {hand_type}")
        joint_order = get_joint_order(control_type, hand_type)
        joint_groups = get_joint_groups(control_type, hand_type)
        print(f"Joint order length: {len(joint_order)}")
        print(f"Joint groups: {list(joint_groups.keys())}")

        snapshot_dir = resolve_snapshot_dir(BASIC_PATH, args.folder_name, args.dataset_name)
        print(f"Snapshot dir: {snapshot_dir}")
        parquet_path = find_first_parquet_file(snapshot_dir)
        print(f"Loading parquet file: {parquet_path}")
        df = load_episode_dataframe(parquet_path, control_type)
        print(f"Loaded rows: {len(df)}")
        print(f"Episode indices found: {sorted(df['episode_index'].unique().tolist())}")

        output_dir = build_output_dir(control_type, hand_type, args.dataset_name)
        print(f"Output dir: {output_dir}")
        for episode_index in range(args.max_episodes):
            episode_df = df[df["episode_index"] == episode_index]
            if episode_df.empty:
                print(f"Episode {episode_index}: no data, skipping")
                continue
            print(f"Episode {episode_index}: {len(episode_df)} frames")
            plot_episode(
                episode_df,
                episode_index,
                joint_order,
                joint_groups,
                output_dir,
                control_type,
                hand_type,
            )
            print(f"Episode {episode_index}: saved plot")
    except Exception:
        import traceback

        print("An error occurred:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()


