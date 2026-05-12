from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Runtime settings ────────────────────────────────────────────────────────
BASIC_PATH = "/mnt/hdd/huggingface/hub"
FOLDER_NAME = "UnifoLM_G1_Brainco_Dataset"
DATASET_NAME = "datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset"

FPS = 30
STATIONARY_FRAMES = FPS  # 에피소드 시작 후 1초 = frame_index 0~29

# ─── Dataset catalog (unannotated_graph_visualization.py 와 동일) ─────────────
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

# ─── Joint order definitions (unannotated_graph_visualization.py 와 동일) ────
UPPER_BODY_JOINT_ORDER_BRAINCO = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
    "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb", "kLeftHandThumbAux", "kLeftHandIndex",
    "kLeftHandMiddle", "kLeftHandRing", "kLeftHandPinky",
    "kRightHandThumb", "kRightHandThumbAux", "kRightHandIndex",
    "kRightHandMiddle", "kRightHandRing", "kRightHandPinky",
]

UPPER_BODY_JOINT_ORDER_DEX3 = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
    "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1",
    "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1",
    "kRightHandMiddle0", "kRightHandMiddle1",
]

WBT_BODY_JOINT_ORDER = [
    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw",
    "kLeftKnee", "kLeftAnklePitch", "kLeftAnkleRoll",
    "kRightHipPitch", "kRightHipRoll", "kRightHipYaw",
    "kRightKnee", "kRightAnklePitch", "kRightAnkleRoll",
    "kWaistYaw", "kWaistRoll", "kWaistPitch",
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
    "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
    "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
]

WBT_HAND_ORDER_BRAINCO = [
    "kLeftHandThumb", "kLeftHandThumbAux", "kLeftHandIndex",
    "kLeftHandMiddle", "kLeftHandRing", "kLeftHandPinky",
    "kRightHandThumb", "kRightHandThumbAux", "kRightHandIndex",
    "kRightHandMiddle", "kRightHandRing", "kRightHandPinky",
]

WBT_HAND_ORDER_INSPIRE = [
    "kLeftHandIndex", "kLeftHandMiddle", "kLeftHandRing", "kLeftHandPinky",
    "kLeftHandThumb", "kLeftHandThumbAux",
    "kRightHandIndex", "kRightHandMiddle", "kRightHandRing", "kRightHandPinky",
    "kRightHandThumb", "kRightHandThumbAux",
]

# ─── Helper functions (unannotated_graph_visualization.py 로딩 로직 그대로 사용) ─
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


def get_joint_order(control_type: str, hand_type: str) -> list[str]:
    if control_type == "WBT":
        if hand_type == "g1_with_brainco":
            return WBT_BODY_JOINT_ORDER + WBT_HAND_ORDER_BRAINCO
        if hand_type == "g1_with_inspire":
            return WBT_BODY_JOINT_ORDER + WBT_HAND_ORDER_INSPIRE
        raise ValueError(f"Unsupported WBT hand type: {hand_type}")
    if hand_type == "g1_with_dex3":
        return UPPER_BODY_JOINT_ORDER_DEX3
    return UPPER_BODY_JOINT_ORDER_BRAINCO


# ─── 데이터 로딩 ──────────────────────────────────────────────────────────────
def load_raw_dataframe(parquet_path: Path, control_type: str) -> pd.DataFrame:
    """분석에 필요한 열만 선택하여 parquet 파일 로드."""
    if control_type == "WBT":
        columns = [
            "observation.state.robot_q_current",
            "observation.state.hand_state",
            "frame_index",
            "episode_index",
        ]
    else:
        columns = ["observation.state", "frame_index", "episode_index"]
    return pd.read_parquet(parquet_path, columns=columns)


# ─── 관절 행렬 추출 ───────────────────────────────────────────────────────────
def extract_joint_matrix(
    ep_df: pd.DataFrame,
    control_type: str,
    hand_type: str,
    joint_order: list[str],
) -> np.ndarray:
    """
    에피소드 DataFrame으로부터 (N_frames, N_joints) shape의 numpy 배열 반환.
    WBT의 경우 robot_q_current 앞 7개(root position/orientation)를 제거한 뒤
    hand_state와 합친다.
    """
    if control_type == "WBT":
        robot_q = np.stack(ep_df["observation.state.robot_q_current"].to_numpy())
        hand_state = np.stack(ep_df["observation.state.hand_state"].to_numpy())
        # robot_q: [root_pos(3) + root_quat(4) + body_joints(29)] = 36
        # 앞 7개 제거 → body_joints 29개만 사용
        body_q = robot_q[:, 7: 7 + len(WBT_BODY_JOINT_ORDER)]
        joint_matrix = np.hstack([body_q, hand_state])
    else:
        joint_matrix = np.stack(ep_df["observation.state"].to_numpy())

    if joint_matrix.shape[1] != len(joint_order):
        raise ValueError(
            f"관절 수 불일치: 실제 {joint_matrix.shape[1]}개, "
            f"예상 {len(joint_order)}개 ({control_type}/{hand_type})"
        )
    return joint_matrix


# ─── std 분석 (정지 구간) ─────────────────────────────────────────────────────
def compute_stop_state_median_std(
    df: pd.DataFrame,
    control_type: str,
    hand_type: str,
    joint_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    각 에피소드의 정지 구간(frame_index 0~29)에서 관절별 std를 계산하고,
    전체 에피소드에 걸친 중간값(median std) 배열을 반환한다.

    Returns:
        median_std : shape (N_joints,) - 각 관절의 median std
        all_stds   : shape (N_episodes, N_joints) - 원시 std 행렬
    """
    # 정지 구간 필터 (frame_index 0 ~ 29, 즉 30fps × 1초)
    stationary_df = df[df["frame_index"] <= STATIONARY_FRAMES - 1]

    episode_stds: list[np.ndarray] = []
    skipped = 0

    for ep_idx, ep_df in stationary_df.groupby("episode_index"):
        if len(ep_df) < 2:
            print(f"  [SKIP] episode {ep_idx}: 정지 구간 프레임 {len(ep_df)}개 (최소 2개 필요)")
            skipped += 1
            continue

        joint_matrix = extract_joint_matrix(ep_df, control_type, hand_type, joint_order)
        # 각 관절의 표준편차 (ddof=0, 모집단 표준편차)
        std_per_joint = np.std(joint_matrix, axis=0)
        episode_stds.append(std_per_joint)

    if not episode_stds:
        raise RuntimeError("유효한 에피소드가 없어 std 계산 불가.")

    print(f"  사용된 에피소드 수: {len(episode_stds)}, 스킵된 에피소드 수: {skipped}")
    all_stds = np.array(episode_stds)          # (N_episodes, N_joints)
    median_std = np.median(all_stds, axis=0)   # (N_joints,)
    return median_std, all_stds


# ─── 결과 출력 ────────────────────────────────────────────────────────────────
def print_median_std_results(
    median_std: np.ndarray,
    all_stds: np.ndarray,
    joint_order: list[str],
) -> None:
    n_episodes, n_joints = all_stds.shape
    print()
    print("=" * 65)
    print(f"  정지 구간(0~1초) 관절별 Jitter 분석 결과")
    print(f"  (에피소드 수: {n_episodes}, 관절 수: {n_joints})")
    print("=" * 65)
    print(f"{'#':>4}  {'관절 이름':<32}  {'Median STD':>12}")
    print("-" * 65)
    for i, (name, val) in enumerate(zip(joint_order, median_std)):
        print(f"{i:>4}  {name:<32}  {val:>12.8f}")
    print("=" * 65)
    print()
    print("[Raw] median std numpy 배열:")
    np.set_printoptions(precision=8, suppress=True, linewidth=120)
    print(median_std)
    print()
    print(f"[Raw] 전체 에피소드 std 행렬 shape: {all_stds.shape}")
    print("(추후 threshold 적용 시 위 배열을 활용)")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    class HelpFormatter(RawTextHelpFormatter):
        def __init__(self, prog: str) -> None:
            super().__init__(prog, max_help_position=36, width=110)

    parser = argparse.ArgumentParser(
        description="정지 구간(에피소드 시작 1초) 관절 Jitter 표준편차 분석",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "--folder-name",
        default=FOLDER_NAME,
        choices=ALL_FOLDER_NAMES,
        metavar="FOLDER_NAME",
        help=(
            f"HuggingFace hub 캐시 내 폴더 이름.\n"
            f"  default : {FOLDER_NAME}\n"
            f"  options : {' | '.join(ALL_FOLDER_NAMES)}"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        metavar="DATASET_NAME",
        help=(
            f"데이터셋 이름 ('datasets--' 접두사 포함).\n"
            f"  default : {DATASET_NAME}"
        ),
    )
    parser.add_argument(
        "--list-supported-datasets",
        action="store_true",
        help="지원되는 폴더/데이터셋 목록 출력 후 종료.",
    )
    return parser.parse_args()


# ─── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.list_supported_datasets:
        print("지원되는 폴더/데이터셋 목록:")
        for folder_name in ALL_FOLDER_NAMES:
            print(f"- {folder_name}")
            for ds_name in FOLDER_TO_DATASETS[folder_name]:
                print(f"    * {ds_name}")
        return

    validate_dataset_selection(args.folder_name, args.dataset_name)
    control_type = classify_control_type(args.folder_name, args.dataset_name)
    hand_type = classify_hand_type(args.folder_name, args.dataset_name)

    print(f"폴더명       : {args.folder_name}")
    print(f"데이터셋명   : {args.dataset_name}")
    print(f"제어 타입    : {control_type}")
    print(f"핸드 타입    : {hand_type}")

    joint_order = get_joint_order(control_type, hand_type)
    print(f"관절 수      : {len(joint_order)}")

    # ── parquet 파일 로드 (unannotated_graph_visualization.py 로직 재사용) ──
    snapshot_dir = resolve_snapshot_dir(BASIC_PATH, args.folder_name, args.dataset_name)
    parquet_path = find_first_parquet_file(snapshot_dir)

    # 디버깅용: 로드한 파일 경로 출력
    # HuggingFace 캐시는 snapshots/의 .parquet이 blobs/의 실제 파일을 가리키는 심볼릭 링크이므로
    # resolve() 시 blobs/ 경로가 표시되는 것은 정상 동작임
    print(f"\n[파일 경로 (심볼릭 링크)] {parquet_path}")
    print(f"[파일 경로 (실제 파일)]   {parquet_path.resolve()}")

    df = load_raw_dataframe(parquet_path, control_type)

    # 데이터 형태 및 열 이름 출력
    print(f"[DataFrame] shape = {df.shape}  (행: {df.shape[0]}, 열: {df.shape[1]})")
    print(f"[DataFrame] 열 이름: {list(df.columns)}")
    print(f"[DataFrame] 에피소드 목록: {sorted(df['episode_index'].unique().tolist())}")
    print()

    # ── 정지 구간 std 분석 ────────────────────────────────────────────────────
    print(f"▶ 정지 구간 필터: frame_index 0 ~ {STATIONARY_FRAMES - 1} ({STATIONARY_FRAMES}프레임, {STATIONARY_FRAMES / FPS:.1f}초)")
    median_std, all_stds = compute_stop_state_median_std(df, control_type, hand_type, joint_order)

    print_median_std_results(median_std, all_stds, joint_order)


if __name__ == "__main__":
    main()
