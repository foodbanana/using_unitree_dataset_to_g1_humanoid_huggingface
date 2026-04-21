from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# 1) Hardcoded runtime settings
# -----------------------------
BASIC_PATH = "/mnt/hdd/huggingface/hub"
FOLDER_NAME = "UnifoLM_G1_Brainco_Dataset"

# IMPORTANT: this must include the "datasets--" prefix.
DATASET_NAME = "datasets--unitreerobotics--G1_Brainco_GraspOreo_Dataset"

# Choose one of: body, left_leg, right_leg, left_arm, right_arm, left_hand, right_hand 
SELECTED_JOINT_GROUP = "left_arm"

# Plot first N episodes only.
MAX_EPISODES_TO_PLOT = 7

# ----------------------------------
# 2) Dataset joint order (fixed spec in huggingface. Lerobot form)
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


# ---------------------------------
# 3) Joint groups for visualization. Dictionary . 
# ---------------------------------
JOINT_GROUPS = {
    "body": [
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
    ],
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


def find_parquet_file(dataset_root: Path) -> Path:
    """Find a parquet file under snapshots/*/data/**."""
    candidates = sorted(dataset_root.glob("snapshots/*/data/**/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet file found under: {dataset_root}")
    return candidates[0] # 일단 지금은 이름상 제일 빠른 1개의 .parquet 파일만 return. 모든 paraquet 파일을 보고 싶다면 return candidates 라 해야한다.


# joint값이 없는 경우 에러가 발생하기에, selected group에 있는 joint들이 dataset joint order에 모두 있는지 체크하는 함수
def get_index_number_of_joints_in_dataset(selected_group: str) -> tuple[list[str], list[int]]:
    # 안전장치 1 : selected_group이 실존하는지 확인 -> 없으면 error 
    if selected_group not in JOINT_GROUPS:
        raise ValueError(
            f"Invalid SELECTED_JOINT_GROUP={selected_group!r}. "
            f"Valid groups: {list(JOINT_GROUPS.keys())}"
        )
    
    # 안정장치 2 : seleceted_group의 joint값 중 dataset에 없는 값이 있는지 확인 -> 있으면 error  ex) SELECTED_JOINT_GROUP =  body로 할 때 속에 있는 joint 값들이 dataset에 없는 값들이라 에러 발생
    selected_joints = JOINT_GROUPS[selected_group]
    missing_joints = [joint for joint in selected_joints if joint not in DATASET_JOINT_ORDER]
    if missing_joints:
        raise ValueError(
            "Selected group contains joints that are not in DATASET_JOINT_ORDER: "
            f"{missing_joints}"
        )
    # joint : 최상단에서 선택한 group 내부의 joint들
    # DATASET_JOINT_ORDER : 다운받은 dataset 내부의 joint 목록들
    # 내가 선택한 그룹의 joint들이 데이터셋의 
    plot_episode_ids = episode_ids[:MAX_EPISODES_TO_PLOT]몇 번 index 인지를 표기한다
    selected_indices = [DATASET_JOINT_ORDER.index(joint) for joint in selected_joints]
    # return 을 2개를 하기에 튜플 형태로 내보내진다 ex)([이름 리스트], [인덱스 리스트])
    # ["kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", ...], # 첫 번째 보따리 (list[str])
    # [0, 1, 2, ...]                                                        # 두 번째 보따리 (list[int])
    return selected_joints, selected_indices


def main() -> None:
    dataset_root = Path(BASIC_PATH) / FOLDER_NAME / DATASET_NAME # 보고자 하는 데이터셋 폴더 경로
    parquet_file_path = find_parquet_file(dataset_root)          # 그 속에서 parquet_file의 경로 자동 찾아줌
    print(f"[INFO] parquet file: {parquet_file_path}")           

    df = pd.read_parquet(parquet_file_path)                        #  pandas의 read_parquet 함수를 이용해서 parquet 파일을 dataframe 형태로 불러온다.
    print(f"[INFO] loaded rows={len(df)}, cols={len(df.columns)}") #  df는 모든 정보가 있는 표. 행 열 정보 출력

    required_columns = {"observation.state", "episode_index"}      # 읽을 열 정의 (frame_index는 아래에 정의)
    missing_columns = required_columns - set(df.columns)           # 
    
    if missing_columns:                    # "observation.state", "episode_index" 열이 데이터셋에 없을 경우 에러 
        raise KeyError(f"Missing required columns: {missing_columns}")

    # selected joint group 속 joint의 이름 , joint의 index 번호 할당하기
    selected_joints, selected_indices = get_index_number_of_joints_in_dataset(SELECTED_JOINT_GROUP)  # joint의 index 숫자 얻기
    selected_column_name = f"{SELECTED_JOINT_GROUP}_state" # ex) "left_arm_state"

    # Extract only selected joint values from observation.state for each row.
    # 아까 df에서 parquet 파일을 dataframe 형태로 불러왔었다.
    # .apply() : pandas의 함수. 모든 행에 대해 lambda 함수를 적용한다. 
    # .apply()로 불러온 observation.state 열의 각 행의 칸 안에는 26개의 관절 값이 들어있는 리스트가 담겨있다.
    # lambda 인자 : 표현식
    # state 그 26개의 관절값 리스트를 부르는 임시 이름 
    # state를 numpy 배열로 변환한 다음, selected_indices에 해당하는 joint 값들만 추출한다.

    # 기존 df에 추가로 오른쪽에 열 1개를 추가한다. 열 이름은 selected_column_name (ex) "left_arm_state") 이다.
    # 추가한 열의 값은 observation.state에서 selected_indices에 해당하는 joint 값들로 구성된 numpy 배열이 된다.
    df[selected_column_name] = df["observation.state"].apply(  
        lambda state: np.asarray(state)[selected_indices]
    )
    print(f"[INFO] rightmost column name: {df.columns[-1]}")
    print(f"[INFO] rightmost column sample (top 3): {df.iloc[:3, -1].tolist()}")
    
    print(f"[INFO] after adding observation state of selected joints, rows={len(df)}, cols={len(df.columns)}")
    # episode_index 열의 값이 변할때 그것을 잘라서 각각을 그룹화 + sort로 오름차순 배열 
    grouped_episodes = df.groupby("episode_index", sort=True) # episode_index 열을 기준으로 dataframe을 자르고 그룹화한다. episode_index는 각 episode의 고유한 ID를 나타낸다. sort=True는 episode_index를 오름차순으로 정렬한다.
    
    # 안전장치 : episode_index 열이 데이터셋에 없거나, 그룹화 후에 episode_id가 하나도 없는 경우 에러
    episode_ids = sorted(grouped_episodes.groups.keys())
    if not episode_ids:
        raise ValueError("No episodes found in 'episode_index'.")

    print(f"[INFO] total episodes: {len(episode_ids)}")
    print(
        f"[INFO] selected group='{SELECTED_JOINT_GROUP}', "
        f"joints={selected_joints}, indices={selected_indices}"
    )



    # 여기부터는 시각화 코드. episode_id 별로 subplot을 하나씩 만들어서, x축은 frame_index (없으면 timestep), y축은 joint value로 하는 선 그래프를 그린다.
    plot_episode_ids = episode_ids[:MAX_EPISODES_TO_PLOT]
    fig, axes = plt.subplots(
        nrows=len(plot_episode_ids),
        ncols=1,
        figsize=(13, 3.2 * len(plot_episode_ids)),
        squeeze=False,
    )

    for row_idx, episode_id in enumerate(plot_episode_ids):
        episode_data = grouped_episodes.get_group(episode_id)
        x = (
            episode_data["frame_index"].to_numpy()
            if "frame_index" in episode_data.columns
            else np.arange(len(episode_data))
        )
        y = np.vstack(episode_data[selected_column_name].values)

        ax = axes[row_idx, 0]
        lines = ax.plot(x, y)
        ax.set_title(
            f"Episode {episode_id} - {SELECTED_JOINT_GROUP} joint angles",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Frame Index" if "frame_index" in episode_data.columns else "Timestep")
        ax.set_ylabel("Joint Value")
        ax.grid(True, linestyle="--", alpha=0.5)

        if row_idx == 0:
            ax.legend(lines, selected_joints, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()

    backend_name = plt.get_backend().lower()
    # 파일로 저장하는 과정 --> 리눅스일 경우 .png파일로 저장
    if "agg" in backend_name:
        output_dir = Path("/home/taeung/g1_datasets_huggingface/joint_angle_graphs/g1_with_brainco_hand")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{SELECTED_JOINT_GROUP}_episodes_preview.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"[INFO] non-interactive backend detected. saved figure: {output_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()