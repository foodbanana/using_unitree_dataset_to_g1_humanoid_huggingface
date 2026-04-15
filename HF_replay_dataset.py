import time

import mujoco
import mujoco.viewer
import numpy as np
from datasets import load_dataset

from replay_args import build_replay_args


# Dataset 28개 관절 이름(순서 중요) -> MuJoCo XML joint 이름 매핑
DATASET_TO_MUJOCO = {
    "kLeftShoulderPitch": "left_shoulder_pitch_joint",
    "kLeftShoulderRoll": "left_shoulder_roll_joint",
    "kLeftShoulderYaw": "left_shoulder_yaw_joint",
    "kLeftElbow": "left_elbow_joint",
    "kLeftWristRoll": "left_wrist_roll_joint",
    "kLeftWristPitch": "left_wrist_pitch_joint",
    "kLeftWristYaw": "left_wrist_yaw_joint",
    "kRightShoulderPitch": "right_shoulder_pitch_joint",
    "kRightShoulderRoll": "right_shoulder_roll_joint",
    "kRightShoulderYaw": "right_shoulder_yaw_joint",
    "kRightElbow": "right_elbow_joint",
    "kRightWristRoll": "right_wrist_roll_joint",
    "kRightWristPitch": "right_wrist_pitch_joint",
    "kRightWristYaw": "right_wrist_yaw_joint",
    "kLeftHandThumb0": "left_hand_thumb_0_joint",
    "kLeftHandThumb1": "left_hand_thumb_1_joint",
    "kLeftHandThumb2": "left_hand_thumb_2_joint",
    "kLeftHandMiddle0": "left_hand_middle_0_joint",
    "kLeftHandMiddle1": "left_hand_middle_1_joint",
    "kLeftHandIndex0": "left_hand_index_0_joint",
    "kLeftHandIndex1": "left_hand_index_1_joint",
    "kRightHandThumb0": "right_hand_thumb_0_joint",
    "kRightHandThumb1": "right_hand_thumb_1_joint",
    "kRightHandThumb2": "right_hand_thumb_2_joint",
    "kRightHandIndex0": "right_hand_index_0_joint",
    "kRightHandIndex1": "right_hand_index_1_joint",
    "kRightHandMiddle0": "right_hand_middle_0_joint",
    "kRightHandMiddle1": "right_hand_middle_1_joint",
}

DATASET_JOINT_ORDER = list(DATASET_TO_MUJOCO.keys())


def parse_args():
    return build_replay_args(
        description="Replay Unitree G1 Dex3 frames in MuJoCo (kinematic qpos replay).",
        default_repo_id="unitreerobotics/G1_Dex3_PickBottle_Dataset",
        default_xml_path="mujoco_menagerie/unitree_g1/g1_with_hands.xml",
        default_num_frames=300,
    )


def load_episode_data(dataset, episode_index: int):
    episode = dataset.filter(lambda x: x["episode_index"] == episode_index)
    states = np.asarray(episode["observation.state"], dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != 28:
        raise ValueError(
            f"Expected observation.state shape [N, 28], got {states.shape}."
        )
    print(f"[INFO] Episode {episode_index}: {len(states)} frames")
    return episode, states


def apply_mapped_qpos(
    data: mujoco.MjData,
    mapping: list[tuple[int, int, str, str]],
    frame: np.ndarray,
) -> None:
    for qpos_adr, col_idx, _, _ in mapping:
        data.qpos[qpos_adr] = frame[col_idx]


def build_qpos_mapping(model: mujoco.MjModel) -> list[tuple[int, int, str, str]]:
    mapping = []
    for col_idx, ds_joint in enumerate(DATASET_JOINT_ORDER):
        mj_joint = DATASET_TO_MUJOCO[ds_joint]
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_joint)
        if jnt_id == -1:
            raise KeyError(f"MuJoCo joint not found in XML: {mj_joint} (from {ds_joint})")
        qpos_adr = int(model.jnt_qposadr[jnt_id])
        mapping.append((qpos_adr, col_idx, ds_joint, mj_joint))
    return mapping


def reset_to_stand_keyframe_if_exists(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_forward(model, data)


def main() -> None:
    args = parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    reset_to_stand_keyframe_if_exists(model, data)

    mapping = build_qpos_mapping(model)
    print("[INFO] 28-joint mapping loaded:")
    for qpos_adr, _, ds_joint, mj_joint in mapping:
        print(f"  {ds_joint:>20s} -> {mj_joint:<28s} (qpos[{qpos_adr}])")

    print(f"[INFO] Loading dataset: {args.repo_id} ({args.split})")
    dataset = load_dataset(args.repo_id, split=args.split)

    start_episode = args.episode_index
    num_episodes = max(1, args.num_episodes)
    start_frame = max(0, args.start_frame)
    num_frames = max(1, args.num_frames)

    sequence: list[tuple[int, np.ndarray]] = []
    for ep in range(start_episode, start_episode + num_episodes):
        _, states = load_episode_data(dataset, ep)
        end_frame = min(len(states), start_frame + num_frames)
        if start_frame >= len(states):
            raise IndexError(
                f"start-frame {start_frame} is out of range for episode {ep} with {len(states)} frames."
            )
        window = states[start_frame:end_frame]
        sequence.append((ep, window))
        print(
            f"[INFO] Episode {ep}: replay frames [{start_frame}:{end_frame}) -> {len(window)} frames"
        )

    total_frames = sum(len(frames) for _, frames in sequence)
    print(
        f"[INFO] Replaying {len(sequence)} episodes (total {total_frames} frames) at {args.fps:.1f} FPS"
    )

    seq_idx = 0
    frame_idx = 0
    frame_dt = 1.0 / args.fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            tick = time.time()

            _, frames = sequence[seq_idx]
            frame = frames[frame_idx]

            # 28개 매핑 관절만 업데이트하고 나머지 DoF(다리/허리 등)는 stand 포즈 유지
            apply_mapped_qpos(data, mapping, frame)

            # 동역학 제어 대신 순수 기구학 갱신으로 시각화
            mujoco.mj_kinematics(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()

            frame_idx += 1
            if frame_idx >= len(frames):
                seq_idx += 1
                frame_idx = 0

                if seq_idx >= len(sequence):
                    if args.loop:
                        seq_idx = 0
                    else:
                        break

                # 에피소드 전환 시 첫 프레임으로 즉시 스냅하고 기구학 갱신
                _, next_frames = sequence[seq_idx]
                apply_mapped_qpos(data, mapping, next_frames[0])
                mujoco.mj_kinematics(model, data)
                mujoco.mj_forward(model, data)
                viewer.sync()

                # 전환 직후 첫 프레임 중복 재생 방지
                frame_idx = 1 if len(next_frames) > 1 else 0

            sleep_t = frame_dt - (time.time() - tick)
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()