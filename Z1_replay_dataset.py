import time

import mujoco
import mujoco.viewer
import numpy as np
from datasets import load_dataset

from replay_args import build_replay_args


# Dataset joint names (fixed order from user requirement)
DATASET_JOINT_ORDER = [
    "kWaist",
    "kShoulder",
    "kElbow",
    "kForearmRoll",
    "kWristAngle",
    "kWristRotate",
    "kGripper",
]

# Required exact mapping: dataset joint name -> MuJoCo joint name
DATASET_TO_MUJOCO = {
    "kWaist": "joint1",
    "kShoulder": "joint2",
    "kElbow": "joint3",
    "kForearmRoll": "joint4",
    "kWristAngle": "joint5",
    "kWristRotate": "joint6",
    "kGripper": "jointGripper",
}


def parse_args():
    return build_replay_args(
        description="Replay Unitree Z1 StackBox dataset in MuJoCo (kinematic qpos replay).",
        default_repo_id="unitreerobotics/Z1_StackBox_Dataset",
        default_xml_path="mujoco_menagerie/unitree_z1/z1_gripper.xml",
        default_num_frames=300,
    )


def joint_qpos_width(joint_type: int) -> int:
    if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return 1
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    raise ValueError(f"Unsupported joint type: {joint_type}")


def load_episode_states(dataset, episode_index: int) -> np.ndarray:
    episode = dataset.filter(lambda x: x["episode_index"] == episode_index)
    states = np.asarray(episode["observation.state"], dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != len(DATASET_JOINT_ORDER):
        raise ValueError(
            f"Expected observation.state shape [N, {len(DATASET_JOINT_ORDER)}], got {states.shape}."
        )
    if len(states) == 0:
        raise ValueError(f"Episode {episode_index} has no frames.")

    print(f"[INFO] Episode {episode_index}: {len(states)} frames")
    return states


def build_mapped_qpos_indices(model: mujoco.MjModel) -> list[tuple[int, int, str, str]]:
    mapping = []
    for col_idx, dataset_joint_name in enumerate(DATASET_JOINT_ORDER):
        mujoco_joint_name = DATASET_TO_MUJOCO[dataset_joint_name]
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint_name)
        if jnt_id == -1:
            raise KeyError(
                f"MuJoCo joint not found in XML: {mujoco_joint_name} (from {dataset_joint_name})"
            )

        qpos_adr = int(model.jnt_qposadr[jnt_id])
        jnt_type = int(model.jnt_type[jnt_id])
        width = joint_qpos_width(jnt_type)
        if width != 1:
            raise ValueError(
                f"Expected 1-DoF joint for {mujoco_joint_name}, got width={width}."
            )

        mapping.append((qpos_adr, col_idx, dataset_joint_name, mujoco_joint_name))

    return mapping


def build_unmapped_qpos_indices(
    model: mujoco.MjModel,
    mapped_qpos_indices: set[int],
) -> list[int]:
    unmapped = []
    for jnt_id in range(model.njnt):
        qpos_adr = int(model.jnt_qposadr[jnt_id])
        jnt_type = int(model.jnt_type[jnt_id])
        width = joint_qpos_width(jnt_type)
        for offset in range(width):
            qpos_idx = qpos_adr + offset
            if qpos_idx not in mapped_qpos_indices:
                unmapped.append(qpos_idx)
    return sorted(set(unmapped))


def apply_frame_qpos(
    data: mujoco.MjData,
    mapped_indices: list[tuple[int, int, str, str]],
    unmapped_qpos_indices: list[int],
    frame: np.ndarray,
) -> None:
    for qpos_idx in unmapped_qpos_indices:
        data.qpos[qpos_idx] = 0.0

    for qpos_idx, col_idx, _, _ in mapped_indices:
        data.qpos[qpos_idx] = frame[col_idx]


def main() -> None:
    args = parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    mapped_indices = build_mapped_qpos_indices(model)
    mapped_qpos_set = {qpos_idx for qpos_idx, _, _, _ in mapped_indices}
    unmapped_qpos_indices = build_unmapped_qpos_indices(model, mapped_qpos_set)

    print("[INFO] Joint mapping loaded:")
    for qpos_idx, _, ds_name, mj_name in mapped_indices:
        print(f"  {ds_name:>14s} -> {mj_name:<12s} (qpos[{qpos_idx}])")
    print(f"[INFO] Unmapped qpos indices set to zero each frame: {len(unmapped_qpos_indices)}")

    print(f"[INFO] Loading dataset: {args.repo_id} ({args.split})")
    dataset = load_dataset(args.repo_id, split=args.split)
    start_episode = args.episode_index
    num_episodes = max(1, args.num_episodes)
    start_frame = max(0, args.start_frame)
    num_frames = max(1, args.num_frames)

    sequence: list[tuple[int, np.ndarray]] = []
    for ep in range(start_episode, start_episode + num_episodes):
        states = load_episode_states(dataset, ep)
        if start_frame >= len(states):
            raise IndexError(
                f"start-frame {start_frame} is out of range for episode {ep} with {len(states)} frames."
            )
        end_frame = min(len(states), start_frame + num_frames)
        window = states[start_frame:end_frame]
        sequence.append((ep, window))
        print(
            f"[INFO] Episode {ep}: replay frames [{start_frame}:{end_frame}) -> {len(window)} frames"
        )

    total_frames = sum(len(frames) for _, frames in sequence)
    print(
        f"[INFO] Replaying {len(sequence)} episodes (total {total_frames} frames) at {args.fps:.1f} FPS"
    )

    frame_dt = 1.0 / max(args.fps, 1e-6)
    seq_idx = 0
    frame_idx = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            tick = time.time()

            _, replay_frames = sequence[seq_idx]
            frame = replay_frames[frame_idx]
            apply_frame_qpos(data, mapped_indices, unmapped_qpos_indices, frame)

            # Kinematic-only update (no dynamics stepping)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()

            frame_idx += 1
            if frame_idx >= len(replay_frames):
                seq_idx += 1
                frame_idx = 0

                if seq_idx >= len(sequence):
                    if args.loop:
                        seq_idx = 0
                    else:
                        break

                # Snap to first frame of the next episode window.
                _, next_frames = sequence[seq_idx]
                apply_frame_qpos(data, mapped_indices, unmapped_qpos_indices, next_frames[0])
                mujoco.mj_kinematics(model, data)
                mujoco.mj_forward(model, data)
                viewer.sync()
                frame_idx = 1 if len(next_frames) > 1 else 0

            sleep_t = frame_dt - (time.time() - tick)
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
