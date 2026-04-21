import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

# Force Hugging Face cache location as requested.
os.environ["HF_HOME"] = "/mnt/hdd/huggingface"

from datasets import load_dataset


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


BRAINCO_COLLECTION_ALIASES = {
	"unitreerobotics/unifolm-g1-brainco-dataset",
	"unifolm-g1-brainco-dataset",
	"unitreerobotics/unifolm_g1_brainco_dataset",
}

HF_BRAINCO_COLLECTION_DIR = Path(os.environ["HF_HOME"]) / "hub" / "UnifoLM_G1_Brainco_Dataset"


def local_cache_dirs_for_repo(repo_id: str) -> list[Path]:
	cache_dir_name = f"datasets--{repo_id.replace('/', '--')}"
	hf_hub_dir = Path(os.environ["HF_HOME"]) / "hub"
	paths = [hf_hub_dir / cache_dir_name, HF_BRAINCO_COLLECTION_DIR / cache_dir_name]

	result = []
	for path in paths:
		if path.exists() and path not in result:
			result.append(path)
	return result


def discover_local_parquet_files(repo_id: str) -> list[str]:
	parquet_files: list[str] = []
	for cache_dir in local_cache_dirs_for_repo(repo_id):
		snapshot_dir = cache_dir / "snapshots"
		if not snapshot_dir.exists():
			continue
		for snapshot_hash_dir in sorted(snapshot_dir.iterdir()):
			if not snapshot_hash_dir.is_dir():
				continue
			data_dir = snapshot_hash_dir / "data"
			if not data_dir.exists():
				continue
			for parquet_path in sorted(data_dir.rglob("*.parquet")):
				parquet_files.append(str(parquet_path))
	return parquet_files


# Candidate list is ordered by preference for robust model-name matching.
DATASET_TO_MUJOCO_CANDIDATES = {
	"kLeftShoulderPitch": ["left_shoulder_pitch_joint"],
	"kLeftShoulderRoll": ["left_shoulder_roll_joint"],
	"kLeftShoulderYaw": ["left_shoulder_yaw_joint"],
	"kLeftElbow": ["left_elbow_joint"],
	"kLeftWristRoll": ["left_wrist_roll_joint"],
	"kLeftWristPitch": ["left_wrist_pitch_joint"],
	"kLeftWristYaw": ["left_wrist_yaw_joint"],
	"kRightShoulderPitch": ["right_shoulder_pitch_joint"],
	"kRightShoulderRoll": ["right_shoulder_roll_joint"],
	"kRightShoulderYaw": ["right_shoulder_yaw_joint"],
	"kRightElbow": ["right_elbow_joint"],
	"kRightWristRoll": ["right_wrist_roll_joint"],
	"kRightWristPitch": ["right_wrist_pitch_joint"],
	"kRightWristYaw": ["right_wrist_yaw_joint"],
	"kLeftHandThumb": ["left_thumb_metacarpal_joint", "left_thumb_0_joint"],
	"kLeftHandThumbAux": [
		"left_thumb_proximal_joint",
		"left_thumb_1_joint",
		"left_thumb_distal_joint",
	],
	"kLeftHandIndex": ["left_index_proximal_joint", "left_index_0_joint"],
	"kLeftHandMiddle": ["left_middle_proximal_joint", "left_middle_0_joint"],
	"kLeftHandRing": ["left_ring_proximal_joint", "left_ring_0_joint"],
	"kLeftHandPinky": ["left_pinky_proximal_joint", "left_pinky_0_joint"],
	"kRightHandThumb": ["right_thumb_metacarpal_joint", "right_thumb_0_joint"],
	"kRightHandThumbAux": [
		"right_thumb_proximal_joint",
		"right_thumb_1_joint",
		"right_thumb_distal_joint",
	],
	"kRightHandIndex": ["right_index_proximal_joint", "right_index_0_joint"],
	"kRightHandMiddle": ["right_middle_proximal_joint", "right_middle_0_joint"],
	"kRightHandRing": ["right_ring_proximal_joint", "right_ring_0_joint"],
	"kRightHandPinky": ["right_pinky_proximal_joint", "right_pinky_0_joint"],
}


DATASET_TO_TOKEN_HINTS = {
	"kLeftShoulderPitch": ("left", "shoulder", "pitch"),
	"kLeftShoulderRoll": ("left", "shoulder", "roll"),
	"kLeftShoulderYaw": ("left", "shoulder", "yaw"),
	"kLeftElbow": ("left", "elbow"),
	"kLeftWristRoll": ("left", "wrist", "roll"),
	"kLeftWristPitch": ("left", "wrist", "pitch"),
	"kLeftWristYaw": ("left", "wrist", "yaw"),
	"kRightShoulderPitch": ("right", "shoulder", "pitch"),
	"kRightShoulderRoll": ("right", "shoulder", "roll"),
	"kRightShoulderYaw": ("right", "shoulder", "yaw"),
	"kRightElbow": ("right", "elbow"),
	"kRightWristRoll": ("right", "wrist", "roll"),
	"kRightWristPitch": ("right", "wrist", "pitch"),
	"kRightWristYaw": ("right", "wrist", "yaw"),
	"kLeftHandThumb": ("left", "thumb", "metacarpal"),
	"kLeftHandThumbAux": ("left", "thumb", "proximal"),
	"kLeftHandIndex": ("left", "index", "proximal"),
	"kLeftHandMiddle": ("left", "middle", "proximal"),
	"kLeftHandRing": ("left", "ring", "proximal"),
	"kLeftHandPinky": ("left", "pinky", "proximal"),
	"kRightHandThumb": ("right", "thumb", "metacarpal"),
	"kRightHandThumbAux": ("right", "thumb", "proximal"),
	"kRightHandIndex": ("right", "index", "proximal"),
	"kRightHandMiddle": ("right", "middle", "proximal"),
	"kRightHandRing": ("right", "ring", "proximal"),
	"kRightHandPinky": ("right", "pinky", "proximal"),
}


@dataclass
class JointBinding:
	dataset_index: int
	dataset_name: str
	mujoco_name: str
	qpos_addr: int
	value_scale: float = 1.0
	qpos_min: float | None = None
	qpos_max: float | None = None


# BrainCo hand data has one channel per finger, while this hand model has
# proximal/distal joints; drive distal with the same value to make replay visible.
DATASET_TO_EXTRA_MUJOCO_CANDIDATES = {
	"kLeftHandThumbAux": [("left_thumb_distal_joint", 1.0)],
	"kLeftHandIndex": [("left_index_distal_joint", 1.0)],
	"kLeftHandMiddle": [("left_middle_distal_joint", 1.0)],
	"kLeftHandRing": [("left_ring_distal_joint", 1.0)],
	"kLeftHandPinky": [("left_pinky_distal_joint", 1.0)],
	"kRightHandThumbAux": [("right_thumb_distal_joint", 1.0)],
	"kRightHandIndex": [("right_index_distal_joint", 1.0)],
	"kRightHandMiddle": [("right_middle_distal_joint", 1.0)],
	"kRightHandRing": [("right_ring_distal_joint", 1.0)],
	"kRightHandPinky": [("right_pinky_distal_joint", 1.0)],
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Replay Unitree G1 BrainCo motion in MuJoCo and plot observation.state."
	)
	parser.add_argument(
		"--repo-id",
		type=str,
		default="unitreerobotics/unifolm-g1-brainco-dataset",
		help="Hugging Face dataset repository.",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		help="Dataset split name.",
	)
	parser.add_argument(
		"--xml-path",
		type=str,
		default="/home/taeung/g1_datasets_huggingface/assets/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.xml",
		help="Path to the MuJoCo XML model.",
	)
	parser.add_argument(
		"--episode-index",
		type=int,
		default=0,
		help="Episode index to replay/plot.",
	)
	parser.add_argument(
		"--start-frame",
		type=int,
		default=0,
		help="Start frame index in the selected episode.",
	)
	parser.add_argument(
		"--num-frames",
		type=int,
		default=-1,
		help="Number of frames to use (-1 means all frames from start-frame).",
	)
	parser.add_argument(
		"--fps",
		type=float,
		default=30.0,
		help="Replay frame rate.",
	)
	parser.add_argument(
		"--loop",
		action="store_true",
		help="Loop replay continuously.",
	)
	parser.add_argument(
		"--local-files-only",
		action="store_true",
		help="Load dataset only from local cache.",
	)
	parser.add_argument(
		"--plot",
		action="store_true",
		help="Show matplotlib plots for the selected episode frames.",
	)
	parser.add_argument(
		"--plot-only",
		action="store_true",
		help="Show plot and skip MuJoCo replay.",
	)
	parser.add_argument(
		"--plot-degrees",
		action="store_true",
		help="Convert values from radians to degrees in plots.",
	)
	return parser.parse_args()


def normalize_name(name: str) -> str:
	return "".join(ch for ch in name.lower() if ch.isalnum())


def get_model_joint_names(model: mujoco.MjModel) -> list[str]:
	names = []
	for jnt_id in range(model.njnt):
		name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
		if name is not None:
			names.append(name)
	return names


def resolve_mujoco_joint_name(dataset_joint: str, model_joint_names: list[str]) -> str:
	lower_to_name = {name.lower(): name for name in model_joint_names}
	norm_to_name = {normalize_name(name): name for name in model_joint_names}

	for candidate in DATASET_TO_MUJOCO_CANDIDATES[dataset_joint]:
		key = candidate.lower()
		if key in lower_to_name:
			return lower_to_name[key]

	for candidate in DATASET_TO_MUJOCO_CANDIDATES[dataset_joint]:
		key = normalize_name(candidate)
		if key in norm_to_name:
			return norm_to_name[key]

	required_tokens = DATASET_TO_TOKEN_HINTS[dataset_joint]
	token_hits = []
	for name in model_joint_names:
		lowered = name.lower()
		if all(token in lowered for token in required_tokens):
			token_hits.append(name)

	if len(token_hits) == 1:
		return token_hits[0]

	if len(token_hits) > 1:
		token_hits.sort(key=lambda s: (0 if s.endswith("_joint") else 1, len(s), s))
		return token_hits[0]

	raise KeyError(
		f"Unable to resolve MuJoCo joint for dataset joint '{dataset_joint}'. "
		f"Checked candidates={DATASET_TO_MUJOCO_CANDIDATES[dataset_joint]} and tokens={required_tokens}."
	)


def resolve_candidate_joint_name(candidate: str, model_joint_names: list[str]) -> str | None:
	lower_to_name = {name.lower(): name for name in model_joint_names}
	norm_to_name = {normalize_name(name): name for name in model_joint_names}
	if candidate.lower() in lower_to_name:
		return lower_to_name[candidate.lower()]
	norm = normalize_name(candidate)
	if norm in norm_to_name:
		return norm_to_name[norm]
	return None


def build_joint_bindings(model: mujoco.MjModel) -> list[JointBinding]:
	model_joint_names = get_model_joint_names(model)
	bindings: list[JointBinding] = []

	for dataset_index, dataset_name in enumerate(DATASET_JOINT_ORDER):
		def append_binding(mj_name: str, scale: float) -> None:
			jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
			if jnt_id == -1:
				raise KeyError(f"Resolved MuJoCo joint '{mj_name}' was not found in model.")
			qpos_addr = int(model.jnt_qposadr[jnt_id])
			qpos_min = None
			qpos_max = None
			if int(model.jnt_limited[jnt_id]) != 0:
				qpos_min = float(model.jnt_range[jnt_id][0])
				qpos_max = float(model.jnt_range[jnt_id][1])

			bindings.append(
				JointBinding(
					dataset_index=dataset_index,
					dataset_name=dataset_name,
					mujoco_name=mj_name,
					qpos_addr=qpos_addr,
					value_scale=scale,
					qpos_min=qpos_min,
					qpos_max=qpos_max,
				)
			)

		primary_name = resolve_mujoco_joint_name(dataset_name, model_joint_names)
		append_binding(primary_name, scale=1.0)

		for extra_candidate, extra_scale in DATASET_TO_EXTRA_MUJOCO_CANDIDATES.get(dataset_name, []):
			extra_name = resolve_candidate_joint_name(extra_candidate, model_joint_names)
			if extra_name is None:
				print(
					f"[WARN] Extra coupled joint '{extra_candidate}' for '{dataset_name}' "
					"was not found; continuing without it."
				)
				continue
			if extra_name == primary_name:
				continue
			append_binding(extra_name, scale=extra_scale)

	return bindings


def print_joint_mapping(bindings: list[JointBinding]) -> None:
	print(f"[INFO] Dataset-to-MuJoCo joint mapping ({len(bindings)} bindings):")
	for b in bindings:
		range_txt = ""
		if b.qpos_min is not None and b.qpos_max is not None:
			range_txt = f" range[{b.qpos_min:.3f}, {b.qpos_max:.3f}]"
		print(
			f"  [{b.dataset_index:02d}] {b.dataset_name:<20s} -> "
			f"{b.mujoco_name:<30s} qpos[{b.qpos_addr}] scale={b.value_scale:.2f}{range_txt}"
		)


def load_episode_states(args: argparse.Namespace) -> np.ndarray:
	def parse_repo_id_from_cache_dir(cache_dir_name: str) -> str | None:
		prefix = "datasets--"
		if not cache_dir_name.startswith(prefix):
			return None
		repo_part = cache_dir_name[len(prefix) :]
		if "--" not in repo_part:
			return None
		return repo_part.replace("--", "/", 1)

	def discover_brainco_repo_ids_from_cache() -> list[str]:
		if not HF_BRAINCO_COLLECTION_DIR.exists():
			return []
		repo_ids: list[str] = []
		for child in sorted(HF_BRAINCO_COLLECTION_DIR.iterdir()):
			if not child.is_dir():
				continue
			repo_id = parse_repo_id_from_cache_dir(child.name)
			if repo_id is None:
				continue
			repo_ids.append(repo_id)
		return repo_ids

	def build_repo_try_list(requested_repo_id: str) -> list[str]:
		normalized = requested_repo_id.strip()
		tries = [normalized]

		if normalized in BRAINCO_COLLECTION_ALIASES:
			local_brainco_repos = discover_brainco_repo_ids_from_cache()
			tries.extend(local_brainco_repos)
			# Fallback to known public naming convention even if cache scan is empty.
			tries.append("unitreerobotics/UnifoLM_G1_Brainco_Dataset")

		# Preserve order and remove duplicates.
		seen = set()
		ordered = []
		for item in tries:
			if item and item not in seen:
				ordered.append(item)
				seen.add(item)
		return ordered

	repo_try_list = build_repo_try_list(args.repo_id)

	print(
		f"[INFO] Loading dataset '{args.repo_id}' split='{args.split}' "
		f"(local_files_only={args.local_files_only})"
	)

	last_error: Exception | None = None
	dataset = None
	selected_repo_id = None
	for repo_id in repo_try_list:
		try:
			dataset = load_dataset(
				repo_id,
				split=args.split,
				local_files_only=args.local_files_only,
			)
			selected_repo_id = repo_id
			break
		except Exception as exc:
			last_error = exc

	if dataset is None:
		for repo_id in repo_try_list:
			local_parquet_files = discover_local_parquet_files(repo_id)
			if not local_parquet_files:
				continue
			try:
				print(
					f"[INFO] Loading local parquet fallback for '{repo_id}' "
					f"from {len(local_parquet_files)} files"
				)
				dataset = load_dataset(
					"parquet",
					data_files={args.split: local_parquet_files},
					split=args.split,
				)
				selected_repo_id = repo_id
				break
			except Exception as exc:
				last_error = exc

	if dataset is None:
		help_lines = [
			"[ERROR] Failed to load dataset with all attempted repo IDs:",
		]
		for rid in repo_try_list:
			help_lines.append(f"  - {rid}")

		if args.repo_id in BRAINCO_COLLECTION_ALIASES:
			cache_repo_ids = discover_brainco_repo_ids_from_cache()
			if cache_repo_ids:
				help_lines.append("[HINT] Cached BrainCo dataset repos discovered:")
				for rid in cache_repo_ids:
					help_lines.append(f"  - {rid}")
				help_lines.append(
					"[HINT] Re-run with one concrete repo, e.g. --repo-id unitreerobotics/G1_Brainco_PickApple_Dataset"
				)
			else:
				help_lines.append(
					"[HINT] No local BrainCo cache found under /mnt/hdd/huggingface/hub/UnifoLM_G1_Brainco_Dataset."
				)

		raise RuntimeError("\n".join(help_lines)) from last_error

	if selected_repo_id != args.repo_id:
		print(f"[INFO] Resolved collection-like repo id -> using dataset repo '{selected_repo_id}'")

	episode = dataset.filter(lambda row: row["episode_index"] == args.episode_index)
	states = np.asarray(episode["observation.state"], dtype=np.float64)

	if states.ndim != 2:
		raise ValueError(f"Expected 2D observation.state, got shape {states.shape}.")
	if states.shape[1] != len(DATASET_JOINT_ORDER):
		raise ValueError(
			"Expected observation.state second dimension to be "
			f"{len(DATASET_JOINT_ORDER)}, got {states.shape[1]}."
		)
	if len(states) == 0:
		raise ValueError(f"Episode {args.episode_index} has no frames in split '{args.split}'.")

	start = max(0, args.start_frame)
	if start >= len(states):
		raise IndexError(
			f"start-frame {start} is out of range for episode {args.episode_index} "
			f"with {len(states)} frames."
		)

	if args.num_frames < 0:
		end = len(states)
	else:
		end = min(len(states), start + max(1, args.num_frames))

	selected = states[start:end]
	print(
		f"[INFO] Episode {args.episode_index}: total={len(states)} frames, "
		f"selected=[{start}:{end}) ({len(selected)} frames)"
	)
	return selected


def plot_joint_states(states: np.ndarray, use_degrees: bool = False) -> None:
	values = np.rad2deg(states) if use_degrees else states
	y_label = "Joint value (deg)" if use_degrees else "Joint value (rad)"
	x = np.arange(values.shape[0])

	groups = [
		(
			"Left Arm (7)",
			[
				"kLeftShoulderPitch",
				"kLeftShoulderRoll",
				"kLeftShoulderYaw",
				"kLeftElbow",
				"kLeftWristRoll",
				"kLeftWristPitch",
				"kLeftWristYaw",
			],
		),
		(
			"Right Arm (7)",
			[
				"kRightShoulderPitch",
				"kRightShoulderRoll",
				"kRightShoulderYaw",
				"kRightElbow",
				"kRightWristRoll",
				"kRightWristPitch",
				"kRightWristYaw",
			],
		),
		(
			"Left BrainCo Hand (6)",
			[
				"kLeftHandThumb",
				"kLeftHandThumbAux",
				"kLeftHandIndex",
				"kLeftHandMiddle",
				"kLeftHandRing",
				"kLeftHandPinky",
			],
		),
		(
			"Right BrainCo Hand (6)",
			[
				"kRightHandThumb",
				"kRightHandThumbAux",
				"kRightHandIndex",
				"kRightHandMiddle",
				"kRightHandRing",
				"kRightHandPinky",
			],
		),
	]

	index_by_name = {name: i for i, name in enumerate(DATASET_JOINT_ORDER)}
	fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
	axes = axes.flatten()

	for ax, (title, joint_names) in zip(axes, groups):
		for joint_name in joint_names:
			idx = index_by_name[joint_name]
			ax.plot(x, values[:, idx], linewidth=1.4, label=joint_name)
		ax.set_title(title)
		ax.set_xlabel("Frame index")
		ax.set_ylabel(y_label)
		ax.grid(True, linestyle="--", alpha=0.4)
		ax.legend(fontsize=8, ncol=2)

	fig.suptitle("Unitree G1 BrainCo observation.state Joint Trajectories", fontsize=14)
	fig.tight_layout()
	plt.show()


def reset_pose_and_get_baseline(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
	mujoco.mj_resetData(model, data)
	mujoco.mj_kinematics(model, data)
	mujoco.mj_forward(model, data)
	return data.qpos.copy()


def apply_frame_qpos(
	data: mujoco.MjData,
	baseline_qpos: np.ndarray,
	bindings: list[JointBinding],
	frame: np.ndarray,
) -> None:
	# Keep non-dataset joints (legs/waist/etc.) at baseline, then overwrite mapped 26 joints.
	data.qpos[:] = baseline_qpos
	for b in bindings:
		value = float(frame[b.dataset_index]) * b.value_scale
		if b.qpos_min is not None and b.qpos_max is not None:
			value = float(np.clip(value, b.qpos_min, b.qpos_max))
		data.qpos[b.qpos_addr] = value


def replay_mujoco(
	model: mujoco.MjModel,
	data: mujoco.MjData,
	bindings: list[JointBinding],
	states: np.ndarray,
	fps: float,
	loop: bool,
) -> None:
	baseline_qpos = reset_pose_and_get_baseline(model, data)
	frame_dt = 1.0 / max(1e-6, fps)
	frame_idx = 0

	print(f"[INFO] Starting MuJoCo replay at {fps:.2f} FPS")
	with mujoco.viewer.launch_passive(model, data) as viewer:
		while viewer.is_running():
			tick = time.time()

			apply_frame_qpos(data, baseline_qpos, bindings, states[frame_idx])
			mujoco.mj_kinematics(model, data)
			mujoco.mj_forward(model, data)
			viewer.sync()

			frame_idx += 1
			if frame_idx >= len(states):
				if loop:
					frame_idx = 0
				else:
					break

			elapsed = time.time() - tick
			sleep_t = frame_dt - elapsed
			if sleep_t > 0:
				time.sleep(sleep_t)


def main() -> None:
	args = parse_args()
	states = load_episode_states(args)

	if args.plot or args.plot_only:
		plot_joint_states(states, use_degrees=args.plot_degrees)

	if args.plot_only:
		return

	model = mujoco.MjModel.from_xml_path(args.xml_path)
	data = mujoco.MjData(model)
	bindings = build_joint_bindings(model)
	print_joint_mapping(bindings)

	replay_mujoco(
		model=model,
		data=data,
		bindings=bindings,
		states=states,
		fps=args.fps,
		loop=args.loop,
	)


if __name__ == "__main__":
	main()
