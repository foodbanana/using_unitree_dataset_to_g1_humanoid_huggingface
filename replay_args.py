import argparse


def build_replay_args(
    *,
    description: str,
    default_repo_id: str,
    default_xml_path: str,
    default_num_frames: int = 300,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--repo-id",
        type=str,
        default=default_repo_id,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to replay.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of consecutive episodes to replay from episode-index.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame within selected episode.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=default_num_frames,
        help="Number of frames to replay.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback fps.",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default=default_xml_path,
        help="Path to MuJoCo XML model.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop selected frame window continuously.",
    )
    return parser.parse_args()
