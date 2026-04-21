import argparse

import numpy as np

from hf_dataset_loader import load_dataset_with_cache_fallback


def parse_args():
    parser = argparse.ArgumentParser(description="HF dataset sample shape/type quick checker")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="unitreerobotics/G1_Dex3_PickBottle_Dataset",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument("--index", type=int, default=0, help="Sample index to inspect")
    parser.add_argument(
        "--ts-window",
        type=int,
        default=11,
        help="Number of timestamp points used to estimate average control period",
    )
    return parser.parse_args()


def value_desc(value):
    if isinstance(value, (list, np.ndarray)):
        return str(np.array(value).shape)
    return type(value).__name__


def print_nested(prefix, value):
    # dict 내부를 재귀 순회해서 observation.state 같은 경로로 출력
    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else k
            print_nested(next_prefix, v)
        return

    print(f"{prefix:<40} | {value_desc(value)}")


def main():
    args = parse_args()

    split_ds = load_dataset_with_cache_fallback(args.repo_id, split=args.split)

    print("=== 데이터셋 퀵 체크 ===")
    print(f"repo_id: {args.repo_id}")
    print(f"split: {args.split}")
    print(f"전체 프레임 수(Rows): {len(split_ds)}")

    if len(split_ds) == 0:
        raise ValueError("선택한 split이 비어 있습니다.")

    if args.index < 0 or args.index >= len(split_ds):
        raise IndexError(f"index 범위 오류: {args.index} (0 ~ {len(split_ds) - 1})")

    sample = split_ds[args.index]

    print(f"\n{'항목(Key)':<40} | {'형태(Shape) / 타입'}")
    print("-" * 80)
    print_nested("", sample)

    if "timestamp" in sample:
        ts_window = max(2, args.ts_window)
        ts_sample = split_ds["timestamp"][:ts_window]
        if len(ts_sample) >= 2:
            avg_dt = np.mean(np.diff(ts_sample))
            print(f"\n평균 제어 주기: {avg_dt:.4f}s ({1/avg_dt:.1f}Hz)")


if __name__ == "__main__":
    main()