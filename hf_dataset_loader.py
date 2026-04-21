import os
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


DEFAULT_HF_HOME = "/mnt/hdd/huggingface"


COLLECTION_SLUG_TO_DIR = {
    "unitreerobotics/unifolm-g1-brainco-dataset": "UnifoLM_G1_Brainco_Dataset",
    "unifolm-g1-brainco-dataset": "UnifoLM_G1_Brainco_Dataset",
    "unitreerobotics/unifolm_g1_brainco_dataset": "UnifoLM_G1_Brainco_Dataset",
    "unitreerobotics/unifolm-g1-dex3-dataset": "UnifoLM_G1_Dex3_Dataset",
    "unitreerobotics/unifolm-g1-dex1-dataset": "UnifoLM_G1_Dex1_Dataset",
    "unitreerobotics/unifolm-g1-dex1-diversemanip-dataset": "UnifoLM_G1_Dex1_DiverseManip_Dataset",
    "unitreerobotics/unifolm-z1-arm-dataset": "UnifoLM_Z1_Arm_Dataset",
    "unitreerobotics/unifolm-wbt-dataset": "UnifoLM_WBT_Dataset",
    "unitreerobotics/unifolm-vla-0": "UnifoLM-VLA-0",
    "unitreerobotics/unifolm-wma-0": "UnifoLM-WMA-0",
}


def _parse_repo_id_from_cache_dir(cache_dir_name: str) -> str | None:
    prefix = "datasets--"
    if not cache_dir_name.startswith(prefix):
        return None
    repo_part = cache_dir_name[len(prefix) :]
    if "--" not in repo_part:
        return None
    return repo_part.replace("--", "/", 1)


def _repo_cache_dir_name(repo_id: str) -> str:
    return f"datasets--{repo_id.replace('/', '--')}"


def _discover_collection_repo_ids(hf_home: Path, collection_slug: str) -> list[str]:
    dir_name = COLLECTION_SLUG_TO_DIR.get(collection_slug)
    if not dir_name:
        return []

    collection_dir = hf_home / "hub" / dir_name
    if not collection_dir.exists():
        return []

    repo_ids = []
    for child in sorted(collection_dir.iterdir()):
        if not child.is_dir():
            continue
        repo_id = _parse_repo_id_from_cache_dir(child.name)
        if repo_id is not None:
            repo_ids.append(repo_id)
    return repo_ids


def _discover_cache_dirs_for_repo(hf_home: Path, repo_id: str) -> list[Path]:
    cache_name = _repo_cache_dir_name(repo_id)
    hub_dir = hf_home / "hub"

    result: list[Path] = []
    direct = hub_dir / cache_name
    if direct.exists():
        result.append(direct)

    if hub_dir.exists():
        for child in hub_dir.iterdir():
            if not child.is_dir():
                continue
            nested = child / cache_name
            if nested.exists() and nested not in result:
                result.append(nested)

    return result


def _discover_local_parquet_files(hf_home: Path, repo_id: str) -> list[str]:
    parquet_files: list[str] = []
    for cache_dir in _discover_cache_dirs_for_repo(hf_home, repo_id):
        snapshots = cache_dir / "snapshots"
        if not snapshots.exists():
            continue
        for snapshot_hash_dir in sorted(snapshots.iterdir()):
            if not snapshot_hash_dir.is_dir():
                continue
            data_dir = snapshot_hash_dir / "data"
            if not data_dir.exists():
                continue
            for parquet_path in sorted(data_dir.rglob("*.parquet")):
                parquet_files.append(str(parquet_path))
    return parquet_files


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_dataset_with_cache_fallback(
    repo_id: str,
    *,
    split: str | None = None,
    local_files_only: bool = False,
    hf_home: str = DEFAULT_HF_HOME,
):
    # Keep existing explicit HF_HOME if already set by the caller.
    os.environ.setdefault("HF_HOME", hf_home)

    hf_home_path = Path(os.environ["HF_HOME"])  # reflect effective environment
    repo_id = repo_id.strip()

    repo_try_list = _unique_preserve_order(
        [repo_id] + _discover_collection_repo_ids(hf_home_path, repo_id)
    )

    last_error: Exception | None = None
    for candidate in repo_try_list:
        try:
            kwargs = {
                "path": candidate,
                "local_files_only": local_files_only,
            }
            if split is not None:
                kwargs["split"] = split
            return load_dataset(**kwargs)
        except Exception as exc:
            last_error = exc

    for candidate in repo_try_list:
        parquet_files = _discover_local_parquet_files(hf_home_path, candidate)
        if not parquet_files:
            continue

        try:
            if split is None:
                return load_dataset("parquet", data_files={"train": parquet_files})
            return load_dataset(
                "parquet",
                data_files={split: parquet_files},
                split=split,
            )
        except Exception as exc:
            last_error = exc

    lines = [
        "[ERROR] Failed to load dataset from hub and local cache fallback.",
        f"Requested repo-id: {repo_id}",
        "Attempted repo-ids:",
    ]
    lines.extend([f"  - {x}" for x in repo_try_list])
    lines.append(f"HF_HOME: {hf_home_path}")
    raise RuntimeError("\n".join(lines)) from last_error