import os
import inspect
from pathlib import Path

# 1. HF 캐시를 직접 마운트된 16TB 하드 드라이브 경로로 강제 지정 (가장 확실한 방법)
target_cache_dir = "/mnt/hdd/huggingface"
os.environ["HF_HOME"] = target_cache_dir

from huggingface_hub import get_collection, snapshot_download

# 2. 대상 컬렉션 정의
namespace = "unitreerobotics"
collection_slugs = [
    "unifolm-g1-dex1-diversemanip-dataset",
    "unifolm-wbt-dataset",
    "unifolm-vla-0",
    "unifolm-wma-0",
    "unifolm-g1-brainco-dataset",
    "unifolm-g1-dex3-dataset",
    "unifolm-z1-arm-dataset",
    "unifolm-g1-dex1-dataset"
]

hub_base_dir = Path(target_cache_dir) / "hub"
slug_to_folder = {
    "unifolm-g1-dex1-diversemanip-dataset": "UnifoLM_G1_Dex1_DiverseManip_Dataset",
    "unifolm-wbt-dataset": "UnifoLM_WBT_Dataset",
    "unifolm-vla-0": "UnifoLM-VLA-0",
    "unifolm-wma-0": "UnifoLM-WMA-0",
    "unifolm-g1-brainco-dataset": "UnifoLM_G1_Brainco_Dataset",
    "unifolm-g1-dex3-dataset": "UnifoLM_G1_Dex3_Dataset",
    "unifolm-z1-arm-dataset": "UnifoLM_Z1_Arm_Dataset",
    "unifolm-g1-dex1-dataset": "UnifoLM_G1_Dex1_Dataset",
}

# True면 다운로드 후 datasets--... 캐시 폴더를 지정 폴더로 이동합니다.
organize_after_download = True


def ensure_cache_dir_ready(cache_dir: str) -> bool:
    """캐시 디렉토리 생성 및 쓰기 가능 여부를 사전 점검합니다."""
    path = Path(cache_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)

        probe_file = path / ".hf_write_probe"
        with probe_file.open("w", encoding="utf-8") as f:
            f.write("ok")
        probe_file.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"[치명적 오류] HF 캐시 경로를 사용할 수 없습니다: {cache_dir}")
        print(f"원인: {e}")
        print("- 디렉토리 마운트 상태와 쓰기 권한을 확인하세요.")
        return False


def classify_error(e: Exception) -> str:
    """운영 시 원인 파악이 쉽도록 예외를 간단히 분류합니다."""
    # huggingface_hub의 HTTP 예외는 보통 response.status_code를 포함합니다.
    status_code = getattr(getattr(e, "response", None), "status_code", None)
    if status_code in (401, 403):
        return "인증/권한 오류 (HF 토큰 필요 가능)"
    if status_code == 404:
        return "리소스 없음 (컬렉션/데이터셋 ID 확인 필요)"
    if status_code is not None:
        return f"허브 HTTP 오류 (status={status_code})"

    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        return "네트워크/스토리지 오류"
    return "기타 오류"


def snapshot_download_compat(repo_id: str) -> str:
    """huggingface_hub 버전에 따라 snapshot_download 인자를 호환 처리합니다."""
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "max_workers": 4,
    }
    # 일부 버전에서는 resume_download가 제거/변경될 수 있어 동적으로 추가합니다.
    if "resume_download" in inspect.signature(snapshot_download).parameters:
        kwargs["resume_download"] = True
    return snapshot_download(**kwargs)


def get_organized_cache_dir(repo_id: str, slug: str) -> Path | None:
    folder_name = slug_to_folder.get(slug)
    if not folder_name:
        return None
    cache_dir_name = f"datasets--{repo_id.replace('/', '--')}"
    return hub_base_dir / folder_name / cache_dir_name


def ensure_organize_folders_ready() -> None:
    for folder_name in slug_to_folder.values():
        (hub_base_dir / folder_name).mkdir(parents=True, exist_ok=True)


def move_cache_dir_to_collection_folder(snapshot_path: str, slug: str) -> None:
    folder_name = slug_to_folder.get(slug)
    if not folder_name:
        return

    # snapshot_path: .../hub/datasets--.../snapshots/<hash>
    src_snapshot = Path(snapshot_path)
    src_cache_dir = src_snapshot.parents[1]
    dst_cache_dir = hub_base_dir / folder_name / src_cache_dir.name

    # 이미 정리된 상태라면 이동하지 않습니다.
    if src_cache_dir == dst_cache_dir or not src_cache_dir.exists():
        return

    if dst_cache_dir.exists():
        print(f"  [i] 이미 정리된 폴더가 존재하여 이동을 건너뜁니다: {dst_cache_dir}")
        return

    src_cache_dir.rename(dst_cache_dir)
    print(f"  [정리] {src_cache_dir.name} -> {folder_name}/")

def main():
    print(f"HF_HOME이 다음 경로로 설정되었습니다: {os.environ['HF_HOME']}")
    print("Unitree 컬렉션 다운로드 파이프라인을 시작합니다...\n")

    if not ensure_cache_dir_ready(target_cache_dir):
        return

    if organize_after_download:
        ensure_organize_folders_ready()

    success_count = 0
    fail_count = 0

    for slug in collection_slugs:
        full_collection_id = f"{namespace}/{slug}"
        print(f"--- 컬렉션 가져오기: {full_collection_id} ---")
        
        try:
            collection = get_collection(full_collection_id)
            datasets = [item for item in collection.items if item.item_type == "dataset"]
            
            if not datasets:
                print(f"  [i] {slug}에서 데이터셋을 찾을 수 없습니다.")
                continue

            for dataset in datasets:
                repo_id = dataset.item_id
                print(f"  -> 데이터셋 다운로드 시작: {repo_id}")
                existing_organized_cache = get_organized_cache_dir(repo_id, slug)
                if (
                    organize_after_download
                    and existing_organized_cache is not None
                    and existing_organized_cache.exists()
                ):
                    print(f"  [i] 이미 정리된 캐시가 존재하여 다운로드를 건너뜁니다: {existing_organized_cache}")
                    success_count += 1
                    continue
                
                try:
                    snapshot_path = snapshot_download_compat(repo_id)
                    if organize_after_download:
                        move_cache_dir_to_collection_folder(snapshot_path, slug)
                    print(f"  [성공] 다운로드 완료: {repo_id}")
                    success_count += 1
                    
                except Exception as e:
                    fail_count += 1
                    error_type = classify_error(e)
                    print(f"  [오류] {repo_id} 다운로드 실패 ({error_type}). 오류 메시지: {e}")
                    if "인증/권한" in error_type:
                        print("  힌트: HF_TOKEN 환경변수 또는 `huggingface-cli login` 상태를 확인하세요.")
                    print("  다음 데이터셋으로 이동합니다...")
                    
        except Exception as e:
            error_type = classify_error(e)
            print(f"[오류] {full_collection_id} 컬렉션을 가져올 수 없습니다 ({error_type}). 오류 메시지: {e}")
            
    print("\n파이프라인 실행이 완료되었습니다.")
    print(f"요약: 성공 {success_count}건, 실패 {fail_count}건")
    print("건너뛴 항목이 있는지 위 로그를 확인하세요.")

if __name__ == "__main__":
    main()