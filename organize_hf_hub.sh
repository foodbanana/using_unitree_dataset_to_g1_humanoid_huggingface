#!/usr/bin/env bash
set -euo pipefail

# Organize Hugging Face hub dataset cache directories into category folders.
# Default hub path can be overridden by the first argument.
# Usage:
#   ./organize_hf_hub.sh
#   ./organize_hf_hub.sh /mnt/hdd/huggingface/hub
#   ./organize_hf_hub.sh --dry-run
#   ./organize_hf_hub.sh -n /mnt/hdd/huggingface/hub

DRY_RUN=0
HUB_DIR="/mnt/hdd/huggingface/hub"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
  esac
fi

if [[ $# -gt 0 ]]; then
  HUB_DIR="$1"
fi

if [[ ! -d "$HUB_DIR" ]]; then
  echo "[error] Hub directory not found: $HUB_DIR" >&2
  exit 1
fi

# Keep this order: specific pattern first, then broad pattern.
PATTERNS=(
  "datasets--unitreerobotics--G1_Dex1_DiverseManip_*|UnifoLM_G1_Dex1_DiverseManip_Dataset"
  "datasets--unitreerobotics--G1_Dex1_*|UnifoLM_G1_Dex1_Dataset"
  "datasets--unitreerobotics--G1_WBT_*|UnifoLM_WBT_Dataset"
  "datasets--unitreerobotics--VLA-0*|UnifoLM-VLA-0"
  "datasets--unitreerobotics--WMA-0*|UnifoLM-WMA-0"
  "datasets--unitreerobotics--G1_Dex3_*|UnifoLM_G1_Dex3_Dataset"
  "datasets--unitreerobotics--G1_Brainco_*|UnifoLM_G1_Brainco_Dataset"
  "datasets--unitreerobotics--Z1_*|UnifoLM_Z1_Arm_Dataset"
)

mkdir_target() {
  local target="$1"
  if [[ ! -d "$target" ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] mkdir -p $target"
    else
      mkdir -p "$target"
    fi
  fi
}

move_one() {
  local src="$1"
  local dst="$2"

  if [[ ! -d "$src" ]]; then
    return 0
  fi

  if [[ -e "$dst" ]]; then
    echo "[skip] target already exists: $dst"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] mv $src $dst"
  else
    mv "$src" "$dst"
    echo "[moved] $(basename "$src") -> $(basename "$(dirname "$dst")")/"
  fi
}

echo "[info] hub_dir: $HUB_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[info] mode: dry-run"
fi

moved_count=0
skip_count=0

for entry in "${PATTERNS[@]}"; do
  pattern="${entry%%|*}"
  folder="${entry##*|}"

  target_dir="$HUB_DIR/$folder"
  mkdir_target "$target_dir"

  # nullglob makes unmatched glob expand to nothing instead of itself.
  shopt -s nullglob
  matches=("$HUB_DIR"/$pattern)
  shopt -u nullglob

  for src in "${matches[@]}"; do
    dst="$target_dir/$(basename "$src")"

    if [[ -e "$dst" ]]; then
      echo "[skip] target already exists: $dst"
      ((skip_count+=1))
      continue
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] mv $src $dst"
      ((moved_count+=1))
    else
      mv "$src" "$dst"
      echo "[moved] $(basename "$src") -> $folder/"
      ((moved_count+=1))
    fi
  done
done

echo "[done] moved: $moved_count, skipped(existing): $skip_count"
