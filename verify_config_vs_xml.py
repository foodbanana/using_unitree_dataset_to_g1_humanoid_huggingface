#!/usr/bin/env python3
"""Verify YAML joint names against MuJoCo XML models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Set
import xml.etree.ElementTree as ET

import yaml


ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def iter_yaml_joints(base_cfg: dict, hand_cfg: dict) -> Iterable[str]:
    # base_cfg: joint_groups -> group_name -> [joint names]
    joint_groups = base_cfg.get("joint_groups", {})
    if isinstance(joint_groups, dict):
        for group_joints in joint_groups.values():
            if isinstance(group_joints, list):
                for joint in group_joints:
                    if isinstance(joint, str):
                        yield joint

    # hand_cfg: left_hand/right_hand -> [joint names]
    for key in ("left_hand", "right_hand"):
        group = hand_cfg.get(key, [])
        if isinstance(group, list):
            for joint in group:
                if isinstance(joint, str):
                    yield joint


def collect_xml_joints(path: Path) -> Set[str]:
    tree = ET.parse(path)
    root = tree.getroot()

    names: Set[str] = set()
    for elem in root.iter():
        if elem.tag in ("joint", "freejoint"):
            name = elem.get("name")
            if name:
                names.add(name)
    return names


def verify_hand(
    hand_key: str,
    base_cfg: dict,
    hand_cfg_file: Path,
    base_xml: Path,
    hand_xml: Path,
) -> int:
    hand_cfg = load_yaml(hand_cfg_file)
    yaml_joints = list(iter_yaml_joints(base_cfg, hand_cfg))

    xml_joints = collect_xml_joints(base_xml)
    xml_joints.update(collect_xml_joints(hand_xml))

    missing = [name for name in yaml_joints if name not in xml_joints]

    print(f"active_hand: {hand_key}")
    print(f"base_config: {base_cfg}")
    print(f"hand_config: {hand_cfg_file}")
    print(f"base_xml: {base_xml}")
    print(f"hand_xml: {hand_xml}")
    print(f"yaml joints: {len(yaml_joints)}")
    print(f"xml joints: {len(xml_joints)}")

    if missing:
        print(f"{ANSI_RED}Missing joints ({len(missing)}):{ANSI_RESET}")
        for name in missing:
            print(f"{ANSI_RED}- {name}{ANSI_RESET}")
        return len(missing)

    print("All YAML joints exist in the XML model(s).")
    return 0


def print_summary_table(rows: list[tuple[str, str, int]]) -> None:
    headers = ("hand", "status", "missing")
    col1 = max(len(headers[0]), *(len(r[0]) for r in rows)) if rows else len(headers[0])
    col2 = max(len(headers[1]), *(len(r[1]) for r in rows)) if rows else len(headers[1])
    col3 = max(len(headers[2]), *(len(str(r[2])) for r in rows)) if rows else len(headers[2])

    sep = f"+{'-' * (col1 + 2)}+{'-' * (col2 + 2)}+{'-' * (col3 + 2)}+"
    print(sep)
    print(f"| {headers[0].ljust(col1)} | {headers[1].ljust(col2)} | {headers[2].rjust(col3)} |")
    print(sep)
    for hand, status, missing in rows:
        print(f"| {hand.ljust(col1)} | {status.ljust(col2)} | {str(missing).rjust(col3)} |")
    print(sep)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify YAML joints vs MuJoCo XML.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all hand models listed in robot_config.yaml",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    assets_dir = repo_root / "assets"

    robot_config_path = assets_dir / "robot_config.yaml"
    if not robot_config_path.exists():
        print(f"Missing robot_config.yaml at: {robot_config_path}")
        return 2

    robot_cfg = load_yaml(robot_config_path)
    active_hand = robot_cfg.get("active_hand")
    if not args.all:
        if not isinstance(active_hand, str) or not active_hand:
            print("robot_config.yaml missing valid active_hand")
            return 2

    base_cfg_path = robot_cfg.get("base_config_path")
    hand_cfg_paths = robot_cfg.get("hand_config_paths", {})
    if not isinstance(base_cfg_path, str) or not base_cfg_path:
        print("robot_config.yaml missing base_config_path")
        return 2
    if not isinstance(hand_cfg_paths, dict):
        print("robot_config.yaml hand_config_paths is invalid")
        return 2

    base_cfg_file = assets_dir / base_cfg_path

    if not base_cfg_file.exists():
        print(f"Missing base config: {base_cfg_file}")
        return 2

    xml_map = {
        "base": assets_dir / "g1_base/g1_29dof_rev_1_0.xml",
        "inspire_dfq": assets_dir / "g1_with_inspire_hand/g1_with_inspire_hand_DFQ.xml",
        "inspire_ftp": assets_dir / "g1_with_inspire_hand/g1_with_inspire_hand_FTP.xml",
        "dex1": assets_dir / "g1_with_dex1_hand/g1_with_dex1_hand.xml",
        "dex3": assets_dir / "g1_with_dex3_hand/g1_with_dex3_hand.xml",
        "brainco": assets_dir / "g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.xml",
    }

    base_xml = xml_map["base"]
    if not base_xml.exists():
        print(f"Missing base XML: {base_xml}")
        return 2

    base_cfg = load_yaml(base_cfg_file)

    if args.all:
        failures = 0
        summary_rows: list[tuple[str, str, int]] = []
        for hand_key in sorted(hand_cfg_paths.keys()):
            hand_cfg_path = hand_cfg_paths.get(hand_key)
            if not isinstance(hand_cfg_path, str) or not hand_cfg_path:
                print(f"{ANSI_RED}Invalid hand_config_paths entry: {hand_key}{ANSI_RESET}")
                failures += 1
                summary_rows.append((hand_key, "invalid", 0))
                continue

            hand_cfg_file = assets_dir / hand_cfg_path
            if not hand_cfg_file.exists():
                print(f"{ANSI_RED}Missing hand config: {hand_cfg_file}{ANSI_RESET}")
                failures += 1
                summary_rows.append((hand_key, "missing_cfg", 0))
                continue

            hand_xml = xml_map.get(hand_key)
            if hand_xml is None:
                print(f"{ANSI_RED}No XML mapping for active_hand '{hand_key}'{ANSI_RESET}")
                failures += 1
                summary_rows.append((hand_key, "missing_xml", 0))
                continue
            if not hand_xml.exists():
                print(f"{ANSI_RED}Missing hand XML: {hand_xml}{ANSI_RESET}")
                failures += 1
                summary_rows.append((hand_key, "missing_xml", 0))
                continue

            missing_count = verify_hand(hand_key, base_cfg, hand_cfg_file, base_xml, hand_xml)
            failures += 1 if missing_count else 0
            status = "ok" if missing_count == 0 else "missing"
            summary_rows.append((hand_key, status, missing_count))
            print("")

        print_summary_table(summary_rows)
        return 1 if failures else 0

    hand_cfg_path = hand_cfg_paths.get(active_hand)
    if not isinstance(hand_cfg_path, str) or not hand_cfg_path:
        print(f"active_hand '{active_hand}' has no hand_config_paths entry")
        return 2

    hand_cfg_file = assets_dir / hand_cfg_path
    if not hand_cfg_file.exists():
        print(f"Missing hand config: {hand_cfg_file}")
        return 2

    hand_xml = xml_map.get(active_hand)
    if hand_xml is None:
        print(f"No XML mapping for active_hand '{active_hand}'")
        return 2
    if not hand_xml.exists():
        print(f"Missing hand XML: {hand_xml}")
        return 2

    missing_count = verify_hand(active_hand, base_cfg, hand_cfg_file, base_xml, hand_xml)
    return 1 if missing_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
