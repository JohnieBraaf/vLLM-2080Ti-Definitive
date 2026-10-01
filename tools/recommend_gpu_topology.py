#!/usr/bin/env python3
"""Recommend TP rank groups from the local NVIDIA topology matrix."""

import argparse
import json
import re
import subprocess
import sys
from functools import lru_cache
from itertools import combinations, permutations

LINK_SCORE = {"PIX": 40, "PXB": 30, "PHB": 20, "NODE": 10, "SYS": 0}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
MAX_SELECTED_GPUS = 12


def read_matrix(text):
    header = None
    rows = {}
    for line in text.splitlines():
        cells = ANSI_ESCAPE.sub("", line).split()
        if not cells:
            continue
        if header is None and re.fullmatch(r"GPU\d+", cells[0]):
            header = []
            for cell in cells:
                if not re.fullmatch(r"GPU\d+", cell):
                    break
                header.append(cell)
            continue
        if header and re.fullmatch(r"GPU\d+", cells[0]):
            if len(cells) < len(header) + 1:
                raise ValueError(f"incomplete topology row for {cells[0]}")
            rows[cells[0]] = dict(zip(header, cells[1 : len(header) + 1]))
    if not header or not rows:
        raise ValueError("no GPU topology matrix returned by nvidia-smi")
    return rows


def link_score(value):
    if re.fullmatch(r"NV\d+", value):
        return 50 + int(value[2:])
    if value not in LINK_SCORE:
        raise ValueError(f"unknown GPU topology link: {value}")
    return LINK_SCORE[value]


def order_pipeline_stages(groups, topology, p2p):
    orientations = [tuple(permutations(group)) for group in groups]
    all_stages = (1 << len(groups)) - 1

    @lru_cache(None)
    def stage_link_score(left, right):
        # vLLM's PP x TP layout connects the same TP rank in adjacent stages.
        score = 0
        for source, target in zip(left, right):
            score += link_score(topology[source][target])
            score += link_score(topology[target][source])
            if p2p is not None and p2p[source][target] == p2p[target][source] == "OK":
                score += 100
        return score

    @lru_cache(None)
    def best_tail(used, previous):
        if used == all_stages:
            return 0, ()
        best = None
        for index, stage_orientations in enumerate(orientations):
            if used & (1 << index):
                continue
            for stage in stage_orientations:
                tail_score, tail = best_tail(used | (1 << index), stage)
                candidate = stage_link_score(previous, stage) + tail_score
                if best is None or candidate > best[0]:
                    best = candidate, (stage, *tail)
        return best

    best = None
    for index, group in enumerate(groups):
        tail_score, tail = best_tail(1 << index, group)
        if best is None or tail_score > best[0]:
            best = tail_score, (group, *tail)
    return best[1]


def recommend(devices, tp_size, topology, p2p=None, pp_size=None):
    if not devices or len(set(devices)) != len(devices) or len(devices) % tp_size:
        raise ValueError("selected GPUs must be unique and divisible by TP size")
    if pp_size is not None and pp_size != len(devices) // tp_size:
        raise ValueError("selected GPU count must equal TP size times PP size")
    if len(devices) > MAX_SELECTED_GPUS:
        raise ValueError(
            f"topology ranking supports at most {MAX_SELECTED_GPUS} selected GPUs"
        )
    names = [f"GPU{device}" for device in devices]
    for left in names:
        for right in names:
            if left != right:
                link_score(topology[left][right])

    if tp_size == 1:
        groups = [(name,) for name in names]
    else:

        @lru_cache(None)
        def partition(remaining):
            if not remaining:
                return 0, ()
            first, *rest = remaining
            best = None
            for others in combinations(rest, tp_size - 1):
                group = (first, *others)
                if p2p is not None and any(
                    p2p[a][b] != "OK" or p2p[b][a] != "OK"
                    for a, b in combinations(group, 2)
                ):
                    continue
                tail = tuple(name for name in remaining if name not in group)
                candidate_score, candidate_groups = partition(tail)
                if candidate_score < 0:
                    continue
                score = candidate_score + sum(
                    link_score(topology[a][b]) + link_score(topology[b][a])
                    for a, b in combinations(group, 2)
                )
                if best is None or score > best[0]:
                    best = score, (group, *candidate_groups)
            return best if best is not None else (-1, ())

        _, groups = partition(tuple(names))
        if not groups:
            raise ValueError("no fully P2P-valid TP grouping found")

    if len(groups) > 1:
        groups = order_pipeline_stages(groups, topology, p2p)

    ordered = [name[3:] for group in groups for name in group]
    if len(groups) > 1:
        summary = "PP stages: " + "  ".join(
            f"PP{index}(TP{tp_size})=[{','.join(name[3:] for name in group)}]"
            for index, group in enumerate(groups)
        )
    else:
        summary = f"TP group: [{','.join(name[3:] for name in groups[0])}]"
    summary += (
        "; P2P read/write checked." if p2p is not None else "; P2P probe unavailable."
    )
    if len(groups) > 1:
        summary += " PP stage links ranked by topology."
    return {"ordered_devices": ",".join(ordered), "summary": summary}


def probe(command):
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=10
    ).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--pp-size", type=int)
    args = parser.parse_args()
    if (
        args.tp_size < 1
        or (args.pp_size is not None and args.pp_size < 1)
        or not re.fullmatch(r"\d+(,\d+)*", args.devices)
    ):
        parser.error("invalid GPU selection or TP size")
    devices = args.devices.split(",")
    try:
        topology = read_matrix(probe(["nvidia-smi", "topo", "-m"]))
        p2p = None
        try:
            reads = read_matrix(probe(["nvidia-smi", "topo", "-p2p", "r"]))
            writes = read_matrix(probe(["nvidia-smi", "topo", "-p2p", "w"]))
            p2p = {
                left: {
                    right: "OK"
                    if reads[left][right] == writes[left][right] == "OK"
                    else "NS"
                    for right in reads[left]
                }
                for left in reads
            }
        except (KeyError, ValueError, subprocess.SubprocessError):
            pass
        if args.tp_size > 1 and p2p is None:
            raise ValueError("P2P read/write probe unavailable")
        result = recommend(devices, args.tp_size, topology, p2p, args.pp_size)
    except (KeyError, ValueError, subprocess.SubprocessError, FileNotFoundError) as exc:
        print(f"Topology probe unavailable: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
