#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Content fingerprint for checkpoints used with persistent KV caches."""

import hashlib
import os
import sys
from pathlib import Path


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _checkpoint_files(root: Path, role: str) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"directory symlink in {role} checkpoint: {path}")
        if path.is_symlink() and not path.exists():
            raise ValueError(f"broken symlink in {role} checkpoint: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError(f"{role} checkpoint has no files: {root}")
    return sorted(files)


def fingerprint_checkpoints(model_dir: str, draft_dir: str = "") -> str:
    digest = hashlib.sha256()
    snapshots = []
    for role, directory in (("target", model_dir), ("draft", draft_dir)):
        if not directory:
            continue
        root = Path(directory).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"{role} checkpoint is not a directory: {directory}")

        files = _checkpoint_files(root, role)
        snapshots.append((root, role, files, {}))
        file_stats = snapshots[-1][3]

        digest.update(role.encode() + b"\0")
        for path in files:
            relative = path.relative_to(root).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            before = path.stat()
            file_stats[path] = _stat_identity(before)
            digest.update(before.st_size.to_bytes(16, "big"))
            with path.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
            after = path.stat()
            if _stat_identity(before) != _stat_identity(after):
                raise ValueError(f"checkpoint file changed while hashing: {path}")
    for root, role, files, file_stats in snapshots:
        if _checkpoint_files(root, role) != files:
            raise ValueError(f"{role} checkpoint file set changed while hashing: {root}")
        for path in files:
            if _stat_identity(path.stat()) != file_stats[path]:
                raise ValueError(f"checkpoint file changed while hashing: {path}")
    return digest.hexdigest()


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: checkpoint_fingerprint.py MODEL_DIR [DRAFT_DIR]")
    try:
        print(fingerprint_checkpoints(*sys.argv[1:]))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: Cannot fingerprint checkpoint: {exc}") from exc
