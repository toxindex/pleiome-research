"""Verify and assemble the byte-identical upstream checkpoint from Git LFS parts."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(directory, name):
    root = Path(directory).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes the model directory")
    return path


def verify(directory="models/pleiome", manifest_path="artifacts/model-manifest.json"):
    manifest = json.loads(Path(manifest_path).read_text())
    for item in manifest["files"]:
        path = checked_path(directory, item["path"])
        if not path.is_file():
            raise ValueError(
                f"Missing artifact {item['path']}; run git lfs install and git lfs pull"
            )
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{item['path']} is an LFS pointer; run git lfs pull")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {item['path']}")
    return manifest


def prepare(directory="models/pleiome", manifest_path="artifacts/model-manifest.json"):
    manifest = verify(directory, manifest_path)
    expected = manifest["checkpoint"]
    output = checked_path(directory, expected["path"])
    if output.exists():
        if output.stat().st_size != expected["bytes"] or sha256(output) != expected["sha256"]:
            raise ValueError(
                "Existing assembled checkpoint has a checksum mismatch; remove it and retry"
            )
        return output
    parts = [item for item in manifest["files"] if item["role"] == "checkpoint_part"]
    if not parts:
        raise ValueError("Manifest has no checkpoint parts")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for item in parts:
                with checked_path(directory, item["path"]).open("rb") as part:
                    shutil.copyfileobj(part, stream, length=8 * 1024**2)
        if temporary.stat().st_size != expected["bytes"] or sha256(temporary) != expected["sha256"]:
            raise ValueError("Assembled checkpoint checksum mismatch")
        temporary.replace(output)
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
