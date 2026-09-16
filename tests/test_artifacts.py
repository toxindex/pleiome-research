import json

import pytest

from pleiome_research.artifacts import prepare, sha256, verify


def small_bundle(tmp_path):
    files = []
    for name, content in [("part0", b"first"), ("part1", b"second")]:
        path = tmp_path / name
        path.write_bytes(content)
        files.append(
            {"path": name, "bytes": len(content), "sha256": sha256(path), "role": "checkpoint_part"}
        )
    combined = tmp_path / "expected"
    combined.write_bytes(b"firstsecond")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": files,
                "checkpoint": {"path": "best.pt", "bytes": 11, "sha256": sha256(combined)},
            }
        )
    )
    return manifest


def test_assembly_verifies_bytes_and_reuses_existing_checkpoint(tmp_path):
    manifest = small_bundle(tmp_path)
    assert prepare(tmp_path, manifest).read_bytes() == b"firstsecond"
    assert prepare(tmp_path, manifest).read_bytes() == b"firstsecond"
    (tmp_path / "best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare(tmp_path, manifest)


def test_corrupt_parts_and_lfs_pointers_fail(tmp_path):
    manifest = small_bundle(tmp_path)
    (tmp_path / "part0").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify(tmp_path, manifest)
    (tmp_path / "part0").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="git lfs pull"):
        verify(tmp_path, manifest)


def test_manifest_cannot_escape_directory(tmp_path):
    manifest = small_bundle(tmp_path)
    content = json.loads(manifest.read_text())
    content["files"][0]["path"] = "../outside"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="escapes"):
        verify(tmp_path, manifest)
