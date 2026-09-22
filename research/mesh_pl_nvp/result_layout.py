"""Small result directories linked to complete, relocatable run archives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil


def relative_path(target, directory):
    return Path(os.path.relpath(Path(target).resolve(), Path(directory).resolve())).as_posix()


def default_archive(output):
    output = Path(output).resolve()
    for parent in output.parents:
        if parent.name == "output" and parent.parent.name == "data":
            return parent.parent / "archive" / output.relative_to(parent)
    return output.parent / "_run_archives" / output.name


def source_store(archive):
    archive = Path(archive).resolve()
    for parent in archive.parents:
        if parent.name == "archive" and parent.parent.name == "data":
            return parent / "_sources"
    return archive.parent / "_sources"


def validate_locations(output, archive):
    output, archive = Path(output).resolve(), Path(archive).resolve()
    if output == archive or output in archive.parents or archive in output.parents:
        raise ValueError("result and archive directories must be separate, non-nested paths")
    return output, archive


def create_result(output, archive):
    output, archive = validate_locations(output, archive)
    locator = relative_path(archive, output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps({
        "format_version": 1, "archive": locator,
        "model": "final.model.pt", "note": "Logs, checkpoints and raw audits live in the archive."
    }, indent=2), encoding="utf-8")
    return output


def resolve_run(path):
    path = Path(path).resolve()
    link = path / "run.json"
    if link.is_file():
        data = json.loads(link.read_text(encoding="utf-8"))
        if data.get("format_version") != 1:
            raise ValueError("unsupported result layout")
        archive = (path / data["archive"]).resolve()
        validate_locations(path, archive)
        if not archive.is_dir():
            raise FileNotFoundError(f"run archive is unavailable: {archive}")
        return archive
    return path


def copy_verified(source, target):
    source, target = Path(source), Path(target)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"refusing to replace different result: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise IOError(f"copy verification failed: {target}")


def expose_final(archive, output):
    archive, output = Path(archive), Path(output)
    for name in ("final.obj", "final.usda"):
        if (archive / name).is_file():
            copy_verified(archive / name, output / name)
    # The OBJ writer places copied material libraries alongside the final mesh.
    for material in archive.glob("*.mtl"):
        copy_verified(material, output / material.name)


def store_source_bytes(path, data, store):
    """Preserve the exact ZIP bytes once; references retain their original SHA256."""
    path, store = Path(path), Path(store).resolve()
    store.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    target = store / (digest + ".zip")
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise IOError("source store hash mismatch")
    else:
        with target.open("xb") as stream:
            stream.write(data)
    reference = {"format_version": 1, "sha256": digest,
                 "archive": relative_path(target, path.parent)}
    path.with_suffix(".ref.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return target


def deduplicate_sources(archive):
    archive = Path(archive).resolve()
    store = source_store(archive)
    for path in sorted(archive.rglob("source.zip")):
        resolved = path.resolve()
        if archive not in resolved.parents or path.is_symlink():
            raise ValueError("source snapshot escaped the selected archive")
        data = path.read_bytes()
        target = store_source_bytes(path, data, store)
        if target.read_bytes() != data:
            raise IOError("source snapshot verification failed")
        path.unlink()
