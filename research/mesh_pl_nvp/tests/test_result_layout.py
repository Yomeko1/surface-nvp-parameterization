from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from research.mesh_pl_nvp.result_layout import (
    copy_verified, create_result, deduplicate_sources, default_archive, resolve_run, source_store,
    store_source_bytes, validate_locations,
)
from research.mesh_pl_nvp.result_visualization import draw_mesh, intersection_counts
from research.mesh_pl_nvp.summarize_v31_audit import validity_checkpoints


def test_result_archive_paths_and_loading(tmp_path):
    result = tmp_path/"data/output/v3.2/00071/ours"
    archive = default_archive(result)
    assert archive == tmp_path/"data/archive/v3.2/00071/ours"
    archive.mkdir(parents=True)
    create_result(result, archive)
    assert resolve_run(result) == archive
    assert resolve_run(archive) == archive
    assert source_store(archive) == tmp_path/"data/archive/_sources"
    with pytest.raises(FileExistsError):
        create_result(result, archive)
    for bad in (result, result/"raw", result.parent):
        with pytest.raises(ValueError):
            validate_locations(result, bad)


def test_missing_archive_is_explicit(tmp_path):
    create_result(tmp_path/"result", tmp_path/"archive")
    with pytest.raises(FileNotFoundError):
        resolve_run(tmp_path/"result")


def test_source_dedup_preserves_exact_original_bytes_and_refs(tmp_path):
    archive = tmp_path/"data/archive/v3.2/00071/ours"
    archive.mkdir(parents=True)
    data = b"exact original source snapshot bytes"
    for directory in (archive, archive/"audit", archive/"report"):
        directory.mkdir(exist_ok=True)
        (directory/"source.zip").write_bytes(data)
    deduplicate_sources(archive)
    assert not list(archive.rglob("source.zip"))
    assert len(list(source_store(archive).glob("*.zip"))) == 1
    for ref in archive.rglob("source.ref.json"):
        payload = json.loads(ref.read_text())
        target = (ref.parent/payload["archive"]).resolve()
        assert target.read_bytes() == data
        assert payload["sha256"] == hashlib.sha256(data).hexdigest()
    store_source_bytes(archive/"source.zip", data, source_store(archive))
    assert len(list(source_store(archive).glob("*.zip"))) == 1


def test_copy_will_not_replace_different_results(tmp_path):
    source, target = tmp_path/"original.obj", tmp_path/"final.obj"
    source.write_bytes(b"original mesh")
    copy_verified(source, target)
    copy_verified(source, target)
    assert source.read_bytes() == target.read_bytes()
    source.write_bytes(b"different mesh")
    with pytest.raises(FileExistsError):
        copy_verified(source, target)
    assert target.read_bytes() == b"original mesh"


def test_all_intersection_checkpoints_include_h2_milestones():
    initial = {"extended_validity": {"intersection_pairs": 0}}
    offsets = {"first_harmonic": 0, "local": 100, "final_harmonic": 600}
    rows = [{"phase": p, "step": s, "extended_validity": {"intersection_pairs": c}}
            for p, s, c in (("first_harmonic", 100, 0), ("local", 500, 0),
                             ("final_harmonic", 50, None), ("final_harmonic", 100, 0),
                             ("final_harmonic", 200, 2), ("final_harmonic", 300, 0))]
    result = validity_checkpoints(initial, rows, offsets)
    assert [r["global_step"] for r in result] == [0, 100, 600, 700, 800, 900]
    assert result[-2]["extended_validity"]["intersection_pairs"] == 2


def test_plot_uses_supplied_float64_face_values():
    fig, ax = plt.subplots()
    try:
        uv = np.array([[0., 0.], [1., 0.], [0., 1.]])
        faces = np.array([[0, 1, 2]])
        values = np.array([4.000000000001], dtype=np.float64)
        artist = draw_mesh(ax, uv, faces, values=values)
        assert np.array_equal(artist.get_array(), values)
        assert artist.get_array().dtype == np.float64
        assert intersection_counts(faces, []).tolist() == [0]
    finally:
        plt.close(fig)
