"""Tests du checkpointing incrémental et de la reprise du benchmark."""

import json
from pathlib import Path

from evaluation_dictee.pipeline.benchmark import _load_processed_copy_ids


def test_load_processed_from_empty(tmp_path: Path) -> None:
    """Fichier absent = aucune copie déjà traitée."""
    assert _load_processed_copy_ids(tmp_path / "nope.jsonl") == set()


def test_load_processed_extracts_copy_ids(tmp_path: Path) -> None:
    """Lit correctement les copy_id depuis un JSONL existant."""
    path = tmp_path / "p.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"copy_id": "c1.png", "item_id": "i1"}) + "\n")
        f.write(json.dumps({"copy_id": "c1.png", "item_id": "i2"}) + "\n")
        f.write(json.dumps({"copy_id": "c2.png", "item_id": "i1"}) + "\n")
    assert _load_processed_copy_ids(path) == {"c1.png", "c2.png"}


def test_load_processed_ignores_truncated_last_line(tmp_path: Path) -> None:
    """Une ligne tronquée par un crash à mi-écriture ne casse pas la reprise."""
    path = tmp_path / "p.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"copy_id": "c1.png", "item_id": "i1"}) + "\n")
        # Ligne tronquée (le crash a coupé au milieu du json)
        f.write('{"copy_id": "c2.png", "item_id":')
    # Doit lire c1 et ignorer la ligne tronquée sans lever
    assert _load_processed_copy_ids(path) == {"c1.png"}


def test_load_processed_ignores_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n")
        f.write(json.dumps({"copy_id": "c1.png", "item_id": "i1"}) + "\n")
        f.write("\n")
    assert _load_processed_copy_ids(path) == {"c1.png"}


def test_load_processed_skips_records_without_copy_id(tmp_path: Path) -> None:
    """Ligne JSON valide mais sans copy_id : ignorée silencieusement."""
    path = tmp_path / "p.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"copy_id": "c1.png", "item_id": "i1"}) + "\n")
        f.write(json.dumps({"unrelated": "junk"}) + "\n")  # pas de copy_id
        f.write(json.dumps({"copy_id": "c2.png", "item_id": "i1"}) + "\n")
    assert _load_processed_copy_ids(path) == {"c1.png", "c2.png"}
