"""Tests du module de comparaison multi-modèles."""

import json
from pathlib import Path

import pytest

from evaluation_dictee.evaluation.multi_model import (
    agreement_per_item,
    confidence_score,
    disagreement_type_summary,
    load_multi_runs,
    referral_curve_multi,
    referral_curve_with_ci,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def multi_runs_data(tmp_path: Path) -> tuple[Path, list[str]]:
    """Crée 3 runs factices : 2 copies, 4 items chacune, 3 modèles."""
    # Copie A : les 3 modèles s'accordent partout (haute confiance)
    # Copie B : divergence sur i2 et i3 (faible confiance)
    truths_a = {"i1": "1", "i2": "9", "i3": "1", "i4": "0"}
    preds_a = {
        "run1": {"i1": "1", "i2": "9", "i3": "1", "i4": "0"},  # parfait
        "run2": {"i1": "1", "i2": "9", "i3": "1", "i4": "0"},  # parfait
        "run3": {"i1": "1", "i2": "9", "i3": "1", "i4": "0"},  # parfait
    }
    truths_b = {"i1": "1", "i2": "9", "i3": "1", "i4": "9"}
    preds_b = {
        "run1": {"i1": "1", "i2": "9", "i3": "9", "i4": "1"},  # 2 désaccords
        "run2": {"i1": "1", "i2": "1", "i3": "1", "i4": "1"},  # 2 désaccords
        "run3": {"i1": "1", "i2": "9", "i3": "1", "i4": "9"},  # parfait
    }

    for run in ["run1", "run2", "run3"]:
        records = []
        for it, truth in truths_a.items():
            records.append(
                {
                    "copy_id": "A",
                    "item_id": it,
                    "y_true": truth,
                    "y_pred": preds_a[run][it],
                    "confidence": 0.9,
                }
            )
        for it, truth in truths_b.items():
            records.append(
                {
                    "copy_id": "B",
                    "item_id": it,
                    "y_true": truth,
                    "y_pred": preds_b[run][it],
                    "confidence": 0.8,
                }
            )
        _write_jsonl(tmp_path / f"{run}_predictions.jsonl", records)

    return tmp_path, ["run1", "run2", "run3"]


def test_load_multi_runs_joins_correctly(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    assert len(df) == 8  # 2 copies × 4 items
    for run in runs:
        assert f"y_pred__{run}" in df.columns
        assert f"conf__{run}" in df.columns
    assert "y_true" in df.columns


def test_load_missing_run_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_multi_runs(["inexistant"], output_dir=tmp_path)


def test_agreement_per_item(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)

    # Copie A : unanimité partout (les 3 modèles s'accordent)
    a_rows = agree[agree["copy_id"] == "A"]
    assert (a_rows["unanimite"]).all()
    assert (a_rows["n_accord_modeles"] == 3).all()

    # Copie B : sur i1 les 3 s'accordent, sur les autres il y a du désaccord
    b_rows = agree[agree["copy_id"] == "B"]
    i1 = b_rows[b_rows["item_id"] == "i1"].iloc[0]
    assert i1["unanimite"] is True or i1["unanimite"]  # unanimité sur i1
    # Sur i4 : run1 dit "1", run2 dit "1", run3 dit "9" → modale = "1", 2 accords
    i4 = b_rows[b_rows["item_id"] == "i4"].iloc[0]
    assert i4["n_accord_modeles"] == 2
    assert i4["modal_pred"] == "1"


def test_confidence_score_ordering(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)
    conf = confidence_score(agree)

    # Copie A doit avoir un score plus élevé que B (unanimité vs divergence)
    assert conf.loc["A", "score_confiance"] > conf.loc["B", "score_confiance"]
    # A a 100 % d'unanimité, B moins
    assert conf.loc["A", "pct_unanime"] == 100.0


def test_referral_curve_monotonicity(multi_runs_data) -> None:
    """Plus le seuil monte, plus on renvoie de copies (monotonie)."""
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)
    conf = confidence_score(agree)
    curve = referral_curve_multi(agree, conf, reference_run="run1")

    # Le % renvoyé doit être monotone croissant en τ
    pct = curve["pct_copies_renvoyees"].values
    assert all(pct[i] <= pct[i + 1] for i in range(len(pct) - 1)), f"Non monotone : {pct}"


def test_disagreement_type_summary(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)
    summary = disagreement_type_summary(agree)

    # Il doit y avoir des lignes pour n_accord_modeles = 2 ou 3
    assert set(summary.index).issubset({1, 2, 3})
    # Somme des % = 100
    assert abs(summary["pct_items"].sum() - 100.0) < 1e-6


def test_referral_curve_with_ci_has_bounds(multi_runs_data) -> None:
    tmp_path, runs = multi_runs_data
    df = load_multi_runs(runs, output_dir=tmp_path)
    agree = agreement_per_item(df)
    conf = confidence_score(agree)
    curve = referral_curve_with_ci(agree, conf, reference_run="run1")
    assert "accord_lo" in curve.columns
    assert "accord_hi" in curve.columns
    # Les bornes encadrent la valeur estimée (là où elle est définie)
    ok = curve.dropna(subset=["pct_accord_retenues"])
    assert (ok["accord_lo"] <= ok["pct_accord_retenues"]).all()
    assert (ok["pct_accord_retenues"] <= ok["accord_hi"]).all()


def test_single_run_fallback(tmp_path: Path) -> None:
    """N=1 : le score de confiance est constant, la structure marche quand même."""
    records = [
        {"copy_id": "A", "item_id": "i1", "y_true": "1", "y_pred": "1", "confidence": 0.9},
        {"copy_id": "A", "item_id": "i2", "y_true": "9", "y_pred": "9", "confidence": 0.9},
    ]
    _write_jsonl(tmp_path / "solo_predictions.jsonl", records)
    df = load_multi_runs(["solo"], output_dir=tmp_path)
    agree = agreement_per_item(df)
    conf = confidence_score(agree)
    # Avec un seul modèle, tous les items sont "unanimes" par construction
    assert conf.loc["A", "score_confiance"] == 100.0
