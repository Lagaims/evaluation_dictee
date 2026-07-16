"""Tests des métriques HTR et du chargement Scoledit (TEI)."""

from evaluation_dictee.transcription.htr_metrics import (
    _levenshtein,
    compute_transcription_metrics,
)
from evaluation_dictee.transcription.scoledit import tei_to_text


# ── Distance d'édition ────────────────────────────────────────────────────────
def test_levenshtein_identique() -> None:
    assert _levenshtein("chat", "chat") == 0


def test_levenshtein_substitution() -> None:
    assert _levenshtein("chat", "chien") == 3  # a→i, t→e, +n


def test_levenshtein_mots() -> None:
    assert _levenshtein(["le", "chat"], ["le", "chien"]) == 1


# ── CER / WER ─────────────────────────────────────────────────────────────────
def test_transcription_parfaite() -> None:
    m = compute_transcription_metrics("le chat dort", "le chat dort")
    assert m.cer == 0.0
    assert m.wer == 0.0


def test_cer_une_faute() -> None:
    # 1 caractère substitué sur 12 → CER = 1/12
    m = compute_transcription_metrics("le chat dort", "le chit dort")
    assert abs(m.cer - 1 / 12) < 1e-9
    assert m.wer == 1 / 3  # 1 mot sur 3 diffère


def test_normalisation_accents_casse() -> None:
    # Différence uniquement d'accent/casse : CER brut > 0, CER normalisé = 0
    m = compute_transcription_metrics("Été", "ete")
    assert m.cer > 0
    assert m.cer_normalise == 0.0


def test_preserve_les_fautes() -> None:
    # La référence contient une faute d'élève ; une transcription fidèle a CER 0
    ref = "il était tune fois"
    m = compute_transcription_metrics(ref, "il était tune fois")
    assert m.cer == 0.0


# ── TEI → texte ───────────────────────────────────────────────────────────────
def test_tei_supprime_balises_et_lb() -> None:
    tei = "<p>il était tune fois un chat <lb/> pérsent. ile se promener</p>"
    txt = tei_to_text(tei)
    assert "<lb/>" not in txt
    assert "<p>" not in txt
    assert txt.startswith("il était tune fois")
    assert "pérsent" in txt  # fautes préservées


def test_tei_espaces_normalises() -> None:
    tei = "<p>a  <lb/>  b</p>"
    assert tei_to_text(tei) == "a b"
