"""Métriques d'évaluation de la transcription (HTR).

Compare une transcription produite par le modèle à une transcription de référence
humaine. Les deux préservent les fautes de l'élève : on mesure la fidélité de
LECTURE, pas la correction orthographique.

Métriques principales :
- **CER** (Character Error Rate) : distance d'édition au niveau caractère,
  normalisée par la longueur de la référence. 0 = transcription parfaite.
- **WER** (Word Error Rate) : idem au niveau mot.

On fournit aussi une variante « normalisée » (minuscules, sans accents ni
ponctuation) pour distinguer les erreurs de lecture substantielles des simples
différences de casse/accents.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def _levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    """Distance d'édition (insertions + suppressions + substitutions) entre a et b.

    Fonctionne sur des chaînes (niveau caractère) ou des listes (niveau mot).

    Args:
        a: séquence de référence.
        b: séquence hypothèse.

    Returns:
        Le nombre minimal d'opérations pour transformer a en b.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _normalize_text(s: str) -> str:
    """Minuscule, sans accents, sans ponctuation (pour la variante normalisée)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower())
    return " ".join(s.split())


@dataclass
class TranscriptionMetrics:
    """Métriques de transcription pour un échantillon ou un corpus.

    Attributes:
        cer: Character Error Rate (0 = parfait).
        wer: Word Error Rate.
        cer_normalise: CER après normalisation (casse/accents/ponctuation ignorés).
        wer_normalise: WER après normalisation.
        n_char_ref: nombre de caractères de la référence.
        n_mots_ref: nombre de mots de la référence.
    """

    cer: float
    wer: float
    cer_normalise: float
    wer_normalise: float
    n_char_ref: int
    n_mots_ref: int


def compute_transcription_metrics(reference: str, hypothesis: str) -> TranscriptionMetrics:
    """Calcule CER et WER (bruts et normalisés) entre référence et hypothèse.

    Args:
        reference: transcription de référence (humaine).
        hypothesis: transcription produite par le modèle.

    Returns:
        Les métriques de transcription.
    """
    ref_c, hyp_c = reference, hypothesis
    ref_w, hyp_w = reference.split(), hypothesis.split()

    n_char = max(len(ref_c), 1)
    n_mots = max(len(ref_w), 1)
    cer = _levenshtein(ref_c, hyp_c) / n_char
    wer = _levenshtein(ref_w, hyp_w) / n_mots

    rn, hn = _normalize_text(reference), _normalize_text(hypothesis)
    rn_w, hn_w = rn.split(), hn.split()
    cer_n = _levenshtein(rn, hn) / max(len(rn), 1)
    wer_n = _levenshtein(rn_w, hn_w) / max(len(rn_w), 1)

    return TranscriptionMetrics(
        cer=cer,
        wer=wer,
        cer_normalise=cer_n,
        wer_normalise=wer_n,
        n_char_ref=len(ref_c),
        n_mots_ref=len(ref_w),
    )
