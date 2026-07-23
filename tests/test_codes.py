"""Tests de la logique de grille (simplification des codes)."""

import pytest

from evaluation_dictee.data import grid


@pytest.mark.parametrize(
    ("code_complet", "attendu"),
    [
        ("1", "1"),  # correct reste correct
        ("0", "0"),  # absent reste absent
        ("3", "9"),  # erreur lexicale -> erreur
        ("4", "9"),  # erreur grammaticale -> erreur
        ("5", "9"),  # les deux -> erreur
        ("9", "9"),  # erreur ponctuation -> erreur
        (" 4 ", "9"),  # tolère les espaces
    ],
)
def test_to_simplified(code_complet: str, attendu: str) -> None:
    assert grid.to_simplified(code_complet) == attendu


def test_normalize_complete_garde_le_code() -> None:
    assert grid.normalize("4", "complete") == "4"


def test_normalize_simplifiee_regroupe() -> None:
    assert grid.normalize("4", "simplifiee") == "9"
