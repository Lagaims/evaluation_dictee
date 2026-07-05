"""Tests de l'option chain_of_thought (chain-of-thought)."""

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.reference import GridItem
from evaluation_dictee.pipeline.prompts import build_dictation_prompt

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", [], [], []),
]


def test_cot_desactive_pas_de_champ_comparaison() -> None:
    prompt = build_dictation_prompt("Le soir", ITEMS, PromptConfig(chain_of_thought=False))
    assert "comparaison" not in prompt
    assert '"code": "1"' in prompt  # schéma classique présent


def test_cot_active_ajoute_champ_comparaison() -> None:
    prompt = build_dictation_prompt("Le soir", ITEMS, PromptConfig(chain_of_thought=True))
    assert "comparaison" in prompt
    # Le schéma JSON attendu inclut le nouveau champ
    assert '"comparaison"' in prompt
    # L'exemple pédagogique est présent
    assert "inquiets" in prompt.lower()  # inclus dans l'exemple donné


def test_cot_ordre_du_raisonnement_est_impose() -> None:
    prompt = build_dictation_prompt("Le soir", ITEMS, PromptConfig(chain_of_thought=True))
    # La consigne indique bien AVANT le code (ordre du raisonnement)
    assert "AVANT de choisir le code" in prompt
