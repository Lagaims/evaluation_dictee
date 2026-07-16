"""Fabrique de scorer : choisit l'approche selon la configuration.

Centralise la construction du bon `Scorer` (approche 1 ou 2) pour que le script
CLI et les notebooks n'aient pas à dupliquer cette logique.
"""

from __future__ import annotations

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.data.reference import GridItem
from evaluation_dictee.models.base import Scorer
from evaluation_dictee.models.two_stage import TwoStageScorer
from evaluation_dictee.models.vlm import VLMScorer


def build_scorer(
    config: ExperimentConfig,
    grid_items: list[GridItem],
    base_url: str,
    api_key: str,
) -> Scorer:
    """Construit le scorer correspondant à l'approche déclarée dans la config.

    Args:
        config: configuration de l'expérience (contient `approach`).
        grid_items: items de la grille (mot attendu).
        base_url: URL de l'API.
        api_key: clé d'API.

    Returns:
        Un VLMScorer (approche end_to_end) ou un TwoStageScorer (approche two_stage).
    """
    if config.approach == "two_stage":
        return TwoStageScorer(
            model_config=config.model,
            prompt_config=config.prompt,
            base_url=base_url,
            api_key=api_key,
            grid_items=grid_items,
            scheme=config.grid.scheme,
            model_config_stage2=config.model_stage2,
        )
    return VLMScorer(
        model_config=config.model,
        prompt_config=config.prompt,
        base_url=base_url,
        api_key=api_key,
        grid_items=grid_items,
        scheme=config.grid.scheme,
    )
