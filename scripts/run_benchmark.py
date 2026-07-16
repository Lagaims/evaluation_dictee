"""Point d'entrée simple pour lancer un benchmark.

Pour les débutant·e·s : ce script est volontairement minimal. Il lit une config,
construit le modèle, lance le benchmark et affiche les métriques. Suivre le fil
de `run_benchmark` (dans src/.../pipeline/benchmark.py) pour comprendre le projet.

Usage :
    python scripts/run_benchmark.py --config configs/dictee_qwen7b_zeroshot.yaml
"""

from __future__ import annotations

import argparse

from langfuse import get_client

from evaluation_dictee.config import Secrets, load_config
from evaluation_dictee.data.reference import load_grid
from evaluation_dictee.evaluation.calibration import referral_curve
from evaluation_dictee.models.factory import build_scorer
from evaluation_dictee.pipeline.benchmark import run_benchmark
from evaluation_dictee.utils.logging import get_logger
from evaluation_dictee.utils.tracking import experiment_run, log_metrics

logger = get_logger(__name__)


def main() -> None:
    """Parse les arguments, lance le benchmark, affiche métriques et calibration."""
    parser = argparse.ArgumentParser(description="Lance un benchmark d'évaluation.")
    parser.add_argument("--config", required=True, help="Chemin du fichier YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    secrets = Secrets()

    grid = load_grid(config.data.grid_path)
    scorer = build_scorer(
        config=config,
        grid_items=grid.items,
        base_url=secrets.llm_base_url,
        api_key=secrets.llm_api_key,
    )

    try:
        with experiment_run(config):
            result = run_benchmark(config, scorer)
            log_metrics(
                {
                    "raw_agreement": result.metrics.raw_agreement,
                    "cohen_kappa": result.metrics.cohen_kappa,
                }
            )
    finally:
        # Langfose envoie les traces de façon asynchrone : sans flush explicite,
        # le script peut se terminer avant l'envoi et perdre les dernières traces.
        get_client().flush()

    logger.info("Accord brut : %.1f%%", result.metrics.raw_agreement * 100)
    logger.info("Kappa de Cohen : %.3f", result.metrics.cohen_kappa)

    logger.info("Courbe de renvoi humain :")
    for point in referral_curve(result.y_true, result.y_pred, result.confidences):
        logger.info(
            "  seuil %.1f → renvoi humain %.0f%% | erreur résiduelle %.1f%%",
            point.threshold,
            point.human_referral_rate * 100,
            point.residual_error_rate * 100,
        )


if __name__ == "__main__":
    main()
