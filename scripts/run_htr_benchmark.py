"""Point d'entrée : évaluer la transcription (HTR) d'un modèle sur Scoledit.

Usage :
    python scripts/run_htr_benchmark.py --config configs/htr_gemma4_scoledit.yaml
"""

from __future__ import annotations

import argparse

import yaml

from evaluation_dictee.config import ModelConfig, Secrets
from evaluation_dictee.transcription.htr_benchmark import VLMTranscriber, run_htr_benchmark
from evaluation_dictee.transcription.scoledit import load_scoledit_dataset
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Charge la config, transcrit le corpus Scoledit, affiche CER/WER."""
    parser = argparse.ArgumentParser(description="Évalue la transcription HTR.")
    parser.add_argument("--config", required=True, help="Chemin du fichier YAML.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    secrets = Secrets()
    model_config = ModelConfig.model_validate(cfg["model"])

    samples = load_scoledit_dataset(
        scans_dir=cfg["data"]["scans_dir"],
        annotations_dir=cfg["data"]["annotations_dir"],
        limit=cfg["data"].get("limit"),
    )
    logger.info("%d échantillons Scoledit chargés.", len(samples))
    if not samples:
        raise RuntimeError(
            "Aucun échantillon chargé. Vérifier scans_dir/annotations_dir et l'accès S3."
        )

    transcriber = VLMTranscriber(
        model_config=model_config,
        base_url=secrets.llm_base_url,
        api_key=secrets.llm_api_key,
        read_final_state=cfg.get("read_final_state", True),
    )

    result = run_htr_benchmark(samples, transcriber, run_name=cfg["name"])

    logger.info("CER moyen : %.1f%%", result.mean_cer * 100)
    logger.info("WER moyen : %.1f%%", result.mean_wer * 100)
    logger.info("Échecs de transcription : %d / %d", result.n_echecs, len(samples))
    logger.info("Prédictions : %s", result.predictions_path)


if __name__ == "__main__":
    main()
