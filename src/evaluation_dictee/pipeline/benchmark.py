"""Orchestration d'un run de benchmark : données → modèle → métriques.

Ce module est le cœur du projet. Il enchaîne le chargement des données, l'appel
au modèle copie par copie, la normalisation des codes selon le schéma choisi, et
le calcul des métriques. Tout est piloté par une `ExperimentConfig`.

Les prédictions détaillées (une ligne JSON par item × copie) sont sauvegardées dans
data/processed/<run_name>_predictions.jsonl pour analyse ultérieure via
evaluation/report.py et le notebook 03_analyse_resultats.ipynb.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from evaluation_dictee.config import ExperimentConfig
from evaluation_dictee.data import grid
from evaluation_dictee.data.loaders import Copy, load_dataset
from evaluation_dictee.data.reference import load_grid
from evaluation_dictee.evaluation.metrics import ScoringMetrics, compute_scoring_metrics
from evaluation_dictee.models.base import Scorer
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Résultat d'un run : métriques + prédictions et labels alignés.

    Attributes:
        metrics: métriques globales (accord, kappa...).
        y_true: codes experts normalisés, aplatis (toutes copies × tous items).
        y_pred: codes prédits normalisés, alignés sur y_true.
        confidences: confiance par item (None si indisponible).
        item_ids: identifiant de l'item pour chaque position (pour l'analyse/item).
        copy_ids: identifiant de la copie pour chaque position.
        predictions_path: chemin du fichier JSONL sauvegardé (None si non sauvegardé).
    """

    metrics: ScoringMetrics
    y_true: list[str]
    y_pred: list[str]
    confidences: list[float | None]
    item_ids: list[str] = field(default_factory=list)
    copy_ids: list[str] = field(default_factory=list)
    predictions_path: Path | None = None
    non_transcribed: list[str] = field(default_factory=list)


def _load_processed_copy_ids(predictions_path: Path) -> set[str]:
    """Lit un fichier de prédictions existant et renvoie les copy_id déjà traités.

    Permet de reprendre un run interrompu : on saute les copies déjà présentes
    dans le fichier `<run>_predictions.jsonl`. Fichier corrompu (dernière ligne
    tronquée par un crash) : on ignore silencieusement la dernière ligne.

    Args:
        predictions_path: chemin du fichier JSONL de prédictions.

    Returns:
        Ensemble des copy_id déjà présents dans le fichier.
    """
    if not predictions_path.exists():
        return set()
    processed: set[str] = set()
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                processed.add(rec["copy_id"])
            except (json.JSONDecodeError, KeyError):
                # Ligne tronquée par un crash : on la laisse tomber
                continue
    return processed


def run_benchmark(
    config: ExperimentConfig,
    scorer: Scorer,
    output_dir: str | Path = "data/processed",
) -> BenchmarkResult:
    """Exécute un benchmark complet pour une config et un modèle donnés.

    Écriture incrémentale : chaque copie évaluée est immédiatement `fsync`-ée
    dans le JSONL, donc un crash à mi-chemin ne perd que la copie en cours.
    Reprise automatique : si un `<run>_predictions.jsonl` existe déjà, les
    copies déjà traitées sont sautées. Pour repartir de zéro, supprimer le
    fichier ou changer `config.name`.

    Erreurs API transitoires : chaque échec sur une copie est loggé et la
    copie est marquée dans `failed_copies.txt`, mais le run continue sur les
    suivantes. Sans ça, un incident réseau à 30 % perdait 8 h de calcul.

    Args:
        config: configuration de l'expérience.
        scorer: modèle implémentant l'interface Scorer.
        output_dir: dossier où sauvegarder le fichier de prédictions détaillées.

    Returns:
        Les métriques et les vecteurs alignés (vrais codes, prédictions, confiances).
    """
    logger.info("Labels : %s", config.data.labels_path)
    logger.info("Images : %s", config.data.images_path)

    copies: list[Copy] = load_dataset(
        images_dir=config.data.images_path,
        labels_csv=config.data.labels_path,
        limit=config.data.limit,
    )

    # Échec explicite si aucune copie — évite la boucle silencieuse
    if not copies:
        raise RuntimeError(
            "Aucune copie chargée. Causes possibles :\n"
            f"  1. Le CSV est inaccessible : {config.data.labels_path}\n"
            f"  2. Les images sont introuvables : {config.data.images_path}\n"
            "     (les noms dans le CSV doivent correspondre aux fichiers du dossier)\n"
            "  3. Les identifiants S3 ne sont pas configurés.\n"
            'Diagnostic : python -c "'
            "from evaluation_dictee.data.loaders import load_labels; "
            f"labels=load_labels('{config.data.labels_path}'); "
            "print(len(labels), 'copies dans le CSV')\""
        )

    # Reprise éventuelle depuis un checkpoint
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config.name}_predictions.jsonl"
    failed_path = output_dir / f"{config.name}_failed_copies.txt"

    processed = _load_processed_copy_ids(out_path)
    if processed:
        logger.info(
            "Reprise détectée : %d copies déjà traitées dans %s. On saute ces copies.",
            len(processed),
            out_path,
        )
    copies_a_traiter = [c for c in copies if c.copy_id not in processed]
    logger.info(
        "%d copies au total, %d à traiter (%d déjà faites).",
        len(copies),
        len(copies_a_traiter),
        len(processed),
    )

    reference = load_grid(config.data.grid_path).reference_text
    scheme = config.grid.scheme

    non_transcrites: list[str] = []
    failed_copies: list[tuple[str, str]] = []  # (copy_id, message d'erreur)

    # Ouverture en mode APPEND : les copies précédemment traitées sont conservées.
    with open(out_path, "a", encoding="utf-8") as f_pred:
        for copy in tqdm(copies_a_traiter, desc=f"Évaluation ({config.name})"):
            # Chaque copie est isolée dans un try/except : une erreur API sur
            # UNE copie ne fait plus perdre les précédentes.
            try:
                prediction = scorer.score_copy(copy, reference)
            except Exception as exc:  # noqa: BLE001 — on veut TOUT rattraper ici
                logger.error(
                    "Échec sur la copie %s : %s. On passe à la suivante.",
                    copy.copy_id,
                    exc,
                )
                failed_copies.append((copy.copy_id, str(exc)))
                continue

            if not prediction.transcribed:
                non_transcrites.append(copy.copy_id)
                logger.warning(
                    "Copie non transcrite après %d tentative(s) : %s (exclue des métriques).",
                    prediction.n_attempts,
                    copy.copy_id,
                )
                continue

            pred_by_id = {it.item_id: it for it in prediction.items}

            # Écriture incrémentale de chaque item de la copie
            for item_id, expert_code in zip(copy.item_ids, copy.expert_codes, strict=True):
                pred = pred_by_id.get(item_id)
                true_code = grid.normalize(expert_code, scheme)
                pred_code = grid.normalize(pred.code, scheme) if pred else "?"
                conf = pred.confidence if pred else 0.0
                record = {
                    "copy_id": copy.copy_id,
                    "item_id": item_id,
                    "y_true": true_code,
                    "y_pred": pred_code,
                    "confidence": conf,
                    "transcription": pred.transcription if pred else None,
                    "comparaison": pred.comparaison if pred else None,
                    "raw_transcription": prediction.raw_transcription,
                }
                f_pred.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Flush + fsync : les données sont sur le disque. Un crash après ce
            # point ne peut plus perdre la copie qu'on vient d'écrire.
            f_pred.flush()
            os.fsync(f_pred.fileno())

    # Sauvegarde de la liste des échecs (utile pour relancer sélectivement)
    if failed_copies:
        with open(failed_path, "w", encoding="utf-8") as f:
            for cid, msg in failed_copies:
                f.write(f"{cid}\t{msg}\n")
        logger.warning(
            "%d copie(s) en échec, listées dans %s (le run continuera à les "
            "reprendre au prochain lancement).",
            len(failed_copies),
            failed_path,
        )

    logger.info("Prédictions sauvegardées : %s", out_path)

    # Recharger l'intégralité du JSONL (celles traitées maintenant + celles
    # déjà présentes en reprise) pour construire les métriques et les vecteurs
    # alignés attendus par BenchmarkResult.
    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float | None] = []
    item_ids: list[str] = []
    copy_ids: list[str] = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            y_true.append(rec["y_true"])
            y_pred.append(rec["y_pred"])
            confidences.append(rec.get("confidence"))
            item_ids.append(rec["item_id"])
            copy_ids.append(rec["copy_id"])
    # Garde-fou : modèle et experts doivent coder dans le MÊME jeu de modalités.
    # Après normalisation, tout code hors de l'alphabet attendu signale une
    # incohérence (ex. prétraitement expert oublié, prompt non aligné sur le schéma).
    attendus = grid.allowed_codes(scheme)
    codes_vus = set(y_true) | set(y_pred)
    intrus = codes_vus - attendus - {"?"}  # "?" = réponse modèle non parsée, traité à part
    if intrus:
        logger.warning(
            "Codes hors du schéma '%s' (attendu %s) détectés : %s. "
            "Vérifier le prétraitement des codes experts et le prompt du modèle.",
            scheme,
            sorted(attendus),
            sorted(intrus),
        )

    if non_transcrites:
        logger.warning(
            "%d copie(s) non transcrite(s) exclue(s) des métriques : %s",
            len(non_transcrites),
            non_transcrites,
        )

    metrics = compute_scoring_metrics(y_true, y_pred)
    return BenchmarkResult(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        confidences=confidences,
        item_ids=item_ids,
        copy_ids=copy_ids,
        predictions_path=out_path,
        non_transcribed=non_transcrites,
    )
