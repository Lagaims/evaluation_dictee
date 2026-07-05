"""Chargement et validation des configurations d'expérience.

Toute la configuration passe par des fichiers YAML (dossier `configs/`) validés
par Pydantic. Aucun chemin ni paramètre ne doit être codé en dur ailleurs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Secrets lus depuis les variables d'environnement (ou le fichier .env).

    Ne contient jamais de valeur en clair dans le code. Voir .env.example.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_endpoint: str = "minio.lab.sspcloud.fr"
    s3_bucket: str = ""

    llm_base_url: str = "https://llm.lab.sspcloud.fr/api/v1"
    llm_api_key: str = "dummy"

    mlflow_tracking_uri: str = ""


class ModelConfig(BaseModel):
    """Paramètres du modèle à interroger."""

    # Nom du modèle servi par vLLM (ex: "Qwen/Qwen2.5-VL-7B-Instruct")
    name: str
    # "vlm" = modèle multimodal image+texte ; "llm" = texte seul ; "htr" = HTR classique
    kind: Literal["vlm", "llm", "htr"] = "vlm"
    temperature: float = 0.0
    max_tokens: int = 2048
    # Demander les log-probabilités pour estimer la confiance par item
    request_logprobs: bool = True
    # Nombre de tentatives si la réponse est vide/non parsable (0 = pas de retry).
    # Une copie restée sans transcription après tous les essais est marquée
    # "non transcrite" et exclue du calcul des métriques de performance.
    max_retries: int = 2
    # Désactiver le mode « thinking » natif des modèles Qwen3 / DeepSeek-R1.
    # Ces modèles génèrent par défaut un bloc <think>...</think> avant la réponse,
    # ce qui casse le parsing JSON et multiplie la latence par 5 à 10.
    # Mettre à True pour tout modèle thinking : Qwen3-235B-A22B, Qwen3.6-35B-A3B,
    # QwQ, DeepSeek-R1, etc.
    # Sans effet sur les modèles classiques (gemma4, Qwen2.5-VL, ...).
    disable_thinking: bool = True


class DataConfig(BaseModel):
    """Localisation et périmètre des données."""

    corpus: Literal["dictee", "production_ecrite"] = "dictee"
    # Dossier des imagettes. Accepte un chemin local OU un préfixe S3, ex :
    #   s3://projet-production-ecrits-depp/dictee_2015/
    images_path: str
    # Fichier CSV des codes de l'annotateur expert. Comme la dictée n'a qu'un seul
    # annotateur, ce fichier fait office de vérité de terrain (gold standard).
    # Accepte un chemin local OU S3, ex :
    #   s3://projet-production-ecrits-depp/resultat_dictee_2015.csv
    labels_path: str
    # Grille de codage au format JSON (versionnée dans configs/). Contient, par
    # item, le mot attendu et les fautes connues. Sert à construire le prompt et
    # à fournir le texte de référence (plus besoin d'un fichier .txt séparé).
    grid_path: str = "configs/grille_dictee_2015.json"
    # Limiter le nombre de copies (utile pour les tests rapides)
    limit: int | None = None


class GridConfig(BaseModel):
    """Schéma de codage cible."""

    # "simplifiee" = 1/erreur/0 (cible principale) ; "complete" = 1/3/4/5/9/0
    scheme: Literal["simplifiee", "complete"] = "simplifiee"


class PromptConfig(BaseModel):
    """Stratégie de prompting."""

    method: Literal["A", "B", "C", "D"] = "C"
    n_few_shot: int = 0  # nombre d'exemples annotés dans le prompt
    enforce_faithful: bool = True  # consigne anti-sur-correction
    read_final_state: bool = True  # règle des ratures : lire l'état final
    # Chain-of-thought : force le modèle à écrire un champ "comparaison" AVANT
    # le code, pour verbaliser la différence lue-attendue. Rend le raisonnement
    # inspectable et réduit les erreurs bien lues mais mal codées (« mis » vs « mit »).
    chain_of_thought: bool = False


class ExperimentConfig(BaseModel):
    """Configuration complète d'une expérience (un fichier YAML = un run)."""

    name: str = Field(..., description="Nom unique du run, utilisé dans MLflow")
    seed: int = 42
    # Approche d'évaluation :
    # - "end_to_end" (approche 2) : un VLM lit l'image ET code en une passe.
    # - "two_stage"  (approche 1) : étape 1 HTR (transcription) puis étape 2
    #   codage du texte transcrit. Permet d'isoler lecture et jugement.
    approach: Literal["end_to_end", "two_stage"] = "end_to_end"
    model: ModelConfig
    # Modèle de l'ÉTAPE 2 (codage textuel) pour l'approche two_stage. Peut être
    # un modèle texte seul, plus léger. Si absent, on réutilise `model`.
    model_stage2: ModelConfig | None = None
    data: DataConfig
    grid: GridConfig = GridConfig()
    prompt: PromptConfig = PromptConfig()


def load_config(path: str | Path) -> ExperimentConfig:
    """Charge et valide une configuration d'expérience depuis un fichier YAML.

    Args:
        path: chemin vers le fichier YAML.

    Returns:
        La configuration validée.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration pour le pipeline HTR (transcription seule, corpus Scoledit)
# ─────────────────────────────────────────────────────────────────────────────


class HTRDataConfig(BaseModel):
    """Localisation des données Scoledit pour l'évaluation HTR."""

    # Dossier des images de copies (local ou s3://.../scans/CE1/).
    scans_path: str
    # Dossier des transcriptions JSON de référence (local ou s3://.../annotation/CE1/).
    annotations_path: str
    # Limiter le nombre d'échantillons (utile pour les tests rapides).
    limit: int | None = None


class HTRExperimentConfig(BaseModel):
    """Configuration complète d'une expérience de transcription (HTR).

    Volontairement distincte d'`ExperimentConfig` car la structure des données
    (scans + annotations JSON) et les métriques (CER/WER) diffèrent de celles
    du codage de dictée. Mêmes principes toutefois : validation Pydantic,
    un fichier YAML = un run reproductible.
    """

    name: str = Field(..., description="Nom unique du run.")
    seed: int = 42
    model: ModelConfig
    data: HTRDataConfig
    # Consigne HTR : en cas de rature, lire l'état final (corrigé par l'élève).
    read_final_state: bool = True


def load_htr_config(path: str | Path) -> HTRExperimentConfig:
    """Charge et valide une configuration HTR depuis un fichier YAML.

    Args:
        path: chemin vers le fichier YAML.

    Returns:
        La configuration HTR validée.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return HTRExperimentConfig.model_validate(raw)