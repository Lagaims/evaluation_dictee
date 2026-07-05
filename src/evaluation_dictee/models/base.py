"""Interface commune à tous les modèles d'évaluation.

Chaque méthode (A/B/C/D) implémente la même interface `Scorer`, ce qui permet
de les comparer avec le même code de benchmark et les mêmes métriques.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from evaluation_dictee.data.loaders import Copy


@dataclass
class ItemPrediction:
    """Prédiction du modèle pour un item.

    Attributes:
        item_id: identifiant de l'item.
        code: code prédit (selon le schéma de la config).
        confidence: score de confiance dans [0, 1] (None si indisponible).
        transcription: ce que le modèle a lu (utile pour l'analyse, optionnel).
        comparaison: description de la différence lue-attendue quand le mode
            chain-of-thought est activé (None sinon). Sert au diagnostic.
    """

    item_id: str
    code: str
    confidence: float | None = None
    transcription: str | None = None
    comparaison: str | None = None


@dataclass
class CopyPrediction:
    """Ensemble des prédictions du modèle pour une copie.

    Attributes:
        copy_id: identifiant de la copie.
        items: prédictions par item.
        transcribed: False si le modèle n'a renvoyé aucune transcription
            exploitable (réponse vide/non parsable après tous les essais).
            Une copie non transcrite est exclue du calcul des métriques.
        n_attempts: nombre de tentatives effectuées pour obtenir une réponse.
    """

    copy_id: str
    items: list[ItemPrediction]
    transcribed: bool = True
    n_attempts: int = 1
    # Transcription brute produite à l'étape 1 (approche two_stage). None en
    # approche end_to_end, où la transcription n'existe que par item.
    raw_transcription: str | None = None


class Scorer(ABC):
    """Contrat que doit respecter tout modèle d'évaluation de copie.

    Implémenter `score_copy` pour une nouvelle méthode suffit à l'intégrer au
    pipeline de benchmark.
    """

    @abstractmethod
    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        """Évalue une copie et renvoie un code (+ confiance) par item.

        Args:
            copy: la copie à évaluer (image + identifiants d'items).
            reference_text: texte de référence de la dictée (None si production libre).

        Returns:
            Les prédictions pour tous les items de la copie.
        """
        raise NotImplementedError
