"""Scorer en deux étapes (APPROCHE 1) : transcription HTR puis codage textuel.

Contraste avec `VLMScorer` (approche 2, end-to-end). Ici on sépare explicitement :

  Étape 1 (HTR) : un modèle multimodal lit l'image et restitue le texte brut
                  écrit par l'élève, SANS connaître le texte de référence ni le
                  découpage en items. On mesure ainsi la lecture pour elle-même.

  Étape 2 (codage) : un modèle (éventuellement texte seul, plus léger) reçoit la
                     transcription de l'étape 1 + le texte de référence, et code
                     chaque item. Il ne voit pas l'image.

Avantages : la transcription intermédiaire est réutilisable et inspectable ;
les sources d'erreur (lecture vs jugement) sont découplées.
Limite : deux étapes = deux sources d'erreur cumulables.

Le scorer respecte la même interface `Scorer` que `VLMScorer`, donc il est évalué
par exactement le même code de benchmark et de métriques (comparaison équitable).
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.loaders import Copy, load_image
from evaluation_dictee.data.reference import GridItem
from evaluation_dictee.models.base import CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.models.vlm import _image_to_data_url
from evaluation_dictee.pipeline.alignment import best_realignment, needs_realignment
from evaluation_dictee.pipeline.prompts import (
    build_text_coding_prompt,
    build_transcription_prompt,
)
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_items_from_content(content: str) -> list[dict]:
    """Extrait la liste d'items d'une réponse modèle, de façon robuste.

    Cascade de tentatives, de la plus stricte à la plus permissive :
    1. JSON complet après nettoyage des ```json``` ;
    2. Recherche d'un objet racine contenant "items" par regex ;
    3. Extraction directe de tous les sous-objets à structure d'item
       (au moins item_id + code) — utile si le JSON global est tronqué.

    Args:
        content: réponse brute du modèle.

    Returns:
        La liste d'items trouvée (vide si rien d'exploitable).
    """
    if not content:
        return []

    # Tentative 1 : JSON complet
    cleaned = content.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
        items = data.get("items")
        if isinstance(items, list) and items:
            return items
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        pass

    # Tentative 2 : trouver le premier objet racine avec une clé "items"
    m = re.search(r'\{[^{}]*"items"\s*:\s*\[', content)
    if m:
        start = m.start()
        # Aller jusqu'au } équilibré correspondant (approximation)
        depth = 0
        end = None
        for i in range(start, len(content)):
            c = content[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            try:
                data = json.loads(content[start:end])
                items = data.get("items")
                if isinstance(items, list) and items:
                    return items
            except json.JSONDecodeError:
                pass

    # Tentative 3 : récupérer TOUS les sous-objets qui ressemblent à un item
    # (utile si la liste "items" est tronquée mais que les objets sont valides)
    items = []
    for match in re.finditer(
        r'\{\s*"item_id"\s*:\s*"[^"]+"[^{}]*"code"\s*:\s*"[^"]+"[^{}]*\}',
        content,
    ):
        try:
            items.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return items


class TwoStageScorer(Scorer):
    """Évalue une copie en deux étapes : transcription HTR puis codage du texte."""

    def __init__(
        self,
        model_config: ModelConfig,
        prompt_config: PromptConfig,
        base_url: str,
        api_key: str,
        grid_items: list[GridItem],
        scheme: str = "simplifiee",
        model_config_stage2: ModelConfig | None = None,
    ) -> None:
        """Initialise les clients des deux étapes.

        Args:
            model_config: modèle de l'étape 1 (HTR) — doit être multimodal.
            prompt_config: stratégie de prompting (ratures...).
            base_url: URL de l'API (compatible OpenAI).
            api_key: clé d'API.
            grid_items: items de la grille (mot attendu, dans l'ordre).
            scheme: schéma de codage ("simplifiee" ou "complete").
            model_config_stage2: modèle de l'étape 2 (codage texte). Si None, on
                réutilise `model_config`. Peut être un modèle texte seul, plus léger.
        """
        self.model_config = model_config
        self.model_config_stage2 = model_config_stage2 or model_config
        self.prompt_config = prompt_config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.grid_items = grid_items
        self.scheme = scheme
        self._items_by_id = {it.item_id: it for it in grid_items}

    # ── Étape 1 : transcription HTR ──────────────────────────────────────────
    def _transcribe(self, copy: Copy) -> tuple[str, int]:
        """Lit l'image et renvoie (transcription, nb_tentatives).

        Réessaie tant que la réponse est vide/non parsable, comme VLMScorer.

        Returns:
            (transcription, n_attempts). Transcription vide si échec total.
        """
        image = load_image(copy.image_path)
        prompt = build_transcription_prompt(read_final_state=self.prompt_config.read_final_state)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }
        ]
        for attempt in range(self.model_config.max_retries + 1):
            temp = self.model_config.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=temp,
                max_tokens=self.model_config.max_tokens,
                messages=messages,
            )
            content = response.choices[0].message.content or "{}"
            transcription = self._parse_transcription(content)
            if transcription.strip():
                return transcription, attempt + 1
        return "", self.model_config.max_retries + 1

    @staticmethod
    def _parse_transcription(content: str) -> str:
        """Extrait le champ 'transcription' du JSON renvoyé à l'étape 1."""
        try:
            cleaned = content.strip().removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
            data = json.loads(cleaned)
            return str(data.get("transcription", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            # Repli : si le modèle a répondu en texte brut, on le prend tel quel.
            return content.strip()

    # ── Étape 2 : codage du texte transcrit ──────────────────────────────────
    def _code_text(self, copy: Copy, reference_text: str, transcription: str) -> CopyPrediction:
        """Code chaque item à partir de la transcription (sans image).

        Réessaie tant que le JSON de réponse est vide ou non parsable, comme
        l'étape 1. Sans ce retry, un échec ponctuel de génération JSON (préambule
        parasite, troncature) codait toute la copie en '?' silencieusement.
        """
        items_a_coder = [self._items_by_id[i] for i in copy.item_ids if i in self._items_by_id]
        prompt = build_text_coding_prompt(
            reference_text=reference_text,
            transcription=transcription,
            items=items_a_coder,
            scheme=self.scheme,
        )
        for attempt in range(self.model_config_stage2.max_retries + 1):
            temp = self.model_config_stage2.temperature + (0.3 if attempt > 0 else 0.0)
            response = self.client.chat.completions.create(
                model=self.model_config_stage2.name,
                temperature=temp,
                max_tokens=self.model_config_stage2.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or ""
            # On regarde d'abord si le JSON est extractible avant de le parser.
            if _extract_items_from_content(content):
                return self._parse_coding(copy, content)
            logger.warning(
                "Étape 2 : JSON non extractible à l'essai %d/%d pour %s. Nouvel essai.",
                attempt + 1,
                self.model_config_stage2.max_retries + 1,
                copy.copy_id,
            )
        # Tous les essais ont échoué : on parse quand même (produira des '?')
        # pour laisser _parse_coding logger et remonter proprement.
        return self._parse_coding(copy, content)

    def _parse_coding(self, copy: Copy, content: str) -> CopyPrediction:
        """Parse le JSON de codage de l'étape 2 (mêmes garanties que VLMScorer).

        Extraction JSON robuste : gère les préambules textuels, blocs ```json,
        et tente une recherche par regex si le parsing direct échoue (utile
        quand le modèle ajoute du texte avant/après le JSON).

        Applique le ré-alignement de sécurité si la transcription par item révèle
        un décalage. Réutilise exactement la logique d'alignement de l'approche 2,
        ce qui garde les deux approches comparables.
        """
        raw_items = _extract_items_from_content(content)
        if not raw_items:
            # Échec total du parsing : le modèle n'a pas rendu de JSON exploitable.
            # On loggue explicitement pour que le silence ne masque plus le problème.
            logger.warning(
                "Étape 2 (codage textuel) : aucun item extractible du JSON pour la "
                "copie %s. Réponse tronquée ou format cassé (longueur brute : %d). "
                "Tous les items seront codés '?'. Extrait : %r",
                copy.copy_id,
                len(content),
                content[:200],
            )

        codes_seq = [str(it.get("code", "?")).strip() for it in raw_items]
        trans_seq = [it.get("transcription") for it in raw_items]
        conf_seq = [it.get("confidence") for it in raw_items]

        expected_words = [
            self._items_by_id[i].attendu for i in copy.item_ids if i in self._items_by_id
        ]

        if codes_seq and needs_realignment(expected_words, trans_seq):
            aligned = best_realignment(expected_words, codes_seq, trans_seq, conf_seq)
            items = [
                ItemPrediction(
                    item_id=item_id,
                    code=a.code,
                    confidence=a.confidence,
                    transcription=a.transcription,
                )
                for item_id, a in zip(copy.item_ids, aligned, strict=False)
            ]
            return CopyPrediction(copy_id=copy.copy_id, items=items)

        by_id = {it.get("item_id"): it for it in raw_items}
        items = []
        for item_id in copy.item_ids:
            entry = by_id.get(item_id)
            if entry is None:
                items.append(ItemPrediction(item_id=item_id, code="?", confidence=0.0))
            else:
                items.append(
                    ItemPrediction(
                        item_id=item_id,
                        code=str(entry.get("code", "?")).strip(),
                        confidence=entry.get("confidence"),
                        transcription=entry.get("transcription"),
                        comparaison=entry.get("comparaison"),
                    )
                )
        return CopyPrediction(copy_id=copy.copy_id, items=items)

    # ── Interface Scorer ─────────────────────────────────────────────────────
    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        """Évalue une copie en enchaînant transcription puis codage.

        Args:
            copy: copie à évaluer.
            reference_text: texte de référence de la dictée.

        Returns:
            Prédictions par item. Si l'étape 1 ne produit aucune transcription,
            la copie est marquée non transcrite (exclue des métriques).
        """
        transcription, n_attempts = self._transcribe(copy)

        if not transcription.strip():
            # Échec de lecture : aucune transcription exploitable.
            items_vides = [
                ItemPrediction(item_id=i, code="?", confidence=0.0) for i in copy.item_ids
            ]
            return CopyPrediction(
                copy_id=copy.copy_id,
                items=items_vides,
                transcribed=False,
                n_attempts=n_attempts,
            )

        prediction = self._code_text(copy, reference_text or "", transcription)
        prediction.n_attempts = n_attempts
        prediction.raw_transcription = transcription
        return prediction