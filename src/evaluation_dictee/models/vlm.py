"""Client pour un VLM open weight servi par vLLM (API compatible OpenAI).

Implémente la méthode C (end-to-end) : on envoie l'image, le texte de référence
et la grille, et le modèle renvoie un code par item au format JSON, avec un score
de confiance. Voir CLAUDE.md §4.
"""

from __future__ import annotations

import base64
import io
import json

from openai import OpenAI
from PIL import Image

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.loaders import Copy, load_image
from evaluation_dictee.data.reference import GridItem
from evaluation_dictee.models.base import CopyPrediction, ItemPrediction, Scorer
from evaluation_dictee.pipeline.alignment import best_realignment, needs_realignment
from evaluation_dictee.pipeline.prompts import build_dictation_prompt


def _image_to_data_url(image: Image.Image) -> str:
    """Encode une image PIL en data URL base64 (format attendu par l'API)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class VLMScorer(Scorer):
    """Évalue une copie via un VLM (méthode C end-to-end)."""

    def __init__(
        self,
        model_config: ModelConfig,
        prompt_config: PromptConfig,
        base_url: str,
        api_key: str,
        grid_items: list[GridItem],
        scheme: str = "simplifiee",
    ) -> None:
        """Initialise le client.

        Args:
            model_config: paramètres du modèle.
            prompt_config: stratégie de prompting.
            base_url: URL de l'API vLLM (compatible OpenAI).
            api_key: clé d'API (factice en interne SSP Cloud).
            grid_items: items de la grille (id + mot attendu), pour ancrer
                l'alignement dans le prompt.
            scheme: schéma de codage demandé au modèle ("simplifiee" ou "complete").
        """
        self.model_config = model_config
        self.prompt_config = prompt_config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.grid_items = grid_items
        self.scheme = scheme
        self._items_by_id = {it.item_id: it for it in grid_items}

    def score_copy(self, copy: Copy, reference_text: str | None) -> CopyPrediction:
        """Évalue une copie complète en un appel au VLM.

        Args:
            copy: copie à évaluer.
            reference_text: texte de référence de la dictée.

        Returns:
            Prédictions par item.
        """
        image = load_image(copy.image_path)
        # On code les items présents dans la copie, dans l'ordre de la grille.
        items_a_coder = [self._items_by_id[i] for i in copy.item_ids if i in self._items_by_id]
        prompt = build_dictation_prompt(
            reference_text=reference_text or "",
            items=items_a_coder,
            config=self.prompt_config,
            scheme=self.scheme,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }
        ]

        # Tentatives successives tant que la réponse n'est pas exploitable.
        # On augmente légèrement la température aux essais suivants pour sortir
        # d'une éventuelle réponse vide déterministe.
        prediction: CopyPrediction | None = None
        for attempt in range(self.model_config.max_retries + 1):
            temp = self.model_config.temperature + (0.3 if attempt > 0 else 0.0)
            # Les modèles thinking (Qwen3, DeepSeek-R1) génèrent par défaut un
            # bloc <think>...</think> qui casse le parsing JSON et multiplie la
            # latence. On le désactive via extra_body quand disable_thinking=True.
            extra_body = (
                {"chat_template_kwargs": {"enable_thinking": False}}
                if self.model_config.disable_thinking
                else {}
            )
            response = self.client.chat.completions.create(
                model=self.model_config.name,
                temperature=temp,
                max_tokens=self.model_config.max_tokens,
                messages=messages,
                extra_body=extra_body or None,
            )
            content = response.choices[0].message.content or "{}"
            prediction = self._parse_response(copy, content)
            prediction.n_attempts = attempt + 1
            if prediction.transcribed:
                return prediction
        # Toutes les tentatives ont échoué : copie non transcrite.
        return prediction  # type: ignore[return-value]

    def _parse_response(self, copy: Copy, content: str) -> CopyPrediction:
        """Parse la réponse JSON du modèle en prédictions par item.

        Le modèle renvoie un objet de la forme :
            {"items": [{"item_id": "...", "transcription": "...", "code": "1",
                        "confidence": 0.97}, ...]}

        Robuste aux réponses imparfaites. Applique en outre un filet de sécurité :
        si la transcription du modèle révèle un décalage (mot scindé/collé par
        l'élève), les prédictions sont ré-alignées sur les mots attendus.
        """
        try:
            cleaned = content.strip().removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
            data = json.loads(cleaned)
            raw_items = data.get("items", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            raw_items = []

        # Séquence des prédictions DANS L'ORDRE renvoyé par le modèle.
        codes_seq = [str(it.get("code", "?")).strip() for it in raw_items]
        trans_seq = [it.get("transcription") for it in raw_items]
        conf_seq = [it.get("confidence") for it in raw_items]

        # Aucune réponse exploitable : copie non transcrite (à réessayer puis exclure).
        n_trans_utiles = sum(1 for t in trans_seq if t and str(t).strip())
        if not raw_items or n_trans_utiles == 0:
            items_vides = [
                ItemPrediction(item_id=i, code="?", confidence=0.0) for i in copy.item_ids
            ]
            return CopyPrediction(copy_id=copy.copy_id, items=items_vides, transcribed=False)

        # Mots attendus dans l'ordre de la copie.
        expected_words = [
            self._items_by_id[i].attendu for i in copy.item_ids if i in self._items_by_id
        ]

        # Filet de sécurité : ré-aligner si un décalage est détecté.
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

        # Sinon : mapping classique par item_id renvoyé par le modèle.
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