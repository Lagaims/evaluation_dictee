"""Pipeline d'évaluation de la TRANSCRIPTION (HTR) sur le corpus Scoledit.

Objectif distinct du codage de dictée : mesurer, isolément, à quel point un modèle
lit fidèlement l'écriture manuscrite d'un enfant (fautes comprises). On compare la
transcription du modèle à la transcription de référence Scoledit via CER/WER.

Architecture volontairement modulaire pour changer de modèle facilement : le
transcripteur encapsule l'appel VLM et n'expose que `transcribe(image_path) -> str`.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from evaluation_dictee.config import ModelConfig
from evaluation_dictee.data.loaders import load_image
from evaluation_dictee.pipeline.prompts import build_transcription_prompt
from evaluation_dictee.transcription.htr_metrics import (
    TranscriptionMetrics,
    compute_transcription_metrics,
)
from evaluation_dictee.transcription.scoledit import ScoledtSample


def _image_to_data_url(image: Image.Image) -> str:
    """Encode une image PIL en data URL base64."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


class VLMTranscriber:
    """Transcripteur HTR basé sur un VLM (API compatible OpenAI).

    Changer de modèle = changer `model_config.name`. Toute l'évaluation aval
    (CER/WER) est indépendante du modèle choisi.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        base_url: str,
        api_key: str,
        read_final_state: bool = True,
    ) -> None:
        """Initialise le client de transcription.

        Args:
            model_config: modèle multimodal à utiliser.
            base_url: URL de l'API.
            api_key: clé d'API.
            read_final_state: consigne de lire l'état final en cas de rature.
        """
        self.model_config = model_config
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.prompt = build_transcription_prompt(read_final_state=read_final_state)

    def transcribe(self, image_path: str) -> str:
        """Transcrit une image manuscrite en texte brut (fautes préservées).

        Args:
            image_path: chemin de l'image (local ou s3://).

        Returns:
            La transcription produite par le modèle (chaîne vide si échec).
        """
        image = load_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
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
            transcription = self._parse(content)
            if transcription.strip():
                return transcription
        return ""

    @staticmethod
    def _parse(content: str) -> str:
        """Extrait le champ 'transcription' du JSON, avec repli sur texte brut."""
        try:
            cleaned = content.strip().removeprefix("```json").removeprefix("```")
            cleaned = cleaned.removesuffix("```").strip()
            return str(json.loads(cleaned).get("transcription", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            return content.strip()


@dataclass
class HTRBenchmarkResult:
    """Résultat d'un benchmark de transcription.

    Attributes:
        per_sample: DataFrame par échantillon (scan, cer, wer, ...).
        mean_cer / mean_wer : moyennes (micro-pondérées par longueur).
        n_echecs: nombre d'images sans transcription exploitable.
        predictions_path: chemin du fichier de prédictions sauvegardé.
    """

    per_sample: pd.DataFrame
    mean_cer: float
    mean_wer: float
    n_echecs: int
    predictions_path: Path | None = None
    failed_scans: list[str] = field(default_factory=list)


def run_htr_benchmark(
    samples: list[ScoledtSample],
    transcriber: VLMTranscriber,
    run_name: str,
    output_dir: str | Path = "data/processed",
) -> HTRBenchmarkResult:
    """Transcrit tous les échantillons et calcule les métriques HTR.

    Args:
        samples: échantillons Scoledit (image + référence).
        transcriber: transcripteur (encapsule le modèle choisi).
        run_name: nom du run (pour le fichier de sortie).
        output_dir: dossier de sauvegarde des prédictions.

    Returns:
        Les résultats agrégés et par échantillon.
    """
    records = []
    failed = []
    for s in tqdm(samples, desc=f"Transcription ({run_name})"):
        hypothese = transcriber.transcribe(s.image_path)
        if not hypothese.strip():
            failed.append(s.scan)
        m: TranscriptionMetrics = compute_transcription_metrics(s.reference, hypothese)
        records.append(
            {
                "scan": s.scan,
                "level": s.level,
                "reference": s.reference,
                "hypothese": hypothese,
                "cer": m.cer,
                "wer": m.wer,
                "cer_normalise": m.cer_normalise,
                "wer_normalise": m.wer_normalise,
                "n_char_ref": m.n_char_ref,
                "n_mots_ref": m.n_mots_ref,
                "transcrit": bool(hypothese.strip()),
            }
        )

    df = pd.DataFrame(records)
    # CER/WER micro-pondérés par la longueur de la référence (échantillons transcrits)
    ok = df[df["transcrit"]]
    mean_cer = (
        float((ok["cer"] * ok["n_char_ref"]).sum() / ok["n_char_ref"].sum())
        if len(ok) and ok["n_char_ref"].sum() > 0
        else float("nan")
    )
    mean_wer = (
        float((ok["wer"] * ok["n_mots_ref"]).sum() / ok["n_mots_ref"].sum())
        if len(ok) and ok["n_mots_ref"].sum() > 0
        else float("nan")
    )

    out_path = Path(output_dir) / f"{run_name}_htr_predictions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return HTRBenchmarkResult(
        per_sample=df,
        mean_cer=mean_cer,
        mean_wer=mean_wer,
        n_echecs=len(failed),
        predictions_path=out_path,
        failed_scans=failed,
    )
