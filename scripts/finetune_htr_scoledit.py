"""Fine-tune un VLM gemma4 pour la transcription (HTR) sur Scoledit tous niveaux.

Objectif : améliorer la fidélité de lecture du modèle sur l'écriture manuscrite
d'enfants du CP au CM2, en fine-tunant avec QLoRA (LoRA en 4-bit).

Corpus : Scoledit CP, CE1, CE2, CM1, CM2 (scans + transcriptions humaines TEI).
Le modèle apprend à reproduire fidèlement le texte de l'élève, FAUTES COMPRISES —
c'est la lecture qu'on entraîne, pas la correction orthographique.

Sortie : un adaptateur LoRA (~50-200 Mo) qui se charge par-dessus le modèle de
base à l'inférence. Le fichier de configuration YAML pilote tous les
hyperparamètres pour la reproductibilité.

Prérequis (à installer sur le service GPU Onyxia) :
    pip install unsloth trl peft transformers bitsandbytes accelerate
                datasets pillow s3fs pyyaml pydantic mlflow

Usage :
    python scripts/finetune_htr_scoledit.py --config configs/finetune_htr_gemma4.yaml

Ce script s'exécute sur une H100/A100 80 Go du SSP Cloud. Ne pas lancer sans GPU.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fsspec
import yaml
from PIL import Image
from pydantic import BaseModel, Field

from evaluation_dictee.transcription.scoledit import ScoledtSample, tei_to_text
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Pydantic dédiée au fine-tuning
# ─────────────────────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    """Localisation des données Scoledit multi-niveaux et split train/val/test."""

    # Racine des scans (contient les sous-dossiers CP, CE1, CE2, CM1, CM2).
    scans_root: str = "s3://projet-production-ecrits-depp/scoledit/scans"
    # Racine des annotations (idem).
    annotations_root: str = "s3://projet-production-ecrits-depp/scoledit/annotation"
    # Niveaux à inclure dans l'entraînement.
    levels: list[str] = Field(default_factory=lambda: ["CP", "CE1", "CE2", "CM1", "CM2"])
    # Répartition des splits : 70 % train, 15 % val, 15 % test.
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # Le reste va dans le test set (jamais vu pendant l'entraînement).
    seed: int = 42
    # Limiter le nombre total d'échantillons (utile pour un test rapide).
    limit: int | None = None


class LoraConfig(BaseModel):
    """Hyperparamètres LoRA."""

    r: int = 16  # rang (16 = bon compromis qualité/mémoire)
    lora_alpha: int = 32  # échelle des adapteurs (typiquement 2 × r)
    lora_dropout: float = 0.05
    # Cibler les couches vision ET langage : essentiel pour l'HTR.
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


class TrainingConfig(BaseModel):
    """Hyperparamètres d'entraînement."""

    output_dir: str = "checkpoints/htr_gemma4_scoledit"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # batch effectif = 2 × 8 = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler_type: Literal["cosine", "linear", "constant"] = "cosine"
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3  # garder les 3 derniers checkpoints
    bf16: bool = True  # H100 supporte bfloat16 nativement
    gradient_checkpointing: bool = True  # -30 % VRAM
    report_to: Literal["mlflow", "none", "tensorboard"] = "mlflow"
    max_seq_length: int = 2048


class FinetuneConfig(BaseModel):
    """Configuration complète du fine-tuning."""

    name: str = Field(..., description="Nom unique du run (MLflow + dossier).")
    # Modèle de base à fine-tuner (identifiant HuggingFace).
    base_model: str = "unsloth/gemma-4-27b-it"
    # Chargement en 4-bit (QLoRA) : ~20 Go VRAM au lieu de 60.
    load_in_4bit: bool = True
    data: DataConfig = Field(default_factory=DataConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    # Prompt fixe utilisé pour transcrire (identique à celui d'inférence !).
    # Point crucial : le prompt d'entraînement et d'inférence DOIVENT matcher.
    read_final_state: bool = True


# Résolution des annotations forward-declared (nécessaire avec
# `from __future__ import annotations` + Pydantic + Literal).
TrainingConfig.model_rebuild()
FinetuneConfig.model_rebuild()


def load_finetune_config(path: str | Path) -> FinetuneConfig:
    """Charge et valide la config de fine-tuning."""
    with open(path, encoding="utf-8") as f:
        return FinetuneConfig.model_validate(yaml.safe_load(f))


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des données Scoledit multi-niveaux
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DatasetSplit:
    """Contient les trois splits train/val/test."""

    train: list[ScoledtSample]
    val: list[ScoledtSample]
    test: list[ScoledtSample]


def _list_annotations(fs, annotations_dir: str) -> list[str]:
    """Liste les fichiers .json d'un dossier, avec ou sans schéma s3://."""
    pattern = annotations_dir.rstrip("/") + "/*.json"
    paths = sorted(fs.glob(pattern))
    if annotations_dir.startswith("s3://"):
        paths = [p if p.startswith("s3://") else "s3://" + p for p in paths]
    return paths


def load_scoledit_all_levels(cfg: DataConfig) -> list[ScoledtSample]:
    """Charge les échantillons Scoledit pour tous les niveaux configurés.

    Args:
        cfg: config des données (chemins, niveaux, limite).

    Returns:
        Liste unifiée d'échantillons (avec le niveau conservé pour la stratification).
    """
    samples: list[ScoledtSample] = []
    fs, _ = fsspec.core.url_to_fs(cfg.annotations_root)
    for level in cfg.levels:
        scans_dir = f"{cfg.scans_root.rstrip('/')}/{level}/"
        annotations_dir = f"{cfg.annotations_root.rstrip('/')}/{level}/"
        json_paths = _list_annotations(fs, annotations_dir)
        logger.info("Niveau %s : %d annotations trouvées.", level, len(json_paths))
        for jpath in json_paths:
            try:
                with fsspec.open(jpath, "rt", encoding="utf-8") as f:
                    meta = json.load(f)
                scan = meta.get("scan", "")
                samples.append(
                    ScoledtSample(
                        scan=scan,
                        level=meta.get("level", level),
                        student_id=int(meta.get("student_id", -1)),
                        image_path=f"{scans_dir.rstrip('/')}/{scan}.jpg",
                        reference=tei_to_text(meta.get("tei", "")),
                        tei_raw=meta.get("tei", ""),
                    )
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Échantillon ignoré (%s) : %s", jpath, exc)
        if cfg.limit is not None and len(samples) >= cfg.limit:
            samples = samples[: cfg.limit]
            break
    logger.info("Total : %d échantillons chargés.", len(samples))
    return samples


def stratified_split(samples: list[ScoledtSample], cfg: DataConfig) -> DatasetSplit:
    """Split train/val/test STRATIFIÉ par niveau scolaire.

    Assure que chaque niveau (CP, CE1, ..., CM2) soit représenté dans les trois
    splits selon les mêmes proportions. Sans stratification, un niveau minoritaire
    pourrait n'apparaître que dans le train, biaisant l'évaluation.
    """
    rng = random.Random(cfg.seed)
    by_level: dict[str, list[ScoledtSample]] = {}
    for s in samples:
        by_level.setdefault(s.level, []).append(s)

    train, val, test = [], [], []
    for level, sub in by_level.items():
        rng.shuffle(sub)
        n = len(sub)
        n_train = int(n * cfg.train_ratio)
        n_val = int(n * cfg.val_ratio)
        train.extend(sub[:n_train])
        val.extend(sub[n_train : n_train + n_val])
        test.extend(sub[n_train + n_val :])
        logger.info(
            "  %s : %d train + %d val + %d test (total %d)",
            level,
            n_train,
            n_val,
            n - n_train - n_val,
            n,
        )
    rng.shuffle(train)  # mélanger les niveaux
    rng.shuffle(val)
    return DatasetSplit(train=train, val=val, test=test)


# ─────────────────────────────────────────────────────────────────────────────
# Conversion en format conversation multimodale pour SFTTrainer
# ─────────────────────────────────────────────────────────────────────────────


_PROMPT_TRANSCRIPTION = (
    "Tu es un expert en lecture d'écriture manuscrite d'enfants.\n"
    "Transcris EXACTEMENT le texte écrit par l'élève, mot pour mot, "
    "FAUTES D'ORTHOGRAPHE COMPRISES. Ne corrige rien, ne complète rien. "
    "Reproduis fidèlement les erreurs, y compris les accents manquants, "
    "les mots mal orthographiés et la ponctuation.\n"
    "Si l'élève a raturé puis réécrit, transcris uniquement la version "
    "FINALE (non barrée).\n"
    "Réponds UNIQUEMENT par un objet JSON de la forme :\n"
    '{"transcription": "le texte exact écrit par l\'élève"}'
)


def _load_image_from_s3_or_local(path: str) -> Image.Image:
    """Charge une image en mémoire (S3 ou local) au format PIL RGB."""
    with fsspec.open(path, "rb") as f:
        data = f.read()
    img = Image.open(io.BytesIO(data))
    # Convertir en RGB (les VLM attendent 3 canaux ; les JPEG le sont déjà)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def sample_to_conversation(s: ScoledtSample) -> dict:
    """Convertit un échantillon Scoledit en conversation multimodale.

    Format compatible SFTTrainer (Unsloth / TRL) : image + prompt utilisateur,
    puis réponse assistant contenant le JSON de transcription. Le modèle
    apprend à imiter la réponse assistant.

    Args:
        s: échantillon Scoledit (image + référence humaine).

    Returns:
        Dict au format {"messages": [...]} attendu par le trainer.
    """
    reponse_attendue = json.dumps({"transcription": s.reference}, ensure_ascii=False)
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _load_image_from_s3_or_local(s.image_path)},
                    {"type": "text", "text": _PROMPT_TRANSCRIPTION},
                ],
            },
            {"role": "assistant", "content": reponse_attendue},
        ]
    }


def build_hf_dataset(samples: list[ScoledtSample]):
    """Construit un `datasets.Dataset` HuggingFace à partir des échantillons."""
    from datasets import Dataset

    return Dataset.from_list([sample_to_conversation(s) for s in samples])


# ─────────────────────────────────────────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────────────────────────────────────────


def run_finetuning(cfg: FinetuneConfig) -> None:
    """Enchaîne : chargement modèle, données, entraînement, sauvegarde adapteur.

    Args:
        cfg: configuration validée du fine-tuning.
    """
    # Imports paresseux : ces libs sont lourdes et pas installées côté CPU
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    logger.info("Chargement du modèle de base : %s", cfg.base_model)
    model, tokenizer = FastVisionModel.from_pretrained(
        cfg.base_model,
        load_in_4bit=cfg.load_in_4bit,
        use_gradient_checkpointing="unsloth" if cfg.training.gradient_checkpointing else False,
    )

    logger.info("Application de LoRA (r=%d, alpha=%d)", cfg.lora.r, cfg.lora.lora_alpha)
    model = FastVisionModel.get_peft_model(
        model,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_language_layers=cfg.lora.finetune_language_layers,
        finetune_attention_modules=cfg.lora.finetune_attention_modules,
        finetune_mlp_modules=cfg.lora.finetune_mlp_modules,
        random_state=cfg.data.seed,
    )

    logger.info("Chargement des données Scoledit multi-niveaux...")
    samples = load_scoledit_all_levels(cfg.data)
    split = stratified_split(samples, cfg.data)
    logger.info(
        "Splits : %d train, %d val, %d test",
        len(split.train),
        len(split.val),
        len(split.test),
    )

    # Sauvegarder la liste des IDs par split (reproductibilité / traçabilité)
    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
    splits_manifest = {
        "train": [s.scan for s in split.train],
        "val": [s.scan for s in split.val],
        "test": [s.scan for s in split.test],
    }
    with open(Path(cfg.training.output_dir) / "splits.json", "w") as f:
        json.dump(splits_manifest, f, ensure_ascii=False, indent=2)
    logger.info("Manifeste des splits sauvegardé (splits.json).")

    logger.info("Conversion en dataset HuggingFace...")
    train_ds = build_hf_dataset(split.train)
    val_ds = build_hf_dataset(split.val)

    logger.info("Configuration du SFTTrainer...")
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=cfg.training.output_dir,
            num_train_epochs=cfg.training.num_train_epochs,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            warmup_ratio=cfg.training.warmup_ratio,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            weight_decay=cfg.training.weight_decay,
            logging_steps=cfg.training.logging_steps,
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            save_strategy="steps",
            save_steps=cfg.training.save_steps,
            save_total_limit=cfg.training.save_total_limit,
            bf16=cfg.training.bf16,
            report_to=cfg.training.report_to,
            run_name=cfg.name,
            max_seq_length=cfg.training.max_seq_length,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},  # nécessaire pour images
            seed=cfg.data.seed,
        ),
    )

    logger.info("=== Début de l'entraînement ===")
    train_result = trainer.train()
    logger.info("Entraînement terminé. Loss finale : %.4f", train_result.training_loss)

    logger.info("Sauvegarde de l'adaptateur LoRA dans %s", cfg.training.output_dir)
    model.save_pretrained(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    # Évaluation finale sur le val set (le test set reste intact pour un bench séparé)
    logger.info("Évaluation finale sur le val set...")
    metrics = trainer.evaluate()
    logger.info("Métriques val : %s", metrics)


def main() -> None:
    """Point d'entrée CLI : charge la config et lance le fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune un VLM pour la transcription HTR sur Scoledit."
    )
    parser.add_argument("--config", required=True, help="Chemin du YAML de config.")
    args = parser.parse_args()

    cfg = load_finetune_config(args.config)
    logger.info("=== Fine-tuning HTR Scoledit : %s ===", cfg.name)
    run_finetuning(cfg)


if __name__ == "__main__":
    main()
