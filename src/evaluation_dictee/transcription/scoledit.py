"""Chargement du corpus Scoledit pour l'évaluation de la transcription (HTR).

Structure des données (sur S3) :
  scans/<niveau>/<scan>.jpg          image de la copie (ex. 100a.jpg)
  annotation/<niveau>/<scan>.json    transcription de référence, ex. :
      {"student_id": 100, "level": "CE1", "scan": "100a",
       "tei": "<p>il était tune fois un chat <lb/> pérsent...</p>"}

La transcription de référence est en TEI léger : balise <p> englobante et <lb/>
pour les retours à la ligne. On la nettoie en texte brut (les <lb/> deviennent des
espaces) tout en PRÉSERVANT les fautes d'orthographe de l'élève, qui sont l'objet
même de l'évaluation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import fsspec


@dataclass
class ScoledtSample:
    """Un échantillon Scoledit : image + transcription de référence.

    Attributes:
        scan: identifiant du scan (ex. "100a").
        level: niveau scolaire (ex. "CE1").
        student_id: identifiant de l'élève.
        image_path: chemin de l'image (local ou s3://).
        reference: transcription de référence en texte brut (fautes préservées).
        tei_raw: transcription TEI brute (pour référence/débogage).
    """

    scan: str
    level: str
    student_id: int
    image_path: str
    reference: str
    tei_raw: str


def tei_to_text(tei: str) -> str:
    """Convertit une transcription TEI légère Scoledit en texte brut.

    Règles :
    - <lb/> (saut de ligne) → espace ;
    - balises englobantes <p>...</p> retirées ;
    - toute autre balise retirée ;
    - espaces multiples réduits, texte découpé/recollé proprement.
    Les fautes d'orthographe de l'élève sont conservées telles quelles.

    Args:
        tei: chaîne TEI (ex. "<p>il était tune fois <lb/> un chat</p>").

    Returns:
        Le texte brut correspondant.
    """
    txt = tei.replace("<lb/>", " ")
    txt = re.sub(r"<[^>]+>", " ", txt)  # retire toute balise restante
    txt = re.sub(r"\s+", " ", txt)  # espaces multiples → un seul
    return txt.strip()


def _join(base: str, name: str) -> str:
    return base.rstrip("/") + "/" + name


def load_scoledit_dataset(
    scans_dir: str,
    annotations_dir: str,
    limit: int | None = None,
) -> list[ScoledtSample]:
    """Charge les échantillons Scoledit en appariant scans et annotations.

    Args:
        scans_dir: dossier des images (local ou s3://.../scans/CE1/).
        annotations_dir: dossier des JSON (local ou s3://.../annotation/CE1/).
        limit: nombre maximal d'échantillons (None = tous).

    Returns:
        Liste de ScoledtSample, ordonnée par nom de scan.
    """
    fs, _, paths = fsspec.get_fs_token_paths(annotations_dir.rstrip("/") + "/*")
    json_files = sorted(p for p in fs.glob(annotations_dir.rstrip("/") + "/*.json"))

    samples: list[ScoledtSample] = []
    for jpath in json_files:
        with fsspec.open(_scheme_prefix(annotations_dir, jpath), "rt", encoding="utf-8") as f:
            meta = json.load(f)
        scan = meta.get("scan", "")
        image_path = _join(scans_dir, f"{scan}.jpg")
        samples.append(
            ScoledtSample(
                scan=scan,
                level=meta.get("level", ""),
                student_id=int(meta.get("student_id", -1)),
                image_path=image_path,
                reference=tei_to_text(meta.get("tei", "")),
                tei_raw=meta.get("tei", ""),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _scheme_prefix(reference_dir: str, path: str) -> str:
    """Restaure le préfixe s3:// perdu par fs.glob quand la source est S3."""
    if reference_dir.startswith("s3://") and not path.startswith("s3://"):
        return "s3://" + path
    return path
