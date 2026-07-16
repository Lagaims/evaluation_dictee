"""Mini-interface HTML pour inspecter visuellement les transcriptions HTR.

Pour chaque copie Scoledit sélectionnée (les N pires, ou N aléatoires), affiche
côte à côte : l'image de la copie, la transcription de référence humaine et la
transcription produite par le modèle, avec les mots divergents surlignés.

Fichier HTML autonome (image en base64), à ouvrir dans le navigateur. Idéal pour
le diagnostic rapide des erreurs de lecture.

ATTENTION DONNÉES SENSIBLES : le HTML contient des images d'élèves mineurs. À ne
jamais sortir de l'environnement sécurisé, à ne jamais committer (dossier data/
ignoré par Git).
"""

from __future__ import annotations

import base64
import difflib
import html
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from evaluation_dictee.data.loaders import load_image


def _img_base64(path: str, max_width: int = 1100) -> str:
    """Charge l'image, la redimensionne si besoin, renvoie une data URL PNG."""
    img: Image.Image = load_image(path)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _surligne_diff(ref_mots: list[str], hyp_mots: list[str]) -> tuple[str, str]:
    """Renvoie les deux séquences HTML avec surlignage des divergences."""
    sm = difflib.SequenceMatcher(a=ref_mots, b=hyp_mots)
    diff_ref, diff_hyp = set(), set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            diff_ref.update(range(i1, i2))
            diff_hyp.update(range(j1, j2))

    def _render(mots: list[str], surlignes: set[int], couleur: str) -> str:
        out = []
        for i, mot in enumerate(mots):
            bg = couleur if i in surlignes else "transparent"
            out.append(
                f"<span style='background:{bg};padding:1px 4px;"
                f"border-radius:3px;margin-right:2px'>{html.escape(mot)}</span>"
            )
        return "".join(out)

    return _render(ref_mots, diff_ref, "#d7f0d7"), _render(hyp_mots, diff_hyp, "#f7d4d4")


def _build_sample_html(scan: str, image_path: str, row: pd.Series) -> str:
    """Construit le HTML d'un échantillon Scoledit."""
    img_data = _img_base64(image_path)
    ref_mots = row["reference"].split()
    hyp_mots = row["hypothese"].split() if row["hypothese"] else []
    ref_html, hyp_html = _surligne_diff(ref_mots, hyp_mots)

    if not hyp_mots:
        hyp_html = "<em style='color:#c62828'>(transcription vide — échec du modèle)</em>"

    return f"""
    <section class="sample">
      <h2>{html.escape(scan)}
        <span class="badge">CER {row["cer"]:.1%} · WER {row["wer"]:.1%} ·
              {row["n_mots_ref"]} mots réf.</span>
      </h2>
      <div class="bloc">
        <h3>1 · Copie numérisée</h3>
        <img src="{img_data}" alt="scan {html.escape(scan)}"/>
      </div>
      <div class="bloc">
        <h3>2 · Transcription de référence <small>(humaine, fautes préservées)</small></h3>
        <p class="txt">{ref_html}</p>
      </div>
      <div class="bloc">
        <h3>3 · Transcription du modèle <small>(vert = accord · rouge = divergence)</small></h3>
        <p class="txt">{hyp_html}</p>
      </div>
    </section>
    """


_PAGE_CSS = """
  body { font-family: system-ui, sans-serif; margin: 24px; color: #222; background:#fafafa; }
  h1 { color:#1f3864; }
  section.sample { background:#fff; border:1px solid #ddd; border-radius:8px;
                   padding:18px; margin-bottom:32px; }
  h2 { color:#2f5496; border-bottom:2px solid #eee; padding-bottom:6px; }
  .badge { font-size:0.6em; background:#eef; color:#2f5496; padding:3px 8px;
           border-radius:10px; margin-left:10px; font-weight:normal; }
  .bloc { margin:14px 0; }
  h3 { font-size:1em; color:#444; margin-bottom:6px; }
  h3 small { font-weight:normal; color:#888; }
  img { max-width:100%; border:1px solid #ccc; border-radius:4px; }
  .txt { background:#f9f9f9; padding:10px; border-radius:4px;
         font-family:Georgia,serif; line-height:1.8; font-size:15px; }
  .legende { color:#555; font-size:0.9em; }
"""


def _wrap_page(fragments: list[str], title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"/>
<title>{html.escape(title)}</title><style>{_PAGE_CSS}</style></head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="legende">Vert = mot en accord référence/modèle · rouge = mot divergent.
  Transcription vide = échec du modèle (aucune sortie exploitable).</p>
  <p><strong>Données sensibles</strong> : ce fichier contient des copies d'élèves.
  Ne pas le sortir de l'environnement sécurisé.</p>
  {"".join(fragments)}
</body></html>"""


def report_worst_transcriptions(
    predictions_df: pd.DataFrame,
    scans_dir: str,
    output_path: str | Path,
    n: int = 10,
) -> Path:
    """Génère le HTML des N transcriptions au plus fort CER.

    Args:
        predictions_df: prédictions HTR (scan, reference, hypothese, cer, wer, ...).
        scans_dir: dossier des images (local ou s3://).
        output_path: chemin du fichier HTML à écrire.
        n: nombre d'échantillons (les pires) à inclure.

    Returns:
        Le chemin du fichier HTML produit.
    """
    pires = predictions_df.sort_values("cer", ascending=False).head(n)
    fragments = [
        _build_sample_html(row["scan"], f"{scans_dir.rstrip('/')}/{row['scan']}.jpg", row)
        for _, row in pires.iterrows()
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _wrap_page(fragments, f"HTR — {n} transcriptions au plus fort CER"),
        encoding="utf-8",
    )
    return out


def report_random_transcriptions(
    predictions_df: pd.DataFrame,
    scans_dir: str,
    output_path: str | Path,
    n: int = 10,
    seed: int = 42,
) -> Path:
    """Génère le HTML de N transcriptions tirées au hasard.

    Utile pour éviter de ne regarder que les cas pathologiques : les tirages
    aléatoires donnent une vue plus représentative de la performance moyenne.

    Args:
        predictions_df: prédictions HTR.
        scans_dir: dossier des images.
        output_path: chemin du fichier HTML.
        n: nombre d'échantillons aléatoires.
        seed: graine aléatoire pour la reproductibilité.

    Returns:
        Le chemin du fichier HTML produit.
    """
    n_effectif = min(n, len(predictions_df))
    echantillon = predictions_df.sample(n=n_effectif, random_state=seed)
    fragments = [
        _build_sample_html(row["scan"], f"{scans_dir.rstrip('/')}/{row['scan']}.jpg", row)
        for _, row in echantillon.iterrows()
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _wrap_page(fragments, f"HTR — {n_effectif} transcriptions aléatoires"),
        encoding="utf-8",
    )
    return out
