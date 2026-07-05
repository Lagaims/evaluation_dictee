"""Génère un rapport HTML autonome à partir du notebook d'analyse.

Le notebook 03 est organisé en sections repérées par un tag de cellule
(ex. `section:synthese`, `section:prevalence_item`, ...). Ce module l'exécute
puis en exporte un HTML dans lequel on ne garde que les sections choisies.

Cas d'usage : partager un rapport ciblé avec l'équipe DEPP (dictée) au SSMEN,
sans exposer les cellules de code brutes ni les sections de diagnostic interne.

Deux entrées principales :
- `list_sections(notebook_path)` : liste des sections disponibles avec leur tag.
- `build_html_report(notebook_path, selected_tags, output_path, ...)` : exécute
  le notebook et exporte les sections sélectionnées en un HTML autonome.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor
from traitlets.config import Config

# Prefix conventionnel pour les tags de section.
_SECTION_PREFIX = "section:"


@dataclass
class SectionInfo:
    """Description d'une section repérable du notebook.

    Attributes:
        tag: identifiant technique (ex. "section:prevalence_item").
        title: titre humain lu dans la première cellule markdown de la section.
    """

    tag: str
    title: str


def _find_first_title(cell: nbformat.NotebookNode) -> str:
    """Extrait le premier titre markdown non vide de la cellule."""
    src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
    for line in src.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return src.strip().split("\n", 1)[0][:80]


def list_sections(notebook_path: str | Path) -> list[SectionInfo]:
    """Liste les sections d'un notebook, dans l'ordre d'apparition.

    Une section est un ensemble de cellules dont la PREMIÈRE porte un tag
    `section:<nom>`. Le titre humain est extrait du markdown de cette première
    cellule pour l'affichage.

    Args:
        notebook_path: chemin du .ipynb.

    Returns:
        Sections détectées, avec leur tag et leur titre humain.
    """
    nb = nbformat.read(str(notebook_path), as_version=4)
    sections: list[SectionInfo] = []
    seen: set[str] = set()
    for cell in nb.cells:
        for tag in cell.metadata.get("tags", []):
            if tag.startswith(_SECTION_PREFIX) and tag not in seen:
                sections.append(SectionInfo(tag=tag, title=_find_first_title(cell)))
                seen.add(tag)
    return sections


def _filter_by_sections(
    nb: nbformat.NotebookNode, selected_tags: list[str]
) -> nbformat.NotebookNode:
    """Ne garde dans le notebook que les cellules des sections sélectionnées.

    La règle : chaque cellule appartient à la DERNIÈRE section rencontrée avant
    elle (elle prolonge la section jusqu'à la prochaine cellule taguée
    `section:...`). Les cellules AVANT la première section sont conservées
    (imports, paramètres) car indispensables à l'affichage.

    Args:
        nb: notebook chargé.
        selected_tags: liste de tags de section à conserver.

    Returns:
        Une copie du notebook filtrée.
    """
    filtered = copy.deepcopy(nb)
    kept_cells: list[nbformat.NotebookNode] = []
    current_section: str | None = None
    seen_first_section = False

    for cell in filtered.cells:
        tags = cell.metadata.get("tags", [])
        section_tags = [t for t in tags if t.startswith(_SECTION_PREFIX)]
        if section_tags:
            current_section = section_tags[0]
            seen_first_section = True

        # Cellules d'en-tête (avant la 1ère section) : toujours conservées.
        # Sinon : conservées seulement si la section courante est sélectionnée.
        if not seen_first_section or (current_section in selected_tags):
            kept_cells.append(cell)

    filtered.cells = kept_cells
    return filtered


def _run_notebook(nb: nbformat.NotebookNode, notebook_dir: Path) -> nbformat.NotebookNode:
    """Exécute le notebook (nécessaire pour que les sorties apparaissent en HTML)."""
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(notebook_dir)}})
    return nb


def _export_html(
    nb: nbformat.NotebookNode,
    hide_code: bool = True,
    template_name: str = "lab",
) -> str:
    """Convertit le notebook exécuté en HTML autonome (assets inlinés)."""
    c = Config()
    # Retirer les cellules explicitement taguées "hide" (utile pour cellules
    # techniques comme le choix de sections lui-même).
    c.TagRemovePreprocessor.remove_cell_tags = {"hide", "remove_cell"}
    if hide_code:
        c.TagRemovePreprocessor.remove_input_tags = {"hide_input"}
    c.TagRemovePreprocessor.enabled = True

    exporter = HTMLExporter(config=c, template_name=template_name)
    exporter.register_preprocessor(TagRemovePreprocessor(config=c), True)
    if hide_code:
        exporter.exclude_input = True  # masque les cellules de code
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True

    body, _ = exporter.from_notebook_node(nb)
    return body


def build_html_report(
    notebook_path: str | Path,
    selected_tags: list[str],
    output_path: str | Path,
    hide_code: bool = True,
    execute: bool = True,
) -> Path:
    """Exécute le notebook et exporte les sections choisies en HTML.

    Args:
        notebook_path: chemin du notebook source.
        selected_tags: tags des sections à inclure (ex. ["section:synthese",
            "section:prevalence_item"]).
        output_path: chemin du fichier HTML à produire.
        hide_code: si True, masque les cellules de code dans le rapport final
            (recommandé pour envoi à des non-développeurs).
        execute: si True, exécute le notebook avant export (nécessaire si les
            sorties ne sont pas déjà présentes dans le fichier source).

    Returns:
        Le chemin du HTML produit.
    """
    notebook_path = Path(notebook_path)
    nb = nbformat.read(str(notebook_path), as_version=4)

    if execute:
        nb = _run_notebook(nb, notebook_path.parent)

    filtered = _filter_by_sections(nb, selected_tags)
    html = _export_html(filtered, hide_code=hide_code)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def bootstrap_prevalence_ci(
    df,
    item_col: str = "item_id",
    label_col: str = "y_true",
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 42,
):
    """Bootstrap groupé par copie de la prévalence d'erreur par item.

    Ré-échantillonne les COPIES (pas les items individuels) pour respecter la
    structure hiérarchique de la donnée. Renvoie pour chaque item l'IC à `level`
    de la prévalence d'erreur (proportion d'items non codés « 1 »).

    Args:
        df: DataFrame long avec colonnes copy_id, item_id, y_true et/ou y_pred.
        item_col: nom de la colonne d'items.
        label_col: colonne dont on calcule la prévalence (y_true = expert,
            y_pred = modèle).
        n_boot: nombre de ré-échantillonnages.
        level: niveau de confiance.
        seed: reproductibilité.

    Returns:
        DataFrame indexé par item_id, avec colonnes `estimate`, `lo`, `hi`
        (pourcentages).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    copies = df["copy_id"].unique()
    items = sorted(df[item_col].unique())

    # Estimation ponctuelle
    est = df.groupby(item_col)[label_col].apply(lambda s: (s != "1").mean() * 100)

    # Bootstrap
    samples = np.zeros((n_boot, len(items)))
    for b in range(n_boot):
        sampled_copies = rng.choice(copies, size=len(copies), replace=True)
        sub = pd.concat([df[df["copy_id"] == c] for c in sampled_copies], ignore_index=True)
        grouped = sub.groupby(item_col)[label_col].apply(lambda s: (s != "1").mean() * 100)
        for j, it in enumerate(items):
            samples[b, j] = grouped.get(it, np.nan)

    alpha = (1 - level) / 2
    lo = np.nanpercentile(samples, alpha * 100, axis=0)
    hi = np.nanpercentile(samples, (1 - alpha) * 100, axis=0)

    return pd.DataFrame(
        {"estimate": [est.get(i, np.nan) for i in items], "lo": lo, "hi": hi},
        index=pd.Index(items, name=item_col),
    )
