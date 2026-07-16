"""Tests du module html_report (sélection de sections + bootstrap prévalence)."""

from pathlib import Path

import nbformat
import pandas as pd

from evaluation_dictee.evaluation.html_report import (
    _filter_by_sections,
    bootstrap_prevalence_ci,
    list_sections,
)


def _make_notebook(tmp_path: Path) -> Path:
    """Crée un mini-notebook avec 3 sections balisées + une cellule d'en-tête."""
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("import pandas as pd\ndf = pd.DataFrame()"),
        nbformat.v4.new_markdown_cell("## Section A", metadata={"tags": ["section:a"]}),
        nbformat.v4.new_code_cell("print('A')"),
        nbformat.v4.new_markdown_cell("## Section B", metadata={"tags": ["section:b"]}),
        nbformat.v4.new_code_cell("print('B')"),
        nbformat.v4.new_markdown_cell("## Section C", metadata={"tags": ["section:c"]}),
        nbformat.v4.new_code_cell("print('C')"),
    ]
    path = tmp_path / "nb.ipynb"
    nbformat.write(nb, str(path))
    return path


def test_list_sections(tmp_path: Path) -> None:
    path = _make_notebook(tmp_path)
    sections = list_sections(path)
    assert [s.tag for s in sections] == ["section:a", "section:b", "section:c"]
    assert sections[0].title == "Section A"


def test_filter_keeps_header_cells(tmp_path: Path) -> None:
    """Les cellules avant la 1re section (imports) sont toujours conservées."""
    path = _make_notebook(tmp_path)
    nb = nbformat.read(str(path), as_version=4)
    filtered = _filter_by_sections(nb, ["section:b"])
    # 1 cellule d'en-tête + 2 cellules pour section B = 3
    assert len(filtered.cells) == 3
    # La 1re cellule (imports) est toujours là
    assert "import pandas" in "".join(filtered.cells[0].source)


def test_filter_by_two_sections(tmp_path: Path) -> None:
    path = _make_notebook(tmp_path)
    nb = nbformat.read(str(path), as_version=4)
    filtered = _filter_by_sections(nb, ["section:a", "section:c"])
    # header + 2 cellules A + 2 cellules C
    assert len(filtered.cells) == 5


def test_filter_empty_selection_keeps_only_header(tmp_path: Path) -> None:
    path = _make_notebook(tmp_path)
    nb = nbformat.read(str(path), as_version=4)
    filtered = _filter_by_sections(nb, [])
    # Seul le header est conservé (aucune section sélectionnée)
    assert len(filtered.cells) == 1


def test_bootstrap_prevalence_ci_shape() -> None:
    """Le bootstrap renvoie une ligne par item, avec estimate/lo/hi cohérents."""
    df = pd.DataFrame(
        [
            {
                "copy_id": c,
                "item_id": f"i{k}",
                "y_true": "9" if k == 0 else "1",
                "y_pred": "9" if k == 0 and c < 5 else "1",
            }
            for c in range(10)
            for k in range(3)
        ]
    )
    ci = bootstrap_prevalence_ci(df, label_col="y_true", n_boot=100, seed=42)
    assert len(ci) == 3
    assert set(ci.columns) == {"estimate", "lo", "hi"}
    # L'estimation de la prévalence sur i0 est 100 % (tous à "9")
    assert ci.loc["i0", "estimate"] == 100.0
    # IC toujours [lo, hi] avec lo <= estimate <= hi
    for it in ci.index:
        assert ci.loc[it, "lo"] <= ci.loc[it, "estimate"] <= ci.loc[it, "hi"]
