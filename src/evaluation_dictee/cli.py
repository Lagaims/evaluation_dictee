"""Interface en ligne de commande du projet.

Exposée via la commande `eval-ecrit` (voir pyproject.toml [project.scripts]).
Exemple : `eval-ecrit benchmark configs/dictee_qwen7b_zeroshot.yaml`
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from evaluation_dictee.config import Secrets, load_config
from evaluation_dictee.data.reference import load_grid
from evaluation_dictee.models.factory import build_scorer
from evaluation_dictee.pipeline.benchmark import run_benchmark
from evaluation_dictee.utils.tracking import experiment_run, log_metrics

app = typer.Typer(help="Évaluation automatique de la production d'écrit (DEPP × SSP Lab).")
console = Console()


@app.command()
def benchmark(config_path: str) -> None:
    """Lance un benchmark à partir d'un fichier de configuration YAML.

    Args:
        config_path: chemin vers la config de l'expérience.
    """
    config = load_config(config_path)
    secrets = Secrets()

    console.print(f"[bold]Run :[/bold] {config.name}")
    console.print(f"Modèle : {config.model.name} | Méthode : {config.prompt.method}")

    scorer = build_scorer(
        config=config,
        grid_items=load_grid(config.data.grid_path).items,
        base_url=secrets.llm_base_url,
        api_key=secrets.llm_api_key,
    )

    with experiment_run(config):
        result = run_benchmark(config, scorer)
        log_metrics(
            {
                "raw_agreement": result.metrics.raw_agreement,
                "cohen_kappa": result.metrics.cohen_kappa,
                "n_items": result.metrics.n_items,
            }
        )

    _print_metrics(result.metrics)


def _print_metrics(metrics) -> None:  # type: ignore[no-untyped-def]
    """Affiche les métriques dans un tableau lisible."""
    table = Table(title="Résultats du benchmark")
    table.add_column("Métrique")
    table.add_column("Valeur", justify="right")
    table.add_row("Items comparés", str(metrics.n_items))
    table.add_row("Accord brut", f"{metrics.raw_agreement:.1%}")
    table.add_row("Kappa de Cohen", f"{metrics.cohen_kappa:.3f}")
    console.print(table)


if __name__ == "__main__":
    app()
