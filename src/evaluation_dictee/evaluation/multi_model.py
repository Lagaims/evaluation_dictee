"""Comparaison multi-modèles pour construire un score de confiance par copie.

Principe : quand plusieurs modèles indépendants s'accordent sur un item, la
prédiction est fiable ; quand ils divergent, c'est un signal d'incertitude qui
appelle un renvoi humain. Le désaccord inter-modèles est un bien meilleur
prédicteur de confiance que le score annoncé par un modèle unique (qui est en
pratique quasi-constant à 1.0 dans nos benchmarks — inexploitable).

Workflow type :
    1. Lancer N runs indépendants (mêmes copies, modèles/prompts différents),
       chacun produit son `<run>_predictions.jsonl`.
    2. `load_multi_runs([run1, run2, ...])` charge les N runs et les joint sur
       (copy_id, item_id) pour construire une prédiction par modèle par ligne.
    3. `agreement_per_item()` calcule le nb de modèles d'accord sur chaque item.
    4. `confidence_score()` agrège au niveau copie : proportion d'items où tous
       les modèles s'accordent = score de confiance de la copie.
    5. `referral_curve_multi()` compare, pour chaque seuil de renvoi, l'accord
       vs un expert humain (si dispo) sur les copies retenues.

Structure agnostique du nombre de modèles : marche de N=1 (score = confiance
d'un modèle unique) à N quelconque.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from evaluation_dictee.evaluation.report import load_predictions


def load_multi_runs(
    run_names: list[str],
    output_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Charge N runs indépendants et les joint sur (copy_id, item_id).

    Chaque run doit avoir été produit par `run_benchmark.py` et posséder son
    fichier `<run>_predictions.jsonl` dans `output_dir`.

    Args:
        run_names: identifiants des runs à comparer (ex. ["dictee_gemma4_zeroshot",
            "dictee_gemma4_cot"]). Le premier de la liste sert de référence pour
            les colonnes partagées (y_true, transcription_ref, etc.).
        output_dir: dossier contenant les JSONL de prédictions.

    Returns:
        DataFrame avec les colonnes :
        - copy_id, item_id  : clé de jointure
        - y_true            : code expert (identique dans tous les runs)
        - y_pred__<run1>, y_pred__<run2>, ... : code prédit par chaque modèle
        - conf__<run1>, conf__<run2>, ...    : confiance de chaque modèle

    Raises:
        FileNotFoundError: si un run est absent.
        ValueError: si les runs ne portent pas sur les mêmes (copy_id, item_id).
    """
    output_dir = Path(output_dir)
    if not run_names:
        raise ValueError("Au moins un run est requis.")

    dfs = {}
    for run in run_names:
        path = output_dir / f"{run}_predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Prédictions manquantes : {path}")
        df_run = load_predictions(path)
        # Alias des colonnes propres au modèle
        renamed = df_run.rename(columns={"y_pred": f"y_pred__{run}", "confidence": f"conf__{run}"})
        dfs[run] = renamed

    # Le premier run fournit les colonnes communes (y_true, item_id, copy_id)
    ref = dfs[run_names[0]][["copy_id", "item_id", "y_true"]].copy()

    # Joint successivement les prédictions de chaque run
    merged = ref
    for run in run_names:
        cols = ["copy_id", "item_id", f"y_pred__{run}", f"conf__{run}"]
        merged = merged.merge(
            dfs[run][cols],
            on=["copy_id", "item_id"],
            how="inner",  # on ne garde que les (copy_id, item_id) présents PARTOUT
        )

    if merged.empty:
        raise ValueError(
            "Aucun (copy_id, item_id) commun aux runs. Vérifier qu'ils portent "
            "sur les mêmes données."
        )
    return merged


def agreement_per_item(df_multi: pd.DataFrame) -> pd.DataFrame:
    """Ajoute au DataFrame multi-runs les colonnes de désaccord par item.

    Args:
        df_multi: sortie de `load_multi_runs`.

    Returns:
        Le DataFrame enrichi de :
        - n_modeles          : nombre total de modèles
        - modal_pred         : prédiction majoritaire parmi les modèles
        - n_accord_modeles   : nombre de modèles alignés sur la modale
        - unanimite          : True si tous les modèles s'accordent
        - modele_vs_expert__<run> : True si ce modèle est d'accord avec l'expert
    """
    pred_cols = [c for c in df_multi.columns if c.startswith("y_pred__")]
    n_modeles = len(pred_cols)

    out = df_multi.copy()

    # Prédiction majoritaire et effectif d'accord
    def _modal_and_count(row):
        counts = Counter(row[c] for c in pred_cols)
        modal, n_acc = counts.most_common(1)[0]
        return pd.Series({"modal_pred": modal, "n_accord_modeles": n_acc})

    modal_df = out[pred_cols].apply(_modal_and_count, axis=1)
    out["modal_pred"] = modal_df["modal_pred"]
    out["n_accord_modeles"] = modal_df["n_accord_modeles"].astype(int)
    out["n_modeles"] = n_modeles
    out["unanimite"] = out["n_accord_modeles"] == n_modeles

    # Accord de chaque modèle avec l'expert
    for c in pred_cols:
        run = c.removeprefix("y_pred__")
        out[f"modele_vs_expert__{run}"] = out[c] == out["y_true"]

    return out


def confidence_score(df_agree: pd.DataFrame) -> pd.DataFrame:
    """Agrège au niveau copie un score de confiance basé sur le désaccord.

    Args:
        df_agree: sortie de `agreement_per_item()`.

    Returns:
        DataFrame indexé par copy_id :
        - n_items          : nb d'items de la copie
        - pct_unanime      : % d'items où tous les modèles s'accordent
        - accord_moyen_mod : nb moyen de modèles alignés sur la modale (max = N)
        - n_desaccord_max  : nb d'items où l'accord inter-modèles est minimum
        - score_confiance  : pct_unanime, mais renommé pour la lisibilité (0-100)
    """
    n_modeles = int(df_agree["n_modeles"].iloc[0])
    rows = []
    for copy_id, grp in df_agree.groupby("copy_id"):
        n = len(grp)
        n_unan = int(grp["unanimite"].sum())
        accord_moy = float(grp["n_accord_modeles"].mean())
        # Combien d'items ont l'accord MINIMAL possible (n_accord = ceil(N/2), ie
        # dispersion maximale possible sur un item catégoriel) ?
        seuil_min = (n_modeles // 2) + 1
        n_dis_max = int((grp["n_accord_modeles"] < seuil_min).sum()) if n_modeles > 1 else 0
        rows.append(
            {
                "copy_id": copy_id,
                "n_items": n,
                "pct_unanime": n_unan / n * 100 if n else 0.0,
                "accord_moyen_mod": accord_moy,
                "n_desaccord_max": n_dis_max,
                "score_confiance": n_unan / n * 100 if n else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("copy_id").sort_values("score_confiance", ascending=False)


def referral_curve_multi(
    df_agree: pd.DataFrame,
    conf: pd.DataFrame,
    reference_run: str,
) -> pd.DataFrame:
    """Courbe de renvoi humain : pour chaque seuil de confiance, quel accord ?

    Simule une stratégie : « renvoyer les copies dont le score de confiance est
    inférieur à τ ». Pour chaque τ, calcule le % de copies renvoyées et l'accord
    résiduel entre le modèle de RÉFÉRENCE et l'expert sur les copies retenues.

    Args:
        df_agree: DataFrame item-niveau enrichi par `agreement_per_item()`.
        conf: scores de confiance par copie (sortie de `confidence_score()`).
        reference_run: run dont on mesure l'accord modèle-expert sur les copies
            retenues (typiquement le modèle qu'on envisage de mettre en production).

    Returns:
        DataFrame par seuil τ :
        - seuil_confiance    : τ (%), copies renvoyées si score_confiance < τ
        - pct_copies_renvoyees
        - pct_accord_retenues : accord modèle-expert sur les copies retenues (%)
        - n_copies_retenues
    """
    pred_col = f"y_pred__{reference_run}"
    if pred_col not in df_agree.columns:
        raise ValueError(
            f"Run de référence {reference_run!r} absent de df_agree. "
            f"Colonnes disponibles : {[c for c in df_agree.columns if c.startswith('y_pred__')]}"
        )

    # Table par copie : score de confiance + accord modèle-expert de la copie
    copies = conf.copy()
    accord_par_copie = df_agree.groupby("copy_id").apply(
        lambda g: (g[pred_col] == g["y_true"]).mean() * 100
    )
    copies["accord_modele_expert"] = accord_par_copie

    n_total = len(copies)
    rows = []
    for tau in range(0, 101, 5):
        retenues = copies[copies["score_confiance"] >= tau]
        renvoyees = copies[copies["score_confiance"] < tau]
        n_ret = len(retenues)
        pct_renv = len(renvoyees) / n_total * 100 if n_total else 0.0
        acc_ret = float(retenues["accord_modele_expert"].mean()) if n_ret else float("nan")
        rows.append(
            {
                "seuil_confiance": tau,
                "pct_copies_renvoyees": pct_renv,
                "pct_accord_retenues": acc_ret,
                "n_copies_retenues": n_ret,
            }
        )
    return pd.DataFrame(rows)


def disagreement_type_summary(df_agree: pd.DataFrame) -> pd.DataFrame:
    """Répartition des items selon le niveau d'accord inter-modèles.

    Args:
        df_agree: DataFrame item-niveau enrichi par `agreement_per_item()`.

    Returns:
        DataFrame résumé :
        - n_accord_modeles (1..N)
        - n_items
        - pct_items
        - accord_avec_expert : parmi ces items, % où la modale = expert
    """
    n_modeles = int(df_agree["n_modeles"].iloc[0])
    n_total = len(df_agree)
    rows = []
    for n_acc in range(1, n_modeles + 1):
        sub = df_agree[df_agree["n_accord_modeles"] == n_acc]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "n_accord_modeles": n_acc,
                "n_items": len(sub),
                "pct_items": len(sub) / n_total * 100,
                "accord_avec_expert": float((sub["modal_pred"] == sub["y_true"]).mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("n_accord_modeles")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de Wilson (pourcentages). Utilisé pour l'accord retenu."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return (max(0.0, centre - margin) * 100, min(1.0, centre + margin) * 100)


def referral_curve_with_ci(
    df_agree: pd.DataFrame,
    conf: pd.DataFrame,
    reference_run: str,
) -> pd.DataFrame:
    """Comme `referral_curve_multi`, avec IC Wilson sur l'accord retenu."""
    pred_col = f"y_pred__{reference_run}"
    if pred_col not in df_agree.columns:
        raise ValueError(f"Run {reference_run!r} absent.")

    # Table par copie : score + booléen accord sur chaque item
    copies_conf = conf["score_confiance"].to_dict()
    df = df_agree.copy()
    df["_correct"] = (df[pred_col] == df["y_true"]).astype(int)
    df["_score"] = df["copy_id"].map(copies_conf)

    n_copies_total = df["copy_id"].nunique()
    rows = []
    for tau in range(0, 101, 5):
        retenues_items = df[df["_score"] >= tau]
        n_items_ret = len(retenues_items)
        n_copies_ret = retenues_items["copy_id"].nunique()
        k_ok = int(retenues_items["_correct"].sum())
        lo, hi = _wilson_ci(k_ok, n_items_ret)
        rows.append(
            {
                "seuil_confiance": tau,
                "pct_copies_renvoyees": (1 - n_copies_ret / n_copies_total) * 100
                if n_copies_total
                else 0.0,
                "pct_accord_retenues": (k_ok / n_items_ret * 100) if n_items_ret else float("nan"),
                "accord_lo": lo,
                "accord_hi": hi,
                "n_copies_retenues": n_copies_ret,
                "n_items_retenus": n_items_ret,
            }
        )
    return pd.DataFrame(rows)
