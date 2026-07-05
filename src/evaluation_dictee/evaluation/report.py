"""Analyse des résultats par item et par copie, avec intervalles de confiance.

Toutes les fonctions renvoient des DataFrames pandas, prêts à tracer ou à
exporter. Conventions :
- « erreur » = code != "1" (dans la grille simplifiée : 9 ou 0)
- détection d'erreur : classe positive = l'expert a codé une erreur

Vocabulaire des désaccords (du point de vue du modèle) :
- SUR-CORRECTION : l'expert voit une erreur, le modèle code correct.
  Le modèle a « corrigé » silencieusement la faute en lisant. Biais VLM connu.
- SUR-DÉTECTION : l'expert code correct, le modèle voit une erreur.
  Le modèle invente une faute (lecture trop sévère ou hallucination).
Les deux se compensent dans l'accord global mais ont des conséquences
opérationnelles opposées : la sur-correction sous-estime les difficultés des
élèves, la sur-détection les surestime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from evaluation_dictee.evaluation.statistics import wilson_interval


def load_predictions(predictions_path: str | Path) -> pd.DataFrame:
    """Charge les prédictions sauvegardées par le benchmark (JSON Lines).

    Args:
        predictions_path: chemin du fichier .jsonl produit par run_benchmark.

    Returns:
        DataFrame avec colonnes copy_id, item_id, y_true, y_pred, confidence.
    """
    records = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def per_item_metrics(df: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Métriques par item, avec intervalle de Wilson sur l'accord et la prévalence.

    Args:
        df: DataFrame des prédictions.
        level: niveau de confiance des intervalles.

    Returns:
        DataFrame indexé par item_id :
        - n, accord, accord_lo, accord_hi (intervalle de Wilson)
        - kappa (NaN si pas de variance)
        - pct_erreur_expert / pct_erreur_modele : prévalence d'erreur (%) selon chacun
        - pct_erreur_expert_lo/hi, pct_erreur_modele_lo/hi : IC Wilson à 95 %
        - rappel_erreur : parmi les erreurs de l'expert, % retrouvées par le modèle
        - precision_erreur : parmi les erreurs du modèle, % confirmées par l'expert
        - n_sur_correction / n_sur_detection : effectifs des deux désaccords
    """
    rows = []
    for item_id, grp in df.groupby("item_id"):
        y_true = grp["y_true"]
        y_pred = grp["y_pred"]
        n = len(grp)
        n_accord = int((y_true == y_pred).sum())
        ci = wilson_interval(n_accord, n, level)

        try:
            kappa = float(cohen_kappa_score(y_true, y_pred))
        except ValueError:
            kappa = float("nan")

        exp_err = y_true != "1"
        mod_err = y_pred != "1"
        n_exp_err = int(exp_err.sum())
        n_mod_err = int(mod_err.sum())
        vrais_pos = int((exp_err & mod_err).sum())
        sur_corr = int((exp_err & ~mod_err).sum())  # expert: erreur, modèle: correct
        sur_det = int((~exp_err & mod_err).sum())  # expert: correct, modèle: erreur

        # Intervalles de Wilson sur la prévalence d'erreur (proportion binomiale)
        ci_exp = wilson_interval(n_exp_err, n, level)
        ci_mod = wilson_interval(n_mod_err, n, level)

        rows.append(
            {
                "item_id": item_id,
                "n": n,
                "accord": ci.estimate,
                "accord_lo": ci.lower,
                "accord_hi": ci.upper,
                "kappa": kappa,
                "pct_erreur_expert": n_exp_err / n * 100,
                "pct_erreur_expert_lo": ci_exp.lower * 100,
                "pct_erreur_expert_hi": ci_exp.upper * 100,
                "pct_erreur_modele": n_mod_err / n * 100,
                "pct_erreur_modele_lo": ci_mod.lower * 100,
                "pct_erreur_modele_hi": ci_mod.upper * 100,
                "rappel_erreur": vrais_pos / n_exp_err if n_exp_err else float("nan"),
                "precision_erreur": vrais_pos / n_mod_err if n_mod_err else float("nan"),
                "n_sur_correction": sur_corr,
                "n_sur_detection": sur_det,
            }
        )
    return pd.DataFrame(rows).set_index("item_id")


def per_copy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Métriques par copie : repère les élèves/copies difficiles pour le modèle.

    Le test pilote a montré que les copies d'élèves faibles (plus de fautes,
    écriture moins normée) sont plus dures à coder. Cette vue le quantifie.

    Args:
        df: DataFrame des prédictions.

    Returns:
        DataFrame indexé par copy_id :
        - n_items, accord
        - n_erreurs_expert, n_erreurs_modele : nombre total d'items non corrects
          (codes 9 ET 0) selon l'expert et le modèle. Sert de proxy du niveau
          global de la copie.
        - n_fautes_expert, n_fautes_modele : nombre d'items code 9 (fautes
          d'orthographe stricto sensu).
        - n_manquants_expert, n_manquants_modele : nombre d'items code 0
          (mots oubliés par l'élève).
        - pct_erreur_expert / pct_erreur_modele : mêmes grandeurs en pourcentage.
        - confiance_moyenne : confiance moyenne du modèle sur la copie.
    """
    rows = []
    for copy_id, grp in df.groupby("copy_id"):
        n = len(grp)
        y_true, y_pred = grp["y_true"], grp["y_pred"]
        n_err_exp = int((y_true != "1").sum())
        n_err_mod = int((y_pred != "1").sum())
        n_fautes_exp = int((y_true == "9").sum())
        n_fautes_mod = int((y_pred == "9").sum())
        n_manq_exp = int((y_true == "0").sum())
        n_manq_mod = int((y_pred == "0").sum())
        rows.append(
            {
                "copy_id": copy_id,
                "n_items": n,
                "accord": float((y_true == y_pred).mean()),
                "n_erreurs_expert": n_err_exp,
                "n_erreurs_modele": n_err_mod,
                "n_fautes_expert": n_fautes_exp,
                "n_fautes_modele": n_fautes_mod,
                "n_manquants_expert": n_manq_exp,
                "n_manquants_modele": n_manq_mod,
                "pct_erreur_expert": n_err_exp / n * 100 if n else 0.0,
                "pct_erreur_modele": n_err_mod / n * 100 if n else 0.0,
                "confiance_moyenne": float(grp["confidence"].dropna().mean())
                if grp["confidence"].notna().any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("copy_id")


def disagreement_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Décompose chaque type de désaccord, globalement.

    Args:
        df: DataFrame des prédictions.

    Returns:
        DataFrame une ligne par type de transition expert→modèle parmi les
        désaccords, avec effectif et pourcentage du total des désaccords.
    """
    dis = df[df["y_true"] != df["y_pred"]]
    if len(dis) == 0:
        return pd.DataFrame(columns=["transition", "n", "pct_desaccords"])
    counts = (
        dis.groupby(["y_true", "y_pred"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    counts["transition"] = "expert:" + counts["y_true"] + " → modèle:" + counts["y_pred"]
    counts["pct_desaccords"] = counts["n"] / len(dis) * 100
    return counts[["transition", "n", "pct_desaccords"]].reset_index(drop=True)


def confusion_df(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    """Matrice de confusion globale expert × modèle.

    Args:
        df: DataFrame des prédictions.
        normalize: si True, normalise chaque ligne (somme = 1).

    Returns:
        DataFrame lignes = expert, colonnes = modèle.
    """
    labels = sorted(set(df["y_true"]) | set(df["y_pred"]))
    matrix = confusion_matrix(df["y_true"], df["y_pred"], labels=labels)
    out = pd.DataFrame(
        matrix,
        index=[f"expert:{c}" for c in labels],
        columns=[f"modèle:{c}" for c in labels],
    )
    if normalize:
        out = out.div(out.sum(axis=1).replace(0, 1), axis=0)
    return out


def copies_by_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """Table des copies triées par taux de désaccord brut décroissant.

    Outil de diagnostic : les copies en tête sont les plus problématiques et les
    premières à inspecter visuellement.

    Args:
        df: DataFrame des prédictions (copy_id, item_id, y_true, y_pred).

    Returns:
        DataFrame indexé par copy_id, trié par pct_desaccord décroissant :
        - n_items : nombre d'items de la copie
        - n_desaccords : nombre d'items en désaccord
        - pct_desaccord : taux de désaccord brut (0–100)
        - accord : taux d'accord (= 100 - pct_desaccord, en proportion)
    """
    rows = []
    for copy_id, grp in df.groupby("copy_id"):
        n = len(grp)
        n_dis = int((grp["y_true"] != grp["y_pred"]).sum())
        rows.append(
            {
                "copy_id": copy_id,
                "n_items": n,
                "n_desaccords": n_dis,
                "pct_desaccord": n_dis / n * 100 if n else 0.0,
                "accord": (n - n_dis) / n if n else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("copy_id").sort_values("pct_desaccord", ascending=False)
