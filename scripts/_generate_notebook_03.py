"""Regénère notebooks/03_analyse_resultats.ipynb — version consolidée.

Intègre les évolutions récentes :
- IC analytiques (Wilson + delta method pour le kappa)
- Prévalence par item corrigée du design effect de Kish
- Distributions à 3 panneaux (total / fautes / absents) + courbe de survie
- Métriques accord/kappa/rappel/précision par niveau de difficulté
- Comparaison multi-modèles pour construire un score de confiance
- Stratégie de renvoi basée sur le désaccord multi-modèles (avec IC)
- Export HTML sélectif pour la DEPP
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path


def md(source: str, tag: str | None = None) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [tag]} if tag else {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str, tag: str | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]} if tag else {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


# ══════════════════════════════════════════════════════════════════════════════
# EN-TÊTE
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "# Analyse du benchmark — comparaison modèle vs annotateur expert\n\n"
        "Notebook organisé en sections balisées `section:<nom>` pour permettre\n"
        "l'export HTML sélectif via la dernière section « Exports ».\n\n"
        "**Question centrale** : au-delà des désaccords item par item, le modèle\n"
        "reproduit-il fidèlement les grandeurs agrégées et distributionnelles qui\n"
        "intéressent la DEPP ?\n\n"
        "**Méthode statistique** : IC analytiques (Wilson pour les proportions,\n"
        "delta method pour le kappa) — instantanés et rigoureux. Le design effect\n"
        "de Kish est utilisé pour la prévalence par item afin de corriger la\n"
        "corrélation intra-copie."
    )
)

CELLS.append(
    code(
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from scipy import stats as scistats\n"
        "from scipy.stats import norm as scinorm\n"
        "from sklearn.metrics import cohen_kappa_score\n\n"
        "from evaluation_dictee.data.reference import load_grid\n"
        "from evaluation_dictee.evaluation.report import (\n"
        "    disagreement_decomposition, load_predictions,\n"
        "    per_copy_metrics, per_item_metrics,\n"
        ")\n\n"
        "# ═════════════════ PARAMÈTRES ═════════════════\n"
        'RUN_NAME = "dictee_gemma4_zeroshot"\n'
        'GRID_PATH = "configs/grille_dictee_2015.json"\n'
        "# ══════════════════════════════════════════════\n\n"
        'BASE = Path("..")\n'
        'PRED = BASE / f"data/processed/{RUN_NAME}_predictions.jsonl"\n'
        'FIG_DIR = BASE / "data/processed" / f"figures_{RUN_NAME}"\n'
        "FIG_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "df = load_predictions(PRED)\n"
        "grid = load_grid(BASE / GRID_PATH)\n"
        "id_to_mot = {it.item_id: it.attendu for it in grid.items}\n"
        "id_to_position = {it.item_id: i + 1 for i, it in enumerate(grid.items)}\n\n"
        "def lib(item_id: str) -> str:\n"
        '    return f"{id_to_position.get(item_id, 0):02d} · {id_to_mot.get(item_id, item_id)}"\n\n'
        "plt.rcParams.update({\n"
        '    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",\n'
        '    "axes.spines.top": False, "axes.spines.right": False,\n'
        '    "axes.grid": True, "grid.alpha": 0.25,\n'
        "})\n"
        'C_EXPERT, C_MODELE = "#1f4e79", "#c44536"\n'
        'C_OK, C_MOYEN, C_PB = "#2e7d32", "#ef6c00", "#c62828"\n'
        "Z95 = scinorm.ppf(0.975)\n\n"
        "print(f\"{len(df):,} lignes | {df['copy_id'].nunique()} copies × {df['item_id'].nunique()} items\")\n"
    )
)

CELLS.append(
    md(
        "Note sur les libellés d'items : format `NN · mot`, où `NN` est la position\n"
        "ordinale dans la dictée. Cela lève l'ambiguïté des mots répétés."
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Synthèse globale
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "## 1. Synthèse globale avec intervalles de confiance\n\n"
        "IC analytiques (Wilson pour les proportions, delta method pour le kappa).\n"
        "Instantanés et rigoureux — remplace le bootstrap qui souffrait de bugs\n"
        "silencieux (bornes identiques) sur certaines configurations.",
        tag="section:synthese",
    )
)

CELLS.append(
    code(
        "def wilson_global(k, n):\n"
        "    if n == 0: return float('nan'), float('nan'), float('nan')\n"
        "    p = k / n\n"
        "    denom = 1 + Z95**2 / n\n"
        "    centre = (p + Z95**2 / (2 * n)) / denom\n"
        "    margin = (Z95 / denom) * (p * (1 - p) / n + Z95**2 / (4 * n**2)) ** 0.5\n"
        "    return p, max(0.0, centre - margin), min(1.0, centre + margin)\n\n"
        "def kappa_ci(y_true, y_pred):\n"
        "    kappa = cohen_kappa_score(y_true, y_pred)\n"
        "    n = len(y_true)\n"
        "    cats = sorted(set(y_true) | set(y_pred))\n"
        "    cat_idx = {c: i for i, c in enumerate(cats)}\n"
        "    conf = np.zeros((len(cats), len(cats)))\n"
        "    for yt, yp in zip(y_true, y_pred):\n"
        "        conf[cat_idx[yt], cat_idx[yp]] += 1\n"
        "    conf /= n\n"
        "    p_e = (conf.sum(axis=0) * conf.sum(axis=1)).sum()\n"
        "    if p_e == 1.0: return kappa, kappa, kappa\n"
        "    A = np.diag(conf) * (1 - (conf.sum(axis=0) + conf.sum(axis=1)) * (1 - kappa))\n"
        "    B = (1 - kappa) ** 2 * (\n"
        "        (conf * (conf.sum(axis=0)[np.newaxis, :] + conf.sum(axis=1)[:, np.newaxis]) ** 2).sum()\n"
        "        - np.diag(conf * (conf.sum(axis=0) + conf.sum(axis=1)) ** 2).sum()\n"
        "    )\n"
        "    var_k = (A.sum() + B - (kappa - p_e * (1 - kappa)) ** 2) / (n * (1 - p_e) ** 2)\n"
        "    se = max(0.0, var_k) ** 0.5\n"
        "    return kappa, kappa - Z95 * se, kappa + Z95 * se\n\n"
        "n_total = len(df)\n"
        "n_accord = int((df['y_true'] == df['y_pred']).sum())\n"
        "exp_err = df['y_true'] != '1'\n"
        "mod_err = df['y_pred'] != '1'\n"
        "n_exp_err, n_mod_err = exp_err.sum(), mod_err.sum()\n"
        "n_vp = (exp_err & mod_err).sum()\n\n"
        "acc, acc_lo, acc_hi = wilson_global(n_accord, n_total)\n"
        "rap, rap_lo, rap_hi = wilson_global(n_vp, n_exp_err)\n"
        "pre, pre_lo, pre_hi = wilson_global(n_vp, n_mod_err)\n"
        "kap, kap_lo, kap_hi = kappa_ci(df['y_true'].values, df['y_pred'].values)\n\n"
        "syn = pd.DataFrame([\n"
        "    {'Métrique': 'Accord brut', 'Valeur': acc, 'IC95 bas': acc_lo, 'IC95 haut': acc_hi},\n"
        "    {'Métrique': 'Kappa de Cohen', 'Valeur': kap, 'IC95 bas': kap_lo, 'IC95 haut': kap_hi},\n"
        "    {'Métrique': 'Rappel erreurs (sensibilité)', 'Valeur': rap, 'IC95 bas': rap_lo, 'IC95 haut': rap_hi},\n"
        "    {'Métrique': 'Précision erreurs', 'Valeur': pre, 'IC95 bas': pre_lo, 'IC95 haut': pre_hi},\n"
        "]).set_index('Métrique')\n\n"
        "print(f\"N = {n_total:,} items | {df['copy_id'].nunique()} copies\")\n"
        "display(syn.style.format('{:.1%}'))\n"
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Prévalence par item (Wilson + Kish, sans bootstrap)
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "## 2. Prévalence d'erreur par item — expert vs modèle\n\n"
        "**Objectif DEPP** : identifier les items les plus difficiles pour les élèves.\n"
        "La hiérarchie de difficulté du modèle correspond-elle à celle de l'expert ?\n\n"
        "**Méthode** : Wilson corrigé du design effect de Kish (corrélation intra-copie).",
        tag="section:prevalence_item",
    )
)

CELLS.append(
    code(
        "def estimate_deff(df, item_col='item_id', label_col='y_true'):\n"
        "    df2 = df.copy()\n"
        "    df2['_err'] = (df2[label_col] != '1').astype(float)\n"
        "    m = df2.groupby('copy_id')[item_col].count().mean()\n"
        "    var_total = df2['_err'].var(ddof=1)\n"
        "    if var_total == 0: return 1.0\n"
        "    copy_means = df2.groupby('copy_id')['_err'].mean()\n"
        "    var_between = copy_means.var(ddof=1)\n"
        "    icc = max(0.0, (var_between - (var_total - var_between) / (m - 1)) /\n"
        "              (var_between + (m - 1) * (var_total - var_between) / (m - 1)))\n"
        "    return 1 + (m - 1) * icc\n\n"
        "deff = max(1.0, estimate_deff(df))\n"
        "print(f'Design effect estimé : {deff:.3f}')\n\n"
        "rows = []\n"
        "for item_id, grp in df.groupby('item_id'):\n"
        "    n = len(grp); n_eff = round(n / deff)\n"
        "    k_exp = int((grp['y_true'] != '1').sum())\n"
        "    k_mod = int((grp['y_pred'] != '1').sum())\n"
        "    _, lo_e, hi_e = wilson_global(k_exp, n_eff)\n"
        "    _, lo_m, hi_m = wilson_global(k_mod, n_eff)\n"
        "    rows.append({\n"
        "        'item_id': item_id, 'position': id_to_position.get(item_id, 0),\n"
        "        'mot_attendu': id_to_mot.get(item_id, item_id),\n"
        "        'pct_expert': k_exp / n * 100, 'IC_expert_lo': lo_e * 100, 'IC_expert_hi': hi_e * 100,\n"
        "        'pct_modele': k_mod / n * 100, 'IC_modele_lo': lo_m * 100, 'IC_modele_hi': hi_m * 100,\n"
        "        'ecart_modele_expert': round((k_mod - k_exp) / n * 100, 2),\n"
        "    })\n"
        "prev = pd.DataFrame(rows).set_index('item_id').sort_values('position')\n\n"
        'print("\\nTop 15 items où le modèle diverge le plus :")\n'
        "top_div = prev.reindex(prev['ecart_modele_expert'].abs().sort_values(ascending=False).index).head(15)\n"
        "display(top_div[['position', 'mot_attendu', 'pct_expert', 'pct_modele', 'ecart_modele_expert']]\n"
        "    .style.format({'pct_expert': '{:.1f} %', 'pct_modele': '{:.1f} %', 'ecart_modele_expert': '{:+.1f} pts'}))\n\n"
        "prev.to_csv(BASE / f'data/processed/{RUN_NAME}_prevalence_par_item.csv',\n"
        "            sep=';', decimal=',', encoding='utf-8-sig')\n"
    )
)

CELLS.append(
    code(
        "fig, ax = plt.subplots(figsize=(9, 8))\n"
        "x, y = prev['pct_expert'].values, prev['pct_modele'].values\n"
        "maxv = max(x.max(), y.max()) * 1.08\n\n"
        "ax.plot([0, maxv], [0, maxv], color='grey', ls='--', lw=1, label='y = x')\n"
        "ax.scatter(x, y, s=42, alpha=0.85, color=C_MODELE, edgecolors='k', linewidths=0.4, zorder=3)\n\n"
        "slope, intercept, r_value, *_ = scistats.linregress(x, y)\n"
        "xs = np.array([0, maxv])\n"
        "ax.plot(xs, intercept + slope * xs, color=C_EXPERT, lw=1.5,\n"
        "        label=f'y = {slope:.2f}x {intercept:+.2f}   |   r = {r_value:.3f}')\n\n"
        "for _, r in prev.iterrows():\n"
        "    ax.annotate(f\"{r['position']:02d}·{r['mot_attendu']}\", (r['pct_expert'], r['pct_modele']),\n"
        "                fontsize=6, xytext=(4, 3), textcoords='offset points', alpha=0.85)\n\n"
        "ax.set_xlabel(\"Prévalence d'erreur selon l'expert (%)\")\n"
        'ax.set_ylabel("Prévalence d\'erreur selon le modèle (%)")\n'
        'ax.set_title(f"Prévalence d\'erreur par item — {RUN_NAME}")\n'
        "ax.legend(loc='lower right')\n"
        "plt.savefig(FIG_DIR / '02_prevalence_expert_vs_modele.png')\n"
        "plt.show()\n\n"
        "biais = (y - x).mean()\n"
        "print(f'Pearson r={np.corrcoef(x, y)[0, 1]:+.3f}  |  Spearman ρ={scistats.spearmanr(x, y).correlation:+.3f}')\n"
        "print(f'Biais moyen : {biais:+.2f} pts    Régression : pente={slope:.3f}, R²={r_value**2:.3f}')\n"
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Distributions (3 panneaux)
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "## 3. Distribution du nombre d'erreurs par copie\n\n"
        "Vue en trois panneaux : erreurs totales, fautes seules (code 9), mots\n"
        "absents (code 0). Complétée d'une courbe de survie qui chiffre la part\n"
        "de copies au-dessus de chaque seuil.",
        tag="section:distribution_erreurs",
    )
)

CELLS.append(
    code(
        "copies_m = per_copy_metrics(df)\n\n"
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n"
        "COLS = [\n"
        "    ('n_erreurs_expert', 'n_erreurs_modele', \"Total d'erreurs (fautes + absents)\"),\n"
        "    ('n_fautes_expert', 'n_fautes_modele', \"Fautes d'orthographe (code 9)\"),\n"
        "    ('n_manquants_expert', 'n_manquants_modele', \"Mots absents (code 0)\"),\n"
        "]\n"
        "for ax, (col_e, col_m, titre) in zip(axes, COLS):\n"
        "    max_v = max(copies_m[col_e].max(), copies_m[col_m].max())\n"
        "    bins = np.arange(0, max_v + 2) - 0.5\n"
        "    ax.hist(copies_m[col_e], bins=bins, alpha=0.65, color=C_EXPERT, label='Expert', edgecolor='white')\n"
        "    ax.hist(copies_m[col_m], bins=bins, alpha=0.65, color=C_MODELE, label='Modèle', edgecolor='white')\n"
        "    ax.axvline(copies_m[col_e].mean(), color=C_EXPERT, ls='--', lw=1.5,\n"
        "               label=f'moy. expert {copies_m[col_e].mean():.1f}')\n"
        "    ax.axvline(copies_m[col_m].mean(), color=C_MODELE, ls='--', lw=1.5,\n"
        "               label=f'moy. modèle {copies_m[col_m].mean():.1f}')\n"
        "    ax.set_xlabel(\"Nombre par copie\"); ax.set_ylabel('Nombre de copies')\n"
        "    ax.set_title(titre, fontsize=9, fontweight='bold'); ax.legend(fontsize=8)\n"
        "plt.tight_layout()\n"
        "plt.savefig(FIG_DIR / '03_distributions.png')\n"
        "plt.show()\n\n"
        "ks = scistats.ks_2samp(copies_m['n_erreurs_expert'], copies_m['n_erreurs_modele'])\n"
        'print(f"KS (nb total d\'erreurs) : D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}")\n\n'
        "def stats_col(c):\n"
        "    return {'moyenne': c.mean(), 'médiane': c.median(), 'écart-type': c.std(),\n"
        "            'q1': c.quantile(0.25), 'q3': c.quantile(0.75), 'max': c.max()}\n"
        "comp = pd.DataFrame({'Expert': stats_col(copies_m['n_erreurs_expert']),\n"
        "                     'Modèle': stats_col(copies_m['n_erreurs_modele'])}).round(2)\n"
        "comp.loc['écart absolu'] = (comp['Modèle'] - comp['Expert']).round(2)\n"
        "display(comp)\n"
    )
)

CELLS.append(
    code(
        "fig, ax = plt.subplots(figsize=(11, 5))\n"
        "max_err = max(copies_m['n_erreurs_expert'].max(), copies_m['n_erreurs_modele'].max())\n"
        "seuils = np.arange(0, int(max_err) + 1)\n"
        "surv_exp = [(copies_m['n_erreurs_expert'] > s).mean() * 100 for s in seuils]\n"
        "surv_mod = [(copies_m['n_erreurs_modele'] > s).mean() * 100 for s in seuils]\n\n"
        "ax.plot(seuils, surv_exp, color=C_EXPERT, lw=2.5, marker='o', ms=4, label='Expert')\n"
        "ax.plot(seuils, surv_mod, color=C_MODELE, lw=2.5, marker='s', ms=4, label='Modèle')\n"
        "ax.fill_between(seuils, surv_exp, surv_mod,\n"
        "                where=[e >= m for e, m in zip(surv_exp, surv_mod)],\n"
        "                alpha=0.12, color=C_EXPERT, label='Modèle sous-estime')\n"
        "ax.fill_between(seuils, surv_exp, surv_mod,\n"
        "                where=[m > e for e, m in zip(surv_exp, surv_mod)],\n"
        "                alpha=0.12, color=C_MODELE, label='Modèle surestime')\n\n"
        "for s_ref, ls in [(5, ':'), (10, '--'), (20, '-.')]:\n"
        "    if s_ref > max_err: continue\n"
        "    pct_e = (copies_m['n_erreurs_expert'] > s_ref).mean() * 100\n"
        "    pct_m = (copies_m['n_erreurs_modele'] > s_ref).mean() * 100\n"
        "    ax.axvline(s_ref, color='grey', ls=ls, lw=1, alpha=0.6)\n"
        "    ax.annotate(f's={s_ref}\\nexpert {pct_e:.0f}%\\nmodèle {pct_m:.0f}%',\n"
        "                xy=(s_ref, max(pct_e, pct_m)), xytext=(s_ref + 0.5, max(pct_e, pct_m) + 3),\n"
        "                fontsize=7.5, color='grey')\n\n"
        'ax.set_xlabel("Seuil s : nombre d\'erreurs par copie")\n'
        'ax.set_ylabel("% de copies avec > s erreurs")\n'
        'ax.set_title("Courbe de survie empirique"); ax.legend(fontsize=9, ncol=2)\n'
        "ax.set_ylim(0, 105)\n"
        "plt.savefig(FIG_DIR / '03b_courbe_survie.png')\n"
        "plt.show()\n"
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Corrélation nb d'erreurs modèle vs expert
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "## 4. Nombre d'erreurs modèle vs expert par copie\n\n"
        "**Objectif DEPP** : le modèle reproduit-il le classement des élèves ?\n"
        "Une corrélation forte + pente proche de 1 = classement fidèle.",
        tag="section:correlation_niveau",
    )
)

CELLS.append(
    code(
        "fig, ax = plt.subplots(figsize=(8, 8))\n"
        "x = copies_m['n_erreurs_expert'].values\n"
        "y = copies_m['n_erreurs_modele'].values\n"
        "maxv = max(x.max(), y.max()) * 1.05\n\n"
        "ax.plot([0, maxv], [0, maxv], color='grey', ls='--', lw=1, label='y = x')\n"
        "ax.scatter(x, y, s=30, alpha=0.55, color=C_MODELE, edgecolors='none')\n\n"
        "slope, intercept, r_value, *_ = scistats.linregress(x, y)\n"
        "xs = np.array([0, maxv])\n"
        "ax.plot(xs, intercept + slope * xs, color=C_EXPERT, lw=1.7,\n"
        "        label=f'y = {slope:.2f}x {intercept:+.2f}   |   r = {r_value:.3f}, R² = {r_value**2:.3f}')\n"
        "ax.set_xlabel(\"Nombre d'erreurs selon l'expert (proxy du niveau)\")\n"
        'ax.set_ylabel("Nombre d\'erreurs selon le modèle")\n'
        "ax.set_title('Corrélation modèle vs expert par copie')\n"
        "ax.legend(loc='upper left')\n"
        "plt.savefig(FIG_DIR / '04_correlation_niveau.png')\n"
        "plt.show()\n\n"
        "rng = np.random.default_rng(42)\n"
        "boot_r = []\n"
        "for _ in range(1000):\n"
        "    idx = rng.integers(0, len(x), len(x))\n"
        "    boot_r.append(np.corrcoef(x[idx], y[idx])[0, 1])\n"
        "lo_r, hi_r = np.percentile(boot_r, [2.5, 97.5])\n"
        "print(f'Pearson  r = {np.corrcoef(x, y)[0, 1]:+.3f}  [IC 95 %: {lo_r:+.3f} ; {hi_r:+.3f}]')\n"
        "print(f'Spearman ρ = {scistats.spearmanr(x, y).correlation:+.3f}')\n"
        "print(f'Kendall  τ = {scistats.kendalltau(x, y).correlation:+.3f}')\n"
        "print(f'Régression : pente = {slope:.3f}, R² = {r_value**2:.3f}')\n"
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Métriques par difficulté (grille 2×4)
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "## 5. Métriques d'accord selon la difficulté de la copie\n\n"
        "Grille 2 × 4 : deux lignes selon la source de difficulté (erreurs expert /\n"
        "modèle), quatre colonnes : accord, kappa, rappel, précision.",
        tag="section:metriques_par_difficulte",
    )
)

CELLS.append(
    code(
        "N_QUANTILES = 8\n\n"
        "def metriques_par_groupe(df_items, copies_df, col_tri, n_q=N_QUANTILES):\n"
        "    try:\n"
        "        quantiles = pd.qcut(copies_df[col_tri], q=n_q, duplicates='drop')\n"
        "    except ValueError:\n"
        "        quantiles = pd.cut(copies_df[col_tri], bins=n_q, duplicates='drop')\n"
        "    copies_df = copies_df.copy(); copies_df['_g'] = quantiles\n"
        "    mid_x = copies_df.groupby('_g', observed=True)[col_tri].mean()\n"
        "    rows = []\n"
        "    for grp_label, grp_copies in copies_df.groupby('_g', observed=True):\n"
        "        sub = df_items[df_items['copy_id'].isin(grp_copies.index)]\n"
        "        if len(sub) < 10: continue\n"
        "        n = len(sub); n_acc = int((sub['y_true'] == sub['y_pred']).sum())\n"
        "        exp_e = sub['y_true'] != '1'; mod_e = sub['y_pred'] != '1'\n"
        "        n_vp = int((exp_e & mod_e).sum()); n_ee = int(exp_e.sum()); n_me = int(mod_e.sum())\n"
        "        _, acc_lo, acc_hi = wilson_global(n_acc, n)\n"
        "        _, rap_lo, rap_hi = wilson_global(n_vp, n_ee) if n_ee else (float('nan'),) * 3\n"
        "        try: kap = cohen_kappa_score(sub['y_true'], sub['y_pred'])\n"
        "        except Exception: kap = float('nan')\n"
        "        rows.append({'x': mid_x[grp_label], 'n_copies': len(grp_copies),\n"
        "                     'accord': n_acc / n, 'acc_lo': acc_lo, 'acc_hi': acc_hi,\n"
        "                     'kappa': kap, 'rappel': (n_vp / n_ee) if n_ee else float('nan'),\n"
        "                     'rap_lo': rap_lo, 'rap_hi': rap_hi,\n"
        "                     'precision': (n_vp / n_me) if n_me else float('nan')})\n"
        "    return pd.DataFrame(rows).sort_values('x')\n\n"
        "fig, axes = plt.subplots(2, 4, figsize=(18, 9))\n"
        'fig.suptitle("Métriques d\'accord selon la difficulté de la copie\\n"\n'
        '             "Ligne pleine = valeur observée · zone ombrée = IC Wilson 95 %", fontsize=12)\n\n'
        "SOURCES = [('n_erreurs_expert', \"Erreurs selon l'expert\"),\n"
        "           ('n_erreurs_modele', \"Erreurs selon le modèle\")]\n"
        "METRIQUES = [('accord', 'Accord brut', 'acc_lo', 'acc_hi', C_EXPERT),\n"
        "             ('kappa', 'Kappa de Cohen', None, None, '#7b2d8b'),\n"
        "             ('rappel', 'Rappel (sensibilité)', 'rap_lo', 'rap_hi', C_MODELE),\n"
        "             ('precision', 'Précision erreurs', None, None, '#e67e22')]\n\n"
        "for col_idx, (col_tri, label_tri) in enumerate(SOURCES):\n"
        "    g = metriques_par_groupe(df, copies_m, col_tri)\n"
        "    xs = g['x'].values\n"
        "    for row_idx, (metric, label, lo_col, hi_col, color) in enumerate(METRIQUES):\n"
        "        ax = axes[col_idx][row_idx]\n"
        "        ax.plot(xs, g[metric].values, color=color, lw=2, marker='o', ms=5)\n"
        "        if lo_col and hi_col:\n"
        "            ax.fill_between(xs, g[lo_col].values, g[hi_col].values, alpha=0.18, color=color)\n"
        "        ax.axhline(g[metric].mean(), color=color, ls=':', lw=1, alpha=0.6)\n"
        "        ax.set_title(label, fontsize=9, fontweight='bold')\n"
        "        ax.set_xlabel(label_tri, fontsize=8)\n"
        "        ax.set_ylim(0, 1.05)\n"
        "        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))\n"
        "    axes[col_idx][0].set_ylabel(label_tri, fontsize=9, fontweight='bold')\n\n"
        "plt.tight_layout()\n"
        "plt.savefig(FIG_DIR / '05_metriques_par_difficulte.png', bbox_inches='tight')\n"
        "plt.show()\n"
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Comparaison multi-modèles (NOUVEAU)
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "## 6. Comparaison multi-modèles (optionnelle)\n\n"
        "**Motivation** : quand plusieurs modèles indépendants convergent, la\n"
        "prédiction est fiable ; quand ils divergent, c'est un signal d'incertitude\n"
        "qui appelle un humain. Le désaccord inter-modèles est un **bien meilleur\n"
        "prédicteur de confiance** que le score annoncé par un modèle unique.\n\n"
        "**Pré-requis** : avoir lancé plusieurs benchmarks sur les mêmes copies\n"
        "(configs différentes : modèles ou prompts variés). Chaque run produit son\n"
        "`<run>_predictions.jsonl`.",
        tag="section:multi_modeles",
    )
)

CELLS.append(
    code(
        "from evaluation_dictee.evaluation.multi_model import (\n"
        "    agreement_per_item, confidence_score, disagreement_type_summary,\n"
        "    load_multi_runs, referral_curve_with_ci,\n"
        ")\n\n"
        "# ═══════════ Sélection des runs à comparer ═══════════\n"
        "RUNS_MULTI = [\n"
        "    RUN_NAME,\n"
        "    # 'dictee_gemma4_cot',\n"
        "    # 'dictee_gemma4_2stages',\n"
        "]\n"
        "# ═════════════════════════════════════════════════════\n\n"
        "RUNS_MULTI = [r for r in RUNS_MULTI\n"
        "              if (BASE / f'data/processed/{r}_predictions.jsonl').exists()]\n"
        "print(f'Runs disponibles : {RUNS_MULTI}')\n\n"
        "if len(RUNS_MULTI) < 2:\n"
        "    print('\\n⚠️  Un seul run disponible.')\n"
        "    print('   Pour la comparaison multi-modèles, lancer d\\'autres benchmarks')\n"
        "    print('   (ex. chain-of-thought, approche 2 étapes) et relancer.')\n"
        "    df_agree = None; conf_copies = None\n"
        "else:\n"
        "    df_multi = load_multi_runs(RUNS_MULTI, output_dir=BASE / 'data/processed')\n"
        "    df_agree = agreement_per_item(df_multi)\n"
        "    conf_copies = confidence_score(df_agree)\n"
        "    print(f'\\n{len(df_multi):,} items × {len(RUNS_MULTI)} modèles')\n"
        "    print(f'{df_agree[\"unanimite\"].mean() * 100:.1f} % d\\'items en unanimité')\n"
        "    print(f'Score de confiance moyen par copie : {conf_copies[\"score_confiance\"].mean():.1f} %')\n"
    )
)

CELLS.append(
    code(
        "if df_agree is not None:\n"
        "    summary = disagreement_type_summary(df_agree)\n"
        '    print("Répartition des items selon le nb de modèles d\'accord :\\n")\n'
        "    display(summary.style.format({'pct_items': '{:.1f} %', 'accord_avec_expert': '{:.1f} %'}))\n"
        "    print(\"\\nLecture : quand tous les modèles s'accordent (dernière ligne), l'accord\")\n"
        "    print(\"avec l'expert est élevé → le désaccord inter-modèles est prédictif d'erreur.\")\n\n"
        "    fig, ax = plt.subplots(figsize=(9, 5))\n"
        "    couleurs = [C_PB if v < 70 else C_MOYEN if v < 90 else C_OK for v in summary['accord_avec_expert']]\n"
        "    ax.bar(summary.index, summary['accord_avec_expert'], color=couleurs, edgecolor='k', linewidth=0.4)\n"
        "    for idx, val in summary['accord_avec_expert'].items():\n"
        "        ax.annotate(f'{val:.1f} %', (idx, val), textcoords='offset points',\n"
        "                    xytext=(0, 4), ha='center', fontsize=9)\n"
        "    ax.set_xlabel(f'Nombre de modèles d\\'accord (sur {int(df_agree[\"n_modeles\"].iloc[0])})')\n"
        '    ax.set_ylabel("Accord de la modale avec l\'expert (%)")\n'
        "    ax.set_title(\"Le consensus inter-modèles prédit l'accord avec l'expert\",\n"
        "                 fontsize=10, fontweight='bold')\n"
        "    ax.set_ylim(0, 105)\n"
        "    plt.savefig(FIG_DIR / '06_consensus_predit_accord.png')\n"
        "    plt.show()\n"
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Stratégie de renvoi basée sur le désaccord (NOUVEAU)
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "## 7. Stratégie de renvoi basée sur le désaccord multi-modèles\n\n"
        "**Question opérationnelle** : combien de copies faut-il renvoyer en\n"
        "correction humaine pour atteindre un accord cible sur les copies retenues ?\n\n"
        "**Stratégie** : trier les copies par score de confiance (% d'items où tous\n"
        "les modèles s'accordent), renvoyer celles sous le seuil τ. **Un seul\n"
        "graphique clair** : accord retenu en fonction du % renvoyé, avec IC Wilson\n"
        "et repères aux seuils clés (10 %, 20 %, 30 %, 50 %).",
        tag="section:strategie_renvoi",
    )
)

CELLS.append(
    code(
        "if df_agree is not None:\n"
        "    curve = referral_curve_with_ci(df_agree, conf_copies, reference_run=RUN_NAME)\n\n"
        "    fig, ax = plt.subplots(figsize=(10, 6))\n"
        "    xs = curve['pct_copies_renvoyees'].values\n"
        "    ys = curve['pct_accord_retenues'].values\n"
        "    ax.plot(xs, ys, color=C_EXPERT, lw=2.5, marker='o', ms=5, label='Accord sur copies retenues')\n"
        "    ax.fill_between(xs, curve['accord_lo'], curve['accord_hi'],\n"
        "                    alpha=0.18, color=C_EXPERT, label='IC Wilson 95 %')\n"
        "    ax.axhline(ys[0], color='grey', ls=':', lw=1,\n"
        "               label=f'accord global sans renvoi = {ys[0]:.1f} %')\n\n"
        "    # Repères aux seuils clés\n"
        "    for pct_ref in [10, 20, 30, 50]:\n"
        "        idx = (curve['pct_copies_renvoyees'] - pct_ref).abs().idxmin()\n"
        "        acc_ref = curve.loc[idx, 'pct_accord_retenues']\n"
        "        if not np.isnan(acc_ref):\n"
        "            ax.plot(pct_ref, acc_ref, 'o', color='#c62828', ms=8, zorder=5)\n"
        "            ax.annotate(f'{pct_ref}% renvoyé\\n→ accord {acc_ref:.1f}%',\n"
        "                        (pct_ref, acc_ref), textcoords='offset points', xytext=(10, -20),\n"
        "                        fontsize=8, arrowprops=dict(arrowstyle='->', color='grey', lw=0.5))\n\n"
        "    ax.set_xlabel('% de copies renvoyées en correction manuelle')\n"
        "    ax.set_ylabel('Accord modèle-expert sur les copies retenues (%)')\n"
        "    ax.set_title('Arbitrage : charge humaine vs qualité automatisée',\n"
        "                 fontsize=10, fontweight='bold')\n"
        "    ax.legend(loc='lower right'); ax.set_xlim(-2, 102)\n"
        "    plt.savefig(FIG_DIR / '07_arbitrage_renvoi.png')\n"
        "    plt.show()\n\n"
        "    print('\\nSeuils clés :')\n"
        "    display(curve[curve['seuil_confiance'].isin([50, 60, 70, 80, 90, 95, 100])]\n"
        "        .style.format({'pct_copies_renvoyees': '{:.1f} %',\n"
        "                       'pct_accord_retenues': '{:.1f} %',\n"
        "                       'accord_lo': '{:.1f} %', 'accord_hi': '{:.1f} %'}))\n"
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTIONS 8-10 — Décomposition, accord par item et par copie
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(md("## 8. Décomposition des désaccords", tag="section:decomposition"))
CELLS.append(
    code(
        "deco = disagreement_decomposition(df)\n"
        "display(deco.style.format({'pct_desaccords': '{:.1f} %'}))\n"
    )
)

CELLS.append(
    md(
        "## 9. Accord par item, avec intervalles de Wilson",
        tag="section:accord_item",
    )
)
CELLS.append(
    code(
        "items = per_item_metrics(df); items_tries = items.sort_values('accord')\n"
        "fig, ax = plt.subplots(figsize=(9.5, 0.24 * len(items_tries) + 1.5))\n"
        "y = np.arange(len(items_tries))\n"
        "couleurs = [C_PB if v < 0.85 else C_MOYEN if v < 0.95 else C_OK for v in items_tries['accord']]\n"
        "err_low = (items_tries['accord'] - items_tries['accord_lo']) * 100\n"
        "err_high = (items_tries['accord_hi'] - items_tries['accord']) * 100\n"
        "ax.barh(y, items_tries['accord'] * 100, xerr=[err_low, err_high],\n"
        "        color=couleurs, error_kw={'lw': 0.8, 'capsize': 2, 'alpha': 0.65})\n"
        "ax.set_yticks(y); ax.set_yticklabels([lib(i) for i in items_tries.index], fontsize=7)\n"
        "ax.axvline(85, color=C_PB, ls=':', lw=1); ax.axvline(95, color=C_MOYEN, ls=':', lw=1)\n"
        "ax.set_xlim(max(0, items_tries['accord_lo'].min() * 100 - 5), 101)\n"
        'ax.set_xlabel("Accord avec l\'expert (%)  — barres = IC Wilson 95 %")\n'
        "ax.set_title('Accord par item')\n"
        "plt.savefig(FIG_DIR / '09_accord_par_item.png'); plt.show()\n"
    )
)

CELLS.append(md("## 10. Accord par copie", tag="section:accord_copie"))
CELLS.append(
    code(
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))\n"
        "ax1.hist(copies_m['accord'] * 100, bins=24, color=C_EXPERT, edgecolor='white', alpha=0.9)\n"
        "ax1.axvline(copies_m['accord'].mean() * 100, color=C_PB, ls='--',\n"
        "            label=f\"moyenne {copies_m['accord'].mean():.1%}\")\n"
        "ax1.set_xlabel('Accord sur la copie (%)'); ax1.set_ylabel('Nombre de copies')\n"
        'ax1.set_title("Distribution de l\'accord par copie"); ax1.legend()\n\n'
        "ax2.scatter(copies_m['pct_erreur_expert'], copies_m['accord'] * 100,\n"
        "            s=22, alpha=0.6, color=C_MODELE, edgecolors='none')\n"
        "nq = min(8, copies_m['pct_erreur_expert'].nunique())\n"
        "q = pd.qcut(copies_m['pct_erreur_expert'], q=nq, duplicates='drop')\n"
        "tend = copies_m.groupby(q, observed=True).agg(x=('pct_erreur_expert', 'mean'), yacc=('accord', 'mean'))\n"
        "ax2.plot(tend['x'], tend['yacc'] * 100, color=C_EXPERT, lw=2, marker='o', label='Moyenne par quantile')\n"
        "ax2.set_xlabel(\"% d'erreur sur la copie\"); ax2.set_ylabel('Accord (%)')\n"
        'ax2.set_title("L\'accord baisse-t-il sur les copies difficiles ?"); ax2.legend()\n'
        "plt.savefig(FIG_DIR / '10_accord_par_copie.png')\n"
        "plt.show()\n"
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — Export HTML sélectif
# ══════════════════════════════════════════════════════════════════════════════
CELLS.append(
    md(
        "## 11. Export HTML pour l'équipe DEPP — sélection des sections",
        tag="section:exports",
    )
)
CELLS.append(
    code(
        "from evaluation_dictee.evaluation.html_report import list_sections, build_html_report\n\n"
        "NOTEBOOK_PATH = Path('03_analyse_resultats.ipynb')\n"
        "for s in list_sections(NOTEBOOK_PATH):\n"
        "    print(f'  {s.tag:35}  {s.title}')\n"
    )
)

CELLS.append(
    code(
        "SECTIONS_A_INCLURE = [\n"
        "    'section:synthese', 'section:prevalence_item', 'section:distribution_erreurs',\n"
        "    'section:correlation_niveau', 'section:metriques_par_difficulte',\n"
        "    # 'section:multi_modeles', 'section:strategie_renvoi',\n"
        "    # 'section:decomposition', 'section:accord_item', 'section:accord_copie',\n"
        "]\n"
        "out = build_html_report(\n"
        "    notebook_path=NOTEBOOK_PATH, selected_tags=SECTIONS_A_INCLURE,\n"
        "    output_path=BASE / f'data/processed/rapport_depp_{RUN_NAME}.html',\n"
        "    hide_code=True, execute=False,\n"
        ")\n"
        "print(f'Rapport généré : {out}')\n"
    )
)


# ──────── Assemblage ────────
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "03_analyse_resultats.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook regénéré : {out} ({len(CELLS)} cellules)")
