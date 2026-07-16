# CLAUDE.md

Ce fichier fournit le contexte du projet à Claude Code (et à tout assistant IA) ainsi
qu'aux développeurs qui rejoignent le projet. Lis-le en entier avant de coder.

---

## 1. En une phrase

Évaluer automatiquement la production écrite manuscrite d'élèves (dictée et rédaction,
du CE1 à la 6e) à l'aide de modèles d'IA open weight, pour assister la correction
experte menée par la DEPP, en mesurant rigoureusement la fiabilité avant tout usage.

## 2. Contexte du projet

- **Commanditaire** : DEPP (Direction de l'évaluation, de la prospective et de la
  performance, Ministère de l'Éducation nationale), bureau B2.
- **Maîtrise d'œuvre technique** : SSP Lab (laboratoire d'innovation data de l'INSEE).
- **Partenaire** : TEKLIA (HTR spécialisé, plateforme Arkindex).
- **Infrastructure** : SSP Cloud (Onyxia, GPU H100, vLLM via `llm.lab`, stockage S3
  MinIO, MLflow, Langfuse, Argo Workflows).

La DEPP corrige depuis 2015 la production d'écrit sur échantillon. La correction est
100 % manuelle (application AGATE), coûteuse (5 à 10 min/copie). L'objectif est de
tester si l'IA peut fiabiliser et accélérer cette correction.

### Deux corpus
1. **Dictée CM2** (1987→2026) : texte de référence CONNU. ~10 000 imagettes
   disponibles (2015 + 2021). 83 items par copie (mots + ponctuation).
2. **Production écrite libre** (nov. 2024) : PAS de texte de référence. 15 027 élèves.
   5 à 13 critères selon le niveau.

> **On commence par la dictée** : la référence connue rend l'évaluation plus simple
> et mesurable (CER/WER exacts, comparaison item par item).

## 3. La tâche, précisément

Pour la dictée, chaque mot/ponctuation de l'élève reçoit un **code** :

| Code | Signification (mots)            | Code | Signification (ponctuation) |
|------|----------------------------------|------|------------------------------|
| 1    | Orthographe correcte             | 1    | Ponctuation correcte         |
| 3    | Erreur lexicale                  | 9    | Erreur                       |
| 4    | Erreur grammaticale              | 0    | Absence de ponctuation       |
| 5    | Erreur lexicale ET grammaticale  |      |                              |
| 0    | Mot absent (oubli)               |      |                              |

**Décision projet (grille simplifiée)** : la cible PRINCIPALE regroupe 3/4/5 en un
seul code « erreur ». On code donc en **1 / erreur / 0**. La grille complète à 6 codes
reste un objectif secondaire. Raison : sur le test pilote, la simplification fait
passer le kappa de 0,785 à 0,879 en supprimant exactement la zone où les correcteurs
humains divergent le plus.

**Règle des ratures** (tranchée par la DEPP) : quand l'élève rature et réécrit, on lit
l'**état final** (version corrigée par l'élève). Cette consigne est dans tous les prompts.

## 4. Approche méthodologique retenue

Quatre méthodes seront comparées sur une grille de métriques commune :

- **A** — Séquentiel : HTR classique (TrOCR/PyLaia) → LLM texte pour le scoring.
- **B** — Séquentiel : VLM transcrit l'image → LLM texte pour le scoring.
- **C** — End-to-end (PRIVILÉGIÉE) : un VLM reçoit l'image + le texte de référence + la
  grille, et produit directement le code de chaque item.
- **D** — Fine-tuning (phase ultérieure) : LoRA/QLoRA sur les ~3 000 copies annotées.

**Modèle principal** : Qwen2.5-VL. Tester le **7B avant le 72B** pour calibrer le coût
GPU. Baselines : TrOCR, PyLaia, PP-OCRv5. Repli si GPU contraint : MiniCPM-V-2.6.

> **Raccourci sans GPU** : `gemma4-26b-moe` est déjà servi sur llm.lab (SSP Cloud)
> et est multimodal (texte + image, OCR et reconnaissance d'écriture). C'est le
> candidat de première intention pour la méthode C : aucun GPU à privatiser,
> appel via l'API compatible OpenAI. Voir `configs/dictee_gemma4_zeroshot.yaml`.
> Attention au mode « thinking » de Gemma 4 : si actif, élaguer le bloc de
> raisonnement avant de parser le JSON (même précaution que `enable_thinking:false`
> sur Qwen3).

**Exigence transverse — score de confiance par item** : chaque code prédit doit
s'accompagner d'un score de confiance (log-probs vLLM, champ structuré, auto-cohérence,
ou désaccord inter-modèles). Objectif : renvoyer vers un codeur humain expert
uniquement les items incertains. Le livrable décisionnel est la courbe
« taux de renvoi humain vs taux d'erreur résiduel ».

## 5. Métriques

- **Scoring** : accord brut, **kappa de Cohen**, QWK (si codes ordinaux), matrice de
  confusion par code, F1 pondéré.
- **HTR** : CER, WER, taux de « illisibles », et surtout **taux de sur-correction**
  (mots que le modèle « corrige » silencieusement — biais critique des VLMs).
- **Confiance** : courbe de fiabilité, ECE (Expected Calibration Error).
- Toujours comparer la performance IA à la **variabilité inter-codeurs humains**
  (modèle Dawid & Skene, package R `rater` ou équivalent Python).

## 6. Données — points d'attention

- Les « .png » fournis sont en réalité des **TIFF bi-level 1 bit** (compression G4,
  1594×2044). Toujours normaliser le format en entrée (`data/loaders.py`).
- La binarisation 1 bit détruit les nuances de gris → accents difficiles à lire.
  Demander à la DEPP si des scans en niveaux de gris existent en amont.
- Les annotations actuelles sont des CODES, pas des transcriptions mot à mot. Pour
  fine-tuner un HTR, il faudra produire un ground truth de transcription séparé.
- **Données sensibles** : écritures d'élèves mineurs. Les microdonnées NE QUITTENT
  JAMAIS le SSP Cloud. Aucune image ni donnée élève dans le dépôt Git
  (voir `.gitignore`). Le dossier `data/` est ignoré par Git.

## 7. Structure du dépôt

```
evaluation_ecrit/
├── CLAUDE.md                 ← ce fichier
├── README.md                 ← démarrage rapide
├── pyproject.toml            ← dépendances + config outils
├── configs/                  ← fichiers YAML d'expériences (un par run)
├── src/evaluation_ecrit/
│   ├── config.py             ← chargement/validation des configs (Pydantic)
│   ├── data/                 ← chargement images, parsing grille, I/O S3
│   ├── models/               ← interfaces modèles (VLM via vLLM, HTR, base)
│   ├── pipeline/             ← orchestration des méthodes A/B/C/D
│   ├── evaluation/           ← métriques, calibration, comparaison annotateurs
│   └── utils/                ← logging, MLflow, helpers
├── scripts/                  ← points d'entrée CLI (run_benchmark, prepare_data…)
├── tests/                    ← tests unitaires (pytest)
├── notebooks/                ← exploration (ne pas y mettre de logique réutilisable)
└── docs/                     ← note de cadrage, grille de codage, décisions
```

## 8. Conventions de code

- **Python ≥ 3.11**, gestion des dépendances avec **uv** (ou pip + venv).
- **Formatage et lint** : `ruff` (formatage + lint). Lancer `ruff check` et
  `ruff format` avant chaque commit.
- **Typage** : annoter toutes les fonctions publiques. `mypy` en CI (non bloquant
  au début).
- **Docstrings** en français, style court (une ligne de résumé + Args/Returns si utile).
- **Pas de chemin en dur** : tout passe par `configs/*.yaml` et `src/.../config.py`.
- **Pas de secret dans le code** : identifiants S3, tokens → variables
  d'environnement (`.env`, jamais commité).
- **Reproductibilité** : chaque expérience = un fichier de config versionné + un run
  MLflow. Fixer les graines aléatoires.
- **Commits** : messages clairs en français, format `type: description`
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).

## 9. Commandes utiles

```bash
# Installer l'environnement
uv sync                         # ou: pip install -e ".[dev]"

# Qualité de code
ruff check src tests            # lint
ruff format src tests           # formatage
pytest                          # tests
mypy src                        # typage

# Lancer un benchmark à partir d'une config
python scripts/run_benchmark.py --config configs/dictee_gemma4_zeroshot.yaml
```

### Runs longs — utiliser screen ou nohup

Un benchmark complet (3469 copies × ~30 s) prend ~30 h. **Ne jamais le lancer
directement dans le terminal du navigateur** : au moindre plantage réseau, mise
en veille, fermeture d'onglet, le processus est tué et tout est perdu.

> **Note Onyxia** : `tmux` n'est pas disponible sur les services vscode-python
> du SSP Cloud (`apt-get install tmux` échoue). Utiliser `screen` ou `nohup`.

**Toujours créer le dossier logs d'abord :**
```bash
mkdir -p logs
```

**Option 1 (recommandée) — screen.** Session détachable, tu peux fermer le
navigateur et revenir le lendemain.
```bash
screen -S dictee                                            # créer la session
python scripts/run_benchmark.py --config configs/dictee_gemma4_cot.yaml
# Détacher : Ctrl+A puis D  (le job continue en arrière-plan)
screen -r dictee                                            # se rattacher plus tard
screen -ls                                                  # lister les sessions
```

**Option 2 — nohup.** Sans interface interactive, log dans un fichier.
```bash
mkdir -p logs
nohup python scripts/run_benchmark.py --config configs/dictee_gemma4_cot.yaml \
      > logs/dictee_gemma4_cot.log 2>&1 &
echo $! > logs/dictee_gemma4_cot.pid          # noter le PID pour arrêter plus tard
tail -f logs/dictee_gemma4_cot.log            # suivre le log en direct
```

### Checkpointing et reprise après crash

Depuis la refonte du benchmark, **chaque copie est écrite sur disque
immédiatement** (`flush + fsync`). Effets :

- Un crash à mi-run (API down, kernel tué, …) ne perd que la copie en cours.
- Relancer la même commande **reprend automatiquement** où on s'était arrêté :
  les copies déjà présentes dans `<run>_predictions.jsonl` sont sautées.
- Les copies qui lèvent une exception API sont loggées dans
  `<run>_failed_copies.txt` et le run continue sur les suivantes. Elles seront
  retentées au prochain lancement.
- Pour repartir de zéro, supprimer `<run>_predictions.jsonl` (ou changer
  `config.name`).

## 10. Pour un⋅e débutant⋅e

- Commence par lire `README.md`, puis `docs/note_de_cadrage.docx`.
- Le point d'entrée le plus simple est `scripts/run_benchmark.py` : il lit une config,
  charge les données, appelle un modèle, calcule les métriques. Suis le fil.
- Les `notebooks/` servent à explorer. Dès qu'un bout de code devient réutilisable,
  déplace-le dans `src/`.
- En cas de doute sur une décision méthodologique, la réponse est probablement dans
  ce fichier ou dans `docs/decisions.md`.

## 11. État d'avancement

- [x] Note de cadrage (docs/)
- [x] Test pilote manuel sur 3 copies (kappa 0,785 → 0,879 grille simplifiée)
- [ ] Phase 0 : accès S3, dataset d'évaluation (200-300 copies gold), MLflow/Langfuse
- [ ] Phase 1 : benchmark HTR sur la dictée
- [ ] Phase 2 : benchmark scoring (méthodes A/B/C)
- [ ] Phase 3 : passage à l'échelle ou fine-tuning