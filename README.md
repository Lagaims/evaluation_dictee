# Évaluation automatique de la dictée des élèves

Évaluer automatiquement la dictée manuscrite d'élèves (CEDRE, école primaire) à
l'aide de modèles multimodaux open weight, et **comparer rigoureusement** le codage
automatique à celui d'un correcteur expert. Collaboration **DEPP × SSP Lab (INSEE)**.

> Contexte complet et décisions : voir **[CLAUDE.md](CLAUDE.md)**, la
> **[note méthodologique](docs/note_methodologique.md)** et les
> **[décisions](docs/decisions.md)** dans [`docs/`](docs/).

---

## Ce que fait le projet, en bref

1. **Charge** les imagettes de dictée (TIFF 1 bit, depuis S3) et les codes de
   l'annotateur expert (gold standard).
2. **Demande à un modèle multimodal** (gemma4 sur llm.lab) de coder chaque mot de
   la dictée — correct / erreur / absent — directement à partir de l'image et du
   texte de référence, sans étape d'OCR séparée.
3. **Compare** les codes du modèle à ceux de l'expert.
4. **Mesure** la fiabilité (kappa, rappel des fautes, sur-correction) et la
   **calibration de la confiance**, pour décider quels items renvoyer à un humain.

La tâche cible est la **grille simplifiée** : `1` correct / `9` erreur / `0` absent
(voir [docs/decisions.md](docs/decisions.md), décision D2).

### Deux approches d'évaluation comparables

Le projet implémente **deux architectures** derrière la même interface `Scorer`,
donc évaluées par le même code de métriques (comparaison rigoureuse) :

- **`end_to_end` (approche 2)** : un VLM lit l'image ET code en une seule passe.
  Approche par défaut. Config : `configs/dictee_gemma4_zeroshot.yaml`.
- **`two_stage` (approche 1)** : étape 1 = transcription HTR (lecture de l'image en
  texte brut, fautes comprises) ; étape 2 = codage du texte transcrit (sans image,
  éventuellement par un modèle texte plus léger via `model_stage2`). Isole lecture
  et jugement. Config : `configs/dictee_gemma4_2stages.yaml`. L'approche se choisit
  via le champ `approach` du YAML.

### Évaluation dédiée de la transcription (HTR) sur Scoledit

Indépendamment du codage, on peut mesurer la **fidélité de lecture** d'un modèle sur
l'écriture manuscrite d'enfants via le corpus **Scoledit** (transcriptions de
référence humaines, fautes préservées). Métriques CER/WER (bruts et normalisés).
Cela permet de comparer les modèles sur la seule lecture et de distinguer les
erreurs de lecture de celles de jugement. Config : `configs/htr_gemma4_scoledit.yaml`,
script : `scripts/run_htr_benchmark.py`, analyse :
`notebooks/05_analyse_transcription_htr.ipynb`.

---

## Démarrage rapide (SSP Cloud / VSCode)

### 1. Installer

```bash
git clone <url-du-depot> evaluation_dictee
cd evaluation_dictee
uv sync
```

> Le paquet s'appelle `evaluation_dictee`. `uv sync` installe toutes les dépendances
> (y compris le groupe `dev`) et configure le mode éditable automatiquement, ce qui
> rend les imports `from evaluation_dictee...` disponibles.

### 2. Configurer les accès

Sur le SSP Cloud (Onyxia), les identifiants S3 sont généralement déjà dans
l'environnement. Pour le modèle, il faut un token llm.lab :

```bash
cp .env.example .env
# éditer .env :
#   LLM_BASE_URL=https://llm.lab.sspcloud.fr/api/v1   (vérifier le chemin exact)
#   LLM_API_KEY=<ton-token-llm.lab>
```

Vérifier l'accès aux données et au modèle :

```bash
# S3 accessible ?
uv run python -c "from evaluation_dictee.data.loaders import load_labels; \
print(len(load_labels('s3://projet-production-ecrits-depp/resultat_dictee_2015.csv')), 'copies')"

# modèle accessible ?
uv run python -c "from openai import OpenAI; from evaluation_dictee.config import Secrets; \
s=Secrets(); c=OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key); \
print(c.chat.completions.create(model='gemma4-26b-moe', \
messages=[{'role':'user','content':'Dis bonjour'}], max_tokens=10).choices[0].message.content)"
```

### 3. Lancer un benchmark

**Commande de base** (test rapide, terminal foreground) :
```bash
uv run scripts/run_benchmark.py --config configs/dictee_gemma4_zeroshot.yaml
```

Cela produit `data/processed/dictee_gemma4_zeroshot_predictions.jsonl`
(une ligne par item × copie) et journalise les métriques dans MLflow.

**Pour un run complet (long)**, utiliser une session détachable — voir la section
« Runs longs (screen / nohup) » plus bas dans ce README.

**Reprise après crash.** Le benchmark écrit sur disque après CHAQUE copie et
reprend automatiquement où il s'était arrêté : si un run est interrompu
(déconnexion, erreur API, kernel tué), il suffit de **relancer la même commande**
et il saute les copies déjà traitées. Voir « Runs longs » pour les détails.

**Pour l'évaluation de la transcription (HTR)** sur Scoledit :

```bash
python scripts/run_htr_benchmark.py --config configs/htr_gemma4_scoledit.yaml
```

Cela produit `data/processed/htr_gemma4_scoledit_htr_predictions.jsonl` et affiche
le CER/WER moyens. Analyse dans `notebooks/05_analyse_transcription_htr.ipynb`.

**Pour le fine-tuning** d'un modèle de transcription (nécessite un GPU H100) :
```bash
python scripts/finetune_htr_scoledit.py --config configs/finetune_htr_gemma4.yaml
```
Voir la documentation détaillée dans le script pour les prérequis d'installation
(`unsloth`, `trl`, `bitsandbytes`).

### 4. Analyser les résultats

Trois notebooks, à ouvrir dans **`notebooks/`** et à exécuter cellule par cellule :

| Notebook | Ce qu'il fait | Prérequis |
|----------|---------------|-----------|
| `03_analyse_resultats.ipynb` | métriques globales, prévalence par item avec IC bootstrap, distributions, corrélation modèle vs expert, seuils critiques. Export HTML sélectif pour la DEPP. | un run de benchmark terminé |
| `04_diagnostic.ipynb` | table des copies triées par désaccord, HTML des N pires copies, HTML d'une copie précise par ID (scan + transcription + comparaison expert/modèle) | un run de benchmark terminé |
| `05_analyse_transcription_htr.ipynb` | CER/WER, distribution du CER, HTML des N pires transcriptions et N aléatoires | un run HTR terminé |

Dans chaque notebook, il suffit de changer la variable `RUN_NAME` en tête pour
analyser un autre run — aucun besoin de relancer le benchmark.

**Générer un rapport HTML pour la DEPP** (à partir du notebook 03) : exécuter la
section « 9. Export HTML pour l'équipe DEPP », choisir les sections à inclure,
et le fichier `data/processed/rapport_depp_<RUN>.html` est autonome (assets
inlinés) prêt à envoyer par mail.

---

## Runs longs (screen / nohup)

Un benchmark complet sur 3469 copies × ~30 s prend ~30 h. **Ne jamais lancer un
tel run dans le terminal du navigateur sans protection** : au moindre plantage
réseau, mise en veille, fermeture d'onglet, le processus est tué. Le
checkpointing sauvera les prédictions déjà écrites, mais pas la copie en cours.

> **Note Onyxia** : `tmux` n'est pas disponible dans les services vscode-python
> du SSP Cloud (`sudo apt-get install tmux` échoue avec « No installation
> candidate »). Utiliser `screen` (Option A) ou `nohup` (Option B).

### Avant tout : créer le dossier logs

```bash
# À faire une seule fois (nohup échoue si le dossier n'existe pas) :
mkdir -p logs
```

### Option A — screen (recommandé sur Onyxia, généralement disponible)

```bash
# Vérifier la disponibilité :
which screen && echo "OK" || echo "absent"

# Créer une session détachable et lancer le run :
screen -S dictee
python scripts/run_benchmark.py --config configs/dictee_gemma4_cot.yaml

# Détacher :         Ctrl+A  puis  D    (le job continue en arrière-plan)
# Rattacher :        screen -r dictee
# Lister sessions :  screen -ls
# Tuer une session : screen -X -S dictee quit
```

### Option B — nohup (toujours disponible, sans interface interactive)

```bash
mkdir -p logs    # créer le dossier si pas encore fait

nohup python scripts/run_benchmark.py --config configs/dictee_gemma4_cot.yaml \
      > logs/dictee_gemma4_cot.log 2>&1 &
echo $! > logs/dictee_gemma4_cot.pid    # noter le PID pour arrêter plus tard

# Suivre le log en direct :
tail -f logs/dictee_gemma4_cot.log

# Vérifier que le process tourne :
ps -p $(cat logs/dictee_gemma4_cot.pid)

# Arrêter proprement (le checkpointing sauvegardera l'état) :
kill $(cat logs/dictee_gemma4_cot.pid)
```

### Surveillance de l'avancement

Pendant un run long, dans une **autre** fenêtre ou onglet, ces commandes donnent
un signal de vie plus fiable que la barre de progression :

```bash
# Compter les copies déjà traitées dans le JSONL (une copie = ~83 lignes) :
wc -l data/processed/dictee_gemma4_cot_predictions.jsonl

# Suivre le compteur en direct (mise à jour toutes les 5 s) :
watch -n 5 "wc -l data/processed/dictee_gemma4_cot_predictions.jsonl"

# Lister les copies en échec (à retenter au prochain lancement) :
cat data/processed/dictee_gemma4_cot_failed_copies.txt
```

### Reprise après crash — mode d'emploi

Le benchmark écrit sur disque après CHAQUE copie évaluée (avec `flush + fsync`).
Conséquences pratiques :

- **Crash ou déconnexion** : relance exactement la même commande. Les copies
  déjà présentes dans `<run>_predictions.jsonl` sont automatiquement sautées,
  et le run reprend à la copie suivante.
- **Erreurs API sur des copies isolées** : elles sont loggées dans
  `<run>_failed_copies.txt`, la copie fautive est sautée mais le run continue.
  Au prochain lancement, ces copies seront retentées.
- **Repartir de zéro** : supprimer `<run>_predictions.jsonl` (ou changer
  `config.name` dans le YAML).

---

## Configurer une expérience

Un fichier YAML dans `configs/` = une expérience reproductible. Exemple commenté :
[`configs/dictee_gemma4_zeroshot.yaml`](configs/dictee_gemma4_zeroshot.yaml).

| Champ | Rôle |
|-------|------|
| `model.name` | nom du modèle servi par llm.lab (ex. `gemma4-26b-moe`) |
| `data.images_path` | dossier des imagettes (local ou `s3://...`) |
| `data.labels_path` | CSV des codes de l'annotateur (local ou `s3://...`) |
| `data.grid_path` | grille de codage JSON (`configs/grille_dictee_2015.json`) |
| `data.limit` | nombre de copies (mettre `null` pour tout le corpus) |
| `grid.scheme` | `simplifiee` (1/9/0) ou `complete` (1/3/4/5/9/0) |
| `prompt.method` | `C` end-to-end (image → code), voir note méthodologique |
| `prompt.read_final_state` | règle des ratures : lire l'état final corrigé |

---

## Structure du dépôt

```
evaluation_dictee/
├── CLAUDE.md                  ← contexte du projet pour humains et IA
├── README.md                  ← ce fichier
├── pyproject.toml             ← dépendances + config ruff/mypy/pytest
├── configs/
│   ├── grille_dictee_2015.json   ← grille de codage (mot attendu + fautes connues)
│   ├── dictee_gemma4_zeroshot.yaml   ← approche end-to-end (1 étape)
│   ├── dictee_gemma4_2stages.yaml    ← approche 2 étapes (HTR + codage texte)
│   ├── dictee_gemma4_cot.yaml        ← 1 étape avec chain-of-thought
│   ├── htr_gemma4_scoledit.yaml      ← évaluation HTR sur corpus Scoledit
│   └── finetune_htr_gemma4.yaml      ← fine-tuning HTR (QLoRA)
├── src/evaluation_dictee/
│   ├── config.py              ← configs validées (Pydantic) + secrets (.env)
│   ├── data/                  ← chargement images (S3, TIFF 1 bit) + grille + labels
│   ├── models/
│   │   ├── base.py            ← interface Scorer + dataclasses de prédiction
│   │   ├── vlm.py             ← scorer end-to-end (approche 1 étape)
│   │   ├── two_stage.py       ← scorer 2 étapes (HTR puis codage texte)
│   │   └── factory.py         ← construit le bon scorer selon la config
│   ├── pipeline/
│   │   ├── prompts.py         ← construction des prompts (dictée + transcription)
│   │   ├── benchmark.py       ← boucle d'évaluation + checkpointing incrémental
│   │   └── alignment.py       ← ré-alignement anti-décalage (Needleman-Wunsch)
│   ├── evaluation/            ← metrics, statistics (bootstrap/Wilson),
│   │   │                        calibration (ECE), report (par item/copie)
│   │   ├── report.py, statistics.py, calibration.py, metrics.py
│   │   ├── diagnostics.py     ← analyse fine des désaccords
│   │   ├── visual_diff.py     ← HTML de comparaison expert / modèle
│   │   └── html_report.py     ← export HTML sélectif (rapport DEPP)
│   ├── transcription/         ← pipeline HTR indépendant (Scoledit)
│   │   ├── scoledit.py        ← loader TEI → texte brut (fautes préservées)
│   │   ├── htr_metrics.py     ← CER, WER (bruts et normalisés)
│   │   ├── htr_benchmark.py   ← run HTR + agrégation métriques
│   │   └── visual_diff.py     ← HTML des pires / N aléatoires
│   └── utils/                 ← logging, suivi MLflow
├── scripts/
│   ├── run_benchmark.py       ← point d'entrée CLI (approches 1 et 2 étapes)
│   ├── run_htr_benchmark.py   ← point d'entrée CLI pour l'évaluation HTR
│   └── finetune_htr_scoledit.py  ← fine-tuning HTR (nécessite un GPU H100)
├── notebooks/
│   ├── 03_analyse_resultats.ipynb   ← analyse statistique + export DEPP
│   ├── 04_diagnostic.ipynb          ← inspection copie par copie
│   └── 05_analyse_transcription_htr.ipynb   ← analyse HTR
├── tests/                     ← 83 tests unitaires (pytest)
└── docs/                      ← note méthodologique, cadrage, décisions, grille
```

---

## Bonnes pratiques développement

Avant chaque commit :

```bash
ruff format src tests scripts        # formatage automatique
ruff check src tests scripts         # lint (attrape les erreurs courantes)
pytest                               # lance toute la suite de tests
pytest tests/test_alignment.py -v    # tester UN fichier précis
pytest -k "chain_of_thought"         # tests dont le nom matche un motif
```

**Ne jamais committer** les données (`data/`), les checkpoints (`checkpoints/`),
les logs (`logs/`) ni les fichiers `.env` : ils sont dans le `.gitignore`.

## Toutes les commandes en un coup d'œil

Section de référence rapide. Chaque commande est détaillée plus haut dans le
README, avec ses prérequis et son contexte d'usage.

```bash
# ─────────── Installation & configuration (une seule fois) ───────────
uv sync                                          # environnement Python
# ou : pip install -e ".[dev]"
cp .env.example .env && nano .env                # renseigner LLM_API_KEY et S3

# ─────────── Vérifier que tout marche ───────────
python -c "from evaluation_dictee.data.loaders import load_labels; \
    print(len(load_labels('s3://projet-production-ecrits-depp/resultat_dictee_2015.csv')), 'copies')"
pytest -q                                        # lancer les tests

# ─────────── Benchmarks : approches 1 et 2 étapes ───────────
python scripts/run_benchmark.py --config configs/dictee_gemma4_zeroshot.yaml
python scripts/run_benchmark.py --config configs/dictee_gemma4_2stages.yaml
python scripts/run_benchmark.py --config configs/dictee_gemma4_cot.yaml   # chain-of-thought

# ─────────── Évaluation de la transcription HTR (Scoledit) ───────────
python scripts/run_htr_benchmark.py --config configs/htr_gemma4_scoledit.yaml

# ─────────── Fine-tuning HTR (GPU H100 requis) ───────────
python scripts/finetune_htr_scoledit.py --config configs/finetune_htr_gemma4.yaml

# ─────────── Runs longs (session détachable) ───────────
mkdir -p logs                                    # toujours créer d'abord

# Option A : screen (recommandé sur Onyxia, généralement disponible)
which screen && screen -S dictee                 # puis Ctrl+A D pour détacher
                                                 # screen -r dictee pour rattacher

# Option B : nohup (toujours dispo, sans interface interactive)
nohup python scripts/run_benchmark.py --config configs/dictee_gemma4_cot.yaml \
      > logs/dictee_gemma4_cot.log 2>&1 &

# ─────────── Surveillance d'un run en cours ───────────
tail -f logs/dictee_gemma4_cot.log
watch -n 5 "wc -l data/processed/dictee_gemma4_cot_predictions.jsonl"

# ─────────── Analyse des résultats ───────────
jupyter lab notebooks/03_analyse_resultats.ipynb   # analyse statistique + export DEPP
jupyter lab notebooks/04_diagnostic.ipynb          # inspection copie par copie
jupyter lab notebooks/05_analyse_transcription_htr.ipynb   # analyse HTR

# ─────────── Qualité de code (avant tout commit) ───────────
ruff format src tests scripts
ruff check src tests scripts
pytest
```

---

## ⚠️ Données sensibles

Les copies sont des **écritures d'élèves mineurs**. Elles **ne quittent jamais le
SSP Cloud** et **ne sont jamais commitées**. Le dossier `/data/` est ignoré par Git
(voir [`.gitignore`](.gitignore)) ; seul le `.env.example` (sans secret) est versionné.