"""Initialise / met à jour les prompts d'évaluation dans Langfuse.

Pousse les trois templates chat (message system + message user) définis dans
`pipeline/prompts.py` vers le gestionnaire de prompts de Langfuse. Chaque appel
crée une NOUVELLE version de chaque prompt et la promeut en production via le
label "production".

Les variables entre doubles accolades ({{reference_text}}, {{grille}}, ...) sont
des placeholders Langfuse. Elles sont remplies à l'exécution par les fonctions
build_* de prompts.py via compile(). La logique conditionnelle (schéma de codage,
flags de PromptConfig) reste dans le code : elle choisit la valeur injectée dans
chaque variable (texte d'un bloc optionnel, ou chaîne vide). On garde donc un seul
prompt versionné par fonction, pas une version par combinaison de flags.

Prérequis : LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY définis dans
l'environnement (ou le fichier .env). Voir .env.example.

Usage :
    uv run add-langfuse-prompt
"""

from __future__ import annotations

from langfuse import get_client

from evaluation_dictee.pipeline.prompts import PROMPT_TEMPLATES
from evaluation_dictee.utils.logging import get_logger

logger = get_logger(__name__)


def push_prompts() -> None:
    """Crée/met à jour chaque template dans Langfuse et le promeut en production."""
    langfuse = get_client()

    for name, messages in PROMPT_TEMPLATES.items():
        prompt = langfuse.create_prompt(
            name=name,
            type="chat",
            prompt=list(messages),
            labels=["production"],  # promeut cette version comme version de prod
        )
        logger.info(
            "Prompt « %s » poussé (version %s), labels : %s",
            prompt.name,
            prompt.version,
            prompt.labels,
        )

    # Langfuse envoie ses requêtes de façon asynchrone : flush explicite avant de
    # terminer le script, sinon les créations peuvent être perdues.
    langfuse.flush()


if __name__ == "__main__":
    push_prompts()
