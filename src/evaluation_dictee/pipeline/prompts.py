"""Construction des prompts d'évaluation.

Les consignes encodent les décisions méthodologiques du projet (CLAUDE.md §3-4) :
grille simplifiée, règle des ratures (état final), consigne anti-sur-correction,
demande de confiance par item, et surtout ALIGNEMENT STRICT mot à mot.

Point critique d'alignement : la grille découpe les élisions (n', S', l', d', qu')
comme des items SÉPARÉS du mot qui suit. Sans consigne explicite, le modèle recolle
ces unités (« n'étaient » au lieu de « n' » + « étaient »), ce qui décale tous les
items suivants et fait chuter l'accord. Le prompt fournit donc la correspondance
explicite item → mot attendu, dérivée de la grille.

Versionnement Langfuse
----------------------
Chaque prompt est un template « chat » (message system + message user) versionné
dans Langfuse. Le template est le MASQUE : il contient des placeholders
`{{variable}}` remplis à l'exécution. La logique conditionnelle (schéma de codage,
flags de PromptConfig) reste ICI, dans le code : elle décide de la VALEUR injectée
dans chaque variable (texte d'un bloc optionnel, ou chaîne vide). On garde donc un
seul prompt par fonction, pas une version par combinaison de flags.

Les templates ci-dessous servent à la fois de source poussée vers Langfuse
(voir `utils/add_langfuse_prompt.py`) et de REPLI hors-ligne : si Langfuse est
indisponible (tests, incident), on compile la structure locale à l'identique.
"""

from __future__ import annotations

from typing import cast

from langfuse import get_client
from langfuse.model import ChatMessageDict

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.reference import GridItem

# Noms sous lesquels les prompts sont versionnés dans Langfuse.
PROMPT_DICTATION = "Dictation"
PROMPT_TRANSCRIPTION = "Transcription"
PROMPT_TEXT_CODING = "Text coding"

_GRILLE_SIMPLIFIEE = (
    "Pour chaque item, attribue un code :\n"
    '- "1" : le mot (ou la ponctuation) attendu est présent et correctement orthographié ;\n'
    '- "9" : le mot est présent mais comporte une erreur (orthographe, accord, '
    "conjugaison, accent, ponctuation erronée...) ;\n"
    "- \"0\" : le mot attendu est absent (l'élève ne l'a pas écrit)."
)

_GRILLE_COMPLETE = (
    "Pour chaque item, attribue un code.\n"
    "Pour un MOT :\n"
    '- "1" : correctement orthographié ;\n'
    '- "3" : erreur LEXICALE (n\'altère pas la prononciation : accent, lettre muette, '
    "mauvais graphème — ex. « soire » pour « soir ») ;\n"
    '- "4" : erreur GRAMMATICALE (accord, conjugaison, confusion de catégorie comme '
    "« on »/« ont », ou toute erreur qui change la prononciation) ;\n"
    '- "5" : erreur À LA FOIS lexicale ET grammaticale ;\n'
    "- \"0\" : le mot est absent (l'élève ne l'a pas écrit).\n"
    'Pour la PONCTUATION : "1" correcte, "9" erronée, "0" absente.'
)

_CONSIGNE_ALIGNEMENT = (
    "2 - RÈGLE D'ALIGNEMENT (la plus importante) : la liste de règles ci-dessous définit des items "
    "FIXES, un par ligne, dans l'ordre du texte. Tu dois rendre EXACTEMENT un code par "
    "item, dans le même ordre, sans en fusionner ni en omettre:\n"
    "a - Aligne-toi sur le MOT ATTENDU de chaque item, JAMAIS sur ta propre découpe de "
    "l'écriture de l'élève. L'élève peut écrire un mot avec un espace au milieu "
    "(« re trouver » au lieu de « retrouver ») ou coller deux mots (« nousles » au "
    "lieu de « nous les ») : ne te laisse pas décaler. Dans ces cas, rattache ce que "
    "tu lis au mot attendu correspondant, et continue d'aligner les items suivants "
    "sur leurs mots attendus respectifs.\n"
    "b - Les apostrophes d'élision sont des items SÉPARÉS du mot qui suit. Par exemple "
    "« n'étaient » se code en DEUX items distincts : « n' » puis « étaient ». "
    "De même « S'ils » = « S' » puis « ils » ; « l'olivier » = « l' » puis « olivier ».\n"
    "c - AVANT de coder, vérifie pour chaque item que la transcription que tu donnes "
    "correspond bien au mot attendu de CE numéro d'item ; si tu remarques un décalage, "
    "recale-toi immédiatement sur le mot attendu. Le nombre d'items de ta réponse doit "
    "être EXACTEMENT celui demandé.\n"
)

_CONSIGNE_FIDELITE = (
    "3 - Transcris EXACTEMENT ce que l'élève a écrit pour cet item, fautes comprises. "
    "Ne corrige jamais silencieusement l'orthographe : une faute non transcrite "
    "fausse l'évaluation. Si le mot écrit diffère du mot attendu, c'est une erreur (9).\n"
)

_CONSIGNE_COMPARAISON = (
    "4 - MÉTHODE DE CODAGE (à appliquer pour chaque item) : compare LETTRE À LETTRE ta "
    "transcription au mot attendu. Le code est 1 SEULEMENT si les deux sont rigoureusement "
    "identiques (mêmes lettres, mêmes accents, même terminaison). La MOINDRE différence, "
    "une lettre, un accent, une terminaison de conjugaison (ex. « mis » au lieu de "
    "« mit ») ou un singulier/pluriel, impose le code 9, même si le mot reste lisible et "
    "plausible. Ne te fie pas au sens : « mis » et « mit » se prononcent pareil mais "
    "l'un est faux. Code d'après la forme écrite exacte, pas d'après ce que l'élève "
    "voulait dire.\n"
)

_CONSIGNE_RATURES_HALLUCINATION = (
    "5 - Quand un passage est raturé/barré : ignore complètement le texte barré et lis "
    "uniquement ce que l'élève a retenu en version finale, DANS L'ORDRE OÙ C'EST ÉCRIT "
    "sur la copie. N'invente pas, ne réordonne pas les mots pour qu'ils collent au texte "
    "attendu : si l'élève a écrit les mots dans un certain ordre, transcris cet ordre réel.\n"
)

_CONSIGNE_RATURES = (
    "6 - Si l'élève a raturé puis réécrit un mot, lis uniquement la version FINALE "
    "(corrigée par l'élève), pas la version barrée.\n"
)

_CONSIGNE_CONFIANCE = (
    "7 - Pour chaque item, fournis un score de confiance entre 0 et 1 reflétant ta "
    "certitude (lisibilité, ambiguïté). Un score bas déclenchera une relecture humaine.\n"
)

_CONSIGNE_COT = (
    "8 - AVANT de choisir le code, écris un champ « comparaison » qui décrit "
    "explicitement en quoi la transcription diffère du mot attendu (ou "
    "précise « identique » si elles correspondent lettre à lettre). "
    "Exemple : attendu « inquiets » lu « inquiet » → « il manque le 's' final ».\n"
)

# Format de sortie JSON de la méthode C, avec ou sans champ « comparaison » (CoT).
_FORMAT_ITEMS_COT = (
    "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :\n"
    '{"items": [{"item_id": "...", "transcription": "ce que l\'élève a écrit", '
    '"comparaison": "identique" OU description brève de la différence, '
    '"code": "1", "confidence": 0.95}, ...]}'
)
_FORMAT_ITEMS_SIMPLE = (
    "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :\n"
    '{"items": [{"item_id": "...", "transcription": "ce que l\'élève a écrit", '
    '"code": "1", "confidence": 0.95}, ...]}\n'
)

# Consigne « ratures » propre à l'étape de transcription (formulation dédiée).
_CONSIGNE_RATURES_TRANSCRIPTION = (
    "Si l'élève a raturé puis réécrit, transcris uniquement la version "
    "FINALE (non barrée). Ignore complètement le texte barré.\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Templates chat (masques versionnés dans Langfuse). Les {{...}} sont remplis
# par les fonctions build_* selon leurs inputs. Ces mêmes structures servent de
# repli hors-ligne.
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE_DICTATION: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu fais partie d'un groupe d'expert composé de professeurs et de formateurs pour une "
            "évaluation nationale de dictée.\n"
            "Ta tâche : On te montre l'image manuscrite de la dictée d'un élève de primaire. "
            "Tu dois noter chaque item de la dictée à l'aide des éléments définis dans la grille de notation.\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Tu dois noter chaque item de la dictée avec la grille de notation disponible : "
            "{{grille}}\n"
            + _CONSIGNE_ALIGNEMENT
            + "{{consignes_optionnelles}}\n\n"
            + _CONSIGNE_CONFIANCE
            + "{{consigne_cot}}"
        ),
    },
    {
        "role": "user",
        "content": (
            "# Texte de référence (ce que l'élève devait écrire) :\n"
            "« {{reference_text}} »\n\n"
            "# Items à coder, dans l'ordre. Chaque ligne = un item fixe "
            "« identifiant → mot attendu » :\n"
            "{{items_list}}\n\n"
            "# Tu dois rendre EXACTEMENT {{n_items}} items, dans cet ordre.\n"
            "{{format_sortie}}"
        ),
    },
]

_TEMPLATE_TRANSCRIPTION: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu es un expert en lecture d'écriture manuscrite d'enfants.\n"
            "On te montre l'image manuscrite de la dictée d'un élève de primaire.\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Transcris EXACTEMENT le texte écrit par l'élève, mot pour mot, "
            "FAUTES D'ORTHOGRAPHE COMPRISES.\n "
            "2 - Ne corrige rien, ne complète rien, "
            "ne réordonne rien.\n "
            " 3 - Reproduis fidèlement les erreurs, y compris les "
            "accents manquants, les mots mal orthographiés et la ponctuation.\n"
            "4 - Respecte l'ordre exact des items écrits sur la copie (mots, ponctuation, chiffres, ...).\n"
            "{{consigne_ratures}}"
        ),
    },
    {
        "role": "user",
        "content": (
            "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :\n"
            '{"transcription": "le texte exact écrit par l\'élève"}'
        ),
    },
]

_TEMPLATE_TEXT_CODING: list[ChatMessageDict] = [
    {
        "role": "system",
        "content": (
            "Tu fais partie d'un groupe d'expert, composé de professeur et de formateurs pour une évaluation nationale de dictée.\n"
            "Ta tâche : Tu ne vois PAS l'image, on te donne uniquement la transcription de ce que l'élève "
            "a écrit (produite par un système de lecture), et le texte de référence.\n"
            "Compare la transcription au mot attendu de chaque item. Si un mot attendu "
            "n'apparaît pas dans la transcription, code-le 0 (absent).\n\n"
            "Règles à respecter impérativement :\n"
            "1 - Tu dois noter chaque item de la dictée avec la grille de notation disponible ci-dessous : "
            "{{grille}}\n\n" + _CONSIGNE_ALIGNEMENT + "\n\n" + _CONSIGNE_COMPARAISON + "\n\n"
            
        ),
    },
    {
        "role": "user",
        "content": (
            "Texte de référence (ce que l'élève devait écrire) :\n"
            "« {{reference_text}} »\n\n"
            "Transcription de la copie de l'élève (fautes comprises) :\n"
            "« {{transcription}} »\n\n"
            "Items à coder, dans l'ordre. Chaque ligne = un item "
            "« identifiant → mot attendu » :\n"
            "{{items_list}}\n\n"
            "Tu dois rendre EXACTEMENT {{n_items}} items, dans cet ordre.\n"
            "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :\n"
            '{"items": [{"item_id": "...", "transcription": "mot lu pour cet item", '
            '"code": "1", "confidence": 0.95}, ...]}'
        ),
    },
]

# Registre exposé pour l'initialisation Langfuse (utils/add_langfuse_prompt.py).
PROMPT_TEMPLATES: dict[str, list[ChatMessageDict]] = {
    PROMPT_DICTATION: _TEMPLATE_DICTATION,
    PROMPT_TRANSCRIPTION: _TEMPLATE_TRANSCRIPTION,
    PROMPT_TEXT_CODING: _TEMPLATE_TEXT_CODING,
}


def _format_items(items: list[GridItem]) -> str:
    """Formate la liste des items « N. identifiant → « mot » (nature) », un par ligne."""
    lignes = []
    for idx, it in enumerate(items, 1):
        nature = "ponctuation" if it.type == "ponctuation" else "mot"
        lignes.append(f"  {idx:>2}. {it.item_id} → « {it.attendu} » ({nature})")
    return "\n".join(lignes)


def _render_local(
    messages: list[ChatMessageDict], variables: dict[str, object]
) -> list[ChatMessageDict]:
    """Compile un template localement (repli si Langfuse est indisponible)."""
    rendered: list[ChatMessageDict] = []
    for message in messages:
        content = message["content"]
        for key, value in variables.items():
            content = content.replace("{{" + key + "}}", str(value))
        rendered.append({"role": message["role"], "content": content})
    return rendered


def _compile_prompt(
    name: str,
    fallback: list[ChatMessageDict],
    variables: dict[str, object],
) -> str:
    """Récupère un prompt chat versionné dans Langfuse, le compile et l'aplatit.

    Le prompt est récupéré via son label de production. `fallback` est la structure
    locale (identique à celle poussée dans Langfuse) : elle sert de repli si le
    prompt n'existe pas encore ou si Langfuse est indisponible (tests, incident).
    Le résultat est aplati en une seule chaîne car les appelants envoient le prompt
    dans un unique bloc de texte, souvent accompagné de l'image.

    Args:
        name: nom du prompt dans Langfuse.
        fallback: messages du template local (repli et source de vérité hors-ligne).
        variables: valeurs à injecter dans les placeholders {{...}}.

    Returns:
        Le prompt compilé, messages system + user concaténés.
    """
    try:
        prompt = get_client().get_prompt(name, type="chat", fallback=fallback)
        messages = prompt.compile(**variables)
    except Exception:  # pragma: no cover - repli défensif si Langfuse indisponible
        messages = _render_local(fallback, variables)
    # compile() peut renvoyer des placeholders de message (sans "content") ; nos
    # prompts n'en contiennent pas, on aplatit donc simplement le champ texte.
    plain = cast("list[dict[str, object]]", messages)
    return "\n\n".join(str(m["content"]) for m in plain if m.get("content"))


def build_dictation_prompt(
    reference_text: str,
    items: list[GridItem],
    config: PromptConfig,
    scheme: str = "simplifiee",
) -> str:
    """Construit le prompt d'évaluation d'une dictée (méthode C, end-to-end).

    Charge le template versionné « dictee_methode_c » depuis Langfuse et le remplit
    selon les inputs : les blocs conditionnels (fidélité, ratures, chain-of-thought)
    et le schéma de codage déterminent la valeur injectée dans chaque variable.

    Args:
        reference_text: texte exact de la dictée attendue.
        items: items de la grille (id + mot attendu + type), dans l'ordre du texte.
        config: stratégie de prompting (fidélité, ratures, few-shot...).
        scheme: "simplifiee" (codes 1/9/0) ou "complete" (codes 1/3/4/5/9/0).
            Détermine les consignes de codage données au modèle, pour que celui-ci
            code dans le MÊME jeu de modalités que les experts après normalisation.

    Returns:
        Le prompt complet à envoyer au modèle.
    """
    grille = _GRILLE_COMPLETE if scheme == "complete" else _GRILLE_SIMPLIFIEE

    blocs: list[str] = []
    if config.enforce_faithful:
        blocs += [_CONSIGNE_FIDELITE, _CONSIGNE_COMPARAISON]
    if config.read_final_state:
        blocs += [_CONSIGNE_RATURES, _CONSIGNE_RATURES_HALLUCINATION]
    consignes_optionnelles = ("\n\n" + "\n\n".join(blocs)) if blocs else ""

    if config.chain_of_thought:
        # Le champ "comparaison" force le modèle à VERBALISER la différence
        # lue-attendue avant de choisir le code, ce qui rend le raisonnement
        # inspectable et réduit les erreurs bien lues mais mal codées.
        consigne_cot = "\n\n" + _CONSIGNE_COT
        format_sortie = _FORMAT_ITEMS_COT
    else:
        consigne_cot = ""
        format_sortie = _FORMAT_ITEMS_SIMPLE

    return _compile_prompt(
        PROMPT_DICTATION,
        fallback=_TEMPLATE_DICTATION,
        variables={
            "grille": grille,
            "consignes_optionnelles": consignes_optionnelles,
            "consigne_cot": consigne_cot,
            "reference_text": reference_text,
            "items_list": _format_items(items),
            "n_items": len(items),
            "format_sortie": format_sortie,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompts pour l'APPROCHE 1 (deux étapes) : transcription HTR puis codage textuel
# ─────────────────────────────────────────────────────────────────────────────


def build_transcription_prompt(read_final_state: bool = True) -> str:
    """Prompt de l'ÉTAPE 1 (HTR) : transcrire fidèlement l'image, sans coder.

    Charge le template versionné « dictee_transcription » depuis Langfuse. Le modèle
    ne reçoit PAS le texte de référence : on veut une lecture brute, non biaisée par
    ce qui était attendu. Il restitue exactement ce que l'élève a écrit, fautes
    comprises.

    Args:
        read_final_state: si True, consigne de lire l'état final en cas de rature.

    Returns:
        Le prompt de transcription.
    """
    consigne_ratures = ("\n" + _CONSIGNE_RATURES_TRANSCRIPTION) if read_final_state else ""
    return _compile_prompt(
        PROMPT_TRANSCRIPTION,
        fallback=_TEMPLATE_TRANSCRIPTION,
        variables={"consigne_ratures": consigne_ratures},
    )


def build_text_coding_prompt(
    reference_text: str,
    transcription: str,
    items: list[GridItem],
    scheme: str = "simplifiee",
) -> str:
    """Prompt de l'ÉTAPE 2 : coder à partir du TEXTE transcrit (sans image).

    Charge le template versionné « dictee_codage_texte » depuis Langfuse. Cette étape
    ne prend que du texte en entrée : la transcription produite à l'étape 1 et le
    texte de référence. Elle peut donc être confiée à un modèle purement textuel.

    Args:
        reference_text: texte de référence de la dictée.
        transcription: transcription produite à l'étape 1.
        items: items de la grille (mot attendu, dans l'ordre).
        scheme: "simplifiee" ou "complete".

    Returns:
        Le prompt de codage textuel.
    """
    grille = _GRILLE_COMPLETE if scheme == "complete" else _GRILLE_SIMPLIFIEE
    return _compile_prompt(
        PROMPT_TEXT_CODING,
        fallback=_TEMPLATE_TEXT_CODING,
        variables={
            "grille": grille,
            "reference_text": reference_text,
            "transcription": transcription,
            "items_list": _format_items(items),
            "n_items": len(items),
        },
    )
