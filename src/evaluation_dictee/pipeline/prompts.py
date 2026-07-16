"""Construction des prompts d'évaluation.

Les consignes encodent les décisions méthodologiques du projet (CLAUDE.md §3-4) :
grille simplifiée, règle des ratures (état final), consigne anti-sur-correction,
demande de confiance par item, et surtout ALIGNEMENT STRICT mot à mot.

Point critique d'alignement : la grille découpe les élisions (n', S', l', d', qu')
comme des items SÉPARÉS du mot qui suit. Sans consigne explicite, le modèle recolle
ces unités (« n'étaient » au lieu de « n' » + « étaient »), ce qui décale tous les
items suivants et fait chuter l'accord. Le prompt fournit donc la correspondance
explicite item → mot attendu, dérivée de la grille.
"""

from __future__ import annotations

from evaluation_dictee.config import PromptConfig
from evaluation_dictee.data.reference import GridItem

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
    "RÈGLE D'ALIGNEMENT (la plus importante) : la liste ci-dessous définit des items "
    "FIXES, un par ligne, dans l'ordre du texte. Tu dois rendre EXACTEMENT un code par "
    "item, dans le même ordre, sans en fusionner ni en omettre.\n"
    "Aligne-toi sur le MOT ATTENDU de chaque item, JAMAIS sur ta propre découpe de "
    "l'écriture de l'élève. L'élève peut écrire un mot avec un espace au milieu "
    "(« re trouver » au lieu de « retrouver ») ou coller deux mots (« nousles » au "
    "lieu de « nous les ») : ne te laisse pas décaler. Dans ces cas, rattache ce que "
    "tu lis au mot attendu correspondant, et continue d'aligner les items suivants "
    "sur leurs mots attendus respectifs.\n"
    "Les apostrophes d'élision sont des items SÉPARÉS du mot qui suit. Par exemple "
    "« n'étaient » se code en DEUX items distincts : « n' » puis « étaient ». "
    "De même « S'ils » = « S' » puis « ils » ; « l'olivier » = « l' » puis « olivier ».\n"
    "AVANT de coder, vérifie pour chaque item que la transcription que tu donnes "
    "correspond bien au mot attendu de CE numéro d'item ; si tu remarques un décalage, "
    "recale-toi immédiatement sur le mot attendu. Le nombre d'items de ta réponse doit "
    "être EXACTEMENT celui demandé."
)

_CONSIGNE_FIDELITE = (
    "Transcris EXACTEMENT ce que l'élève a écrit pour cet item, fautes comprises. "
    "Ne corrige jamais silencieusement l'orthographe : une faute non transcrite "
    "fausse l'évaluation. Si le mot écrit diffère du mot attendu, c'est une erreur (9)."
)

_CONSIGNE_COMPARAISON = (
    "MÉTHODE DE CODAGE (à appliquer pour chaque item) : compare LETTRE À LETTRE ta "
    "transcription au mot attendu. Le code est 1 SEULEMENT si les deux sont rigoureusement "
    "identiques (mêmes lettres, mêmes accents, même terminaison). La MOINDRE différence "
    "— une lettre, un accent, une terminaison de conjugaison (ex. « mis » au lieu de "
    "« mit »), un singulier/pluriel — impose le code 9, même si le mot reste lisible et "
    "plausible. Ne te fie pas au sens : « mis » et « mit » se prononcent pareil mais "
    "l'un est faux. Code d'après la forme écrite exacte, pas d'après ce que l'élève "
    "voulait dire."
)

_CONSIGNE_RATURES_HALLUCINATION = (
    "Quand un passage est raturé/barré : ignore complètement le texte barré et lis "
    "uniquement ce que l'élève a retenu en version finale, DANS L'ORDRE OÙ C'EST ÉCRIT "
    "sur la copie. N'invente pas, ne réordonne pas les mots pour qu'ils collent au texte "
    "attendu : si l'élève a écrit les mots dans un certain ordre, transcris cet ordre réel."
)

_CONSIGNE_RATURES = (
    "Si l'élève a raturé puis réécrit un mot, lis uniquement la version FINALE "
    "(corrigée par l'élève), pas la version barrée."
)

_CONSIGNE_CONFIANCE = (
    "Pour chaque item, fournis un score de confiance entre 0 et 1 reflétant ta "
    "certitude (lisibilité, ambiguïté). Un score bas déclenchera une relecture humaine."
)


def build_dictation_prompt(
    reference_text: str,
    items: list[GridItem],
    config: PromptConfig,
    scheme: str = "simplifiee",
) -> str:
    """Construit le prompt d'évaluation d'une dictée (méthode C, end-to-end).

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
    parts: list[str] = [
        "Tu es correcteur expert pour une évaluation nationale de dictée.",
        "On te montre l'image manuscrite de la dictée d'un élève.",
        "",
        "Texte de référence (ce que l'élève devait écrire) :",
        f"« {reference_text} »",
        "",
        grille,
        "",
        _CONSIGNE_ALIGNEMENT,
    ]

    if config.enforce_faithful:
        parts += ["", _CONSIGNE_FIDELITE, "", _CONSIGNE_COMPARAISON]
    if config.read_final_state:
        parts += ["", _CONSIGNE_RATURES, "", _CONSIGNE_RATURES_HALLUCINATION]
    parts += ["", _CONSIGNE_CONFIANCE]

    # Correspondance explicite item -> mot/ponctuation attendu : c'est ce qui
    # ancre l'alignement et empêche le modèle de recoller les élisions.
    parts += [
        "",
        "Items à coder, dans l'ordre. Chaque ligne = un item fixe « identifiant → mot attendu » :",
    ]
    for idx, it in enumerate(items, 1):
        nature = "ponctuation" if it.type == "ponctuation" else "mot"
        parts.append(f"  {idx:>2}. {it.item_id} → « {it.attendu} » ({nature})")

    parts += [
        "",
        f"Tu dois rendre EXACTEMENT {len(items)} items, dans cet ordre.",
    ]

    if config.chain_of_thought:
        # Le champ "comparaison" force le modèle à VERBALISER la différence
        # lue-attendue avant de choisir le code, ce qui rend le raisonnement
        # inspectable et réduit les erreurs bien lues mais mal codées.
        parts += [
            "AVANT de choisir le code, écris un champ « comparaison » qui décrit "
            "explicitement en quoi la transcription diffère du mot attendu (ou "
            "précise « identique » si elles correspondent lettre à lettre). "
            "Exemple : attendu « inquiets » lu « inquiet » → « il manque le 's' final ».",
            "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :",
            '{"items": [{"item_id": "...", "transcription": "ce que l\'élève a écrit", '
            '"comparaison": "identique" OU description brève de la différence, '
            '"code": "1", "confidence": 0.95}, ...]}',
        ]
    else:
        parts += [
            "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :",
            '{"items": [{"item_id": "...", "transcription": "ce que l\'élève a écrit", '
            '"code": "1", "confidence": 0.95}, ...]}',
        ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts pour l'APPROCHE 1 (deux étapes) : transcription HTR puis codage textuel
# ─────────────────────────────────────────────────────────────────────────────


def build_transcription_prompt(read_final_state: bool = True) -> str:
    """Prompt de l'ÉTAPE 1 (HTR) : transcrire fidèlement l'image, sans coder.

    Le modèle ne reçoit PAS le texte de référence : on veut une lecture brute,
    non biaisée par ce qui était attendu, pour mesurer la transcription pour
    elle-même. Il restitue exactement ce que l'élève a écrit, fautes comprises.

    Args:
        read_final_state: si True, consigne de lire l'état final en cas de rature.

    Returns:
        Le prompt de transcription.
    """
    parts = [
        "Tu es un expert en lecture d'écriture manuscrite d'enfants.",
        "On te montre l'image manuscrite de la dictée d'un élève de primaire.",
        "",
        "Transcris EXACTEMENT le texte écrit par l'élève, mot pour mot, "
        "FAUTES D'ORTHOGRAPHE COMPRISES. Ne corrige rien, ne complète rien, "
        "ne réordonne rien. Reproduis fidèlement les erreurs, y compris les "
        "accents manquants, les mots mal orthographiés et la ponctuation.",
        "Respecte l'ordre exact d'écriture sur la copie.",
    ]
    if read_final_state:
        parts.append(
            "Si l'élève a raturé puis réécrit, transcris uniquement la version "
            "FINALE (non barrée). Ignore complètement le texte barré."
        )
    parts += [
        "",
        "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :",
        '{"transcription": "le texte exact écrit par l\'élève"}',
    ]
    return "\n".join(parts)


def build_text_coding_prompt(
    reference_text: str,
    transcription: str,
    items: list[GridItem],
    scheme: str = "simplifiee",
) -> str:
    """Prompt de l'ÉTAPE 2 : coder à partir du TEXTE transcrit (sans image).

    Cette étape ne prend que du texte en entrée : la transcription produite à
    l'étape 1 et le texte de référence. Elle peut donc être confiée à un modèle
    purement textuel (panel plus large, moins coûteux). Le modèle aligne la
    transcription sur les items attendus et attribue un code par item.

    Args:
        reference_text: texte de référence de la dictée.
        transcription: transcription produite à l'étape 1.
        items: items de la grille (mot attendu, dans l'ordre).
        scheme: "simplifiee" ou "complete".

    Returns:
        Le prompt de codage textuel.
    """
    grille = _GRILLE_COMPLETE if scheme == "complete" else _GRILLE_SIMPLIFIEE
    parts = [
        "Tu es correcteur expert pour une évaluation nationale de dictée.",
        "Tu ne vois PAS l'image : on te donne la transcription de ce que l'élève "
        "a écrit (produite par un système de lecture), et le texte de référence.",
        "",
        "Texte de référence (ce que l'élève devait écrire) :",
        f"« {reference_text} »",
        "",
        "Transcription de la copie de l'élève (fautes comprises) :",
        f"« {transcription} »",
        "",
        grille,
        "",
        _CONSIGNE_ALIGNEMENT,
        "",
        _CONSIGNE_COMPARAISON,
        "",
        "Compare la transcription au mot attendu de chaque item. Si un mot attendu "
        "n'apparaît pas dans la transcription, code-le 0 (absent).",
        "",
        "Items à coder, dans l'ordre. Chaque ligne = un item « identifiant → mot attendu » :",
    ]
    for idx, it in enumerate(items, 1):
        nature = "ponctuation" if it.type == "ponctuation" else "mot"
        parts.append(f"  {idx:>2}. {it.item_id} → « {it.attendu} » ({nature})")
    parts += [
        "",
        f"Tu dois rendre EXACTEMENT {len(items)} items, dans cet ordre.",
        "Réponds UNIQUEMENT par un objet JSON, sans texte autour, de la forme :",
        '{"items": [{"item_id": "...", "transcription": "mot lu pour cet item", '
        '"code": "1", "confidence": 0.95}, ...]}',
    ]
    return "\n".join(parts)
