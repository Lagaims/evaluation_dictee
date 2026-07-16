"""Tests du parseur JSON robuste utilisé par l'étape 2 du TwoStageScorer."""

from evaluation_dictee.models.two_stage import _extract_items_from_content


def test_json_propre() -> None:
    items = _extract_items_from_content(
        '{"items": [{"item_id": "i1", "code": "1", "confidence": 0.9}]}'
    )
    assert len(items) == 1
    assert items[0]["item_id"] == "i1"


def test_json_dans_bloc_markdown() -> None:
    items = _extract_items_from_content('```json\n{"items": [{"item_id": "i1", "code": "1"}]}\n```')
    assert len(items) == 1


def test_preambule_textuel_avant_json() -> None:
    """Cas fréquent : le modèle ajoute un préambule qui casse json.loads direct."""
    contenu = (
        "Voici mes codes pour cette copie :\n"
        '{"items": [{"item_id": "i1", "code": "1", "confidence": 0.9}, '
        '{"item_id": "i2", "code": "9", "confidence": 0.7}]}\n'
        "N'hésitez pas si vous avez des questions."
    )
    items = _extract_items_from_content(contenu)
    assert len(items) == 2
    assert items[1]["code"] == "9"


def test_json_tronque_recupere_items_valides() -> None:
    """Cas critique : réponse coupée par max_tokens, JSON global invalide."""
    contenu = (
        '{"items": ['
        '{"item_id": "i1", "code": "1", "confidence": 0.9},'
        '{"item_id": "i2", "code": "9", "confidence": 0.8},'
        '{"item_id": "i3", "cod'  # tronqué
    )
    items = _extract_items_from_content(contenu)
    # On récupère au moins les 2 items complets
    assert len(items) >= 2
    assert items[0]["item_id"] == "i1"


def test_contenu_vide() -> None:
    assert _extract_items_from_content("") == []
    assert _extract_items_from_content("juste du texte sans json") == []


def test_json_valide_sans_cle_items() -> None:
    # Si "items" absent, l'extraction directe ne trouve rien → tentative 3
    assert _extract_items_from_content('{"autre": "chose"}') == []
