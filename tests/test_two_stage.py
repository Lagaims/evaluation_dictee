"""Tests du TwoStageScorer (approche 1 : transcription puis codage)."""

import json
from types import SimpleNamespace

from evaluation_dictee.config import ModelConfig, PromptConfig
from evaluation_dictee.data.loaders import Copy
from evaluation_dictee.data.reference import GridItem
from evaluation_dictee.models.two_stage import TwoStageScorer

ITEMS = [
    GridItem("i1", "Le", "mot", [], [], []),
    GridItem("i2", "soir", "mot", [], [], []),
    GridItem("i3", "tombait", "mot", [], [], []),
]


class _FakeClient:
    """Client OpenAI factice : renvoie des réponses scriptées dans l'ordre."""

    def __init__(self, reponses: list[str]) -> None:
        self._reponses = list(reponses)
        self.appels: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):  # noqa: ANN003
        self.appels.append(kwargs)
        contenu = self._reponses.pop(0)
        msg = SimpleNamespace(content=contenu)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _make_scorer(reponses: list[str], monkeypatch) -> TwoStageScorer:
    # Évite le vrai chargement d'image
    monkeypatch.setattr("evaluation_dictee.models.two_stage.load_image", lambda path: "FAKE_IMAGE")
    monkeypatch.setattr(
        "evaluation_dictee.models.two_stage._image_to_data_url", lambda img: "data:fake"
    )
    scorer = TwoStageScorer(
        model_config=ModelConfig(name="m1", max_retries=1),
        prompt_config=PromptConfig(),
        base_url="http://x",
        api_key="k",
        grid_items=ITEMS,
    )
    scorer.client = _FakeClient(reponses)
    return scorer


def _copy() -> Copy:
    return Copy(
        copy_id="c.png",
        image_path="/tmp/c.png",
        item_ids=["i1", "i2", "i3"],
        expert_codes=["1", "1", "1"],
    )


def test_deux_etapes_nominal(monkeypatch) -> None:
    transcription = json.dumps({"transcription": "Le soir tombait"})
    codage = json.dumps(
        {
            "items": [
                {"item_id": "i1", "transcription": "Le", "code": "1", "confidence": 0.9},
                {"item_id": "i2", "transcription": "soir", "code": "1", "confidence": 0.9},
                {"item_id": "i3", "transcription": "tombait", "code": "1", "confidence": 0.9},
            ]
        }
    )
    scorer = _make_scorer([transcription, codage], monkeypatch)
    pred = scorer.score_copy(_copy(), "Le soir tombait")
    assert pred.transcribed is True
    assert len(pred.items) == 3
    assert [it.code for it in pred.items] == ["1", "1", "1"]


def test_deux_appels_distincts(monkeypatch) -> None:
    """Vérifie qu'il y a bien DEUX appels : étape 1 (image) puis étape 2 (texte)."""
    transcription = json.dumps({"transcription": "Le soir tombait"})
    codage = json.dumps(
        {
            "items": [
                {"item_id": "i1", "transcription": "Le", "code": "1", "confidence": 0.9},
                {"item_id": "i2", "transcription": "soir", "code": "1", "confidence": 0.9},
                {"item_id": "i3", "transcription": "tombait", "code": "1", "confidence": 0.9},
            ]
        }
    )
    scorer = _make_scorer([transcription, codage], monkeypatch)
    scorer.score_copy(_copy(), "Le soir tombait")
    assert len(scorer.client.appels) == 2
    # 1er appel = multimodal (contenu = liste avec image)
    contenu1 = scorer.client.appels[0]["messages"][0]["content"]
    assert isinstance(contenu1, list)
    # 2e appel = texte seul (contenu = chaîne)
    contenu2 = scorer.client.appels[1]["messages"][0]["content"]
    assert isinstance(contenu2, str)


def test_transcription_vide_marque_non_transcrite(monkeypatch) -> None:
    # L'étape 1 échoue à toutes les tentatives → copie non transcrite
    vide = json.dumps({"transcription": ""})
    scorer = _make_scorer([vide, vide], monkeypatch)  # max_retries=1 → 2 essais
    pred = scorer.score_copy(_copy(), "Le soir tombait")
    assert pred.transcribed is False
    # Pas d'appel à l'étape 2 si pas de transcription
    assert len(scorer.client.appels) == 2  # 2 tentatives étape 1, 0 étape 2


def test_modele_stage2_distinct(monkeypatch) -> None:
    """Le modèle de l'étape 2 doit être utilisé pour le second appel."""
    monkeypatch.setattr("evaluation_dictee.models.two_stage.load_image", lambda path: "IMG")
    monkeypatch.setattr(
        "evaluation_dictee.models.two_stage._image_to_data_url", lambda img: "data:x"
    )
    scorer = TwoStageScorer(
        model_config=ModelConfig(name="htr-model"),
        prompt_config=PromptConfig(),
        base_url="http://x",
        api_key="k",
        grid_items=ITEMS,
        model_config_stage2=ModelConfig(name="text-model", kind="llm"),
    )
    transcription = json.dumps({"transcription": "Le soir tombait"})
    codage = json.dumps(
        {
            "items": [
                {"item_id": "i1", "transcription": "Le", "code": "1", "confidence": 0.9},
                {"item_id": "i2", "transcription": "soir", "code": "1", "confidence": 0.9},
                {"item_id": "i3", "transcription": "tombait", "code": "1", "confidence": 0.9},
            ]
        }
    )
    scorer.client = _FakeClient([transcription, codage])
    scorer.score_copy(_copy(), "Le soir tombait")
    assert scorer.client.appels[0]["model"] == "htr-model"
    assert scorer.client.appels[1]["model"] == "text-model"


def test_parse_transcription_repli_texte_brut() -> None:
    # Si le modèle répond en texte brut (pas de JSON), on récupère quand même
    assert TwoStageScorer._parse_transcription("Le soir tombait") == "Le soir tombait"
