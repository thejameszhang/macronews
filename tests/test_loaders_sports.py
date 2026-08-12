import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from macronews.loaders import load_articles

SPORTS_DIR = REPO / "data" / "sports_news_1994_2000"


def test_sports_loader_returns_standard_schema():
    arts = load_articles(dataset="sports", sample_dir=SPORTS_DIR, max_articles=3)
    assert len(arts) == 3
    for a in arts:
        assert isinstance(a["id"], str)
        assert isinstance(a["paragraphs"], list) and a["paragraphs"]


def test_sports_loader_token_filter_fast_path_and_guard(tmp_path):
    # max_tokens lets the sports loader skip over-long articles; short text stays
    # under the fast-path char limit (no tokenizer), long text needs tokenizer_path.
    from macronews.loaders import load_sports_articles
    (tmp_path / "a.json").write_text(json.dumps(
        {"title": "t", "text": "Short recap. Team won the game."}))
    arts = load_sports_articles(tmp_path, max_tokens=50, tokenizer_path=None)
    assert len(arts) == 1
    (tmp_path / "b.json").write_text(json.dumps({"title": "t", "text": "word " * 200}))
    with pytest.raises(ValueError):  # long article needs a tokenizer; none given
        load_sports_articles(tmp_path, max_tokens=50, tokenizer_path=None)
