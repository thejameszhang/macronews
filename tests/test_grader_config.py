"""The grader takes the same validated config shape as the mapper."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

from macronews import invariants
from macronews.config.runconfig import GraderConfig
from macronews.mapping.grading import runner

DJNW_SHARD = Path(
    "/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/2014-05c_clean.jsonl"
)


class StubGrader:
    def __init__(self, **kw):
        self.kw = kw

    def grade_batch(self, inputs, max_tokens=1024):
        return []


def test_grade_takes_a_config(monkeypatch, tmp_path):
    """The mapper output is a stub: GraderConfig only checks that it is a file, and
    load_articles is stubbed out anyway. Pointing this at a real gitignored production
    shard meant it SKIPPED on a fresh clone, and went silently green the day that
    shard moved."""
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    invariants.apply()

    mapper_out = tmp_path / "2014-05c.jsonl"
    mapper_out.write_text('{"article_id": "1", "mappings": []}\n')

    built = {}
    monkeypatch.setattr(runner, "load_articles", lambda **k: [])
    monkeypatch.setattr(runner, "LLMGrader",
                        lambda **k: (built.update(k) or StubGrader(**k)))
    monkeypatch.setattr(runner, "write_sidecar", lambda *a, **k: {"graded": 0})

    cfg = GraderConfig(mapper_output=mapper_out, output=tmp_path / "o.jsonl",
                       dataset="djnw", input_file=DJNW_SHARD)
    runner.grade(cfg)

    assert built["max_model_len"] == 40_960
    assert built["tensor_parallel_size"] == 1     # the invariant, not a flag


def test_tensor_parallel_size_is_gone_from_the_grader():
    src = (REPO / "src" / "macronews" / "mapping" / "grading" / "runner.py").read_text()
    assert "--tensor-parallel-size" not in src
