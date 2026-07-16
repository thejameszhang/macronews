"""The config layer earns its keep by catching defaults that were already wrong."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import pytest
from pydantic import ValidationError

from macronews import invariants
from macronews.config.paths import MAPPER_MODEL
from macronews.config.runconfig import (
    GraderConfig, MapperConfig, model_max_context,
)

DJNW_SHARD = Path(
    "/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/2014-05c_clean.jsonl"
)

@pytest.fixture
def mapper_out(tmp_path):
    """GraderConfig only checks that mapper_output IS A FILE, so a stub satisfies it.

    These tests used to point at a real gitignored production shard and SKIP when it
    was absent -- so they never ran on a fresh clone, and they went silently green the
    day that shard was moved. A fixture no reorg can mute is worth more than one that
    reads real bytes it never looks at.
    """
    p = tmp_path / "2014-05c.jsonl"
    p.write_text('{"article_id": "1", "mappings": []}\n')
    return p


def _mapper(**kw):
    base = dict(dataset="djnw", input_file=DJNW_SHARD, output_dir=Path("out"))
    return MapperConfig(**{**base, **kw})


def _grader(mapper_output, **kw):
    base = dict(mapper_output=mapper_output, output=Path("o.jsonl"),
                dataset="djnw", input_file=DJNW_SHARD)
    return GraderConfig(**{**base, **kw})


def test_the_production_model_is_the_default_and_it_exists():
    """The old default was ~/models/gemma-4-31b-it: a nonexistent path, and the
    superseded 31B. Nobody noticed because every launcher passed --model."""
    cfg = _mapper()
    assert cfg.model == MAPPER_MODEL
    assert cfg.model.name == "gemma-4-26b-a4b-it"
    assert (cfg.model / "config.json").is_file()


def test_max_model_len_defaults_to_the_production_value():
    """The CLI default was 8192 while every launcher passed 65536 -- and it drives the
    loader's token filter, so 8192 silently dropped 440 articles."""
    assert _mapper().max_model_len == 65_536


def test_max_model_len_is_capped_by_the_model_not_guessed():
    cap = model_max_context(MAPPER_MODEL)
    assert cap == 262_144, "Gemma 4 26B-A4B declares a 256K context"
    with pytest.raises(ValidationError, match="exceeds what"):
        _mapper(max_model_len=cap + 1)
    _mapper(max_model_len=cap)  # exactly the cap is fine


def test_a_missing_model_fails_loud_not_at_load_time():
    with pytest.raises(ValidationError, match="no model at"):
        _mapper(model=Path("/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-99b-it"))


def test_tensor_parallel_size_is_not_a_field():
    with pytest.raises(ValidationError, match="tensor_parallel_size"):
        _mapper(tensor_parallel_size=2)


def test_an_unknown_key_is_an_error_not_a_silent_default():
    """extra='forbid'. A renamed key must not quietly do nothing -- that is how the
    gate's keywords sat unread in group_universe.yaml for a month."""
    with pytest.raises(ValidationError):
        _mapper(keyword_gates=True)          # typo
    with pytest.raises(ValidationError):
        _mapper(max_model_length=65_536)     # typo


@pytest.mark.parametrize("dataset,gated", [
    ("djnw", True), ("gold", False), ("sports", False),
])
def test_the_gate_defaults_from_the_dataset(dataset, gated):
    """The gate is a production cost optimization. gold/sports are instruments:
    gating them takes the model out of the experiment.

    NOT parametrized over wikigaming: data/WikiGaming.jsonl does not exist, so
    constructing that config now raises (see the test below). Listing it here would
    contradict that test and the suite could never go green.
    """
    assert MapperConfig(dataset=dataset, output_file=Path("o.jsonl")).keyword_gate is gated


def test_an_explicit_gate_flag_wins():
    assert MapperConfig(dataset="gold", output_file=Path("o.jsonl"),
                        keyword_gate=True).keyword_gate is True


def test_a_shard_derives_its_own_output_name():
    assert _mapper().output_path == Path("out/2014-05c.jsonl")


def test_sampling_a_shard_is_allowed():
    """--mode prod used to reject --max-articles for no reason."""
    assert _mapper(max_articles=100).max_articles == 100


def test_exactly_one_output_destination():
    with pytest.raises(ValidationError, match="exactly one"):
        MapperConfig(dataset="gold", output_file=Path("a"), output_dir=Path("b"))
    with pytest.raises(ValidationError, match="exactly one"):
        MapperConfig(dataset="gold")


def test_an_input_file_must_be_djnw():
    with pytest.raises(ValidationError, match="djnw shard"):
        MapperConfig(dataset="gold", input_file=DJNW_SHARD, output_dir=Path("out"))


def test_a_shard_implies_djnw():
    """--input-file IS a djnw shard, so --dataset is inferable -- the same reasoning
    that deletes --mode. The prod launcher passes only --input-file/--output-dir."""
    cfg = MapperConfig(input_file=DJNW_SHARD, output_dir=Path("out"))
    assert cfg.dataset == "djnw"
    assert cfg.keyword_gate is True


def test_neither_dataset_nor_input_file_is_an_error():
    with pytest.raises(ValidationError, match="give --dataset, or --input-file"):
        MapperConfig(output_file=Path("o.jsonl"))


def test_sample_dirs_are_absolute():
    """`macronews mapper run --dataset gold` must work from ANY cwd. A relative
    data/articles_sample would resolve against the caller's directory."""
    cfg = MapperConfig(dataset="gold", output_file=Path("o.jsonl"))
    assert cfg.sample_dir.is_absolute()
    assert cfg.sample_dir.is_dir()


def test_a_dataset_with_no_data_on_disk_is_rejected():
    """data/WikiGaming.jsonl does not exist. Without this check the config validates
    clean and the run dies later inside load_articles -- which would make the whole
    'a run cannot start with a value nobody meant' claim false."""
    with pytest.raises(ValidationError, match="it does not exist"):
        MapperConfig(dataset="wikigaming", output_file=Path("o.jsonl"))


def test_the_two_stages_have_DIFFERENT_headroom(mapper_out):
    """The grader reserves 3000 tokens, the mapper 2000, because the grader generates
    1024 tokens to the mapper's 512. Collapsing these into one shared constant
    silently changes which articles each stage loads."""
    assert _mapper().max_article_tokens == 65_536 - 2000
    assert _grader(mapper_out).max_article_tokens == 40_960 - 3000


def test_the_corpus_filter_has_a_floor():
    assert _mapper(max_model_len=2048).max_article_tokens == 1024


def test_the_record_carries_the_invariants(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    invariants.apply()                        # the entrypoint does this
    rec = _mapper().record()
    assert rec["max_model_len"] == 65_536
    assert rec["max_article_tokens"] == 63_536
    assert rec["invariants"]["vllm_batch_invariant"] == "1"
    assert rec["invariants"]["attention_backend"] == "TRITON_ATTN"


def test_grader_defaults_to_qwq_at_its_own_max(mapper_out):
    cfg = _grader(mapper_out)
    assert cfg.model.name == "qwq-32b"
    assert cfg.max_model_len == 40_960 == model_max_context(cfg.model)


def test_grader_djnw_needs_the_source_shard(mapper_out):
    with pytest.raises(ValidationError, match="needs input_file"):
        GraderConfig(mapper_output=mapper_out, output=Path("o.jsonl"), dataset="djnw")


def test_a_missing_input_file_fails_loud_not_deep_in_the_stage(mapper_out):
    """MapperConfig._resolve() derived sample_dir = input_file.parent and checked
    THAT exists -- so a transposed digit in a shard filename validated clean and
    only died later inside load_articles. This is the production path: the launcher
    passes exactly --input-file/--output-dir."""
    bogus = DJNW_SHARD.parent / "9999-99z_clean.jsonl"
    with pytest.raises(ValidationError, match="no djnw shard at"):
        _mapper(input_file=bogus)
    with pytest.raises(ValidationError, match="no djnw shard at"):
        _grader(mapper_out, input_file=bogus)


def test_grader_sample_dir_must_exist(mapper_out):
    """MapperConfig already rejects a sample_dir that does not exist; GraderConfig
    did not, though it accepts the same field."""
    with pytest.raises(ValidationError, match="it does not exist"):
        _grader(mapper_out, dataset="gold", input_file=None, sample_dir=Path("/nonexistent"))


def test_a_corrupt_model_config_is_not_a_usage_error(tmp_path, monkeypatch):
    """json.JSONDecodeError subclasses ValueError, so left unguarded it would be
    wrapped by pydantic into a generic ValidationError -- indistinguishable in kind
    from a bad flag. A corrupt/half-downloaded weights file is not a usage mistake,
    so it must raise something other than ValidationError."""
    bad_model = tmp_path / "gemma-4-26b-a4b-it"
    bad_model.mkdir()
    (bad_model / "config.json").write_text("{not valid json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        model_max_context(bad_model)
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _mapper(model=bad_model)
