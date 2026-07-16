from macronews.kg.link_groups import (
    ASSET_CLASS_TYPES, ALL_ASSET_TYPES, build_group_configs, candidates,
)
from macronews.utils.groups import load_group_universe


def _configs():
    return build_group_configs(load_group_universe())


def test_all_asset_types_is_union():
    assert ALL_ASSET_TYPES == set().union(*ASSET_CLASS_TYPES.values())
    assert "COMMODITY" in ALL_ASSET_TYPES and "CENTRAL_BANK" not in ALL_ASSET_TYPES


def test_group_names_are_unique():
    cfgs = _configs()
    names = [c.name for c in cfgs]
    assert len(names) == len(set(names)) == 50


def test_type_gate_blocks_central_bank_from_rates():
    # The Canada Rates edge case: "Bank of Canada" (CENTRAL_BANK) must NOT be a
    # candidate for Canada Rates (rates -> {GOV_BOND, INTEREST_RATE}), even though
    # the "Canada" keyword matches. Drivers are not assets.
    cfgs = _configs()
    assert not any(c.name == "Canada Rates"
                   for c in candidates("Bank of Canada", "CENTRAL_BANK", cfgs))
    # but the actual rate IS a candidate
    assert any(c.name == "Canada Rates"
               for c in candidates("Canadian Bond Yields", "GOV_BOND", cfgs))


def test_keyword_word_boundary_not_substring():
    cfgs = _configs()
    assert not any(c.name == "Crude Oil"
                   for c in candidates("political turmoil", "COMMODITY", cfgs))
    assert any(c.name == "Crude Oil"
               for c in candidates("Nymex crude oil", "COMMODITY", cfgs))


def test_driver_types_get_no_candidates():
    cfgs = _configs()
    for ty in ("CENTRAL_BANK", "SOVEREIGN", "PERSON", "GOV_BODY", "CONCEPT", "COMPANY"):
        assert candidates("Bank of Canada", ty, cfgs) == [] or ty == "COMPANY"
    # COMPANY (e.g. Royal Bank of Canada) is also a non-asset type -> no candidates
    assert candidates("Royal Bank of Canada", "COMPANY", cfgs) == []


import json
from macronews.kg.link_groups import build_mapper_index, accumulate_links  # noqa: E402


def _write(p, rows):
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_build_mapper_index_filters_05(tmp_path):
    m = tmp_path / "m.jsonl"
    _write(m, [{"article_id": "a1", "mappings": [
        {"group": "Crude Oil", "relevance_score": 0.6, "evidence_paragraphs": [2]},
        {"group": "US Rates", "relevance_score": 0.5, "evidence_paragraphs": [1]},   # boundary: 0.5 dropped (>0.5, not >=)
        {"group": "Natural Gas", "relevance_score": 0.3, "evidence_paragraphs": [4]}]}])
    assert build_mapper_index(m) == {"a1": {"Crude Oil": {2}}}     # <=0.5 dropped


def _disambig(tmp_path, subj, subj_t, obj, obj_t, paras, aid="a1", stmt="s"):
    p = tmp_path / "d.jsonl"
    _write(p, [{"article_id": aid, "events": [
        {"statement": stmt, "evidence_paragraphs": paras, "triplets": [
            {"subject": subj, "subject_type": subj_t, "relation": "CAUSES_RISE_IN",
             "object": obj, "object_type": obj_t, "value": None}]}]}])
    return p


def test_mapper_para_join_confirms(tmp_path):
    cfgs = _configs()
    d = _disambig(tmp_path, "supply", "CONCEPT", "Oil", "COMMODITY", paras=[2])
    m = tmp_path / "m.jsonl"
    _write(m, [{"article_id": "a1", "mappings": [
        {"group": "Crude Oil", "relevance_score": 0.6, "evidence_paragraphs": [2]}]}])
    links = accumulate_links(d, cfgs, build_mapper_index(m), para_strict=True)
    el = links["oil"]
    assert el.confirmed.get("crude_oil") == "mapper-para"
    assert "crude_oil" not in el.residual_keys


def test_mapper_article_only_falls_to_residual_under_para_strict(tmp_path):
    cfgs = _configs()
    d = _disambig(tmp_path, "supply", "CONCEPT", "Oil", "COMMODITY", paras=[9])
    m = tmp_path / "m.jsonl"
    _write(m, [{"article_id": "a1", "mappings": [
        {"group": "Crude Oil", "relevance_score": 0.6, "evidence_paragraphs": [2]}]}])
    links = accumulate_links(d, cfgs, build_mapper_index(m), para_strict=True)
    el = links["oil"]
    assert "crude_oil" not in el.confirmed
    assert "crude_oil" in el.residual_keys
    row = next(r for r in el.residual_rows if r["group_key"] == "crude_oil")
    assert row["mapper_article_tagged"] is True and row["mapper_evidence_paragraphs"] == [2]


def test_mapper_article_level_confirms_under_no_para_strict(tmp_path):
    # PARA_STRICT=False: an article-level tag (no paragraph overlap) confirms, with
    # the "mapper-article" label. Exercises the non-default path + the no-downgrade guard.
    cfgs = _configs()
    d = _disambig(tmp_path, "supply", "CONCEPT", "Oil", "COMMODITY", paras=[9])
    m = tmp_path / "m.jsonl"
    _write(m, [{"article_id": "a1", "mappings": [
        {"group": "Crude Oil", "relevance_score": 0.6, "evidence_paragraphs": [2]}]}])
    links = accumulate_links(d, cfgs, build_mapper_index(m), para_strict=False)
    el = links["oil"]
    assert el.confirmed.get("crude_oil") == "mapper-article"
    assert "crude_oil" not in el.residual_keys


def test_exact_links_without_mapper(tmp_path):
    cfgs = _configs()
    d = _disambig(tmp_path, "supply", "CONCEPT", "Brent Crude Oil", "COMMODITY", paras=[1])
    links = accumulate_links(d, cfgs, mapper_index={}, para_strict=True)
    assert "crude_oil" in links["brent crude oil"].exact


def test_heating_oil_keyword_fp_goes_to_residual_not_link(tmp_path):
    cfgs = _configs()
    d = _disambig(tmp_path, "demand", "CONCEPT", "heating oil", "COMMODITY", paras=[3])
    links = accumulate_links(d, cfgs, mapper_index={}, para_strict=True)
    el = links["heating oil"]
    assert "crude_oil" not in el.confirmed and "crude_oil" not in el.exact
    assert "crude_oil" in el.residual_keys


from macronews.kg.link_groups import link_entities  # noqa: E402


def test_link_entities_no_llm_writes_sidecars(tmp_path):
    d = _disambig(tmp_path, "supply", "CONCEPT", "Brent Crude Oil", "COMMODITY", paras=[1])
    out = tmp_path / "k.entity_groups.json"
    res = tmp_path / "k.residual.jsonl"
    gm = tmp_path / "k.group_members.json"
    summary = link_entities(disambig_path=d, mapper_path=None, output_path=out,
                            residual_path=res, group_members_path=gm,
                            use_llm=False, para_strict=True)
    # (a) machine view: entity -> groups
    entry = json.loads(out.read_text())["brent crude oil"]
    assert entry["type"] == "COMMODITY"
    assert {g["key"] for g in entry["groups"]} == {"crude_oil"}
    assert entry["groups"][0]["method"] == "exact"
    assert summary["linked_entities"] >= 1
    # (b) human view: group -> entities under its umbrella (mirrors clusters.json)
    members = json.loads(gm.read_text())
    assert members["Crude Oil"] == [
        {"entity": "Brent Crude Oil", "type": "COMMODITY", "method": "exact"}]


def test_link_entities_mapper_para_method_end_to_end(tmp_path):
    # mapper-confirmed (para-overlap) link flows through to BOTH sidecars with the
    # "mapper-para" method label, end-to-end through link_entities with a mapper file.
    d = _disambig(tmp_path, "supply", "CONCEPT", "Oil", "COMMODITY", paras=[2])
    m = tmp_path / "m.jsonl"
    _write(m, [{"article_id": "a1", "mappings": [
        {"group": "Crude Oil", "relevance_score": 0.6, "evidence_paragraphs": [2]}]}])
    out = tmp_path / "k.entity_groups.json"
    res = tmp_path / "k.residual.jsonl"
    gm = tmp_path / "k.group_members.json"
    link_entities(disambig_path=d, mapper_path=m, output_path=out,
                  residual_path=res, group_members_path=gm,
                  use_llm=False, para_strict=True)
    entry = json.loads(out.read_text())["oil"]
    assert entry["groups"] == [{"key": "crude_oil", "method": "mapper-para"}]
    members = json.loads(gm.read_text())
    assert members["Crude Oil"] == [
        {"entity": "Oil", "type": "COMMODITY", "method": "mapper-para"}]


def test_link_entities_residual_dropped_under_no_llm(tmp_path):
    d = _disambig(tmp_path, "demand", "CONCEPT", "heating oil", "COMMODITY", paras=[3])
    out = tmp_path / "k.entity_groups.json"
    res = tmp_path / "k.residual.jsonl"
    gm = tmp_path / "k.group_members.json"
    link_entities(disambig_path=d, mapper_path=None, output_path=out,
                  residual_path=res, group_members_path=gm,
                  use_llm=False, para_strict=True)
    eg = json.loads(out.read_text())
    # the crude_oil keyword-FP residual is NOT promoted into entity_groups (heating
    # oil DOES link to refined_petroleum via exact-match, so the entity is present —
    # we assert the residual *group* was dropped, not the whole entity).
    assert not any(g["key"] == "crude_oil"
                   for g in eg.get("heating oil", {}).get("groups", []))
    residual = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]
    assert any(r["group_key"] == "crude_oil" and r["entity"] == "heating oil"
               for r in residual)                        # logged for the (future) judge
