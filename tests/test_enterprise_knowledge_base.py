#!/usr/bin/env python3
"""Comprehensive test suite for ``src/knowledge/enterprise_knowledge_base.py``.

Pure Python / NumPy, no database, vector store, LLM, or embeddings.  Coverage
(180+ tests):

- KnowledgeCategory enum & normalisation
- KnowledgeDocument construction, validation, serialisation
- KnowledgeBaseConfig validation
- Registry (register / remove / get / list / has)
- Search (keyword / tag / category / exact)
- Relevance scoring (keyword / tag / category overlap)
- Collections (document / category / fleet)
- Statistics (counts, distributions, coverage)
- Validation (duplicate ids, empty docs, invalid category, invalid metadata)
- Serialization (to_dict / from_dict / JSON round-trips)
- Determinism
- Tracker
- Edge cases (empty KB, large corpora, unicode, filters)

Run::

    pytest tests/test_enterprise_knowledge_base.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.knowledge.enterprise_knowledge_base import (
    KB_NAME,
    KNOWLEDGE_BASE_REGISTRY,
    EnterpriseKnowledgeBase,
    KnowledgeBaseConfig,
    KnowledgeCategory,
    KnowledgeCollection,
    KnowledgeDocument,
    KnowledgeStatistics,
    SearchResult,
    build_knowledge_base,
    list_knowledge_bases,
    normalize_category,
    register_knowledge_base,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _doc(did="D1", *, title="Bearing Inspection Guide",
         category=KnowledgeCategory.INSPECTION_PROCEDURE.value,
         tags=("bearing", "vibration"), content="Inspect bearing vibration faults.",
         meta=None, dtype="reference", source="eng", created="2026-01-01",
         version="1.0") -> KnowledgeDocument:
    return KnowledgeDocument(document_id=did, title=title, document_type=dtype,
                             source=source, category=category, tags=tags,
                             content=content, created_at=created, version=version,
                             metadata=meta or {})


def _kb(n=0, **kw) -> EnterpriseKnowledgeBase:
    kb = EnterpriseKnowledgeBase(KnowledgeBaseConfig(**kw) if kw else None)
    cats = list(KnowledgeCategory)
    for i in range(n):
        kb.register_document(_doc(
            f"D{i:03d}", title=f"Document {i} bearing gearbox",
            category=cats[i % len(cats)].value,
            tags=("bearing", "gearbox") if i % 2 == 0 else ("oil", "lubrication"),
            content=f"Content body number {i} about maintenance and inspection."))
    return kb


# ===========================================================================
# Categories
# ===========================================================================


class TestCategories:
    def test_ten_categories(self) -> None:
        assert len(list(KnowledgeCategory)) == 10

    def test_expected_values(self) -> None:
        vals = {c.value for c in KnowledgeCategory}
        assert {"maintenance_manual", "inspection_procedure", "operating_procedure",
                "failure_catalog", "root_cause_playbook", "safety_procedure",
                "asset_specification", "engineering_report", "executive_report",
                "unknown"} == vals

    def test_normalize_from_enum(self) -> None:
        assert normalize_category(KnowledgeCategory.SAFETY_PROCEDURE) == "safety_procedure"

    def test_normalize_from_string(self) -> None:
        assert normalize_category("failure_catalog") == "failure_catalog"

    def test_normalize_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid category"):
            normalize_category("nonsense")

    def test_category_is_str(self) -> None:
        assert KnowledgeCategory.UNKNOWN == "unknown"


# ===========================================================================
# Document construction & validation
# ===========================================================================


class TestDocument:
    def test_basic(self) -> None:
        d = _doc()
        assert d.document_id == "D1" and d.title

    def test_all_fields(self) -> None:
        d = _doc(meta={"fleet_id": "F1"})
        for f in ("document_id", "title", "document_type", "source", "category",
                  "tags", "content", "created_at", "version", "metadata"):
            assert hasattr(d, f)

    def test_frozen(self) -> None:
        d = _doc()
        with pytest.raises((AttributeError, TypeError)):
            d.title = "x"  # type: ignore[misc]

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="document_id"):
            _doc("")

    def test_whitespace_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="document_id"):
            _doc("   ")

    def test_empty_title_rejected(self) -> None:
        with pytest.raises(ValueError, match="title"):
            KnowledgeDocument(document_id="X", title="", content="c")

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValueError, match="content"):
            KnowledgeDocument(document_id="X", title="t", content="")

    def test_whitespace_content_rejected(self) -> None:
        with pytest.raises(ValueError, match="content"):
            KnowledgeDocument(document_id="X", title="t", content="   ")

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid category"):
            KnowledgeDocument(document_id="X", title="t", content="c", category="bad")

    def test_default_category_unknown(self) -> None:
        assert KnowledgeDocument(document_id="X", title="t", content="c").category == "unknown"

    def test_tags_normalized_lowercase(self) -> None:
        d = _doc(tags=("Bearing", "VIBRATION"))
        assert d.tags == ("bearing", "vibration")

    def test_tags_stripped(self) -> None:
        d = _doc(tags=("  bearing  ",))
        assert d.tags == ("bearing",)

    def test_tags_from_list(self) -> None:
        d = _doc(tags=["a", "b"])
        assert d.tags == ("a", "b")

    def test_empty_tag_rejected(self) -> None:
        with pytest.raises(ValueError, match="tag"):
            _doc(tags=("bearing", ""))

    def test_non_string_tag_rejected(self) -> None:
        with pytest.raises(ValueError, match="tag"):
            _doc(tags=("bearing", 5))

    def test_tags_default_empty(self) -> None:
        assert KnowledgeDocument(document_id="X", title="t", content="c").tags == ()

    def test_invalid_metadata_object_rejected(self) -> None:
        with pytest.raises(ValueError, match="metadata"):
            _doc(meta={"k": object()})

    def test_metadata_must_be_dict(self) -> None:
        with pytest.raises(ValueError, match="metadata"):
            KnowledgeDocument(document_id="X", title="t", content="c", metadata=["a"])

    def test_metadata_nested_ok(self) -> None:
        d = _doc(meta={"a": {"b": [1, 2, 3]}, "c": True, "d": None})
        assert d.metadata["a"]["b"] == [1, 2, 3]

    def test_metadata_default_empty(self) -> None:
        assert KnowledgeDocument(document_id="X", title="t", content="c").metadata == {}

    def test_version_default(self) -> None:
        assert KnowledgeDocument(document_id="X", title="t", content="c").version == "1.0"

    def test_token_sets(self) -> None:
        title_t, content_t, tag_t = _doc().token_sets()
        assert "bearing" in tag_t and "inspect" in content_t


# ===========================================================================
# Document serialization
# ===========================================================================


class TestDocumentSerialization:
    def test_to_dict_keys(self) -> None:
        d = _doc().to_dict()
        for k in ("document_id", "title", "document_type", "source", "category",
                  "tags", "content", "created_at", "version", "metadata"):
            assert k in d

    def test_to_dict_tags_list(self) -> None:
        assert isinstance(_doc().to_dict()["tags"], list)

    def test_round_trip(self) -> None:
        d = _doc(meta={"fleet_id": "F1"})
        d2 = KnowledgeDocument.from_dict(d.to_dict())
        assert d2.document_id == d.document_id and d2.tags == d.tags
        assert d2.metadata == d.metadata

    def test_json_serializable(self) -> None:
        assert isinstance(json.dumps(_doc().to_dict()), str)

    def test_from_dict_defaults(self) -> None:
        d = KnowledgeDocument.from_dict({"document_id": "X", "title": "t", "content": "c"})
        assert d.category == "unknown" and d.version == "1.0"

    def test_from_dict_requires_mapping(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            KnowledgeDocument.from_dict([1, 2, 3])

    def test_round_trip_via_json(self) -> None:
        d = _doc()
        d2 = KnowledgeDocument.from_dict(json.loads(json.dumps(d.to_dict())))
        assert d2.content == d.content


# ===========================================================================
# Config
# ===========================================================================


class TestConfig:
    def test_defaults(self) -> None:
        c = KnowledgeBaseConfig()
        assert c.title_weight == 3.0 and c.default_top_n == 10

    def test_negative_title_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="title_weight"):
            KnowledgeBaseConfig(title_weight=-1)

    def test_zero_keyword_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to > 0"):
            KnowledgeBaseConfig(title_weight=0, content_weight=0, tag_weight=0)

    def test_negative_component_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="tag_component_weight"):
            KnowledgeBaseConfig(tag_component_weight=-0.1)

    def test_top_n_positive(self) -> None:
        with pytest.raises(ValueError, match="default_top_n"):
            KnowledgeBaseConfig(default_top_n=0)

    def test_frozen(self) -> None:
        c = KnowledgeBaseConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.title_weight = 9  # type: ignore[misc]

    def test_keyword_weight_sum(self) -> None:
        assert KnowledgeBaseConfig().keyword_weight_sum == 6.0


# ===========================================================================
# Registry: register / remove / get / list
# ===========================================================================


class TestRegistry:
    def test_register_and_get(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc()
        kb.register_document(d)
        assert kb.get_document("D1") is d

    def test_register_non_document_rejected(self) -> None:
        kb = EnterpriseKnowledgeBase()
        with pytest.raises(TypeError, match="KnowledgeDocument"):
            kb.register_document({"document_id": "X"})  # type: ignore[arg-type]

    def test_duplicate_rejected(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc())
        with pytest.raises(ValueError, match="duplicate"):
            kb.register_document(_doc())

    def test_remove_returns_doc(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc())
        removed = kb.remove_document("D1")
        assert removed.document_id == "D1" and not kb.has_document("D1")

    def test_remove_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown"):
            EnterpriseKnowledgeBase().remove_document("ZZ")

    def test_get_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown"):
            EnterpriseKnowledgeBase().get_document("ZZ")

    def test_has_document(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc())
        assert kb.has_document("D1") and not kb.has_document("ZZ")

    def test_len(self) -> None:
        assert len(_kb(7)) == 7

    def test_list_all(self) -> None:
        assert len(_kb(5).list_documents()) == 5

    def test_list_sorted_by_id(self) -> None:
        ids = [d.document_id for d in _kb(8).list_documents()]
        assert ids == sorted(ids)

    def test_list_filter_category(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.SAFETY_PROCEDURE.value))
        kb.register_document(_doc("B", category=KnowledgeCategory.MAINTENANCE_MANUAL.value))
        assert len(kb.list_documents(category=KnowledgeCategory.SAFETY_PROCEDURE.value)) == 1

    def test_list_filter_tag(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing",)))
        kb.register_document(_doc("B", tags=("oil",)))
        assert len(kb.list_documents(tag="bearing")) == 1

    def test_list_filter_tag_case_insensitive(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("Bearing",)))
        assert len(kb.list_documents(tag="BEARING")) == 1

    def test_list_empty(self) -> None:
        assert EnterpriseKnowledgeBase().list_documents() == ()


# ===========================================================================
# Keyword search
# ===========================================================================


class TestKeywordSearch:
    def test_finds_match(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", content="bearing vibration analysis"))
        assert kb.search_keywords("bearing")[0].document_id == "A"

    def test_empty_query(self) -> None:
        assert _kb(3).search_keywords("") == ()

    def test_stopwords_only_query(self) -> None:
        assert _kb(3).search_keywords("the and of") == ()

    def test_score_bounded(self) -> None:
        res = _kb(6).search_keywords("bearing gearbox maintenance")
        assert all(0 <= r.score <= 1 for r in res)

    def test_ranked_descending(self) -> None:
        res = _kb(8).search_keywords("bearing gearbox")
        scores = [r.score for r in res]
        assert scores == sorted(scores, reverse=True)

    def test_top_n(self) -> None:
        assert len(_kb(20).search_keywords("maintenance", top_n=3)) <= 3

    def test_default_top_n(self) -> None:
        assert len(_kb(30).search_keywords("maintenance", top_n=None)) <= 10

    def test_min_score(self) -> None:
        res = _kb(10).search_keywords("bearing", min_score=0.99)
        assert all(r.score > 0.99 for r in res)

    def test_matched_terms(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", content="bearing vibration"))
        assert "bearing" in kb.search_keywords("bearing")[0].matched_terms

    def test_no_match_empty(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", title="cooling", content="thermal system", tags=("oil",)))
        assert kb.search_keywords("zebra") == ()

    def test_title_weighted_above_content(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("TITLE", title="bearing fault", content="unrelated text", tags=("x",)))
        kb.register_document(_doc("BODY", title="cooling guide", content="bearing fault here", tags=("y",)))
        res = {r.document_id: r.score for r in kb.search_keywords("bearing fault")}
        assert res["TITLE"] > res["BODY"]

    def test_tied_scores_break_by_id(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("B", title="bearing", content="bearing", tags=("bearing",)))
        kb.register_document(_doc("A", title="bearing", content="bearing", tags=("bearing",)))
        res = kb.search_keywords("bearing")
        assert [r.document_id for r in res] == ["A", "B"]


# ===========================================================================
# Tag search
# ===========================================================================


class TestTagSearch:
    def test_finds_by_tag(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing", "vibration")))
        assert kb.search_tags(["bearing"])[0].document_id == "A"

    def test_overlap_score(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing", "vibration")))
        assert kb.search_tags(["bearing", "vibration"])[0].score == 1.0

    def test_partial_overlap(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing",)))
        assert kb.search_tags(["bearing", "vibration"])[0].score == 0.5

    def test_match_all(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing",)))
        kb.register_document(_doc("B", tags=("bearing", "vibration")))
        res = kb.search_tags(["bearing", "vibration"], match_all=True)
        assert [r.document_id for r in res] == ["B"]

    def test_empty_tags(self) -> None:
        assert _kb(3).search_tags([]) == ()

    def test_case_insensitive(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing",)))
        assert len(kb.search_tags(["BEARING"])) == 1

    def test_ranked(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing",)))
        kb.register_document(_doc("B", tags=("bearing", "vibration")))
        res = kb.search_tags(["bearing", "vibration"])
        assert res[0].document_id == "B"

    def test_top_n(self) -> None:
        kb = _kb(20)
        assert len(kb.search_tags(["bearing"], top_n=2)) <= 2


# ===========================================================================
# Category & exact search
# ===========================================================================


class TestCategorySearch:
    def test_returns_category(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.FAILURE_CATALOG.value))
        kb.register_document(_doc("B", category=KnowledgeCategory.SAFETY_PROCEDURE.value))
        res = kb.search_category(KnowledgeCategory.FAILURE_CATALOG.value)
        assert len(res) == 1 and res[0].document_id == "A"

    def test_score_one(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.FAILURE_CATALOG.value))
        assert kb.search_category("failure_catalog")[0].score == 1.0

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid category"):
            EnterpriseKnowledgeBase().search_category("bogus")

    def test_sorted_by_id(self) -> None:
        kb = EnterpriseKnowledgeBase()
        for i in [3, 1, 2]:
            kb.register_document(_doc(f"D{i}", category=KnowledgeCategory.UNKNOWN.value))
        ids = [r.document_id for r in kb.search_category("unknown")]
        assert ids == sorted(ids)

    def test_empty_category(self) -> None:
        assert EnterpriseKnowledgeBase().search_category("failure_catalog") == ()


class TestExactSearch:
    def test_finds_phrase(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", content="emergency lockout procedure"))
        assert kb.search_exact("lockout procedure")[0].document_id == "A"

    def test_case_insensitive(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", content="Emergency Lockout"))
        assert len(kb.search_exact("emergency lockout")) == 1

    def test_title_searched(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", title="Special Calibration Sheet", content="body"))
        assert len(kb.search_exact("special calibration")) == 1

    def test_empty_phrase(self) -> None:
        assert _kb(3).search_exact("") == ()

    def test_no_match(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", content="nothing here"))
        assert kb.search_exact("quantum flux") == ()

    def test_fields_restricted(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", title="alpha", content="beta gamma"))
        assert len(kb.search_exact("beta", fields=("title",))) == 0


# ===========================================================================
# Relevance scoring
# ===========================================================================


class TestRelevanceScoring:
    def test_keyword_component(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc("A", content="bearing vibration")
        assert kb.relevance_score(d, query="bearing") > 0

    def test_tag_component(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc("A", tags=("bearing", "vibration"))
        assert kb.relevance_score(d, tags=["bearing"]) > 0

    def test_category_match_one(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc("A", category=KnowledgeCategory.SAFETY_PROCEDURE.value)
        assert kb.relevance_score(d, category="safety_procedure") == pytest.approx(1.0)

    def test_category_mismatch_zero(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc("A", category=KnowledgeCategory.SAFETY_PROCEDURE.value)
        assert kb.relevance_score(d, category="failure_catalog") == 0.0

    def test_no_signal_zero(self) -> None:
        assert EnterpriseKnowledgeBase().relevance_score(_doc()) == 0.0

    def test_combined_bounded(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc("A", tags=("bearing",), content="bearing",
                 category=KnowledgeCategory.SAFETY_PROCEDURE.value)
        s = kb.relevance_score(d, query="bearing", tags=["bearing"], category="safety_procedure")
        assert 0 <= s <= 1

    def test_deterministic(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc()
        assert kb.relevance_score(d, query="bearing") == kb.relevance_score(d, query="bearing")

    def test_tag_overlap_fraction(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc("A", tags=("bearing", "vibration"))
        # one of two query tags overlaps
        assert kb.relevance_score(d, tags=["bearing", "thermal"]) == pytest.approx(
            kb.config.tag_component_weight * 0.5 / kb.config.tag_component_weight)


# ===========================================================================
# Collections
# ===========================================================================


class TestCollections:
    def test_make_collection(self) -> None:
        kb = _kb(4)
        coll = kb.make_collection("set1", ["D000", "D001"])
        assert isinstance(coll, KnowledgeCollection) and len(coll) == 2

    def test_make_collection_dedup_sorted(self) -> None:
        kb = _kb(4)
        coll = kb.make_collection("set1", ["D001", "D000", "D001"])
        assert coll.document_ids == ("D000", "D001")

    def test_make_collection_unknown_id(self) -> None:
        with pytest.raises(ValueError, match="unknown document_id"):
            _kb(2).make_collection("bad", ["ZZZ"])

    def test_make_collection_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            _kb(2).make_collection("", ["D000"])

    def test_category_collection(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.FAILURE_CATALOG.value))
        kb.register_document(_doc("B", category=KnowledgeCategory.FAILURE_CATALOG.value))
        coll = kb.category_collection("failure_catalog")
        assert len(coll) == 2 and coll.kind == "category"

    def test_fleet_collection_by_metadata(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", meta={"fleet_id": "FLEET-A"}))
        kb.register_document(_doc("B", meta={"fleet_id": "FLEET-B"}))
        coll = kb.fleet_collection("FLEET-A")
        assert coll.document_ids == ("A",) and coll.kind == "fleet"

    def test_fleet_collection_by_tag(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("fleet-a", "bearing")))
        coll = kb.fleet_collection("FLEET-A")
        assert "A" in coll.document_ids

    def test_get_collection(self) -> None:
        kb = _kb(3)
        kb.make_collection("s", ["D000"])
        assert kb.get_collection("s").name == "s"

    def test_get_collection_missing(self) -> None:
        with pytest.raises(KeyError, match="unknown collection"):
            _kb(1).get_collection("nope")

    def test_list_collections(self) -> None:
        kb = _kb(3)
        kb.make_collection("b", ["D000"])
        kb.make_collection("a", ["D001"])
        assert kb.list_collections() == ("a", "b")

    def test_collection_to_dict(self) -> None:
        kb = _kb(3)
        coll = kb.make_collection("s", ["D000"])
        d = coll.to_dict()
        assert d["name"] == "s" and isinstance(d["document_ids"], list)

    def test_collection_len(self) -> None:
        kb = _kb(3)
        assert len(kb.make_collection("s", ["D000", "D001"])) == 2


# ===========================================================================
# Statistics
# ===========================================================================


class TestStatistics:
    def test_document_count(self) -> None:
        assert _kb(7).statistics().document_count == 7

    def test_empty_kb(self) -> None:
        s = EnterpriseKnowledgeBase().statistics()
        assert s.document_count == 0 and s.coverage_score == 0.0

    def test_coverage_bounded(self) -> None:
        assert 0 <= _kb(10).statistics().coverage_score <= 1

    def test_category_distribution_sorted(self) -> None:
        s = _kb(12).statistics()
        counts = [c[1] for c in s.category_distribution]
        assert counts == sorted(counts, reverse=True)

    def test_tag_distribution_sorted(self) -> None:
        s = _kb(12).statistics()
        counts = [t[1] for t in s.tag_distribution]
        assert counts == sorted(counts, reverse=True)

    def test_distinct_tags(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing", "oil")))
        kb.register_document(_doc("B", tags=("bearing", "vibration")))
        assert kb.statistics().distinct_tags == 3

    def test_categories_covered(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.SAFETY_PROCEDURE.value))
        kb.register_document(_doc("B", category=KnowledgeCategory.FAILURE_CATALOG.value))
        assert kb.statistics().categories_covered == 2

    def test_unknown_excluded_from_coverage(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.UNKNOWN.value))
        assert kb.statistics().categories_covered == 0

    def test_full_coverage(self) -> None:
        kb = EnterpriseKnowledgeBase()
        for i, c in enumerate(KnowledgeCategory):
            if c == KnowledgeCategory.UNKNOWN:
                continue
            kb.register_document(_doc(f"D{i}", category=c.value))
        assert kb.statistics().coverage_score == pytest.approx(1.0)

    def test_to_dict(self) -> None:
        d = _kb(5).statistics().to_dict()
        for k in ("document_count", "category_distribution", "tag_distribution",
                  "coverage_score", "categories_covered", "distinct_tags"):
            assert k in d

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(_kb(5).statistics().to_dict()), str)


# ===========================================================================
# Serialization
# ===========================================================================


class TestKBSerialization:
    def test_to_dict_structure(self) -> None:
        d = _kb(3).to_dict()
        assert "documents" in d and "collections" in d

    def test_round_trip(self) -> None:
        kb = _kb(5)
        kb2 = EnterpriseKnowledgeBase.from_dict(kb.to_dict())
        assert len(kb2) == len(kb)

    def test_round_trip_preserves_docs(self) -> None:
        kb = _kb(5)
        kb2 = EnterpriseKnowledgeBase.from_dict(kb.to_dict())
        assert kb2.get_document("D000").title == kb.get_document("D000").title

    def test_round_trip_collections(self) -> None:
        kb = _kb(4)
        kb.make_collection("s", ["D000", "D001"])
        kb2 = EnterpriseKnowledgeBase.from_dict(kb.to_dict())
        assert "s" in kb2.list_collections()

    def test_to_json(self) -> None:
        assert isinstance(_kb(3).to_json(), str)

    def test_json_round_trip(self) -> None:
        kb = _kb(4)
        kb2 = EnterpriseKnowledgeBase.from_dict(json.loads(kb.to_json()))
        assert len(kb2) == 4

    def test_from_dict_requires_mapping(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            EnterpriseKnowledgeBase.from_dict([1, 2])

    def test_to_dict_documents_sorted(self) -> None:
        ids = [d["document_id"] for d in _kb(6).to_dict()["documents"]]
        assert ids == sorted(ids)


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    def test_keyword_search_deterministic(self) -> None:
        kb = _kb(10)
        a = [r.document_id for r in kb.search_keywords("bearing gearbox")]
        b = [r.document_id for r in kb.search_keywords("bearing gearbox")]
        assert a == b

    def test_across_instances(self) -> None:
        kb = _kb(8)
        kb2 = EnterpriseKnowledgeBase.from_dict(kb.to_dict())
        a = [r.document_id for r in kb.search_keywords("maintenance")]
        b = [r.document_id for r in kb2.search_keywords("maintenance")]
        assert a == b

    def test_statistics_deterministic(self) -> None:
        kb = _kb(10)
        assert kb.statistics().to_dict() == kb.statistics().to_dict()

    def test_list_deterministic(self) -> None:
        kb = _kb(10)
        assert [d.document_id for d in kb.list_documents()] == \
            [d.document_id for d in kb.list_documents()]


# ===========================================================================
# Class registry
# ===========================================================================


class TestClassRegistry:
    def test_registered(self) -> None:
        assert KB_NAME in KNOWLEDGE_BASE_REGISTRY and KB_NAME in list_knowledge_bases()

    def test_build(self) -> None:
        assert isinstance(build_knowledge_base(KB_NAME), EnterpriseKnowledgeBase)

    def test_build_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown knowledge base"):
            build_knowledge_base("nope")

    def test_registry_name(self) -> None:
        assert EnterpriseKnowledgeBase._registry_name == KB_NAME

    def test_duplicate_registration_raises(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            @register_knowledge_base(KB_NAME)
            class _Other:
                pass


# ===========================================================================
# Tracker
# ===========================================================================


class TestTracker:
    def test_logs_operations(self) -> None:
        logged = []

        class FakeTracker:
            def log_metrics(self, m, step=None):
                logged.append(m)

        kb = EnterpriseKnowledgeBase(experiment_tracker=FakeTracker())
        kb.register_document(_doc())
        assert logged and "kb_operations" in logged[0]

    def test_broken_tracker_survives(self) -> None:
        class BrokenTracker:
            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

        kb = EnterpriseKnowledgeBase(experiment_tracker=BrokenTracker())
        kb.register_document(_doc())
        assert len(kb) == 1

    def test_no_tracker_ok(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc())
        assert len(kb) == 1


# ===========================================================================
# Edge cases & scale
# ===========================================================================


class TestEdgeCases:
    def test_large_corpus(self) -> None:
        kb = _kb(500)
        assert len(kb) == 500
        assert len(kb.search_keywords("bearing", top_n=10)) <= 10

    def test_large_corpus_stats(self) -> None:
        s = _kb(300).statistics()
        assert s.document_count == 300 and 0 <= s.coverage_score <= 1

    def test_unicode_content(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("U", title="Análisis", content="vibración rodamiento análisis"))
        assert kb.has_document("U")

    def test_long_content(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("L", content="bearing " * 5000))
        assert kb.search_keywords("bearing")[0].document_id == "L"

    def test_many_tags(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("M", tags=tuple(f"tag{i}" for i in range(50))))
        assert len(kb.get_document("M").tags) == 50

    def test_register_remove_register(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A"))
        kb.remove_document("A")
        kb.register_document(_doc("A"))  # should not raise after removal
        assert kb.has_document("A")

    def test_search_empty_kb(self) -> None:
        kb = EnterpriseKnowledgeBase()
        assert kb.search_keywords("anything") == ()
        assert kb.search_tags(["x"]) == ()
        assert kb.search_exact("x") == ()

    def test_duplicate_after_roundtrip_safe(self) -> None:
        kb = _kb(3)
        kb2 = EnterpriseKnowledgeBase.from_dict(kb.to_dict())
        with pytest.raises(ValueError, match="duplicate"):
            kb2.register_document(_doc("D000"))

    def test_metadata_isolated(self) -> None:
        meta = {"k": 1}
        d = _doc("A", meta=meta)
        meta["k"] = 999
        assert d.metadata["k"] == 1  # stored a copy


# ===========================================================================
# SearchResult object
# ===========================================================================


class TestSearchResult:
    def test_fields(self) -> None:
        r = SearchResult("A", "Title", "unknown", 0.5, ("x",))
        assert r.document_id == "A" and r.score == 0.5

    def test_frozen(self) -> None:
        r = SearchResult("A", "Title", "unknown", 0.5, ("x",))
        with pytest.raises((AttributeError, TypeError)):
            r.score = 0.9  # type: ignore[misc]

    def test_to_dict(self) -> None:
        d = SearchResult("A", "Title", "unknown", 0.5, ("x",)).to_dict()
        for k in ("document_id", "title", "category", "score", "matched_terms"):
            assert k in d

    def test_to_dict_json(self) -> None:
        assert isinstance(json.dumps(
            SearchResult("A", "T", "unknown", 0.5, ("x",)).to_dict()), str)


# ===========================================================================
# Further coverage to reach the 180+ target
# ===========================================================================


class TestDocumentDepth:
    def test_document_type_must_be_string(self) -> None:
        with pytest.raises(ValueError, match="document_type"):
            KnowledgeDocument(document_id="X", title="t", content="c", document_type=5)

    def test_source_must_be_string(self) -> None:
        with pytest.raises(ValueError, match="source"):
            KnowledgeDocument(document_id="X", title="t", content="c", source=[1])

    def test_created_at_must_be_string(self) -> None:
        with pytest.raises(ValueError, match="created_at"):
            KnowledgeDocument(document_id="X", title="t", content="c", created_at=2026)

    def test_version_must_be_string(self) -> None:
        with pytest.raises(ValueError, match="version"):
            KnowledgeDocument(document_id="X", title="t", content="c", version=1.0)

    def test_metadata_with_list_of_dicts(self) -> None:
        d = _doc(meta={"items": [{"a": 1}, {"b": 2}]})
        assert d.metadata["items"][0]["a"] == 1

    def test_metadata_non_string_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="metadata"):
            _doc(meta={1: "x"})

    def test_duplicate_tags_preserved(self) -> None:
        # tags are not de-duplicated at the document level
        d = _doc(tags=("bearing", "bearing"))
        assert d.tags == ("bearing", "bearing")

    def test_token_sets_three(self) -> None:
        assert len(_doc().token_sets()) == 3


class TestKeywordSearchDepth:
    def test_query_normalised(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", content="bearing fault"))
        assert kb.search_keywords("BEARING")[0].document_id == "A"

    def test_punctuation_ignored(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", content="bearing, vibration; fault."))
        assert kb.search_keywords("bearing vibration")[0].document_id == "A"

    def test_top_n_none_returns_all_matches(self) -> None:
        kb = EnterpriseKnowledgeBase()
        for i in range(15):
            kb.register_document(_doc(f"D{i:02d}", content="maintenance bearing"))
        # default_top_n caps at 10 when top_n omitted
        assert len(kb.search_keywords("bearing")) == 10

    def test_matched_terms_sorted(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", content="vibration bearing alpha"))
        mt = kb.search_keywords("vibration bearing alpha")[0].matched_terms
        assert list(mt) == sorted(mt)

    def test_score_full_match(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", title="bearing", content="bearing", tags=("bearing",)))
        assert kb.search_keywords("bearing")[0].score == pytest.approx(1.0)


class TestTagSearchDepth:
    def test_no_overlap_excluded(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("oil",)))
        assert kb.search_tags(["bearing"]) == ()

    def test_whitespace_tags_ignored(self) -> None:
        assert _kb(3).search_tags(["  ", ""]) == ()

    def test_match_all_partial_excluded(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing",)))
        assert kb.search_tags(["bearing", "vibration", "oil"], match_all=True) == ()


class TestRelevanceDepth:
    def test_query_and_category(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc("A", content="bearing", category=KnowledgeCategory.SAFETY_PROCEDURE.value)
        s = kb.relevance_score(d, query="bearing", category="safety_procedure")
        assert 0 < s <= 1

    def test_weights_respected(self) -> None:
        kb = EnterpriseKnowledgeBase(KnowledgeBaseConfig(
            keyword_component_weight=1.0, tag_component_weight=0.0,
            category_component_weight=0.0))
        d = _doc("A", content="bearing", category=KnowledgeCategory.SAFETY_PROCEDURE.value)
        # category contributes nothing under these weights
        s_cat = kb.relevance_score(d, category="failure_catalog")
        assert s_cat == 0.0

    def test_empty_query_string_no_keyword(self) -> None:
        kb = EnterpriseKnowledgeBase()
        d = _doc("A", tags=("bearing",))
        # empty query falsy -> only tags counted
        assert kb.relevance_score(d, query="", tags=["bearing"]) == pytest.approx(1.0)


class TestCollectionsDepth:
    def test_category_collection_empty(self) -> None:
        kb = _kb(0)
        assert len(kb.category_collection("failure_catalog")) == 0

    def test_fleet_collection_none_match(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", meta={"fleet_id": "OTHER"}))
        assert len(kb.fleet_collection("FLEET-X")) == 0

    def test_make_collection_custom_kind(self) -> None:
        kb = _kb(3)
        coll = kb.make_collection("s", ["D000"], kind="custom")
        assert coll.kind == "custom"

    def test_make_collection_metadata(self) -> None:
        kb = _kb(3)
        coll = kb.make_collection("s", ["D000"], metadata={"owner": "eng"})
        assert coll.metadata["owner"] == "eng"

    def test_collection_stored(self) -> None:
        kb = _kb(3)
        kb.make_collection("s", ["D000"])
        assert "s" in kb.list_collections()

    def test_category_collection_stored(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.SAFETY_PROCEDURE.value))
        kb.category_collection("safety_procedure")
        assert "category:safety_procedure" in kb.list_collections()


class TestStatisticsDepth:
    def test_single_document(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.SAFETY_PROCEDURE.value))
        s = kb.statistics()
        assert s.document_count == 1 and s.categories_covered == 1

    def test_coverage_fraction(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.SAFETY_PROCEDURE.value))
        # 1 of 9 non-unknown categories
        assert kb.statistics().coverage_score == pytest.approx(1/9)

    def test_tag_counts(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", tags=("bearing",)))
        kb.register_document(_doc("B", tags=("bearing",)))
        td = dict(kb.statistics().tag_distribution)
        assert td["bearing"] == 2

    def test_category_counts(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", category=KnowledgeCategory.FAILURE_CATALOG.value))
        kb.register_document(_doc("B", category=KnowledgeCategory.FAILURE_CATALOG.value))
        cd = dict(kb.statistics().category_distribution)
        assert cd["failure_catalog"] == 2


class TestSerializationDepth:
    def test_empty_kb_serializes(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb2 = EnterpriseKnowledgeBase.from_dict(kb.to_dict())
        assert len(kb2) == 0

    def test_from_dict_skips_unknown_collection_ids(self) -> None:
        data = {"documents": [_doc("A").to_dict()],
                "collections": [{"name": "c", "kind": "document",
                                 "document_ids": ["A", "GHOST"], "metadata": {}}]}
        kb = EnterpriseKnowledgeBase.from_dict(data)
        assert kb.get_collection("c").document_ids == ("A",)

    def test_metadata_survives_round_trip(self) -> None:
        kb = EnterpriseKnowledgeBase()
        kb.register_document(_doc("A", meta={"fleet_id": "F", "nested": {"x": [1, 2]}}))
        kb2 = EnterpriseKnowledgeBase.from_dict(kb.to_dict())
        assert kb2.get_document("A").metadata["nested"]["x"] == [1, 2]

    def test_config_passed_to_from_dict(self) -> None:
        kb = _kb(3)
        kb2 = EnterpriseKnowledgeBase.from_dict(
            kb.to_dict(), config=KnowledgeBaseConfig(default_top_n=2))
        assert kb2.config.default_top_n == 2