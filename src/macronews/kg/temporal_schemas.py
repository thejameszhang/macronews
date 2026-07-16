"""Bitemporal, statement-first KG schemas (OpenAI-cookbook-mirrored).

Supersedes KGFact/modality in kg.schemas. Reuses ENTITY_TYPES (19 types) /
RELATION_TYPES (15 relations) from kg.schemas.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from macronews.kg.schemas import ENTITY_TYPES, RELATION_TYPES


class StatementType(StrEnum):
    FACT = "FACT"
    OPINION = "OPINION"
    PREDICTION = "PREDICTION"


class TemporalType(StrEnum):
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    ATEMPORAL = "ATEMPORAL"


class RawStatement(BaseModel):
    """Pass-1 output: one atomic, self-contained claim + its two type axes."""
    evidence_paragraphs: list[int] = Field(default_factory=list)
    statement: str
    statement_type: StatementType
    temporal_type: TemporalType


class RawStatementList(BaseModel):
    """Pass-1 guided-decoding wrapper (vLLM needs a top-level object schema)."""
    statements: list[RawStatement] = Field(default_factory=list)


class TemporalValidityRange(BaseModel):
    """Pass-2 output."""
    valid_at: datetime | None = None
    invalid_at: datetime | None = None


class RawTriplet(BaseModel):
    """Pass-3 output: a typed macro triple. `value` carries the magnitude."""
    subject: str
    subject_type: ENTITY_TYPES
    relation: RELATION_TYPES
    object: str
    object_type: ENTITY_TYPES
    value: str | None = None


class RawTripletList(BaseModel):
    """Pass-3 guided-decoding wrapper."""
    triplets: list[RawTriplet] = Field(default_factory=list)


class TemporalEvent(BaseModel):
    """The spine record — one per statement. No set_expired_at validator
    (unlike the cookbook): expired_at is set ONLY by the invalidation pass."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    article_id: str
    statement: str
    statement_type: StatementType
    temporal_type: TemporalType
    triplets: list[RawTriplet] = Field(default_factory=list)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    created_at: datetime | None = None   # = article publication date; None only for a date-less article
    expired_at: datetime | None = None
    invalidated_by: uuid.UUID | None = None
    invalidation_method: str | None = None   # "llm" (invalidation agent) | None
    evidence_paragraphs: list[int] = Field(default_factory=list)
    embedding: list[float] | None = None
