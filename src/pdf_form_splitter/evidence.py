"""Build auditable page-boundary evidence from recognised OCR blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import unicodedata
from typing import Any, Iterable

from src.layout_analysis.pagination import TICKET_PATTERN, detect_page_reference


COVER_FEATURES: tuple[tuple[str, float], ...] = (
    ("phieu chinh dinh ro le bao ve", 0.20),
    ("mo ta chung", 0.18),
    ("thiet bi duoc bao ve", 0.18),
    ("nguyen tac hoat dong", 0.18),
    ("muc dich ban hanh phieu", 0.16),
    ("yeu cau cua trung tam dieu do", 0.10),
)


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    plain = "".join(char for char in decomposed if unicodedata.category(char) != "Mn").replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


@dataclass(frozen=True)
class PageEvidence:
    page_index: int
    page_reference: str | None
    current_page: int | None
    total_pages: int | None
    pagination_label: str | None
    ticket_number: str | None
    cover_score: float
    cover_features: list[str] = field(default_factory=list)
    source_block_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.current_page is not None and self.total_pages is not None and self.current_page == self.total_pages

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageEvidence":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__ if name in payload})


def build_page_evidence(
    page_index: int,
    blocks: Iterable[dict[str, Any]],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
) -> PageEvidence:
    """Combine pagination, ticket and cover-template evidence for one page."""

    source = [dict(block) for block in blocks if str(block.get("text", "")).strip()]
    pagination = detect_page_reference(source, page_width=page_width, page_height=page_height)
    joined = _normalise(" ".join(str(block.get("text", "")) for block in source))
    features = [phrase for phrase, _ in COVER_FEATURES if phrase in joined]
    score = min(1.0, sum(weight for phrase, weight in COVER_FEATURES if phrase in joined))
    ticket_candidates = [
        (float(block["bbox_pixel"][1]), -(float(block.get("recognition_score") or 0.0)), match.group(0))
        for block in source
        for match in TICKET_PATTERN.finditer(str(block.get("text", "")))
    ]
    ticket = min(ticket_candidates)[2] if ticket_candidates else None
    warnings: list[str] = []
    if pagination is None:
        warnings.append("pagination_not_detected")
    elif pagination["total_pages"] is None:
        warnings.append("pagination_total_missing")
    if score < 0.20:
        warnings.append("no_cover_signature")
    return PageEvidence(
        page_index=page_index,
        page_reference=None if pagination is None else pagination["text"],
        current_page=None if pagination is None else int(pagination["page_number"]),
        total_pages=None if pagination is None or pagination["total_pages"] is None else int(pagination["total_pages"]),
        pagination_label=None if pagination is None else pagination["matched_label"],
        ticket_number=ticket,
        cover_score=round(score, 4),
        cover_features=features,
        source_block_count=len(source),
        warnings=warnings,
    )
