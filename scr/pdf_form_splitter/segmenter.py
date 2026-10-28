"""Pure multi-signal segmentation of combined relay-form page evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .evidence import PageEvidence


STRONG_COVER_SCORE = 0.45
SUPPORTING_COVER_SCORE = 0.28


@dataclass(frozen=True)
class DocumentSegment:
    segment_index: int
    start_page: int
    end_page: int
    page_count: int
    ticket_number: str | None
    expected_total_pages: int | None
    start_reasons: list[str]
    end_reason: str
    confidence: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _most_common(values: Iterable[str | None]) -> str | None:
    populated = [value for value in values if value]
    return Counter(populated).most_common(1)[0][0] if populated else None


def _start_reasons(page: PageEvidence, active: list[PageEvidence], previous_was_terminal: bool) -> list[str]:
    reasons: list[str] = []
    if not active:
        reasons.append("after_terminal_page" if previous_was_terminal else "first_or_orphan_page")
    if page.cover_score >= STRONG_COVER_SCORE:
        reasons.append("strong_page1_signature")
    if page.current_page == 1 and page.cover_score >= SUPPORTING_COVER_SCORE:
        reasons.append("page_one_pagination_with_cover_signature")
    active_ticket = _most_common(item.ticket_number for item in active)
    if active and active_ticket and page.ticket_number and page.ticket_number != active_ticket:
        if page.current_page == 1 or page.cover_score >= SUPPORTING_COVER_SCORE:
            reasons.append("ticket_changed_with_start_evidence")
    return reasons


def _segment_warnings(pages: list[PageEvidence], end_reason: str) -> list[str]:
    warnings: list[str] = []
    first = pages[0]
    if first.current_page not in (None, 1):
        warnings.append("segment_does_not_start_at_logical_page_1")
    totals = {page.total_pages for page in pages if page.total_pages is not None}
    if len(totals) > 1:
        warnings.append("inconsistent_total_pages")
    tickets = {page.ticket_number for page in pages if page.ticket_number}
    if len(tickets) > 1:
        warnings.append("ticket_changed_inside_segment")
    numbered = [page for page in pages if page.current_page is not None]
    for previous, current in zip(numbered, numbered[1:]):
        physical_delta = current.page_index - previous.page_index
        logical_delta = current.current_page - previous.current_page
        if logical_delta != physical_delta:
            warnings.append(f"pagination_jump:{previous.current_page}->{current.current_page}")
    expected_total = _most_common(str(page.total_pages) if page.total_pages is not None else None for page in pages)
    if expected_total and first.current_page == 1 and len(pages) != int(expected_total):
        warnings.append(f"physical_page_count_mismatch:{len(pages)}!={expected_total}")
    if end_reason != "pagination_terminal":
        warnings.append("terminal_page_not_confirmed")
    return list(dict.fromkeys(warnings))


def _finalise(
    pages: list[PageEvidence],
    segment_index: int,
    start_reasons: list[str],
    end_reason: str,
) -> DocumentSegment:
    totals = [page.total_pages for page in pages if page.total_pages is not None]
    expected_total = Counter(totals).most_common(1)[0][0] if totals else None
    warnings = _segment_warnings(pages, end_reason)
    start_strength = 0.95 if "strong_page1_signature" in start_reasons else 0.82
    if "after_terminal_page" in start_reasons:
        start_strength = max(start_strength, 0.90)
    end_strength = 0.98 if end_reason == "pagination_terminal" else 0.62
    confidence = max(0.0, min(1.0, (start_strength + end_strength) / 2 - 0.04 * len(warnings)))
    return DocumentSegment(
        segment_index=segment_index,
        start_page=pages[0].page_index,
        end_page=pages[-1].page_index,
        page_count=len(pages),
        ticket_number=_most_common(page.ticket_number for page in pages),
        expected_total_pages=expected_total,
        start_reasons=start_reasons,
        end_reason=end_reason,
        confidence=round(confidence, 4),
        warnings=warnings,
    )


def segment_pages(pages: Iterable[PageEvidence]) -> list[DocumentSegment]:
    """Split ordered page evidence without trusting pagination alone."""

    ordered = sorted(pages, key=lambda page: page.page_index)
    if not ordered:
        return []
    if [page.page_index for page in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("Page evidence must cover each physical page exactly once from 1..N.")

    segments: list[DocumentSegment] = []
    active: list[PageEvidence] = []
    active_reasons: list[str] = []
    previous_was_terminal = False
    for page in ordered:
        reasons = _start_reasons(page, active, previous_was_terminal)
        boundary_reasons = [
            reason for reason in reasons if reason in {
                "strong_page1_signature",
                "page_one_pagination_with_cover_signature",
                "ticket_changed_with_start_evidence",
            }
        ]
        if active and boundary_reasons:
            segments.append(_finalise(active, len(segments) + 1, active_reasons, "next_page_start_evidence"))
            active = []
            previous_was_terminal = False
            reasons = [*boundary_reasons]
        if not active:
            active_reasons = reasons or ["unconfirmed_orphan_start"]
        active.append(page)
        if page.is_terminal:
            segments.append(_finalise(active, len(segments) + 1, active_reasons, "pagination_terminal"))
            active = []
            active_reasons = []
            previous_was_terminal = True
        else:
            previous_was_terminal = False
    if active:
        segments.append(_finalise(active, len(segments) + 1, active_reasons, "input_eof"))
    return segments
