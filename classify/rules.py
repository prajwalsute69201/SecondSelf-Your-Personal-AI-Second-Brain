"""SecondSelf — Deterministic Rule-Based Heuristic Signal Extraction & Scoring.

Implements pure, unit-testable heuristic signal extraction and scoring for the
PARA taxonomy. Operates strictly in-memory with zero I/O or network dependencies.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from classify.models import PARAClassification, Signal
from classify.taxonomy import PARACategory, get_criteria


# ------------------------------------------------------------------------------
# Pattern Definitions (Compiled for Performance & Determinism)
# ------------------------------------------------------------------------------

# 1. Project Patterns (Deadlines, deliverables, sprint goals, action items)
RE_NEGATIVE_DEADLINE = re.compile(
    r"(?i)\b(?:no|without|zero|not\s+have\s+any)\s+(?:active\s+)?deadlines?\b"
)
RE_DEADLINE = re.compile(
    r"(?i)\b(?:due(?:\s+by|\s+date|\s*:)|deadline(?:\s*:|\s+is|\s+of)|target(?:\s+date)?\s*:|deliver(?:\s+by)?\s*:?|finish(?:\s+by)?)\s*[:=]?\s*([A-Za-z0-9\-/\s,]+?)(?:\.|\n|$|;)",
)
RE_TIME_RELATIVE = re.compile(
    r"(?i)\b(?:by\s+(?:end\s+of\s+day|eod|tomorrow|friday|monday|tuesday|wednesday|thursday|sunday|saturday|next\s+week|this\s+month|end\s+of\s+week|eow|q[1-4]|end\s+of\s+quarter))\b",
)
RE_DATE_ISO = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
RE_SPRINT_MILESTONE = re.compile(
    r"(?i)\b(?:sprint\s+goal|milestone|release\s+target|launch\s+date|roadmap\s+target|mvp\s+release|v\d+\.\d+(?:\.\d+)?\s+release)\b",
)
RE_DELIVERABLES = re.compile(
    r"(?i)\b(?:action\s+items?|deliverables?|to-do\s+list|tasks?\s+to\s+complete|implementation\s+plan)\b",
)
RE_ACTION_VERBS = re.compile(
    r"(?i)\b(?:ship|launch|deliver|implement|build\s+v\d+|deploy\s+to\s+prod(?:uction)?)\b",
)

# 2. Area Patterns (Ongoing responsibilities, recurring cadences, standards, routines)
RE_RECURRING_CADENCE = re.compile(
    r"(?i)\b(?:weekly\s+review|monthly\s+review|daily\s+habit|recurring\s+cadence|weekly\s+sync|monthly\s+budget|daily\s+routine|quarterly\s+audit)\b",
)
RE_MAINTENANCE = re.compile(
    r"(?i)\b(?:continuous\s+maintenance|ongoing\s+maintenance|system\s+uptime|infrastructure\s+monitoring|continuous\s+upkeep|service\s+health)\b",
)
RE_SOP_STANDARDS = re.compile(
    r"(?i)\b(?:standard\s+operating\s+procedure|sop|standards?\s+to\s+maintain|guidelines?\s+to\s+follow|operating\s+protocols?|quality\s+standards?)\b",
)
RE_RESPONSIBILITIES = re.compile(
    r"(?i)\b(?:health\s+tracking|workout\s+routine|budget\s+management|personal\s+finances?|team\s+1-on-1s?|career\s+development|kpis?|okrs?|personal\s+governance)\b",
)

# 3. Resource Patterns (Reference, cheatsheets, tutorials, guides, documentation)
RE_CHEATSHEET_REFERENCE = re.compile(
    r"(?i)\b(?:cheatsheet|cheat\s+sheet|syntax\s+reference|api\s+reference|reference\s+manual|technical\s+handbook|quick\s+reference)\b",
)
RE_DOCUMENTATION_GUIDE = re.compile(
    r"(?i)\b(?:documentation|docs|tutorial|guide\s+to|how-to|deep\s+dive|introduction\s+to|overview\s+of|summary\s+of|walkthrough|crash\s+course)\b",
)
RE_CURATED_LIST = re.compile(
    r"(?i)\b(?:curated\s+list|awesome-[a-zA-Z0-9_\-]+|reading\s+list|bookmarks?|bibliography|glossary|library\s+catalog)\b",
)
RE_SPEC_RESEARCH = re.compile(
    r"(?i)\b(?:whitepaper|specification|rfc\s*\d+|research\s+paper|theoretical\s+foundations?|architecture\s+notes?)\b",
)

# 4. Archive Patterns (Completed, deprecated, legacy, retrospective, stale)
RE_COMPLETED = re.compile(
    r"(?i)\b(?:status\s*:\s*(?:done|completed|shipped|finished|closed)|\[x\]\s+completed|all\s+tasks\s+finished|successfully\s+shipped)\b",
)
RE_STATUS_EXPLICIT = re.compile(
    r"(?i)\b(?:DONE|COMPLETED|SHIPPED|FINISHED|RESOLVED|CLOSED)\b",
)
RE_DEPRECATED_LEGACY = re.compile(
    r"(?i)\b(?:deprecated|obsolete|superseded|legacy\s+code|abandoned|retired|discontinued|historical\s+archive)\b",
)
RE_RETROSPECTIVE = re.compile(
    r"(?i)\b(?:post-mortem|postmortem|retrospective|lessons\s+learned|project\s+wrap-up|incident\s+report)\b",
)


# Deterministic tie-breaker priority (stable ordering)
TIE_BREAKER_ORDER: tuple[PARACategory, ...] = (
    PARACategory.PROJECT,
    PARACategory.AREA,
    PARACategory.RESOURCE,
    PARACategory.ARCHIVE,
)


def extract_signals(
    content: str,
    metadata: dict[str, Any] | None = None,
    archive_staleness_days: int = 180,
) -> list[Signal]:
    """Extracts deterministic evidence signals from raw text and sidecar metadata.

    Args:
        content: Clean plain text or markdown content.
        metadata: Optional dictionary with capture attributes (created_at, source_uri, etc.).
        archive_staleness_days: Number of days before an inactive note qualifies for Archive.

    Returns:
        List of Signal instances sorted deterministically.
    """
    signals: list[Signal] = []
    text = content.strip()
    meta = metadata or {}

    # --------------------------------------------------------------------------
    # 1. Project Signals
    # --------------------------------------------------------------------------
    has_negative_deadline = bool(RE_NEGATIVE_DEADLINE.search(text))
    deadline_match = RE_DEADLINE.search(text) if not has_negative_deadline else None
    if deadline_match:
        matched_str = deadline_match.group(0).strip()
        signals.append(
            Signal(
                name="explicit_deadline",
                category=PARACategory.PROJECT,
                weight=2.5,
                matched_text=matched_str[:80],
                source="rule_content",
                description=f"Explicit deadline pattern detected: '{matched_str[:40]}'",
            )
        )

    time_rel_match = RE_TIME_RELATIVE.search(text) if not has_negative_deadline else None
    if time_rel_match:
        signals.append(
            Signal(
                name="relative_time_bound",
                category=PARACategory.PROJECT,
                weight=1.8,
                matched_text=time_rel_match.group(0),
                source="rule_content",
                description=f"Time-bound relative horizon detected: '{time_rel_match.group(0)}'",
            )
        )

    iso_match = RE_DATE_ISO.search(text)
    if iso_match and not deadline_match:
        signals.append(
            Signal(
                name="iso_date_marker",
                category=PARACategory.PROJECT,
                weight=0.8,
                matched_text=iso_match.group(0),
                source="rule_content",
                description=f"Explicit calendar date marker found: '{iso_match.group(0)}'",
            )
        )

    sprint_match = RE_SPRINT_MILESTONE.search(text)
    if sprint_match:
        signals.append(
            Signal(
                name="sprint_milestone_marker",
                category=PARACategory.PROJECT,
                weight=1.6,
                matched_text=sprint_match.group(0),
                source="rule_content",
                description=f"Sprint or milestone marker identified: '{sprint_match.group(0)}'",
            )
        )

    deliverable_match = RE_DELIVERABLES.search(text)
    if deliverable_match:
        signals.append(
            Signal(
                name="actionable_deliverables",
                category=PARACategory.PROJECT,
                weight=1.4,
                matched_text=deliverable_match.group(0),
                source="rule_content",
                description=f"Actionable deliverables or task list detected: '{deliverable_match.group(0)}'",
            )
        )

    action_verb_match = RE_ACTION_VERBS.search(text)
    if action_verb_match:
        signals.append(
            Signal(
                name="project_action_verb",
                category=PARACategory.PROJECT,
                weight=1.0,
                matched_text=action_verb_match.group(0),
                source="rule_content",
                description=f"Goal-oriented execution verb: '{action_verb_match.group(0)}'",
            )
        )

    # --------------------------------------------------------------------------
    # 2. Area Signals
    # --------------------------------------------------------------------------
    recurring_match = RE_RECURRING_CADENCE.search(text)
    if recurring_match:
        signals.append(
            Signal(
                name="recurring_cadence_marker",
                category=PARACategory.AREA,
                weight=2.2,
                matched_text=recurring_match.group(0),
                source="rule_content",
                description=f"Recurring cadence or ongoing routine marker: '{recurring_match.group(0)}'",
            )
        )

    sop_match = RE_SOP_STANDARDS.search(text)
    if sop_match:
        signals.append(
            Signal(
                name="sop_standards_marker",
                category=PARACategory.AREA,
                weight=2.0,
                matched_text=sop_match.group(0),
                source="rule_content",
                description=f"Standard operating procedure or standard of performance: '{sop_match.group(0)}'",
            )
        )

    maint_match = RE_MAINTENANCE.search(text)
    if maint_match:
        signals.append(
            Signal(
                name="ongoing_maintenance_marker",
                category=PARACategory.AREA,
                weight=1.8,
                matched_text=maint_match.group(0),
                source="rule_content",
                description=f"Continuous maintenance and upkeep domain: '{maint_match.group(0)}'",
            )
        )

    resp_match = RE_RESPONSIBILITIES.search(text)
    if resp_match:
        signals.append(
            Signal(
                name="personal_responsibility_domain",
                category=PARACategory.AREA,
                weight=1.5,
                matched_text=resp_match.group(0),
                source="rule_content",
                description=f"Ongoing domain of responsibility: '{resp_match.group(0)}'",
            )
        )

    # --------------------------------------------------------------------------
    # 3. Resource Signals
    # --------------------------------------------------------------------------
    cheat_match = RE_CHEATSHEET_REFERENCE.search(text)
    if cheat_match:
        signals.append(
            Signal(
                name="cheatsheet_reference_marker",
                category=PARACategory.RESOURCE,
                weight=2.4,
                matched_text=cheat_match.group(0),
                source="rule_content",
                description=f"Cheatsheet or reference manual indicator: '{cheat_match.group(0)}'",
            )
        )

    doc_match = RE_DOCUMENTATION_GUIDE.search(text)
    if doc_match:
        signals.append(
            Signal(
                name="documentation_guide_marker",
                category=PARACategory.RESOURCE,
                weight=1.8,
                matched_text=doc_match.group(0),
                source="rule_content",
                description=f"Tutorial, guide, or conceptual documentation: '{doc_match.group(0)}'",
            )
        )

    curated_match = RE_CURATED_LIST.search(text)
    if curated_match:
        signals.append(
            Signal(
                name="curated_list_marker",
                category=PARACategory.RESOURCE,
                weight=1.6,
                matched_text=curated_match.group(0),
                source="rule_content",
                description=f"Curated list or library compilation: '{curated_match.group(0)}'",
            )
        )

    spec_match = RE_SPEC_RESEARCH.search(text)
    if spec_match:
        signals.append(
            Signal(
                name="specification_research_marker",
                category=PARACategory.RESOURCE,
                weight=1.4,
                matched_text=spec_match.group(0),
                source="rule_content",
                description=f"Specification or technical research reference: '{spec_match.group(0)}'",
            )
        )

    # Metadata source boost for URLs / PDFs (typically reference material unless time-bound)
    src_type = str(meta.get("source_type", "")).lower()
    if src_type in ("url", "pdf") and not deadline_match and not recurring_match:
        signals.append(
            Signal(
                name="external_reference_source",
                category=PARACategory.RESOURCE,
                weight=0.7,
                matched_text=f"source_type={src_type}",
                source="rule_metadata",
                description=f"External ingestion source ({src_type}) indicates reference material",
            )
        )

    # --------------------------------------------------------------------------
    # 4. Archive Signals
    # --------------------------------------------------------------------------
    completed_match = RE_COMPLETED.search(text) or RE_STATUS_EXPLICIT.search(text)
    if completed_match:
        signals.append(
            Signal(
                name="completed_status_marker",
                category=PARACategory.ARCHIVE,
                weight=3.0,
                matched_text=completed_match.group(0),
                source="rule_content",
                description=f"Completed / done / shipped status marker: '{completed_match.group(0)}'",
            )
        )

    deprecated_match = RE_DEPRECATED_LEGACY.search(text)
    if deprecated_match:
        signals.append(
            Signal(
                name="deprecated_legacy_marker",
                category=PARACategory.ARCHIVE,
                weight=2.8,
                matched_text=deprecated_match.group(0),
                source="rule_content",
                description=f"Deprecated or historical legacy status: '{deprecated_match.group(0)}'",
            )
        )

    retro_match = RE_RETROSPECTIVE.search(text)
    if retro_match:
        signals.append(
            Signal(
                name="retrospective_postmortem_marker",
                category=PARACategory.ARCHIVE,
                weight=2.5,
                matched_text=retro_match.group(0),
                source="rule_content",
                description=f"Retrospective or post-mortem archive indicator: '{retro_match.group(0)}'",
            )
        )

    # Staleness check based on metadata timestamps
    created_at_raw = meta.get("created_at") or meta.get("timestamp")
    if created_at_raw:
        try:
            # Handle ISO formats
            dt_str = str(created_at_raw).replace("Z", "+00:00")
            parsed_dt = datetime.datetime.fromisoformat(dt_str)
            now = datetime.datetime.now(datetime.timezone.utc)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=datetime.timezone.utc)
            age_days = (now - parsed_dt).days
            if age_days >= archive_staleness_days:
                # Add stale signal
                signals.append(
                    Signal(
                        name="historical_staleness",
                        category=PARACategory.ARCHIVE,
                        weight=2.2,
                        matched_text=f"age={age_days}d",
                        source="rule_metadata",
                        description=f"Record age ({age_days} days) exceeds staleness threshold ({archive_staleness_days} days)",
                    )
                )
        except Exception:
            pass

    # Sort signals deterministically by weight descending, then name
    signals.sort(key=lambda s: (-s.weight, s.name))
    return signals


def score_rules(
    signals: list[Signal],
    confidence_threshold: float = 0.60,
    default_fallback: PARACategory = PARACategory.RESOURCE,
) -> tuple[PARACategory, float, dict[str, float], bool, str]:
    """Computes categorical score distribution and determines the winning category.

    Args:
        signals: List of extracted evidence signals.
        confidence_threshold: Threshold below which `needs_review` is flagged.
        default_fallback: Default category when no signals exist.

    Returns:
        Tuple of (selected_category, confidence, category_scores, needs_review, rationale).
    """
    # Initialize zero score accumulator for all 4 categories
    raw_scores: dict[PARACategory, float] = {cat: 0.0 for cat in PARACategory}

    for sig in signals:
        raw_scores[sig.category] += sig.weight

    total_weight = sum(raw_scores.values())

    # Case 1: Zero signals detected -> Safe fallback with low confidence
    if total_weight <= 0.0:
        scores_dist = {cat.value: 0.25 for cat in PARACategory}
        rationale = (
            f"Zero definitive PARA heuristic signals matched; defaulted to '{default_fallback.value}' "
            "with baseline low confidence (0.25) awaiting human verification."
        )
        return (default_fallback, 0.25, scores_dist, True, rationale)

    # Case 2: Normalize scores into a valid probability distribution (0.0 to 1.0)
    category_scores: dict[str, float] = {
        cat.value: raw_scores[cat] / total_weight for cat in PARACategory
    }

    # Deterministic winner selection: sort by score descending, then tie-breaker order
    def score_sort_key(cat: PARACategory) -> tuple[float, int]:
        # High score first, then tie breaker priority
        priority_index = len(TIE_BREAKER_ORDER) - TIE_BREAKER_ORDER.index(cat)
        return (raw_scores[cat], priority_index)

    sorted_categories = sorted(PARACategory, key=score_sort_key, reverse=True)
    winner = sorted_categories[0]
    runner_up = sorted_categories[1]

    winner_score = category_scores[winner.value]
    runner_up_score = category_scores[runner_up.value]

    # Evaluate confidence and ambiguity
    needs_review = winner_score < confidence_threshold
    margin = winner_score - runner_up_score
    if margin <= 0.05 and winner_score > 0.0:
        needs_review = True

    # Build human-readable rationale
    top_signals = [s for s in signals if s.category == winner]
    signal_reasons = "; ".join(s.description for s in top_signals[:3])
    if not signal_reasons:
        signal_reasons = "No direct signals"

    if needs_review:
        rationale = (
            f"Classified as '{winner.value}' with marginal confidence ({winner_score:.2f}) "
            f"below threshold ({confidence_threshold:.2f}). Runner-up: '{runner_up.value}' ({runner_up_score:.2f}). "
            f"Signals used: {signal_reasons}."
        )
    else:
        rationale = (
            f"Confidently classified as '{winner.value}' ({winner_score:.2f} score). "
            f"Supported by primary signals: {signal_reasons}."
        )

    return (winner, winner_score, category_scores, needs_review, rationale)
