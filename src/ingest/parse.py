"""HTML -> clean text -> labelled sections for SEC 10-K filings.

Phase 1 stopped at "HTML -> one big blob of text" and chunked it blindly. That
blob has no idea where Risk Factors ends and the MD&A begins, so a naive chunk can
straddle two unrelated topics and retrieval can't filter by section. Phase 2 fixes
that here: we still strip the HTML to text, but then we *segment* the text into the
10-K's canonical Items so each chunk can be tagged with the section it came from.

### How section detection works (and where it's fragile)

A 10-K names its sections with headings like ``Item 1A. Risk Factors`` and
``Item 7. Management's Discussion and Analysis``. The catch: every one of those
headings appears at least twice — once in the table of contents near the top, and
again where the real section starts. Two tricks keep us honest:

1. **Line-anchoring.** We match an Item heading only at the start of a line
   (``^Item 1A`` with ``re.MULTILINE``). Cross-references buried in prose
   ("...as described in Item 8...") sit mid-line and are ignored. This is why
   :func:`html_to_text` emits one logical line per source line.
2. **Last occurrence wins.** Of the (usually two) line-anchored matches for a given
   Item, the body heading is the later one, because the whole TOC precedes the body.
   We take each Item's last match as its section start.

For line-anchoring to work the text must first be whitespace-normalised: SEC HTML
writes headings with non-breaking spaces (``Item&nbsp;8.``), and an unnormalised
U+00A0 between "Item" and the number makes the heading regex miss it entirely — that
is exactly how NVIDIA's Item 8 went undetected before we folded NBSP to a plain space.

Known rough edges (acceptable for this learning project, documented on purpose):
a stray line-anchored "Item N" after the real section, or a filing that repeats its
Items in an exhibit index, can mis-place a boundary. A production parser would lean
on the HTML structure (heading tags / anchors) instead of text heuristics — that's a
deliberate later upgrade, not a Phase 2 requirement.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Modern 10-K primary documents are Inline XBRL (XHTML/XML); get_text() still pulls
# clean text out of them, so silence the (correct but noisy) "looks like XML" warning.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# The 10-K Items, in the order they appear in the filing. Used both to recognise
# headings and to keep the emitted sections in canonical document order.
ITEM_ORDER: tuple[str, ...] = (
    "1", "1A", "1B", "1C", "2", "3", "4",
    "5", "6", "7", "7A", "8", "9", "9A", "9B",
    "10", "11", "12", "13", "14", "15",
)

# Friendly labels for the sections we actually care about downstream (retrieval
# filtering + citations). Anything not listed falls back to ``Item {id}``.
SECTION_LABELS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "7": "MD&A",
    "7A": "Market Risk",
    "8": "Financial Statements",
}

# Unicode space separators SEC HTML uses inside headings (non-breaking, narrow
# no-break, thin). Folded to a plain space so line-anchored heading detection works.
_UNICODE_SPACES = re.compile("[     ]")

# Line-anchored Item heading: optional leading whitespace, "Item", the id (a number
# with an optional letter suffix), then a separator (period, colon, space, dash) so we
# don't match "Item 10" when looking for "Item 1". Case-insensitive for "ITEM 1A".
_ITEM_HEADING = re.compile(
    r"(?im)^[ \t]*Item[ \t]+(\d{1,2}[A-C]?)[ \t]*[.:)\-–]",
)


@dataclass(frozen=True)
class Section:
    """One labelled slice of a filing.

    Attributes:
        item: The Item id as it appears in the filing, e.g. ``"1A"`` or ``"7"``.
        label: Human-friendly section name (``SECTION_LABELS`` or ``Item {id}``).
        text: The section's text, heading line included.
        char_start: Offset of the section start in the cleaned full text.
        char_end: Offset of the section end (exclusive); ``text == full[start:end]``.
    """

    item: str
    label: str
    text: str
    char_start: int
    char_end: int


def html_to_text(html: str) -> str:
    """Strip a filing's HTML to clean, line-oriented plain text.

    Same idea as Phase 1's stripper (drop script/style, collapse whitespace) with two
    deliberate changes: we join on newlines rather than spaces so each source line stays
    on its own line (that line structure is what lets :func:`segment_sections`
    line-anchor Item headings), and we fold unicode spaces (NBSP &c.) to plain spaces so
    headings written ``Item&nbsp;8.`` are still recognised.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = _UNICODE_SPACES.sub(" ", text)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def segment_sections(text: str) -> list[Section]:
    """Split cleaned filing text into labelled :class:`Section` slices.

    Returns sections in document order, each spanning from its Item heading up to the
    next section's heading. Everything before the first body section (cover page, table
    of contents) is intentionally dropped — it's navigation, not content.
    """
    # Last line-anchored occurrence per Item id == the body heading (the TOC precedes it).
    last_pos: dict[str, int] = {}
    for m in _ITEM_HEADING.finditer(text):
        item = m.group(1).upper()
        if item in ITEM_ORDER:
            last_pos[item] = m.start()

    if not last_pos:
        return []

    # Order the body headings by where they sit in the document. Sorting by position
    # (rather than canonical order) means a filing that lists Items out of sequence
    # still yields contiguous, non-overlapping slices.
    starts = sorted(last_pos.items(), key=lambda kv: kv[1])

    sections: list[Section] = []
    for idx, (item, start) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(text)
        label = SECTION_LABELS.get(item, f"Item {item}")
        sections.append(
            Section(
                item=item,
                label=label,
                text=text[start:end],
                char_start=start,
                char_end=end,
            )
        )
    return sections
