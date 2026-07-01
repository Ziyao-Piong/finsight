"""Shared pytest fixtures for the ingestion tests.

The fixtures here deliberately avoid the network and the real (messy) SEC HTML.
Phase 2's hard, test-worthy logic is *section segmentation* and *section-aware
chunking* -- not HTTP. So we feed those functions a small synthetic filing that
reproduces the one structural trap that matters: every "Item" heading appears
twice, once in the table of contents near the top and once in the body. A correct
parser must split on the body occurrence, not the TOC one.
"""

from __future__ import annotations

import pytest

# A miniature 10-K in *cleaned-text* form (i.e. what parse.html_to_text would emit:
# one logical line per line, no HTML). It has:
#   * a table of contents listing Items 1, 1A, 7, 8 with page numbers, then
#   * the body, where each Item heading is followed by unique marker tokens
#     (TOKEN_BUSINESS / TOKEN_RISK / TOKEN_MDA / TOKEN_FIN) so tests can assert
#     exactly which body text landed in which section.
_SYNTHETIC_FILING = """\
APPLE INC. FORM 10-K
For the fiscal year ended September 30, 2023
TABLE OF CONTENTS
Item 1. Business 4
Item 1A. Risk Factors 10
Item 7. Management's Discussion and Analysis 25
Item 8. Financial Statements and Supplementary Data 40
PART I
Item 1. Business
The Company designs, manufactures and markets smartphones and personal computers.
TOKEN_BUSINESS describes the company's products and markets in detail.
Item 1A. Risk Factors
The Company's business is subject to numerous risks and uncertainties.
TOKEN_RISK macroeconomic conditions, competition, and supply-chain disruption.
The following risk factors should be read carefully by investors.
Item 7. Management's Discussion and Analysis of Financial Condition
TOKEN_MDA net sales increased 8% year over year driven by services revenue.
Management believes liquidity remains strong heading into the next fiscal year.
Item 8. Financial Statements and Supplementary Data
TOKEN_FIN total net sales were 383,285 million dollars for the fiscal year.
The accompanying notes are an integral part of these financial statements.
"""


@pytest.fixture
def synthetic_filing_text() -> str:
    """Cleaned-text 10-K with a TOC + body, each Item heading appearing twice."""
    return _SYNTHETIC_FILING
