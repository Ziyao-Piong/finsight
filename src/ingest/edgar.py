"""Discover and fetch SEC filings from EDGAR by ticker + fiscal year.

Phase 1 hardcoded one Apple URL. Phase 2 needs to find *any* company's 10-K
programmatically, which means three EDGAR hops:

1. **Ticker -> CIK.** SEC publishes a ticker-to-CIK map at
   ``/files/company_tickers.json``. We download it once, cache it, and look up the
   zero-padded 10-digit CIK every other endpoint wants.
2. **CIK -> filing.** ``/submissions/CIK##########.json`` lists a company's filings as
   parallel arrays (form type, accession number, primary document, report date...).
   We scan for the requested form and fiscal year. The "recent" block only holds the
   latest ~1000 filings, so for busy filers we transparently page into the older
   ``files`` shards until we find the match.
3. **Filing -> document.** The primary document lives at a predictable Archives URL
   built from the CIK and accession number.

### Two things SEC requires (and we honour)
* A descriptive **User-Agent** with contact info (see
  :attr:`~src.config.Settings.edgar_user_agent`); requests without it get 403s.
* **Fair-access rate limits** (<=10 requests/second). We throttle to one request per
  0.2s and reuse a single session.

### A subtlety worth internalising: what "fiscal year" means
We match on the *period of report* (``reportDate``), not the filing date, and call its
year the fiscal year. Fiscal calendars differ — Apple's FY2023 ends 30 Sep 2023 while
NVIDIA's "fiscal 2024" ends in late January 2024 — so keying on the report date is the
only consistent way to line companies up by year for cross-document comparison later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings, get_settings

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SUBMISSIONS_SHARD_URL = "https://data.sec.gov/submissions/{name}"
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

_DATA_DIR = Path("data")
_TICKER_CACHE = _DATA_DIR / "company_tickers.json"

# SEC fair-access guidance is <=10 req/s; 0.2s between calls keeps us comfortably under.
_MIN_INTERVAL_S = 0.2
_last_request_at = 0.0
_session = None  # lazily created requests.Session


@dataclass(frozen=True)
class FilingRef:
    """Everything needed to fetch a filing and stamp its chunks with provenance."""

    ticker: str
    company: str
    cik: str  # zero-padded 10-digit
    accession: str  # dashed form, e.g. 0000320193-23-000106
    form: str
    fiscal_year: int
    report_date: str
    primary_document: str
    source_url: str


def _http(user_agent: str):
    """Return a shared requests.Session carrying the SEC-required User-Agent."""
    global _session
    if _session is None:
        import requests  # lazy: importing this module shouldn't require requests

        _session = requests.Session()
        _session.headers.update({"User-Agent": user_agent})
    return _session


def _get(url: str, user_agent: str):
    """GET ``url`` with throttling and the required header; raise on HTTP errors."""
    global _last_request_at
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    resp = _http(user_agent).get(url, timeout=30)
    _last_request_at = time.monotonic()
    resp.raise_for_status()
    return resp


def load_ticker_map(settings: Settings | None = None) -> dict[str, tuple[str, str]]:
    """Return ``{TICKER: (cik10, company_title)}``, downloading + caching once."""
    settings = settings or get_settings()
    if _TICKER_CACHE.exists():
        import json

        raw = json.loads(_TICKER_CACHE.read_text(encoding="utf-8"))
    else:
        raw = _get(_TICKERS_URL, settings.edgar_user_agent).json()
        _DATA_DIR.mkdir(exist_ok=True)
        import json

        _TICKER_CACHE.write_text(json.dumps(raw), encoding="utf-8")

    mapping: dict[str, tuple[str, str]] = {}
    for row in raw.values():
        cik10 = str(row["cik_str"]).zfill(10)
        mapping[row["ticker"].upper()] = (cik10, row["title"])
    return mapping


def _rows(block: dict):
    """Yield per-filing dicts from a submissions array block (recent or a shard)."""
    forms = block["form"]
    accs = block["accessionNumber"]
    docs = block.get("primaryDocument", [""] * len(forms))
    reports = block["reportDate"]
    for i in range(len(forms)):
        yield {
            "form": forms[i],
            "accession": accs[i],
            "primary": docs[i],
            "report_date": reports[i],
        }


def _iter_filings(cik10: str, settings: Settings):
    """Yield every filing for a CIK: the recent block first, then older shards lazily."""
    data = _get(_SUBMISSIONS_URL.format(cik=cik10), settings.edgar_user_agent).json()
    yield from _rows(data["filings"]["recent"])
    # Busy filers overflow the ~1000-row "recent" block into dated shards; only fetched
    # if the loop below exhausts "recent" without a match.
    for shard in data["filings"].get("files", []):
        older = _get(
            _SUBMISSIONS_SHARD_URL.format(name=shard["name"]), settings.edgar_user_agent
        ).json()
        yield from _rows(older)


def filing_url(cik10: str, accession: str, primary_document: str) -> str:
    """Build the Archives URL for a filing's primary document."""
    return _ARCHIVES_URL.format(
        cik=int(cik10),
        accession=accession.replace("-", ""),
        document=primary_document,
    )


def resolve_filing(
    ticker: str, fiscal_year: int, form: str, settings: Settings | None = None
) -> FilingRef:
    """Find the ``form`` filing for ``ticker`` whose report date falls in ``fiscal_year``."""
    settings = settings or get_settings()
    ticker = ticker.upper()
    tmap = load_ticker_map(settings)
    if ticker not in tmap:
        raise ValueError(f"Unknown ticker {ticker!r} (not in SEC company_tickers.json).")
    cik10, company = tmap[ticker]

    for row in _iter_filings(cik10, settings):
        if row["form"] == form and row["report_date"][:4] == str(fiscal_year):
            return FilingRef(
                ticker=ticker,
                company=company,
                cik=cik10,
                accession=row["accession"],
                form=form,
                fiscal_year=int(fiscal_year),
                report_date=row["report_date"],
                primary_document=row["primary"],
                source_url=filing_url(cik10, row["accession"], row["primary"]),
            )
    raise LookupError(f"No {form} found for {ticker} with fiscal year {fiscal_year}.")


def fetch_html(ref: FilingRef, settings: Settings | None = None) -> str:
    """Return the filing's raw HTML, downloading once and caching it under data/."""
    settings = settings or get_settings()
    form_slug = ref.form.lower().replace("/", "-")
    cache = _DATA_DIR / f"{ref.ticker.lower()}-{ref.fiscal_year}-{form_slug}.htm"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    resp = _get(ref.source_url, settings.edgar_user_agent)
    _DATA_DIR.mkdir(exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    return resp.text
