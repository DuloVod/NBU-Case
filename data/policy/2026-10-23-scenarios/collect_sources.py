"""Targeted primary-source snapshots; downloading is explicit, default is offline."""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCES = [
    dict(id="NG_WINTER", file="raw/naftogaz-winter-2026-10-22.html", url="https://www.naftogaz.com/news/naftogaz-prepares-24-winter-scenarios-factoring-in-russian-attacks-and-weather-conditions", published_at="2026-10-22T13:38:00+03:00", status="dated_primary_operator_release"),
    dict(id="IEA_2025", file="raw/iea-energy-security-2025.html", url="https://www.iea.org/reports/ukraines-energy-security", published_at="2025-10-22", status="prior_winter_background_not_2026_capacity"),
    dict(id="IEA_ASSESSMENT", file="raw/iea-prewinter-assessment-2025.html", url="https://www.iea.org/reports/ukraines-energy-security/a-pre-winter-assessment", published_at="2025-10-22", status="chapter_of_dated_2025_report"),
    dict(id="IEA_RESILIENCE_2026", file="raw/iea-resilience-2026.html", url="https://www.iea.org/reports/energy-system-resilience", published_at="2026-02-11", status="historical_energy_resilience_not_winter_2026_forecast"),
    dict(id="IEA_EXEC_2026", file="raw/iea-resilience-executive-summary-2026.html", url="https://www.iea.org/reports/energy-system-resilience/executive-summary", published_at="2026-02-11", status="chapter_of_dated_2026_report_historical_january_not_next_winter"),
    dict(id="FUEL_TAX", file="raw/dps-code-transition.html", url="https://tax.gov.ua/nk/rozdil-xx--perehidni-polojen/", published_at=None, status="official_consolidated_tax_summary_publication_unknown"),
    dict(id="FUEL_LAW", file="raw/rada-law-3878.html", url="https://zakon.rada.gov.ua/laws/show/3878-20#Text", published_at=None, adopted_at="2024-07-18", status="amending_statute_adoption_date_not_publication_date"),
    dict(id="NBU_CPI", file="raw/nbu-cpi-september.html", url="https://bank.gov.ua/ua/news/all/komentar-natsionalnogo-banku-schodo-rivnya-inflyatsiyi-u-veresni-2026-roku", published_at="2026-10-12", status="dated_primary_commentary"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    (HERE / "raw").mkdir(exist_ok=True)
    prior_file = HERE / "sources_manifest.json"
    prior = {r["id"]: r for r in json.loads(prior_file.read_text())} if prior_file.exists() else {}
    records = []
    for source in SOURCES:
        row = dict(source, information_cutoff="2026-10-22")
        path = HERE / row["file"]
        if args.download and not path.exists():
            try:
                request = Request(row["url"], headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request, timeout=35) as response:
                    path.write_bytes(response.read())
                    row.update(http_status=response.status, retrieved_at=datetime.now(timezone.utc).isoformat(), final_url=response.url)
            except Exception as error:
                row.update(download_error=str(error), retrieved_at=datetime.now(timezone.utc).isoformat(), available=False)
        elif row["id"] in prior:
            for key in ("http_status", "retrieved_at", "final_url", "download_error"):
                if key in prior[row["id"]]:
                    row[key] = prior[row["id"]][key]
        if path.exists():
            row.update(available=True, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
        else:
            row["available"] = False
        records.append(row)
    for id_, relative, url, published in [
        ("BOS_Q3", "data/raw/BOS_2026-Q3.pdf", "https://bank.gov.ua/admin_uploads/article/BOS_2026-Q3.pdf", "2026-10-15"),
        ("IODA_SEPT", "data/research/2026-10-22/snapshots/monthly-business-september.html", "https://bank.gov.ua/ua/news/all/biznes-strimano-otsiniv-rezultati-svoyeyi-diyalnosti--pidsumki-opituvannya-pidpriyemstv-u-veresni", "2026-10-01"),
        ("EXPECT_INFL", "data/raw/Surveys_price.xlsx", "https://bank.gov.ua/files/macro/Surveys_price.xlsx", None),
        ("EXPECT_FX", "data/raw/Surveys_rate.xlsx", "https://bank.gov.ua/files/macro/Surveys_rate.xlsx", None),
        ("CPI_GROUPS", "data/raw/CPI_m.xlsx", "https://bank.gov.ua/files/macro/CPI_m.xlsx", None),
    ]:
        path = ROOT / relative
        records.append(dict(id=id_, file=relative, url=url, published_at=published,
            retrieved_at="existing_project_snapshot_original_retrieval_not_reconstructed", information_cutoff="2026-10-22",
            status="reused_read_only_reference", sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size, available=True))
    prior_file.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{k: r.get(k) for k in ["id", "available", "bytes", "download_error"]} for r in records], ensure_ascii=False))


if __name__ == "__main__":
    main()
