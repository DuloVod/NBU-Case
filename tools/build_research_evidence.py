"""Rebuild traceable research evidence offline; never alter original/model inputs."""
from pathlib import Path
import csv
import calendar
from datetime import date as Date
import hashlib
import html
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/research/2026-10-22"
AS_OF = "2026-10-22"
FIELDS = ["date", "variable", "value", "unit", "source_url", "published_at",
          "status", "frequency", "source_file", "locator", "availability", "note"]


def read(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields, excel=False):
    with path.open("w", encoding="utf-8-sig" if excel else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";" if excel else ",")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if excel:
                row["value"] = str(row["value"]).replace(".", ",")
            writer.writerow(row)


def main():
    sources = {r["file"]: r["url"] for r in read("data/documentation/sources.csv")}
    sources["nbu_policy_archive_rows.json"] = "https://bank.gov.ua/ua/monetary/archive-rish"
    rows = []

    def add(date, variable, value, unit, source_file, locator, status="observed",
            published_at="", source_url="", note="", availability="snapshot_present_publication_unknown"):
        if value in ("", None) or date[:4] < "2010" or date[:4] > "2026":
            return
        if "Q" in date:
            month = 3 * int(date[-1])
            period_end = Date(int(date[:4]), month, calendar.monthrange(int(date[:4]), month)[1])
        elif len(date) == 7:
            year, month = map(int, date.split("-"))
            period_end = Date(year, month, calendar.monthrange(year, month)[1])
        else:
            period_end = Date.fromisoformat(date)
        if period_end.isoformat() > AS_OF:
            raise ValueError("Incomplete/future period cannot be an observed research value: " + date)
        if published_at and published_at[:10] > AS_OF:
            raise ValueError("Publication after cutoff")
        frequency = "Q" if "Q" in date else "D" if len(date) == 10 else "M"
        rows.append(dict(zip(FIELDS, [date, variable, value, unit,
            source_url or sources.get(source_file, ""), published_at, status,
            frequency, source_file, locator, availability, note])))

    inputs = ["data/inputs/observations_long.csv", "data/documentation/auxiliary_provenance.csv",
              "data/inputs/surveys_monthly.csv", "data/inputs/business_quarterly_latest.csv",
              "data/inputs/business_survey_detail.csv", "data/documentation/sources.csv"]
    for path in inputs[:4]:
        for r in read(path):
            if r.get("status") == "missing":
                continue
            add(r["period"], r["series"], r["value"], r["unit"], r["source"],
                r.get("locator", ""), status=r.get("status", "observed"),
                published_at=r.get("publication_date", ""), note=r.get("note", ""),
                availability="published_by_cutoff" if r.get("publication_date") else
                "snapshot_present_publication_unknown")

    # Main BAEI rows are traceable to the same source cells as the detailed extraction.
    baei = {"14": "BAEI_ALL", "18": "BAEI_INDUSTRY", "56": "BAEI_CONSTRUCTION",
            "94": "BAEI_TRADE", "124": "BAEI_SERVICES"}
    for r in read("data/inputs/business_survey_detail.csv"):
        if r["source_row"] in baei:
            pub = "2026-10-01" if r["period"] == "2026-09" else ""
            add(r["period"], baei[r["source_row"]], r["value"], "index_neutral_50",
                r["source"], r["locator"], status="survey", published_at=pub,
                note="September publication verified against saved monthly-business-september.html",
                availability="published_by_cutoff" if pub else "snapshot_present_publication_unknown")

    reserve_url = "https://bank.gov.ua/ua/news/all/mijnarodni-rezervi-stanovili-471-mlrd-dol-ssha-za-pidsumkami-veresnya"
    reserve_file = OUT / "snapshots/reserves-september.html"
    raw = html.unescape(reserve_file.read_text())
    text = re.sub("<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    # Literal numeric anchors fail loudly if the saved primary document is changed.
    release = [
        ("2026-10-01", "GROSS_RESERVES_USD", 47101.8, "47 101,8", "preliminary_stock", "Stock at 01 October; not a September flow"),
        ("2026-09", "GROSS_RESERVES_MOM", -3.2, "3,2%", "preliminary_change", "Published percentage decline"),
        ("2026-09", "NBU_FX_SALES_BALANCE_USD", 5449.9, "5 449,9", "observed_flow", "Balance-sheet settlement basis; not automatically identical to transaction-date intervention series"),
        ("2026-09", "GOV_FX_RECEIPTS_WB_USD", 890.4, "890,4", "observed_flow", "World Bank accounts; separate from USL conversion"),
        ("2026-09", "GOV_USL_CONVERSION_USD", 3810, "3,81", "observed_flow", "Conversion of earmarked EU Ukraine Support Loan funds; USD million conversion from USD billion"),
        ("2026-09", "GOV_FX_DEBT_PAYMENTS_EX_IMF_USD", 195.7, "195,7", "observed_flow", "Government debt servicing and repayment; IMF separately"),
        ("2026-09", "IMF_PAYMENTS_USD", 254.3, "254,3", "observed_flow", "IMF payments separate from government payment subtotal"),
        ("2026-09", "RESERVES_REVALUATION_OTHER_USD", -368.3, "368,3", "observed_flow", "Valuation effects and other factors; decrease"),
    ]
    for date, variable, value, anchor, status, note in release:
        assert anchor in text, (variable, anchor)
        add(date, variable, value, "pct_mom" if variable.endswith("MOM") else "USD_million",
            str(reserve_file.relative_to(ROOT)), "article-content; numeric anchor " + anchor,
            status, "2026-10-07T15:10:00+03:00", reserve_url, note, "published_by_cutoff")

    # Calendar-average rates are observed aggregates, never forecasts. September CPI is
    # independently corroborated by the dated October12 commentary (policy researcher).
    for r in rows:
        if r["date"] == "2026-09" and r["variable"] in {"CPI_UA_MOM", "CPI_UA_YOY_PUBLISHED", "CORE_CPI_UA_MOM", "CORE_CPI_UA_YOY_PUBLISHED"}:
            r["note"] += "; September values cross-checked with NBU October12 commentary; workbook first-publication date not inferred"
    rows.sort(key=lambda r: (r["variable"], r["date"], r["source_file"]))
    keys = [(r["date"], r["variable"], r["source_file"]) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicate source observations"
    assert all(r["source_url"] for r in rows), "Unmapped original source URL"
    assert all(math.isfinite(float(r["value"])) for r in rows), "Nonfinite evidence"
    september_baei = {r["variable"]: float(r["value"]) for r in rows if r["date"] == "2026-09" and r["variable"].startswith("BAEI_")}
    expected_baei = {r["period"]: r for r in read("data/inputs/business_monthly.csv")}["2026-09"]
    assert all(abs(v - float(expected_baei[k])) < 1e-9 for k, v in september_baei.items())
    latest_values = {r["series"]: float(r["value"]) for r in read("data/inputs/observations_long.csv") if r["period"] == "2026-09" and r["value"]}
    for key, expected in {"CPI_UA_MOM": 2.1, "CPI_UA_YOY_PUBLISHED": 10,
                          "CORE_CPI_UA_MOM": 2, "CORE_CPI_UA_YOY_PUBLISHED": 8.8}.items():
        assert abs(latest_values[key] - expected) < 1e-9, key
    aug = {r["variable"]: float(r["value"]) for r in rows if r["date"] == "2026-08"}
    for suffix in ("goods", "services"):
        assert abs(aug["exports_" + suffix] - aug["imports_" + suffix] - aug["trade_balance_" + suffix]) < 1e-9
    write(OUT / "evidence_long.csv", rows, FIELDS)
    latest = {}
    for row in rows:
        latest[row["variable"]] = row
    write(OUT / "latest_evidence.csv", list(latest.values()), FIELDS)
    (OUT / "excel_exports").mkdir(exist_ok=True)
    write(OUT / "excel_exports/evidence_long_excel.csv", rows, FIELDS, excel=True)
    write(OUT / "excel_exports/latest_evidence_excel.csv", list(latest.values()), FIELDS, excel=True)
    all_inputs = [ROOT / p for p in inputs] + list((OUT / "snapshots").glob("*.html"))
    manifest = {"as_of": AS_OF, "rows": len(rows), "variables": len(latest),
        "publication_date_known_rows": sum(bool(r["published_at"]) for r in rows),
        "note": "Unknown publication dates deliberately remain empty; this evidence table does not certify real-time availability of historical workbook vintages.",
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(all_inputs)}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "inputs_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
