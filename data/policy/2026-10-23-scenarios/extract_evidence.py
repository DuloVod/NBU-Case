"""Offline, traceable extraction of selected facts; no structural calibration."""
from pathlib import Path
import csv
import hashlib
import html
import json
import re
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    sources = {r["id"]: r for r in json.loads((HERE / "sources_manifest.json").read_text())}
    rows = []
    def add(source_id, period, variable, value, unit, status, locator, note=""):
        s = sources[source_id]
        rows.append(dict(period=period, variable=variable, value=value, unit=unit, status=status,
            source_id=source_id, source_url=s["url"], published_at=s.get("published_at") or "unknown",
            adopted_at=s.get("adopted_at", ""), retrieved_at=s["retrieved_at"], sha256=s["sha256"],
            locator=locator, information_cutoff="2026-10-22", note=note))
    reader = PdfReader(ROOT / "data/raw/BOS_2026-Q3.pdf")
    texts = {page: reader.pages[page-1].extract_text() for page in (3, 4, 8, 10, 31, 56, 59)}
    for variable, value in {"BOS_CONSTRAINT_CAPACITY":14.5, "BOS_CONSTRAINT_ENERGY_PRICES":41.7,
            "BOS_CONSTRAINT_RAW_MATERIALS":39., "BOS_CONSTRAINT_SKILLS":51.1,
            "BOS_CONSTRAINT_WORKING_CAPITAL":23., "BOS_CONSTRAINT_CREDIT":6.2,
            "BOS_CONSTRAINT_DEMAND":30.9, "BOS_CONSTRAINT_FX_VOLATILITY":17.5,
            "BOS_CONSTRAINT_WAR":82.8}.items():
        assert str(value).rstrip("0").rstrip(".") in texts[56]
        add("BOS_Q3", "2026Q3", variable, value, "pct_respondents_multiple_choice", "survey",
            "PDF p.56 Table4 total row; visually verified", "Fieldwork 2026-07-30 to 2026-08-28; not % loss of output")
    for variable, value in {"BOS_PRICE_FACTOR_ENERGY":64.3, "BOS_PRICE_FACTOR_RAW_MATERIALS":61.4,
            "BOS_PRICE_FACTOR_LABOUR":55.6, "BOS_PRICE_FACTOR_FX":57.8,
            "BOS_PRICE_FACTOR_DEMAND":19.1, "BOS_PRICE_FACTOR_LOGISTICS":37.,
            "BOS_PRICE_FACTOR_INTEREST_RATE":5.7}.items():
        assert str(value).rstrip("0").rstrip(".") in texts[59]
        add("BOS_Q3", "2026Q3", variable, value, "pct_respondents_multiple_choice", "survey",
            "PDF p.59 Table6 total row; visually verified", "Factor of own selling-price expectations; not CPI contribution or structural elasticity")
    add("BOS_Q3", "2026Q3", "BOS_UNIT_COST_EXPECTATION_BALANCE", 69.7, "pct_positive_minus_negative",
        "survey", "PDF p.31", "Previous quarter69.2; not forecast % increase in unit cost")
    add("BOS_Q3", "2026Q3", "BOS_INFL_EXPECTATION_12M", 11.5, "pct_expected_inflation",
        "survey", "PDF p.3", "PreviousQ2=11.6; high level is not proof of worsening in this wave")
    add("BOS_Q3", "2026Q3", "BOS_PRODUCTION_EXPECTATION_BALANCE", -11.9, "pct_positive_minus_negative",
        "survey", "PDF p.3", "PreviousQ2=7.6; expectations, not realized GDP")
    article = html.unescape((HERE / "raw/nbu-cpi-september.html").read_text())
    for variable, value, anchor in [("CPI_FUEL_YOY",46.2,"46,2"),("CPI_PROCESSED_FOOD_YOY",10.7,"10,7"),
                                    ("CPI_SERVICES_YOY",13.7,"13,7")]:
        assert anchor in article
        add("NBU_CPI", "2026-09", variable, value, "pct_yoy", "official_commentary_statistic",
            "article-content; component paragraph", "Fuel heading has typographic dash before46.2; narrative explicitly describes rising gasoline/diesel prices")
    law = html.unescape((HERE / "raw/rada-law-3878.html").read_text())
    assert "чинний" in law and "01.09.2024" in law
    for fuel, old, new in [("GASOLINE",300.8,329.9),("DIESEL",253.8,291.9),("LPG",198.,223.)]:
        for year, value in [(2026,old),(2027,new)]:
            anchor = str(value).replace(".", ",").replace(",0", "")
            assert anchor in law
            add("FUEL_LAW", str(year), "EXCISE_"+fuel, value, "EUR_per_1000_litres_at_15C",
                "enacted_tax_schedule_not_observed_retail_price", "Law3878-IX sectionI, transition paragraph48 tax table",
                "Known statutory schedule; adopted2024-07-18, effective2024-09-01; future2027 rate is not a realized price")
    with (ROOT / "data/inputs/surveys_monthly.csv").open() as handle:
        surveys=list(csv.DictReader(handle))
    for group in ("banks", "households", "analysts"):
        values=[r for r in surveys if r["series"]=="EXP_INFL_12M_"+group and r["value"] and r["period"]<="2026-09"]
        r=max(values,key=lambda item:item["period"])
        add("EXPECT_INFL", r["period"], r["series"], float(r["value"]), r["unit"], "survey",
            r["locator"], "Latest nonmissing survey observation; groups are not synchronous")
    with (HERE / "shock_evidence.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (HERE / "shock_evidence_excel.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter=";");writer.writeheader()
        for row in rows:
            writer.writerow(dict(row,value=str(row["value"]).replace(".",",")))
    derived=[]
    for name in ("bos-table4-review.png","bos-table6-review.png"):
        p=HERE/"raw"/name
        derived.append(dict(file="raw/"+name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source="data/raw/BOS_2026-Q3.pdf",role="rendered_review_not_new_source"))
    (HERE/"extraction_checks.json").write_text(json.dumps(dict(rows=len(rows),source_ids=sorted({r["source_id"] for r in rows}),
        source_publication_unknown="EXCISE and legacy survey workbook dates not invented",derived_review_assets=derived,
        fuel_sign="Positive46.2 reflects text describing price growth; dash in source is not parsed as a minus sign",
        note="No future winter energy deficit, causal elasticity, or scenario probability is inferred"),ensure_ascii=False,indent=2)+"\n")
    print("Extracted",len(rows),"records offline")


if __name__ == "__main__":
    main()
