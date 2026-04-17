"""
Create json data for downstream economic application.

For each statement in example_data_new_uncertainty.xlsx, build a record with
sentence, full context, call time, financial context from yfinance, and subset label.
"""

import json
import math
import yfinance as yf
from datetime import timedelta
from dateutil import parser as dateparser
import pandas as pd


def get_firm_context(ticker_str, call_date):
    t = yf.Ticker(ticker_str)

    end = call_date
    start = call_date - timedelta(days=90)
    hist = t.history(start=start, end=end)

    if hist.empty or len(hist) < 2:
        return None

    returns = hist["Close"].pct_change().dropna()
    info = t.info

    beta = info.get("beta", float("nan"))
    market_cap = info.get("marketCap", 0)

    return {
        "sector":       info.get("sector", "N/A"),
        "return_3m":    round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 1),
        "vol_30d":      round(returns.tail(21).std() * (252 ** 0.5) * 100, 1),
        "beta":         round(beta, 2) if not (isinstance(beta, float) and math.isnan(beta)) else None,
        "market_cap_b": round(market_cap / 1e9, 1) if market_cap else None,
    }


def parse_call_date(call_datetime_str):
    """Parse varied call datetime strings to a date object."""
    try:
        # Strip timezone names that dateutil can't handle (e.g. 'ET', 'EDT')
        cleaned = str(call_datetime_str).replace(" ET", "").replace(" EDT", "").replace(" EST", "")
        return dateparser.parse(cleaned, fuzzy=True).date()
    except Exception:
        return None


def rows_from_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df.to_dict(orient="records")


def main():
    path = "data/example_data_new_uncertainty.xlsx"
    records = []

    for subset in ("examples", "contrastive"):
        rows = rows_from_sheet(path, subset)
        for row in rows:
            records.append((subset, row))

    output = []
    for idx, (subset, row) in enumerate(records, start=1):
        ticker = str(row.get("ticker", "")).strip()
        call_date = parse_call_date(row.get("call_datetime", ""))

        financial_context = None
        if ticker and call_date:
            try:
                financial_context = get_firm_context(ticker, call_date)
            except Exception as e:
                print(f"[{idx}] yfinance error for {ticker}: {e}")

        entry = {
            "id":                str(idx),
            "ticker":            ticker,
            "company":           row.get("company", ""),
            "sentence":          row.get("sentence", ""),
            "full_context":      row.get("full_answer", ""),
            "call_time":         str(row.get("call_datetime", "")),
            "financial_context": financial_context,
            "subset":            subset,
        }
        output.append(entry)
        print(f"[{idx}/{len(records)}] {ticker} ({subset}) done")

    out_path = "data/sentences_with_context.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nWrote {len(output)} records to {out_path}")


if __name__ == "__main__":
    main()
