"""
Information Systems cleaning: API only for *unique* (Author_name, Author_Address) pairs,
in *batches* (multiple pairs per request). Saves requests and stays within 3 RPM / 200 RPD.

Preview (no API, no writes):  python datacleaning.py --preview
"""
import os
import sys
import json
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

INPUT_CSV = "Journal_of_Information_Systems/Information_Systems_article_data.csv"
OUTPUT_CSV = "Journal_of_Information_Systems/Information_Systems_API_cleaned.csv"
# Cache: unique pair -> standardized result (so we never call API twice for same pair)
CACHE_JSON = "Journal_of_Information_Systems/standardization_cache.json"

RPM_LIMIT = 3
RPD_LIMIT = 200
DELAY_BETWEEN_REQUESTS = 60.0 / RPM_LIMIT
# How many unique pairs per API call (fewer calls, same quality)
BATCH_SIZE = 10

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

data = pd.read_csv(INPUT_CSV)
total_rows = len(data)


def _cell(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    return "" if s == "nan" else s


def _pair(row):
    a = _cell(row.get("Author_name", ""))
    b = _cell(row.get("Author_Address", ""))
    return (a, b)


def _address(row):
    return " ".join([_pair(row)[0], _pair(row)[1]]).strip()


SYSTEM_PROMPT = """You extract and standardize author/university/location from text.
Return ONLY valid JSON. For each input text return one object in the "results" array, in the SAME ORDER.
Each object: Standardized_Author, University, Department, State, Country, Pincode (use "" or null if missing).
Rules: English only, title case names, full university names (no abbreviations), full country names (e.g. United States not USA)."""


def extract_batch(texts: list) -> list:
    """Call API once for multiple author+address strings; return list of result dicts."""
    if not texts:
        return []
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user = f"Process these {len(texts)} texts in order and return a JSON object with key 'results' (array of {len(texts)} objects, each with Standardized_Author, University, Department, State, Country, Pincode):\n\n{numbered}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip().strip("```json").strip("```")
    try:
        out = json.loads(raw)
        results = out.get("results", [])
    except Exception as e:
        print("  Parse error:", e)
        results = []
    # Coerce to list of dicts with string values
    def _v(x):
        return x if x is not None and str(x).strip() else ""
    out_list = []
    for i in range(len(texts)):
        r = results[i] if i < len(results) else {}
        out_list.append({
            "Standardized_Author": _v(r.get("Standardized_Author")),
            "University": _v(r.get("University")),
            "Department": _v(r.get("Department")),
            "State": _v(r.get("State")),
            "Country": _v(r.get("Country")),
            "Pincode": _v(r.get("Pincode")),
        })
    return out_list


# --- Unique pairs (we only call API for these)
rows_by_pair = {}
for i, row in data.iterrows():
    pair = _pair(row)
    if pair not in rows_by_pair:
        rows_by_pair[pair] = []
    rows_by_pair[pair].append(i)
unique_pairs = list(rows_by_pair.keys())

# Skip empty address: no API, we'll use original author as standardized
def pair_to_text(pair):
    return " ".join(p for p in pair if p).strip()

pairs_needing_api = [p for p in unique_pairs if pair_to_text(p)]
unique_count = len(pairs_needing_api)
print(f"Total rows: {total_rows}. Unique (Author_name, Author_Address) pairs: {len(unique_pairs)} (non-empty: {unique_count}).")

# --- Load cache (resume across runs)
Path(CACHE_JSON).parent.mkdir(parents=True, exist_ok=True)
cache = {}
if Path(CACHE_JSON).exists():
    try:
        with open(CACHE_JSON, "r", encoding="utf-8") as f:
            cache = json.load(f)
        # key as tuple (JSON stores list)
        cache = {tuple(k): v for k, v in cache.items()}
        print(f"Loaded cache: {len(cache)} pairs already standardized.")
    except Exception as e:
        print("Cache load failed:", e)

# --- Which pairs still need API? Cap at 200 requests/day (each request = up to BATCH_SIZE pairs)
pending = [p for p in pairs_needing_api if p not in cache]
max_pairs_this_run = RPD_LIMIT * BATCH_SIZE  # 200 * 10 = 2000 pairs per day
to_fetch = pending[:max_pairs_this_run]
num_calls = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE

# --- Preview: show counts and exit without calling API or writing files
if "--preview" in sys.argv or "--dry-run" in sys.argv:
    print("\n--- PREVIEW (no API calls, no files written) ---")
    print(f"  Input file:           {INPUT_CSV}")
    print(f"  Total rows:            {total_rows}")
    print(f"  Unique (name, addr):   {len(unique_pairs)}")
    print(f"  Non-empty (need API):  {unique_count}")
    print(f"  Already in cache:     {len(cache)}")
    print(f"  Pending (need API):   {len(pending)}")
    print(f"  This run would:       process {len(to_fetch)} pairs in {num_calls} API call(s)")
    print(f"  Est. time this run:   ~{num_calls * DELAY_BETWEEN_REQUESTS / 60:.1f} min")
    if len(pending) > len(to_fetch):
        print(f"  After this run:        {len(pending) - len(to_fetch)} pairs left (run again next day)")
    print("--- Run without --preview to execute. ---\n")
    sys.exit(0)

if not to_fetch:
    print("No new pairs to process. Writing output from cache.")
else:
    print(f"This run: {len(to_fetch)} new pairs in {num_calls} API call(s) (max {RPD_LIMIT} calls/day).")
    for start in range(0, len(to_fetch), BATCH_SIZE):
        batch_pairs = to_fetch[start : start + BATCH_SIZE]
        texts = [pair_to_text(p) for p in batch_pairs]
        try:
            results = extract_batch(texts)
            for p, res in zip(batch_pairs, results):
                cache[p] = res
            print(f"  Cached {start + len(batch_pairs)}/{len(to_fetch)} unique pairs.")
        except Exception as e:
            print(f"  API error: {e}")
            for p in batch_pairs:
                cache[p] = {
                    "Standardized_Author": p[0],
                    "University": "",
                    "Department": "",
                    "State": "",
                    "Country": "",
                    "Pincode": "",
                }
        time.sleep(DELAY_BETWEEN_REQUESTS)
    # Persist cache for next run
    with open(CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump({list(k): v for k, v in cache.items()}, f, indent=0, ensure_ascii=False)
    print(f"Cache saved ({len(cache)} pairs).")

# --- Fill cache for empty pairs (no API)
for p in unique_pairs:
    if p not in cache:
        cache[p] = {
            "Standardized_Author": p[0],
            "University": "",
            "Department": "",
            "State": "",
            "Country": "",
            "Pincode": "",
        }

# --- Write full output (all rows, in original order)
OUTPUT_HEADER = [
    "URL", "Journal_Title", "Article_Title", "Volume_Issue", "Month_Year",
    "Abstract", "Keywords", "Author_name", "Standardized_Author", "Author_email",
    "Author_Address", "Standardized_University", "Author_Department",
    "Author_State", "Author_Country", "Author_Pincode",
]
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(OUTPUT_HEADER)
    for i in range(total_rows):
        row = data.iloc[i]
        pair = _pair(row)
        res = cache.get(pair, {})
        if not res:
            res = {"Standardized_Author": _cell(row.get("Author_name")), "University": "", "Department": "", "State": "", "Country": "", "Pincode": ""}
        w.writerow([
            _cell(row.get("URL")),
            _cell(row.get("Journal_Title")),
            _cell(row.get("Article_Title")),
            _cell(row.get("Volume_Issue")),
            _cell(row.get("Month_Year")),
            _cell(row.get("Abstract")),
            _cell(row.get("Keywords")),
            _cell(row.get("Author_name")),
            res.get("Standardized_Author", ""),
            _cell(row.get("Author_email")),
            _cell(row.get("Author_Address")),
            res.get("University", ""),
            res.get("Department", ""),
            res.get("State", ""),
            res.get("Country", ""),
            res.get("Pincode", ""),
        ])

print(f"Done. Wrote {total_rows} rows to {OUTPUT_CSV}. Input unchanged.")
remaining = len([p for p in pairs_needing_api if p not in cache])
if remaining:
    print(f"Run again tomorrow to process {remaining} remaining unique pairs (200 RPD).")
