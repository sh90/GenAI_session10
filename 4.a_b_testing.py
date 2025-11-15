# ab_test_llm_prompts.py
from __future__ import annotations
import os, sys, argparse, time
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from statsmodels.stats.contingency_tables import mcnemar

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_A = """You are a lead qualification assistant.
Decide if the lead is QUALIFIED based on the text.
Return ONLY YES or NO.

Lead:
{input}
"""

PROMPT_B = """You are a senior SDR.
Classify if this lead should be prioritized for outreach.
Return strictly 'YES' (qualified) or 'NO' (not qualified). No extra text.

Lead:
{input}
"""

def classify(prompt: str, text: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI once, parse YES/NO robustly."""
    resp = client.responses.create(
        model=model,
        input=prompt.format(input=text),
        temperature=0
    )
    out = (resp.output_text or "").strip().upper()
    # Robust parse: look for YES/NO token anywhere
    if "YES" in out and "NO" not in out:
        return "YES"
    if "NO" in out and "YES" not in out:
        return "NO"
    # fallback: first token heuristic
    return "YES" if out.startswith("Y") else "NO"

def run_variant(prompt: str, inputs: List[str], sleep_s: float = 0.0) -> List[str]:
    preds = []
    for i, x in enumerate(inputs, 1):
        y = classify(prompt, x)
        preds.append(y)
        if sleep_s: time.sleep(sleep_s)
    return preds

def accuracy(preds: List[str], gold: List[str]) -> float:
    correct = sum(p.strip().upper() == g.strip().upper() for p, g in zip(preds, gold))
    return correct / max(1, len(gold))

def build_mcnemar_table(predA: List[str], predB: List[str], gold: List[str]) -> Tuple[int,int,int,int]:
    """Returns contingency counts: a, b, c, d where:
       a: A wrong, B wrong
       b: A right, B wrong
       c: A wrong, B right
       d: A right, B right
       McNemar uses b & c.
    """
    a=b=c=d=0
    for pa, pb, g in zip(predA, predB, gold):
        ca = (pa.upper()==g.upper())
        cb = (pb.upper()==g.upper())
        if ca and cb: d += 1
        elif ca and not cb: b += 1
        elif (not ca) and cb: c += 1
        else: a += 1
    return a,b,c,d

def main():
    ap = argparse.ArgumentParser(description="A/B test two prompts on a labeled dataset (YES/NO).")
    ap.add_argument("--data", default="dataset.csv", help="CSV with columns: input,label")
    ap.add_argument("--limit", type=int, default=0, help="limit rows for a quick run")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"Dataset not found: {args.data}")
        print("Create dataset.csv with columns: input,label (label ∈ {YES,NO})")
        sys.exit(1)

    df = pd.read_csv(args.data)
    if "input" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: input,label")

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    inputs = df["input"].astype(str).tolist()
    gold   = [s.strip().upper() for s in df["label"].astype(str).tolist()]

    print(f"Running A on {len(inputs)} items...")
    predA = run_variant(PROMPT_A, inputs)

    print(f"Running B on {len(inputs)} items...")
    predB = run_variant(PROMPT_B, inputs)

    accA = accuracy(predA, gold)
    accB = accuracy(predB, gold)
    a,b,c,d = build_mcnemar_table(predA, predB, gold)

    print("\n=== Results ===")
    print(f"A accuracy: {accA:.3f}")
    print(f"B accuracy: {accB:.3f}")
    print(f"Contingency (a,b,c,d) = ({a},{b},{c},{d})  [A right/B wrong = b, A wrong/B right = c]")

    # McNemar’s test (exact=False → chi-squared with continuity correction)
    if (b + c) == 0:
        print("McNemar: no discordant pairs; the variants tie on this dataset.")
    else:
        res = mcnemar([[a,b],[c,d]], exact=False, correction=True)
        print(f"McNemar chi2={res.statistic:.4f}, p-value={res.pvalue:.4g}")
        if res.pvalue < 0.05:
            winner = "A" if accA > accB else ("B" if accB > accA else "None (metric tie)")
            print(f"➡ Statistically significant difference at 5% level. Winner: {winner}")
        else:
            print("➡ Not statistically significant at 5% level.")

    # Optional: dump per-row outcomes for error analysis
    out = pd.DataFrame({
        "input": inputs, "gold": gold,
        "predA": predA, "predB": predB,
        "A_correct": [p==g for p,g in zip(predA,gold)],
        "B_correct": [p==g for p,g in zip(predB,gold)]
    })
    out.to_csv("ab_outputs.csv", index=False, encoding="utf-8")
    print("\nWrote per-row results to ab_outputs.csv")

if __name__ == "__main__":
    main()
