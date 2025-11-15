from __future__ import annotations
import os, sys, argparse, time
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv

# --- LangSmith tracing for raw OpenAI calls ---
from langsmith.wrappers import wrap_openai
from langsmith import Client

# OpenAI
from openai import OpenAI

# Stats
from statsmodels.stats.contingency_tables import mcnemar

# ========= Setup =========
load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "Session10-AB-Prompts"))

# Wrap OpenAI so every call is traced to LangSmith automatically
client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
ls_client = Client()  # to write feedback metrics

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

def classify_chat(prompt: str, text: str, model: str = "gpt-4o-mini") -> str:
    """Chat Completions (traced by LangSmith wrapper). Parse robust YES/NO."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer strictly YES or NO."},
            {"role": "user", "content": prompt.format(input=text)},
        ],
        temperature=0,
    )
    out = (resp.choices[0].message.content or "").strip().upper()
    if "YES" in out and "NO" not in out:
        return "YES"
    if "NO" in out and "YES" not in out:
        return "NO"
    return "YES" if out.startswith("Y") else "NO"

def run_variant(name: str, prompt: str, inputs: List[str], model: str, sleep_s: float = 0.0) -> tuple[List[str], float]:
    """Run a variant across inputs. Returns predictions + avg latency (s)."""
    preds, lat = [], []
    for x in inputs:
        t0 = time.perf_counter()
        y = classify_chat(prompt, x, model=model)
        lat.append(time.perf_counter() - t0)
        preds.append(y)
        if sleep_s:
            time.sleep(sleep_s)
    avg_latency = (sum(lat) / len(lat)) if lat else 0.0
    return preds, avg_latency

def accuracy(preds: List[str], gold: List[str]) -> float:
    correct = sum(p.strip().upper() == g.strip().upper() for p, g in zip(preds, gold))
    return correct / max(1, len(gold))

def build_mcnemar_table(predA: List[str], predB: List[str], gold: List[str]) -> Tuple[int,int,int,int]:
    """Return (a,b,c,d) where:
       a: A wrong, B wrong
       b: A right, B wrong
       c: A wrong, B right
       d: A right, B right
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

def post_project_feedback(project_name: str, metrics: dict):
    """Attach metrics to the LangSmith project as feedback entries."""
    try:
        proj = ls_client.read_project(project_name)
        for k, v in metrics.items():
            # Feedback must be attached to a run_id, trace_id, or project_id.
            ls_client.create_feedback(project_id=proj.id, key=k, score=float(v))
    except Exception as e:
        print(f"[warn] Could not post feedback to LangSmith: {e}")

def main():
    ap = argparse.ArgumentParser(description="A/B test two prompts with LangSmith tracing + feedback.")
    ap.add_argument("--data", default="dataset.csv", help="CSV with columns: input,label (YES/NO)")
    ap.add_argument("--limit", type=int, default=0, help="limit rows for a quick run")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--project", default=os.getenv("LANGCHAIN_PROJECT", "Session10-AB-Prompts"))
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"Dataset not found: {args.data}")
        sys.exit(1)

    df = pd.read_csv(args.data)
    if "input" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: input,label (label ∈ {YES,NO})")

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    inputs = df["input"].astype(str).tolist()
    gold   = [s.strip().upper() for s in df["label"].astype(str).tolist()]

    print(f"[LangSmith] Project: {args.project}")
    print(f"Running Variant A on {len(inputs)} items...")
    predA, latA = run_variant("A", PROMPT_A, inputs, model=args.model)

    print(f"Running Variant B on {len(inputs)} items...")
    predB, latB = run_variant("B", PROMPT_B, inputs, model=args.model)

    accA = accuracy(predA, gold)
    accB = accuracy(predB, gold)
    a,b,c,d = build_mcnemar_table(predA, predB, gold)

    print("\n=== Results ===")
    print(f"A accuracy: {accA:.3f} | avg latency: {latA:.3f}s")
    print(f"B accuracy: {accB:.3f} | avg latency: {latB:.3f}s")
    print(f"Contingency (a,b,c,d) = ({a},{b},{c},{d})")

    if (b + c) == 0:
        pval = 1.0
        print("McNemar: no discordant pairs; tie.")
    else:
        res = mcnemar([[a,b],[c,d]], exact=False, correction=True)
        pval = float(res.pvalue)
        print(f"McNemar chi2={res.statistic:.4f}, p-value={pval:.4g}")

    # Post metrics to LangSmith project as feedback
    metrics = {
        "accA": accA, "accB": accB,
        "avg_latency_A_s": latA, "avg_latency_B_s": latB,
        "mcnemar_p": pval,
        "acc_delta_B_minus_A": (accB - accA),
    }
    post_project_feedback(args.project, metrics)

    # Save row-wise outputs
    out = pd.DataFrame({
        "input": inputs, "gold": gold,
        "predA": predA, "predB": predB,
        "A_correct": [p==g for p,g in zip(predA,gold)],
        "B_correct": [p==g for p,g in zip(predB,gold)]
    })
    out.to_csv("ab_outputs.csv", index=False, encoding="utf-8")
    print("\nWrote per-row results to ab_outputs.csv")
    print("Metrics posted to LangSmith project feedback (if keys were set).")

if __name__ == "__main__":
    main()
