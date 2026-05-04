"""GPT-based LLM judge for VLM evaluation results.

Auto-detects schema and field names so the same script works on both:

  Old line (results_*.json):
    {"config": {...}, "summary": {...},
     "results": [{"prompt": "USER: <image>\\nQ\\nASSISTANT:",
                  "generated_text": "...",
                  "answer": "<VQAv2 gold>"}, ...]}

  New line (sweep_*.json):
    [{"question": "...", "baseline_answer": "...",
      "pruned_answer": "...", ...}]

Adds three fields to each record (in place):
  llm_judge_score_0_3 : 0 (wrong) / 1 (slightly) / 2 (partial) / 3 (correct)
  llm_judge_reason    : one-sentence explanation
  llm_judge_question  : the question that was judged

Usage:
  export OPENAI_API_KEY=sk-...
  python llm_judge.py --input sweep_200samples.json --output sweep_200_judged.json
  python llm_judge.py --input results_clip_attention.json --output results_clip_judged.json
  python llm_judge.py --input sweep_200_judged.json --output sweep_200_judged.json --resume
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


# Original midterm prompt — single user message (no system role).
# Keeping byte-for-byte identical to old code so old-line results reproduce exactly.
JUDGE_PROMPT = """
You are evaluating a VQA model answer.

Question: {question}
Reference answer: {reference}
Model prediction: {prediction}

Score the prediction from 0 to 3:
0 = completely incorrect or contradicts the reference
1 = partially related but mostly wrong / missing key answer
2 = mostly correct but imprecise or incomplete
3 = fully correct, semantically matches the reference

Return JSON only:
{{
  "score": 0,
  "reason": "brief explanation"
}}
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Input JSON (old- or new-line)")
    p.add_argument("--output", required=True, help="Output JSON with judge fields added")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id (default: gpt-4o-mini)")
    p.add_argument("--concurrency", type=int, default=8, help="Parallel API requests (default: 8)")
    p.add_argument("--limit", type=int, default=0, help="Cap records (0 = all)")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Write partial results every N records (0 = only at end)")
    p.add_argument("--resume", action="store_true",
                   help="Skip records that already have llm_judge_score_0_3 in --output")
    p.add_argument("--question-key", default="auto",
                   help="Field name for question (auto detects)")
    p.add_argument("--reference-key", default="auto",
                   help="Field name for reference answer (auto detects 'answer' or 'baseline_answer')")
    p.add_argument("--prediction-key", default="auto",
                   help="Field name for prediction (auto detects 'generated_text' or 'pruned_answer')")
    return p.parse_args()


def detect_schema(data: Any):
    """Returns (records_list, write_back_fn).
    write_back_fn(records) -> dict/list ready to dump as JSON, preserving original outer structure."""
    if isinstance(data, list):
        return data, lambda recs: recs
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        outer = dict(data)
        def write_back(recs):
            return {**outer, "results": recs}
        return data["results"], write_back
    raise SystemExit(f"Unrecognized JSON schema: top-level={type(data).__name__}")


def autodetect_fields(record: dict):
    """Returns (question_key, reference_key, prediction_key) or None if can't decide."""
    keys = set(record.keys())

    # Old line:  prompt + generated_text + answer
    if "generated_text" in keys and "prompt" in keys:
        ref_key = "answer" if "answer" in keys else None
        if ref_key is None:
            return None
        return ("prompt", ref_key, "generated_text")

    # New line:  question + baseline_answer + pruned_answer
    if "pruned_answer" in keys and "question" in keys:
        ref_key = "baseline_answer" if "baseline_answer" in keys else (
                  "answer" if "answer" in keys else None)
        if ref_key is None:
            return None
        return ("question", ref_key, "pruned_answer")

    return None


def extract_question(record: dict, key: str) -> str:
    """For old line, parse 'USER: <image>\\n<Q>\\nASSISTANT:' down to just <Q>."""
    val = record.get(key, "")
    if not isinstance(val, str):
        return str(val)
    if "USER:" in val and "ASSISTANT:" in val:
        m = re.search(r"USER:\s*(?:<image>)?\s*(.+?)\s*ASSISTANT:", val, re.DOTALL)
        if m:
            return m.group(1).strip()
    return val.strip()


async def judge_one(client: AsyncOpenAI, model: str, question: str, reference: str, prediction: str):
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": JUDGE_PROMPT.format(
                question=question, reference=reference, prediction=prediction)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    obj = json.loads(resp.choices[0].message.content)
    score = int(obj["score"])
    score = max(0, min(3, score))  # clamp
    return score, str(obj.get("reason", ""))[:500]


async def judge_record(client, model, record, q_key, r_key, p_key, sem):
    async with sem:
        question = extract_question(record, q_key)
        reference = str(record.get(r_key, "") or "").strip()
        prediction = str(record.get(p_key, "") or "").strip()

        if not reference or not prediction:
            record["llm_judge_score_0_3"] = None
            record["llm_judge_reason"] = "[skipped: empty reference or prediction]"
            record["llm_judge_question"] = question
            return record

        try:
            score, reason = await judge_one(client, model, question, reference, prediction)
            record["llm_judge_score_0_3"] = score
            record["llm_judge_reason"] = reason
        except Exception as e:
            record["llm_judge_score_0_3"] = None
            record["llm_judge_reason"] = f"[error: {type(e).__name__}: {e}]"
        record["llm_judge_question"] = question
        return record


def summarize_judged(records):
    scores = [r["llm_judge_score_0_3"] for r in records
              if r.get("llm_judge_score_0_3") is not None]
    if not scores:
        return {"num_judged": 0}
    dist = {str(s): scores.count(s) for s in [0, 1, 2, 3]}
    return {
        "num_judged": len(scores),
        "avg_llm_judge_score_0_3": round(sum(scores) / len(scores), 3),
        "normalized_llm_judge_accuracy": round(sum(scores) / (len(scores) * 3), 4),
        "llm_judge_distribution": dist,
    }


def per_config_summary(records):
    """For new-line records (have layer/keep_ratio), break out judge accuracy per config."""
    if not records or "layer" not in records[0]:
        return None
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in records:
        s = r.get("llm_judge_score_0_3")
        if s is not None:
            by_cfg[(r.get("layer"), r.get("keep_ratio"))].append(s)
    rows = []
    for (l, kr), scores in sorted(by_cfg.items()):
        rows.append({
            "layer": l,
            "keep_ratio": kr,
            "n": len(scores),
            "avg_score": round(sum(scores) / len(scores), 3),
            "normalized_accuracy": round(sum(scores) / (len(scores) * 3), 4),
        })
    return rows


async def main_async(args):
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY environment variable.")

    in_data = json.loads(Path(args.input).read_text())
    records, write_back = detect_schema(in_data)

    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No records found in input.")

    # Detect schema
    auto = autodetect_fields(records[0])
    q_key = args.question_key if args.question_key != "auto" else (auto[0] if auto else None)
    r_key = args.reference_key if args.reference_key != "auto" else (auto[1] if auto else None)
    p_key = args.prediction_key if args.prediction_key != "auto" else (auto[2] if auto else None)
    if not (q_key and r_key and p_key):
        raise SystemExit(
            f"Cannot auto-detect fields. Pass --question-key, --reference-key, --prediction-key.\n"
            f"  Available keys: {list(records[0].keys())}"
        )
    print(f"Field mapping  question={q_key!r}  reference={r_key!r}  prediction={p_key!r}")

    # Resume: copy existing judge fields by index
    if args.resume and Path(args.output).exists():
        try:
            existing_data = json.loads(Path(args.output).read_text())
            existing_records, _ = detect_schema(existing_data)
            n = min(len(records), len(existing_records))
            n_resumed = 0
            for i in range(n):
                s = existing_records[i].get("llm_judge_score_0_3")
                if s is not None:
                    records[i]["llm_judge_score_0_3"] = s
                    records[i]["llm_judge_reason"] = existing_records[i].get("llm_judge_reason")
                    records[i]["llm_judge_question"] = existing_records[i].get("llm_judge_question")
                    n_resumed += 1
            print(f"Resumed: {n_resumed} records skipped (already judged)")
        except Exception as e:
            print(f"[warn] resume failed: {e}; starting fresh")

    todo = [i for i, r in enumerate(records) if r.get("llm_judge_score_0_3") is None]
    print(f"Judging {len(todo)} records with {args.model}  (concurrency={args.concurrency})")

    if not todo:
        print("Nothing to do — all records already have judge scores.")
    else:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)
        coros = [judge_record(client, args.model, records[i], q_key, r_key, p_key, sem) for i in todo]

        completed = 0
        for coro in tqdm_asyncio.as_completed(coros, total=len(coros)):
            await coro
            completed += 1
            if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
                Path(args.output).write_text(json.dumps(write_back(records), indent=2, ensure_ascii=False))

    # Build final summary
    overall = summarize_judged(records)
    out_obj = write_back(records)
    if isinstance(out_obj, dict):
        out_obj.setdefault("summary", {}).update(overall)
    Path(args.output).write_text(json.dumps(out_obj, indent=2, ensure_ascii=False))

    print(f"\nSaved to {args.output}")
    print("\n--- Overall ---")
    print(json.dumps(overall, indent=2))

    per_cfg = per_config_summary(records)
    if per_cfg:
        print("\n--- Per-config (new line) ---")
        for row in per_cfg:
            print(f"  layer={row['layer']:>3}  keep={row['keep_ratio']:.2f}  "
                  f"acc={row['normalized_accuracy']*100:5.1f}%  "
                  f"avg={row['avg_score']:.2f}  n={row['n']}")


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
