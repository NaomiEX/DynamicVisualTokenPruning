#LLM-judge accuracy for timing_qwen.py outputs.
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


JUDGE_MODEL_DEFAULT = "unsloth/Meta-Llama-3.1-8B-Instruct"
JUDGE_MAX_NEW_TOKENS = 1000


def judge_prompt(question: str, ground_truth_answer: str, model_answer: str) -> str:
    return f"""You are an expert at judging whether the given model answer matches a VQA ground truth answer.

You are NOT judging whether the model answer is reasonable from the image or question.
You are ONLY judging whether the model answer contains or entails the ground truth answer.

Ground truth answer: {ground_truth_answer}

Rules:
- Judge carefully and precisely.
- Correct means the answer clearly gives the ground truth answer as its final answer or one of its accepted meanings.
- Incorrect means the answer does not give the ground truth answer.
- If the answer says the ground truth cannot be determined, it is incorrect.
- If the answer says a different answer instead of the ground truth, it is incorrect.
- Extra explanation is allowed.
- If the answer contains multiple answers, it can still be correct as long as the ground truth answer is in the answer.
- If the answer is uncertain but contains the ground truth answer, it is still correct.
- Ignore whether the answer seems reasonable from the question or image.

Question, for context only: {question}
Model answer: {model_answer}

Return only valid JSON in exactly this format:
{{
  "correct": <true or false>,
  "explanation": "<short explanation>"
}}

Do not include markdown, code fences, or any text outside the JSON object.
Do not write <true or false> literally; replace it with either true or false based on the rules."""


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"Expected boolean judge value, got {value!r}")


def parse_judge_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        if "true/false" in candidate.lower():
            raise ValueError(f"Judge returned placeholder true/false: {text!r}")
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            candidate = re.sub(r"\bTrue\b", "true", candidate)
            candidate = re.sub(r"\bFalse\b", "false", candidate)
            data = json.loads(candidate)
        return {
            "correct": parse_bool(data["correct"]),
            "explanation": str(data.get("explanation", "")),
        }
    pair = re.search(r'"?correct"?\s*[:=]\s*(true|false|True|False)\b', text)
    if pair:
        explanation = ""
        m = re.search(r'"?explanation"?\s*[:=]\s*"([^"]*)"', text)
        if m:
            explanation = m.group(1)
        return {"correct": parse_bool(pair.group(1)), "explanation": explanation}
    raise ValueError(f"Judge did not return parseable correctness field: {text!r}")


def judge_one(tokenizer, model, question, ground_truth_answer, model_answer):
    prompt = judge_prompt(question, ground_truth_answer, model_answer)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    out_ids = ids[:, inputs["input_ids"].shape[1]:]
    judge_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    parsed = parse_judge_json(judge_text)
    parsed["judge_text"] = judge_text.strip()
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="timing_qwen.py JSON output")
    ap.add_argument("--output", default=None, help="judged JSON output (default: <input>_judged.json)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_judged.json")

    payload = json.loads(in_path.read_text())
    results = payload["results"]
    if args.limit is not None:
        results = results[: args.limit]

    print(f"loading judge: {args.judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    judged = []
    n_ok = 0
    n_correct = 0
    for i, row in enumerate(results):
        if "error" in row or "generated_text" not in row or row.get("answer") is None:
            judged.append({**row, "judge_skipped": True})
            continue
        try:
            j = judge_one(tokenizer, model, row["question"], row["answer"], row["generated_text"])
            new_row = {**row, "judge_correct": int(j["correct"]),
                       "judge_explanation": j["explanation"],
                       "judge_text": j["judge_text"]}
            n_ok += 1
            n_correct += int(j["correct"])
        except Exception as e:
            new_row = {**row, "judge_error": str(e)}
        judged.append(new_row)
        if i % 25 == 0:
            print(f"[{i}/{len(results)}] correct so far: {n_correct}/{n_ok}")

    payload["results"] = judged
    payload.setdefault("summary", {})
    payload["summary"]["judge_num_ok"] = n_ok
    payload["summary"]["judge_accuracy"] = round(n_correct / n_ok, 4) if n_ok else None
    payload["summary"]["judge_model"] = args.judge_model

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    print(f"judge accuracy: {payload['summary']['judge_accuracy']} ({n_correct}/{n_ok})")


if __name__ == "__main__":
    main()
