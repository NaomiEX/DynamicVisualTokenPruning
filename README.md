# DVT: Dynamic Visual Token Pruning

Prompt-conditioned visual-token pruning for Qwen2.5-VL. A lightweight Query-Aware Pruner is trained offline against debiased provenance scores from the full VLM, then drops ~60-80% of visual tokens at inference time *before* they enter the LLM decoder. The result is a ~2× LLM-prefill speedup with no loss (and on perception-localized inputs, a small *gain*) in accuracy.

This README covers reproducing the MME-RealWorld-Lite results in `mme_lite_results.md`. 
---

## Setup

```bash
conda create -n dvtp python=3.11 -y
conda activate dvtp
pip install "torch>=2.10" "transformers>=5.8" accelerate "datasets==3.6.0" qwen-vl-utils pillow
```

Hardware: single NVIDIA A100 80 GB. The Qwen2.5-VL-7B weights (~15 GB) and the MME-RealWorld-Lite parquet are pulled from HuggingFace on first run and cached.

---

## Reproducing the results

### 1. Learned-pruner eval (the main result)

Requires the trained checkpoint at `mme_data/best_pruner_mme.pt` (shipped with this repo). Runs both unpruned and pruned generation per sample on the 669-row camera-perception subset, recording exact-match accuracy and per-phase latency (`vision_ms`, `pruner_ms`, `prefill_ms`, `ttft_ms`, `avg_tbt_ms`, `total_latency_ms`).

```bash
python - <<'EOF'
import sys
sys.path.insert(0, 'mme_data')
import evaluate_learned_pruner_mme_lite_exact_match as ev
for kr, suffix in [(0.2, '0p2'), (0.4, '0p4')]:
    ev.KEEP_RATIO = kr
    ev.OUTPUT_PATH = ev.DATA_DIR / f'learned_pruner_lite_exact_match_results_{suffix}.jsonl'
    ev.SUMMARY_PATH = ev.DATA_DIR / f'learned_pruner_lite_exact_match_summary_{suffix}.json'
    ev.main()
EOF
```

Writes per-sample JSONL + aggregate summary JSON for each keep ratio under `mme_data/`. ~7-8 min per keep ratio on A100.

### 2. Training-free baselines (FastV / Random / No Pruning)

```bash
bash run_mme_eval.sh
```

Sweeps `accuracy_mme_realworld_qwen.py` over five configs and writes `mme_{none,fastv_k2_r02,fastv_k2_r04,random_r02,random_r04}.json` in the repo root. ~30 min total on A100.

---

## Output schema (both pipelines)

Each per-config JSON contains a `summary` block with accuracy aggregates and average per-phase latency, plus per-sample rows. See `mme_lite_results.md` for the consolidated table.

---

## Notes

- Long sweeps run cleanly under `tmux` if you want to log off (`tmux new -s mme_eval` → launch → `Ctrl-b d` to detach, `tmux attach -t mme_eval` to come back).
- Both scripts run a 2-iter warmup before timed samples and bracket each phase with `torch.cuda.synchronize()` so reported latencies are real wall-clock.
- `accuracy_mme_realworld_qwen.py` uses bfloat16 + SDPA; the learned-pruner script uses float16 + SDPA to match the training setup. The two unpruned baselines differ by ~0.7 pp on greedy-decoded accuracy; within-script comparisons are apples-to-apples.
