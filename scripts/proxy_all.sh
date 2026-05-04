#!/bin/bash
# Master orchestrator: launch all 6 proxy-baseline runs in parallel.
#   3 models × 2 gaze modes (no-gaze + gaze_viz) = 6 runs
#   Each run internally parallelizes 10 task JSONs via MAX_PARALLEL
#   Peak concurrency: 6 × MAX_PARALLEL Python procs (default 60)
#
# The Pluto proxy demuxes by provider family, so inter-family parallelism
# is "free" in terms of upstream rate budget.

set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

# Per-provider knobs (already baked into per-model scripts, but overridable).
export MAX_PARALLEL="${MAX_PARALLEL:-10}"

echo "============================================"
echo "Launching 6 proxy runs in parallel"
echo "  ROOT_DIR = $ROOT_DIR"
echo "  MAX_PARALLEL = $MAX_PARALLEL"
echo "============================================"

# --- Gemini 2.5-pro ---------------------------------------------------------
bash scripts/proxy_gemini25pro.sh                         > logs/gemini_nogaze.log 2>&1 &
GEM_NG=$!
bash scripts/proxy_gemini25pro.sh --use_gaze_instruction  > logs/gemini_gaze.log   2>&1 &
GEM_G=$!

# --- Claude Opus 4.6 --------------------------------------------------------
bash scripts/proxy_opus46.sh                              > logs/opus_nogaze.log   2>&1 &
OPUS_NG=$!
bash scripts/proxy_opus46.sh --use_gaze_instruction       > logs/opus_gaze.log     2>&1 &
OPUS_G=$!

# --- GPT-5.4 ----------------------------------------------------------------
bash scripts/proxy_gpt54.sh                               > logs/gpt54_nogaze.log  2>&1 &
GPT_NG=$!
bash scripts/proxy_gpt54.sh --use_gaze_instruction        > logs/gpt54_gaze.log    2>&1 &
GPT_G=$!

echo "PIDs: gemini(ng=$GEM_NG g=$GEM_G)  opus(ng=$OPUS_NG g=$OPUS_G)  gpt54(ng=$GPT_NG g=$GPT_G)"
echo "Logs: logs/{gemini,opus,gpt54}_{nogaze,gaze}.log"
echo ""

# Collect exit codes without bailing out on first failure.
FAIL=0
for pair in "gemini_nogaze:$GEM_NG" "gemini_gaze:$GEM_G" \
            "opus_nogaze:$OPUS_NG" "opus_gaze:$OPUS_G" \
            "gpt54_nogaze:$GPT_NG" "gpt54_gaze:$GPT_G"; do
    name="${pair%%:*}"; pid="${pair##*:}"
    wait "$pid"; st=$?
    if [ $st -eq 0 ]; then
        echo "  ✓ $name (pid $pid)"
    else
        echo "  ✗ $name (pid $pid, exit $st)"
        FAIL=1
    fi
done

echo "============================================"
if [ $FAIL -eq 0 ]; then
    echo "All 6 runs completed successfully."
else
    echo "One or more runs failed — see logs/*.log"
fi
echo "============================================"
exit $FAIL
