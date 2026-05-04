#!/bin/bash
# Claude Opus 4.6 via Pluto LLM proxy.

USE_GAZE_VIZ=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_gaze_instruction) USE_GAZE_VIZ=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"
export DECORD_EOF_RETRY_MAX=40960

export LLM_PROXY_ENDPOINT="${LLM_PROXY_ENDPOINT:-<model-endpoint>}"
export LLM_PROXY_KEY="${LLM_PROXY_KEY:-sk-REDACTED}"

MAX_PARALLEL="${MAX_PARALLEL:-10}"
export INNER_THREADS="${INNER_THREADS:-2}"
export STREAMGAZE_INNER_THREADS="$INNER_THREADS"

MODEL_NAME="ProxyOpus46"
WORKDIR="$ROOT_DIR/results/$MODEL_NAME"
if [ "$USE_GAZE_VIZ" = "true" ]; then
    LOGDIR="$WORKDIR/logs_viz";     RESULTS_DIR="$WORKDIR/results_viz"
else
    LOGDIR="$WORKDIR/logs";         RESULTS_DIR="$WORKDIR/results"
fi
mkdir -p "$LOGDIR" "$RESULTS_DIR"

QA_DIR="$ROOT_DIR/dataset/qa"
VIDEO_ROOT="$ROOT_DIR/dataset/videos/original_video"
GAZE_VIZ_VIDEO_ROOT="$ROOT_DIR/dataset/videos/gaze_viz_video"

echo "🔧 Model: $MODEL_NAME  MAX_PARALLEL=$MAX_PARALLEL  INNER_THREADS=$INNER_THREADS  gaze_viz=$USE_GAZE_VIZ"

PAST_TASKS=(past_scene_recall.json past_object_transition_prediction.json past_gaze_sequence_matching.json past_non_fixated_object_identification.json)
PRESENT_TASKS=(present_object_attribute_recognition.json present_object_identification_easy.json present_object_identification_hard.json)
FUTURE_TASKS=(present_future_action_prediction.json)
REMIND_TASKS=(proactive_gaze_triggered_alert.json proactive_object_appearance_alert.json)

cd "$ROOT_DIR/src"
PIDS=(); ALL_TASKS=()

launch() {
    local TASK_FILE="$1"; local BENCH="$2"
    local TASK_NAME="${TASK_FILE%.json}"
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do sleep 2; done
    echo "  Launching: $TASK_NAME ($BENCH)"
    if [ "$USE_GAZE_VIZ" = "true" ]; then
        python eval.py --model_name "$MODEL_NAME" --benchmark_name "$BENCH" \
            --data_file "$QA_DIR/$TASK_FILE" \
            --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
            --video_root "$VIDEO_ROOT" \
            --use_gaze_instruction --gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT" \
            > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    else
        python eval.py --model_name "$MODEL_NAME" --benchmark_name "$BENCH" \
            --data_file "$QA_DIR/$TASK_FILE" \
            --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
            --video_root "$VIDEO_ROOT" \
            > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    fi
    PIDS+=($!); ALL_TASKS+=("$TASK_FILE")
    echo "    → PID: $! | Log: $LOGDIR/${TASK_NAME}.log"
}

echo ""; echo "=== Past Tasks ==="
for T in "${PAST_TASKS[@]}"; do launch "$T" StreamingBenchGaze_Past_StreamGaze; done
echo ""; echo "=== Present Tasks ==="
for T in "${PRESENT_TASKS[@]}"; do launch "$T" StreamingBenchGaze_StreamGaze; done
echo ""; echo "=== Future Tasks ==="
for T in "${FUTURE_TASKS[@]}"; do launch "$T" StreamingBenchGaze_StreamGaze; done
echo ""; echo "=== Remind Tasks ==="
for T in "${REMIND_TASKS[@]}"; do launch "$T" StreamingBenchRemind_StreamGaze; done

echo ""; echo "Waiting for ${#PIDS[@]} tasks..."
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"; ST=$?
    TN="${ALL_TASKS[$i]%.json}"
    if [ $ST -eq 0 ]; then echo "  ✓ $TN"; else echo "  ✗ $TN (exit $ST)"; FAILED=1; fi
done

python "$ROOT_DIR/evaluate_results.py" "$RESULTS_DIR" "$MODEL_NAME" || true
echo "Done: $MODEL_NAME  gaze_viz=$USE_GAZE_VIZ  failed=$FAILED"
echo "Results: $RESULTS_DIR"
