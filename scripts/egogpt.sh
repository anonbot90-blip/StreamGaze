#!/bin/bash

# ============================================================================
# EgoGPT-7b-EgoIT Evaluation Script for StreamGaze Benchmark
# 
# Usage:
#   bash scripts/egogpt.sh                    # Without gaze visualization
#   bash scripts/egogpt.sh --use_gaze_instruction  # With gaze visualization
#
# Results will be saved to:
#   - results/EgoGPT/results/         (without gaze viz)
#   - results/EgoGPT/results_viz/     (with gaze viz)
# ============================================================================

# Parse command line arguments
USE_GAZE_VIZ=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --use_gaze_instruction)
            USE_GAZE_VIZ=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--use_gaze_instruction]"
            exit 1
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src"

# ===== MODEL CONFIGURATION =====
MODEL_NAME="EgoGPT"
# ===============================

WORKDIR="$ROOT_DIR/results/$MODEL_NAME"

# Select video and results paths based on gaze instruction flag
if [ "$USE_GAZE_VIZ" = "true" ]; then
    LOGDIR="$WORKDIR/logs_viz"
    RESULTS_DIR="$WORKDIR/results_viz"
    VIDEO_ROOT="$ROOT_DIR/dataset/videos/gaze_viz_video"
    echo "========================================="
    echo "🎯 Using GAZE VISUALIZATION videos"
    echo "   (with green dot + red circle overlay)"
    echo "========================================="
else
    LOGDIR="$WORKDIR/logs"
    RESULTS_DIR="$WORKDIR/results"
    VIDEO_ROOT="$ROOT_DIR/dataset/videos/original_video"
    echo "========================================="
    echo "📹 Using ORIGINAL videos"
    echo "   (no gaze overlay)"
    echo "========================================="
fi

mkdir -p "$LOGDIR"
mkdir -p "$RESULTS_DIR"

QA_DIR="$ROOT_DIR/dataset/qa"
GAZE_VIZ_VIDEO_ROOT="$ROOT_DIR/dataset/videos/gaze_viz_video"

echo "🔧 Model: $MODEL_NAME"
echo "📁 Results: $RESULTS_DIR"
echo "📝 Logs: $LOGDIR"
echo "========================================="
echo ""

# Define tasks
PAST_TASKS=(
    "past_scene_recall.json"
    "past_object_transition_prediction.json"
    "past_gaze_sequence_matching.json"
    "past_non_fixated_object_identification.json"
)

PRESENT_TASKS=(
    "present_object_identification_easy.json"
    "present_object_identification_hard.json"
    "present_object_attribute_recognition.json"
)

FUTURE_TASKS=(
    "present_future_action_prediction.json"
)

REMIND_TASKS=(
    "proactive_gaze_triggered_alert.json"
    "proactive_object_appearance_alert.json"
)

PIDS=()
GPU_ID=0  # Track which GPU to use

# Change to src directory for Python imports
cd "$ROOT_DIR/src" || exit 1

# Run Past Tasks
echo "=== 🔙 Past Tasks (Memory & Temporal Recall) ==="
for TASK_FILE in "${PAST_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    echo "  📌 Launching: $TASK_NAME on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchGaze_Past_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        ${USE_GAZE_VIZ:+--use_gaze_instruction} \
        ${USE_GAZE_VIZ:+--gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT"} \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    
    PIDS+=($!)
    echo "     → PID: $! | GPU: $GPU_ID | Log: $LOGDIR/${TASK_NAME}.log"
    GPU_ID=$(((GPU_ID + 1) % 8))  # Cycle through GPUs 0-7
done

# Run Present Tasks
echo ""
echo "=== 👁️  Present Tasks (Real-time Perception) ==="
for TASK_FILE in "${PRESENT_TASKS[@]}" "${FUTURE_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    echo "  📌 Launching: $TASK_NAME on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchGaze_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        ${USE_GAZE_VIZ:+--use_gaze_instruction} \
        ${USE_GAZE_VIZ:+--gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT"} \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    
    PIDS+=($!)
    echo "     → PID: $! | GPU: $GPU_ID | Log: $LOGDIR/${TASK_NAME}.log"
    GPU_ID=$(((GPU_ID + 1) % 8))  # Cycle through GPUs 0-7
done

# Run Proactive Tasks
echo ""
echo "=== 🔮 Proactive Tasks (Anticipation & Alerting) ==="
for TASK_FILE in "${REMIND_TASKS[@]}"; do
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    echo "  📌 Launching: $TASK_NAME on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
        --model_name $MODEL_NAME \
        --benchmark_name StreamingBenchRemind_StreamGaze \
        --data_file "$QA_DIR/$TASK_FILE" \
        --output_file "$RESULTS_DIR/${TASK_NAME}_output.json" \
        --video_root "$VIDEO_ROOT" \
        ${USE_GAZE_VIZ:+--use_gaze_instruction} \
        ${USE_GAZE_VIZ:+--gaze_viz_video_root "$GAZE_VIZ_VIDEO_ROOT"} \
        > "$LOGDIR/${TASK_NAME}.log" 2>&1 &
    
    PIDS+=($!)
    echo "     → PID: $! | GPU: $GPU_ID | Log: $LOGDIR/${TASK_NAME}.log"
    GPU_ID=$(((GPU_ID + 1) % 8))  # Cycle through GPUs 0-7
done

# Wait for all tasks
echo ""
echo "========================================="
echo "⏳ Waiting for all tasks to complete..."
echo "   Total tasks: $((${#PAST_TASKS[@]} + ${#PRESENT_TASKS[@]} + ${#FUTURE_TASKS[@]} + ${#REMIND_TASKS[@]}))"
echo "   Monitor logs: tail -f $LOGDIR/*.log"
echo "========================================="
echo ""

FAILED=0
ALL_TASKS=("${PAST_TASKS[@]}" "${PRESENT_TASKS[@]}" "${FUTURE_TASKS[@]}" "${REMIND_TASKS[@]}")

for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    STATUS=$?
    TASK_FILE="${ALL_TASKS[$i]}"
    TASK_NAME=$(echo "$TASK_FILE" | sed 's/\.json//')
    
    if [ $STATUS -eq 0 ]; then
        echo "  ✅ Completed: $TASK_NAME"
    else
        echo "  ❌ Failed: $TASK_NAME (exit code: $STATUS)"
        echo "     → Check log: $LOGDIR/${TASK_NAME}.log"
        FAILED=1
    fi
done

echo ""
echo "========================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ All evaluations completed successfully!"
    echo "   - Past tasks: ${#PAST_TASKS[@]}"
    echo "   - Present tasks: ${#PRESENT_TASKS[@]}"
    echo "   - Future tasks: ${#FUTURE_TASKS[@]}"
    echo "   - Proactive tasks: ${#REMIND_TASKS[@]}"
else
    echo "❌ Some evaluations failed. Check logs above."
fi
echo "📁 Results saved to: $RESULTS_DIR/"
echo "========================================="

# Run automatic evaluation
echo ""
echo "========================================="
echo "🔬 Running Automatic Evaluation..."
echo "========================================="

cd "$ROOT_DIR" || exit 1
python evaluate_results.py "$RESULTS_DIR" "$MODEL_NAME"

echo ""
echo "========================================="
echo "✅ All tasks completed!"
echo "========================================="


