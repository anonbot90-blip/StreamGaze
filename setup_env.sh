#!/bin/bash
# Quick setup script for StreamGaze evaluation

# Activate conda environment
source /home/colligo/miniconda3/etc/profile.d/conda.sh
conda activate streamgaze

# Set paths
export STREAMGAZE_ROOT="/home/colligo/sensei-fs-link/main/StreamGaze"
export PYTHONPATH="$STREAMGAZE_ROOT/src:$PYTHONPATH"
export STREAMGAZE_DATASET="/mnt/localssd/streamgaze_work/dataset"

# Create dataset directory on localssd
mkdir -p $STREAMGAZE_DATASET

echo "✅ StreamGaze environment activated"
echo "📁 Code: $STREAMGAZE_ROOT"
echo "📁 Dataset: $STREAMGAZE_DATASET"
echo ""
echo "To download dataset:"
echo "  cd $STREAMGAZE_DATASET"
echo "  huggingface-cli download danaleee/StreamGaze --repo-type dataset"

