#!/bin/bash
# Check StreamGaze setup status

echo "========================================="
echo "StreamGaze Setup Status Check"
echo "========================================="
echo ""

# Check conda environment
echo "🔧 Conda Environment:"
source /home/colligo/miniconda3/etc/profile.d/conda.sh
conda activate streamgaze 2>/dev/null && echo "  ✅ streamgaze environment active" || echo "  ❌ streamgaze environment not found"
echo ""

# Check dataset extraction
echo "📦 Dataset Extraction:"
RUNNING=$(ps aux | grep "tar -xf" | grep -v grep | wc -l)
echo "  ⏳ Tar extractions running: $RUNNING/6"
echo ""

# Check video files
echo "📹 Videos:"
ORIG_COUNT=$(ls /mnt/localssd/streamgaze_work/dataset/videos/original_video/*.mp4 2>/dev/null | wc -l)
VIZ_COUNT=$(ls /mnt/localssd/streamgaze_work/dataset/videos/gaze_viz_video/*.mp4 2>/dev/null | wc -l)
echo "  Original videos: $ORIG_COUNT / 285"
echo "  Gaze viz videos: $VIZ_COUNT / 285"
echo ""

# Check QA files
echo "📝 QA Files:"
QA_COUNT=$(ls /mnt/localssd/streamgaze_work/dataset/qa/*.json 2>/dev/null | wc -l)
echo "  QA files: $QA_COUNT / 10"
[ $QA_COUNT -eq 10 ] && echo "  ✅ All QA files present" || echo "  ❌ QA files missing"
echo ""

# Check symlink
echo "🔗 Symlink:"
if [ -L /home/colligo/sensei-fs-link/main/StreamGaze/dataset ]; then
    echo "  ✅ Dataset symlink exists"
    TARGET=$(readlink -f /home/colligo/sensei-fs-link/main/StreamGaze/dataset)
    echo "  → Points to: $TARGET"
else
    echo "  ❌ Dataset symlink missing"
fi
echo ""

# Check disk space
echo "💾 Disk Space:"
echo "  Localssd usage:"
df -h /mnt/localssd | tail -1
echo ""

# Summary
echo "========================================="
if [ $QA_COUNT -eq 10 ] && [ $ORIG_COUNT -gt 0 ] && [ $VIZ_COUNT -gt 0 ]; then
    if [ $RUNNING -eq 0 ]; then
        echo "✅ Setup COMPLETE - Ready to run!"
        echo ""
        echo "Next steps:"
        echo "  1. Update GPT-4o credentials in src/model/GPT4o.py"
        echo "  2. Run: source setup_env.sh"
        echo "  3. Run: bash scripts/gpt4o.sh"
    else
        echo "⏳ Extraction in progress ($RUNNING jobs running)"
        echo "   Run this script again to check status"
    fi
else
    echo "⚠️  Setup incomplete"
    echo "   Waiting for extraction to finish..."
fi
echo "========================================="

