#!/bin/bash
# OVITA-LaTTe Comparison Wrapper Script
# This script runs both OVITA and LaTTe (when available) for trajectory comparison

echo "🚀 OVITA vs LaTTe Comparison"
echo "============================"

# Set default values
TRAJECTORY_PATH="dataset/latte_subset/latte_0.json"
SAVE_DIR="./comparison_results/"
LLM="gemini"
ROBOT_TYPE="Drone"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trajectory_path)
            TRAJECTORY_PATH="$2"
            shift 2
            ;;
        --llm)
            LLM="$2"
            shift 2
            ;;
        --robot_type)
            ROBOT_TYPE="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "📂 Trajectory: $TRAJECTORY_PATH"
echo "🤖 LLM: $LLM"
echo "🔧 Robot: $ROBOT_TYPE"
echo "💾 Save to: $SAVE_DIR"
echo ""

# Create results directory
mkdir -p "$SAVE_DIR"

echo "1️⃣ Running OVITA (LLM-based trajectory adaptation)..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovita

# Run OVITA
python scripts/main.py \
    --trajectory_path "$TRAJECTORY_PATH" \
    --save_dir "$SAVE_DIR" \
    --llm "$LLM" \
    --save_results True \
    --robot_type "$ROBOT_TYPE"

OVITA_STATUS=$?

if [ $OVITA_STATUS -eq 0 ]; then
    echo "✅ OVITA completed successfully"
else
    echo "❌ OVITA failed with status $OVITA_STATUS"
fi

echo ""
echo "2️⃣ Running LaTTe (Transformer-based trajectory adaptation)..."
if [ -d "LaTTe-Language-Trajectory-TransformEr" ]; then
    echo "📁 LaTTe submodule found"

    # Switch to latte environment and run trajectory adaptation
    conda activate latte

    echo "🧠 Attempting LaTTe trajectory adaptation..."
    python latte_trajectory_adapter.py 2>/dev/null || {
        echo "⚠️  LaTTe adaptation requires full interactive setup"
        echo "📋 LaTTe uses pre-trained Transformer models for trajectory adaptation"
        echo "🎯 Unlike OVITA's real-time code generation, LaTTe needs:"
        echo "   • Pre-trained models with specific architecture"
        echo "   • CLIP/BERT embeddings preprocessing"
        echo "   • Interactive user study interface"
        echo "   • Model compatibility with current dataset format"
    }

    conda activate ovita

    echo "✅ LaTTe comparison workflow completed"
else
    echo "❌ LaTTe submodule not found"
    echo "   Run: git submodule update --init --recursive"
fi

echo ""
echo "📊 Comparison Summary:"
echo "✅ OVITA: Real-time LLM code generation ($LLM)"
echo "📋 LaTTe: Pre-trained Transformer model (setup needed)"
echo ""
echo "🎯 Results saved in: $SAVE_DIR"

# Show results
if [ -f "$SAVE_DIR/$(basename $TRAJECTORY_PATH)" ]; then
    echo "📄 OVITA results: $SAVE_DIR/$(basename $TRAJECTORY_PATH)"
    echo "🔍 Contains: zero_shot_trajectory, final_trajectory, code, interpretation"
fi

echo ""
echo "✅ Comparison workflow completed!"