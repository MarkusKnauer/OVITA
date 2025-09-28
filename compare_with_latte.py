#!/usr/bin/env python3
"""
Simple OVITA vs LaTTe Comparison Script
Usage: python compare_with_latte.py [trajectory_file]
"""
import sys
import os
import json
import subprocess
import argparse

def run_latte_adaptation(trajectory_file, instruction=None):
    """Run LaTTe trajectory adaptation"""
    print("Running LaTTe trajectory adaptation...")

    if instruction is None:
        # Load instruction from trajectory file
        with open(trajectory_file, 'r') as f:
            data = json.load(f)
            instruction = data.get('instruction', 'modify the trajectory')

    cmd = [
        'python', 'simple_latte_adapter.py',
        '--trajectory_path', trajectory_file,
        '--instruction', instruction,
        '--output_dir', 'comparison_results/'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("LaTTe adaptation completed")
        return True
    else:
        print("LaTTe failed:", result.stderr)
        return False

def run_ovita_adaptation(trajectory_file):
    """Run OVITA trajectory adaptation"""
    print("Running OVITA trajectory adaptation...")

    # Set API key
    os.environ['GOOGLE_API_KEY'] = "AIzaSyBYmi6MvCOMmBwqQCCufthfXj8IOvnOEBE"

    cmd = [
        'python', 'scripts/main.py',
        '--trajectory_path', trajectory_file,
        '--save_dir', 'comparison_results/',
        '--llm', 'gemini',
        '--save_results', 'True',
        '--robot_type', 'Drone'
    ]

    # Auto-accept trajectory
    result = subprocess.run(cmd, input='yes\n', capture_output=True, text=True)
    if result.returncode == 0:
        print("OVITA adaptation completed")
        return True
    else:
        print("OVITA failed:", result.stderr)
        return False

def compare_results(trajectory_name):
    """Compare and display results"""
    print("\nComparison Results:")

    # Check for LaTTe results
    latte_file = "comparison_results/latte_adapted_" + trajectory_name
    if os.path.exists(latte_file):
        print("LaTTe results:", latte_file)
    else:
        print("LaTTe results not found")

    # Check for OVITA results
    ovita_file = "comparison_results/" + trajectory_name
    if os.path.exists(ovita_file):
        print("OVITA results:", ovita_file)
    else:
        print("OVITA results not found")

    print("\nKey Differences:")
    print("- LaTTe: Pre-trained Transformer, fast inference, limited to training patterns")
    print("- OVITA: Real-time LLM code generation, flexible, handles novel scenarios")

    print("\nView results in: comparison_results/")

def main():
    parser = argparse.ArgumentParser(description="Compare OVITA vs LaTTe on trajectory adaptation")
    parser.add_argument("trajectory_file", nargs='?', default="dataset/latte_subset/latte_0.json",
                       help="Path to trajectory file")
    parser.add_argument("--instruction", type=str, help="Custom instruction for adaptation")

    args = parser.parse_args()

    if not os.path.exists(args.trajectory_file):
        print("Trajectory file not found:", args.trajectory_file)
        sys.exit(1)

    trajectory_name = os.path.basename(args.trajectory_file)

    print("OVITA vs LaTTe Trajectory Adaptation Comparison")
    print("=" * 55)
    print("Trajectory:", args.trajectory_file)

    # Create results directory
    os.makedirs("comparison_results", exist_ok=True)

    # Run both methods
    print("\n=== Phase 1: LaTTe ===")
    latte_success = run_latte_adaptation(args.trajectory_file, args.instruction)

    print("\n=== Phase 2: OVITA ===")
    ovita_success = run_ovita_adaptation(args.trajectory_file)

    # Compare results
    print("\n=== Phase 3: Comparison ===")
    compare_results(trajectory_name)

    if latte_success and ovita_success:
        print("\nComparison completed successfully!")
    else:
        print("\nSome methods failed - check error messages above")

if __name__ == "__main__":
    main()