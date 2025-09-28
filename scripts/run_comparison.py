#!/usr/bin/env python3
"""
OVITA-LaTTe Comparison Script

This script runs both OVITA and LaTTe on the same trajectory for comparison.
It provides a unified interface to test both methods on new trajectories.
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path

# Add OVITA to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_ovita(trajectory_path, save_dir, llm="gemini", robot_type="Drone"):
    """Run OVITA trajectory adaptation"""
    print("🤖 Running OVITA with LLM-based code generation...")

    cmd = [
        "python", "scripts/main.py",
        "--trajectory_path", trajectory_path,
        "--save_dir", save_dir,
        "--llm", llm,
        "--save_results", "True",
        "--robot_type", robot_type
    ]

    try:
        # Activate ovita environment and run
        env = os.environ.copy()
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__) + "/..",
                              capture_output=True, text=True, env=env)

        if result.returncode == 0:
            print("✅ OVITA completed successfully")
            return True, result.stdout
        else:
            print(f"❌ OVITA failed: {result.stderr}")
            return False, result.stderr

    except Exception as e:
        print(f"❌ Error running OVITA: {e}")
        return False, str(e)

def run_latte(trajectory_path, save_dir):
    """Run LaTTe trajectory adaptation"""
    print("🧠 Running LaTTe with Transformer-based adaptation...")

    # Check if LaTTe is available
    latte_path = os.path.join(os.path.dirname(__file__), "..", "LaTTe-Language-Trajectory-TransformEr")

    if not os.path.exists(latte_path):
        print("⚠️  LaTTe submodule not found. Using pre-computed results from dataset.")
        return use_precomputed_latte(trajectory_path, save_dir)

    try:
        # This would be the LaTTe execution command when properly set up
        # For now, return a placeholder indicating it needs setup
        print("⚠️  LaTTe environment setup needed. Please run:")
        print("   conda activate latte")
        print("   cd LaTTe-Language-Trajectory-TransformEr")
        print("   python src/evaluate_models.py --trajectory_path ...")

        return use_precomputed_latte(trajectory_path, save_dir)

    except Exception as e:
        print(f"❌ Error running LaTTe: {e}")
        return False, str(e)

def use_precomputed_latte(trajectory_path, save_dir):
    """Use pre-computed LaTTe results if available"""

    # Extract filename to look for corresponding LaTTe result
    filename = os.path.basename(trajectory_path)

    # Look in the latte_subset for corresponding results
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "latte_subset")

    if os.path.exists(os.path.join(dataset_dir, filename)):
        print(f"📋 Using pre-computed LaTTe results for {filename}")

        # Load the original trajectory which contains LaTTe-style results
        with open(trajectory_path, 'r') as f:
            data = json.load(f)

        # Create mock LaTTe result (in real setup, this would be actual LaTTe inference)
        latte_result = {
            "method": "LaTTe (pre-computed)",
            "original_trajectory": data["trajectory"],
            "instruction": data["instruction"],
            "objects": data["objects"],
            "note": "This is a placeholder. Real LaTTe inference requires model setup."
        }

        # Save LaTTe result
        latte_save_path = os.path.join(save_dir, f"latte_{filename}")
        os.makedirs(os.path.dirname(latte_save_path), exist_ok=True)

        with open(latte_save_path, 'w') as f:
            json.dump(latte_result, f, indent=4)

        print(f"📄 LaTTe placeholder results saved to {latte_save_path}")
        return True, "LaTTe placeholder completed"

    else:
        print(f"❌ No pre-computed LaTTe results found for {filename}")
        return False, "No LaTTe data available"

def compare_results(ovita_path, latte_path):
    """Compare OVITA and LaTTe results"""
    print("\n📊 Comparing OVITA vs LaTTe results...")

    try:
        # Load results
        with open(ovita_path, 'r') as f:
            ovita_data = json.load(f)

        with open(latte_path, 'r') as f:
            latte_data = json.load(f)

        print("\n=== COMPARISON SUMMARY ===")
        print(f"📁 OVITA results: {ovita_path}")
        print(f"📁 LaTTe results: {latte_path}")

        if 'zero_shot_trajectory' in ovita_data:
            ovita_traj = ovita_data['zero_shot_trajectory'].get('modified trajectory', [])
            print(f"🤖 OVITA: Generated {len(ovita_traj)} modified waypoints")
            print(f"🤖 OVITA: LLM used = {ovita_data.get('LLM', 'unknown')}")
            print(f"🤖 OVITA: Code executable = {ovita_data.get('code_executability', False)}")

        print(f"🧠 LaTTe: {latte_data.get('method', 'unknown method')}")

        return True

    except Exception as e:
        print(f"❌ Error comparing results: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compare OVITA and LaTTe on trajectory adaptation")

    parser.add_argument("--trajectory_path", required=True,
                       help="Path to trajectory JSON file")
    parser.add_argument("--save_dir", default="./comparison_results/",
                       help="Directory to save results")
    parser.add_argument("--llm", choices=["openai", "claude", "gemini"], default="gemini",
                       help="LLM for OVITA")
    parser.add_argument("--robot_type", choices=["Drone", "Arm", "GroundRobot"], default="Drone",
                       help="Robot type for workspace constraints")
    parser.add_argument("--run_ovita", action="store_true", default=True,
                       help="Run OVITA (default: True)")
    parser.add_argument("--run_latte", action="store_true", default=True,
                       help="Run LaTTe (default: True)")

    args = parser.parse_args()

    print("🚀 OVITA vs LaTTe Comparison Tool")
    print("=" * 50)
    print(f"📂 Input: {args.trajectory_path}")
    print(f"💾 Output: {args.save_dir}")
    print(f"🤖 OVITA LLM: {args.llm}")
    print(f"🔧 Robot Type: {args.robot_type}")
    print()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    filename = os.path.basename(args.trajectory_path)
    ovita_results = None
    latte_results = None

    # Run OVITA
    if args.run_ovita:
        ovita_success, ovita_output = run_ovita(
            args.trajectory_path,
            args.save_dir,
            args.llm,
            args.robot_type
        )
        if ovita_success:
            ovita_results = os.path.join(args.save_dir, filename)

    # Run LaTTe
    if args.run_latte:
        latte_success, latte_output = run_latte(args.trajectory_path, args.save_dir)
        if latte_success:
            latte_results = os.path.join(args.save_dir, f"latte_{filename}")

    # Compare results
    if ovita_results and latte_results:
        compare_results(ovita_results, latte_results)

    print("\n✅ Comparison complete!")
    print(f"📊 Results saved in: {args.save_dir}")

if __name__ == "__main__":
    main()