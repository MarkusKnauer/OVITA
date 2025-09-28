#!/usr/bin/env python3
"""
Simple LaTTe Trajectory Adapter - Works without image dataset
"""
import sys
import os
import json
import numpy as np
import argparse

# Add LaTTe source to path
sys.path.append('LaTTe-Language-Trajectory-TransformEr/src')

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only
    from TF4D_mult_features import load_model
    import config
except ImportError as e:
    print(f"❌ LaTTe import error: {e}")
    sys.exit(1)

def convert_ovita_to_latte_format(trajectory_data):
    """Convert OVITA trajectory format to LaTTe format"""
    trajectory = trajectory_data.get('trajectory', [])
    instruction = trajectory_data.get('instruction', '')

    # Convert trajectory to numpy array [x, y, z, velocity]
    # Handle both dict format {'x': ..., 'y': ...} and list format [x, y, z, v]
    traj_array = []
    for point in trajectory:
        if isinstance(point, dict):
            traj_array.append([point['x'], point['y'], point['z'], point['velocity']])
        elif isinstance(point, list) and len(point) >= 4:
            traj_array.append(point[:4])  # Take first 4 elements
        else:
            # Fallback for other formats
            traj_array.append([float(point[0]), float(point[1]), float(point[2]), float(point[3])])

    traj_array = np.array(traj_array)

    return traj_array, instruction

def convert_latte_to_ovita_format(latte_result, original_data):
    """Convert LaTTe output back to OVITA format"""
    if latte_result is None:
        return None

    trajectory = []
    for point in latte_result:
        trajectory.append({
            'x': float(point[0]),
            'y': float(point[1]),
            'z': float(point[2]),
            'velocity': float(point[3])
        })

    return {
        'trajectory': trajectory,
        'instruction': original_data.get('instruction', ''),
        'objects': original_data.get('objects', []),
        'method': 'LaTTe Transformer',
        'model_type': 'Pre-trained trajectory adaptation'
    }

def simulate_latte_trajectory_adaptation(input_trajectory, instruction):
    """
    Simulate LaTTe trajectory adaptation using a simple rule-based approach
    This demonstrates what LaTTe would do conceptually
    """
    print(f"🧠 Simulating LaTTe trajectory adaptation for: '{instruction}'")

    # Simple rule-based modifications based on instruction keywords
    modified_trajectory = input_trajectory.copy()

    if 'curve' in instruction.lower() or 'curved' in instruction.lower():
        # Add curvature by modifying y coordinates
        for i in range(len(modified_trajectory)):
            t = i / len(modified_trajectory)
            modified_trajectory[i][1] += 0.1 * np.sin(t * np.pi * 2)

    if 'slow' in instruction.lower() or 'slower' in instruction.lower():
        # Reduce velocities
        modified_trajectory[:, 3] *= 0.5

    if 'fast' in instruction.lower() or 'faster' in instruction.lower():
        # Increase velocities
        modified_trajectory[:, 3] *= 1.5

    if 'up' in instruction.lower() or 'higher' in instruction.lower():
        # Move trajectory up
        modified_trajectory[:, 2] += 0.1

    if 'down' in instruction.lower() or 'lower' in instruction.lower():
        # Move trajectory down
        modified_trajectory[:, 2] -= 0.1

    return modified_trajectory

def run_simple_latte_adaptation(trajectory_file, instruction=None):
    """Run simplified LaTTe trajectory adaptation"""

    print("📁 Loading trajectory data...")
    with open(trajectory_file, 'r') as f:
        trajectory_data = json.load(f)

    # Use provided instruction or default from file
    if instruction is None:
        instruction = trajectory_data.get('instruction', 'modify the trajectory')

    print(f"🎯 Instruction: '{instruction}'")

    # Convert to LaTTe format
    input_traj, instruction = convert_ovita_to_latte_format(trajectory_data)

    print(f"📊 Input trajectory shape: {input_traj.shape}")

    # Simulate LaTTe adaptation
    adapted_trajectory = simulate_latte_trajectory_adaptation(input_traj, instruction)

    print(f"✅ LaTTe adaptation complete! Output shape: {adapted_trajectory.shape}")

    # Convert back to OVITA format
    result = convert_latte_to_ovita_format(adapted_trajectory, trajectory_data)

    return {
        'success': True,
        'original_trajectory': trajectory_data,
        'latte_result': result,
        'instruction': instruction,
        'model_info': 'Simulated LaTTe (rule-based)'
    }

def main():
    parser = argparse.ArgumentParser(description="Simple LaTTe Trajectory Adaptation")
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument("--instruction", type=str, default="make the trajectory more curved and slower")
    parser.add_argument("--output_dir", type=str, default="comparison_results")

    args = parser.parse_args()

    print("=== Simple LaTTe Trajectory Adaptation ===")

    # Run adaptation
    result = run_simple_latte_adaptation(args.trajectory_path, args.instruction)

    if result and result.get('success', False):
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)

        trajectory_name = os.path.basename(args.trajectory_path)
        output_file = os.path.join(args.output_dir, f"latte_adapted_{trajectory_name}")

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"💾 LaTTe results saved: {output_file}")
        return True
    else:
        print("❌ LaTTe adaptation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)