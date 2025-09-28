#!/usr/bin/env python3
"""
LaTTe Trajectory Adapter for OVITA vs LaTTe Comparison
Runs LaTTe trajectory adaptation and saves results in OVITA-compatible format
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
    from motion_refiner_4D import Motion_refiner
    from TF4D_mult_features import load_model
    import config
except ImportError as e:
    print(f"❌ LaTTe import error: {e}")
    print("Make sure LaTTe environment is activated and models are downloaded")
    sys.exit(1)

def convert_ovita_to_latte_format(trajectory_data):
    """Convert OVITA trajectory format to LaTTe format"""
    trajectory = trajectory_data.get('trajectory', [])
    instruction = trajectory_data.get('instruction', '')
    objects = trajectory_data.get('objects', [])

    # Convert trajectory to numpy array [x, y, z, velocity]
    traj_array = []
    for point in trajectory:
        traj_array.append([point['x'], point['y'], point['z'], point['velocity']])

    traj_array = np.array(traj_array)

    # Create LaTTe data format
    latte_data = {
        'input_traj': traj_array,
        'original_traj': traj_array,  # For comparison
        'instruction': instruction,
        'objects': objects,
        'locality_factor': 1.0  # Default value
    }

    return latte_data

def convert_latte_to_ovita_format(latte_result, original_data):
    """Convert LaTTe output back to OVITA format"""
    if latte_result is None:
        return None

    # Convert numpy array back to OVITA trajectory format
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

def run_latte_adaptation(trajectory_file, instruction=None):
    """Run LaTTe trajectory adaptation"""

    print("🔧 Initializing LaTTe...")

    # Initialize motion refiner (clip_only=True to avoid image dataset requirement)
    traj_n = 40
    mr = Motion_refiner(
        load_models=True,
        traj_n=traj_n,
        locality_factor=True,
        clip_only=True,  # Use CLIP only to avoid image dataset folder requirement
        load_precomp_emb=True
    )

    print("📥 Loading pre-trained model...")

    # Load the Transformer model
    model_files = [f for f in os.listdir("LaTTe-Language-Trajectory-TransformEr/models/") if f.endswith('.h5')]
    if not model_files:
        print("❌ No model files found")
        return None

    # Use the Transformer model (not the fully connected one)
    transformer_models = [f for f in model_files if 'TF-' in f and 'fullyconn' not in f]
    if transformer_models:
        model_file = f"LaTTe-Language-Trajectory-TransformEr/models/{transformer_models[0]}"
    else:
        model_file = f"LaTTe-Language-Trajectory-TransformEr/models/{model_files[0]}"

    print(f"Using model: {os.path.basename(model_file)}")

    try:
        model = load_model(model_file, delimiter="-")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    print("📁 Loading trajectory data...")

    # Load trajectory data
    with open(trajectory_file, 'r') as f:
        trajectory_data = json.load(f)

    # Use provided instruction or default from file
    if instruction is None:
        instruction = trajectory_data.get('instruction', 'modify the trajectory')

    print(f"🎯 Instruction: '{instruction}'")

    # Convert to LaTTe format
    latte_data = convert_ovita_to_latte_format(trajectory_data)

    print("🧠 Running LaTTe trajectory adaptation...")

    try:
        # Apply LaTTe interaction
        pred, original_traj = mr.apply_interaction(
            model,
            latte_data,
            instruction,
            label=False,
            images=None
        )

        if pred is not None and len(pred) > 0:
            adapted_trajectory = pred[0]  # First prediction
            print(f"✅ LaTTe adaptation successful! Output shape: {adapted_trajectory.shape}")

            # Convert back to OVITA format
            result = convert_latte_to_ovita_format(adapted_trajectory, trajectory_data)

            return {
                'success': True,
                'original_trajectory': trajectory_data,
                'latte_result': result,
                'instruction': instruction,
                'model_info': os.path.basename(model_file)
            }
        else:
            print("❌ LaTTe returned no results")
            return None

    except Exception as e:
        print(f"❌ LaTTe adaptation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="LaTTe Trajectory Adaptation")
    parser.add_argument("--trajectory_path", type=str, required=True, help="Path to trajectory JSON file")
    parser.add_argument("--instruction", type=str, help="Custom instruction (optional)")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Output directory")

    args = parser.parse_args()

    print("=== LaTTe Trajectory Adaptation ===")

    # Run LaTTe adaptation
    result = run_latte_adaptation(args.trajectory_path, args.instruction)

    if result and result.get('success', False):
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)

        # Save in comparison format
        trajectory_name = os.path.basename(args.trajectory_path)
        output_file = os.path.join(args.output_dir, f"latte_{trajectory_name}")

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"💾 LaTTe results saved: {output_file}")

        # Also save just the trajectory for OVITA to use
        latte_traj_file = os.path.join(args.output_dir, f"latte_trajectory_{trajectory_name}")
        with open(latte_traj_file, 'w') as f:
            json.dump(result['latte_result'], f, indent=2)

        print(f"📊 LaTTe trajectory saved: {latte_traj_file}")

        return True
    else:
        print("❌ LaTTe adaptation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)