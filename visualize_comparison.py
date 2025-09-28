#!/usr/bin/env python3
"""
Visualization Script for LaTTe vs OVITA Trajectory Comparison
Usage: python visualize_comparison.py [trajectory_name]
Example: python visualize_comparison.py latte_10
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from mpl_toolkits.mplot3d import Axes3D

def load_latte_trajectory(filepath):
    """Load LaTTe trajectory from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'adapted_trajectory' in data:
        trajectory = data['adapted_trajectory']['trajectory']
    elif 'original_trajectory' in data:
        trajectory = data['original_trajectory']['trajectory']
    else:
        trajectory = data['trajectory']

    # Convert [x, y, z, velocity] format to separate arrays
    trajectory = np.array(trajectory)
    return {
        'x': trajectory[:, 0],
        'y': trajectory[:, 1],
        'z': trajectory[:, 2],
        'velocity': trajectory[:, 3]
    }

def load_ovita_trajectory(filepath):
    """Load OVITA trajectory from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Try different possible trajectory keys
    trajectory_data = None
    for key in ['zero_shot_trajectory', 'final_trajectory', 'trajectory']:
        if key in data and 'trajectory' in data[key]:
            trajectory_data = data[key]['trajectory']
            break
        elif key in data:
            trajectory_data = data[key]
            break

    if trajectory_data is None:
        raise ValueError("Could not find trajectory data in OVITA file")

    # Convert dict format to arrays
    x = [point['x'] for point in trajectory_data]
    y = [point['y'] for point in trajectory_data]
    z = [point['z'] for point in trajectory_data]
    velocity = [point['velocity'] for point in trajectory_data]

    return {
        'x': np.array(x),
        'y': np.array(y),
        'z': np.array(z),
        'velocity': np.array(velocity)
    }

def create_comparison_visualization(latte_data, ovita_data, trajectory_name, save_path=None):
    """Create a comprehensive comparison visualization."""

    fig = plt.figure(figsize=(20, 12))

    # 1. 3D Trajectory Comparison
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(latte_data['x'], latte_data['y'], latte_data['z'],
             'b-', linewidth=2, marker='o', markersize=4, label='LaTTe', alpha=0.8)
    ax1.plot(ovita_data['x'], ovita_data['y'], ovita_data['z'],
             'r-', linewidth=2, marker='s', markersize=4, label='OVITA', alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Trajectory Comparison - {trajectory_name}')
    ax1.legend()
    ax1.grid(True)

    # 2. X-Y Projection
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(latte_data['x'], latte_data['y'], 'b-o', linewidth=2, markersize=4,
             label='LaTTe', alpha=0.8)
    ax2.plot(ovita_data['x'], ovita_data['y'], 'r-s', linewidth=2, markersize=4,
             label='OVITA', alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('X-Y Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # 3. Velocity Comparison
    ax3 = fig.add_subplot(2, 3, 3)
    time_latte = np.arange(len(latte_data['velocity']))
    time_ovita = np.arange(len(ovita_data['velocity']))
    ax3.plot(time_latte, latte_data['velocity'], 'b-o', linewidth=2,
             label='LaTTe', alpha=0.8)
    ax3.plot(time_ovita, ovita_data['velocity'], 'r-s', linewidth=2,
             label='OVITA', alpha=0.8)
    ax3.set_xlabel('Waypoint Index')
    ax3.set_ylabel('Velocity')
    ax3.set_title('Velocity Profile Comparison')
    ax3.legend()
    ax3.grid(True)

    # 4. X Position over time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(time_latte, latte_data['x'], 'b-o', linewidth=2, label='LaTTe', alpha=0.8)
    ax4.plot(time_ovita, ovita_data['x'], 'r-s', linewidth=2, label='OVITA', alpha=0.8)
    ax4.set_xlabel('Waypoint Index')
    ax4.set_ylabel('X Position')
    ax4.set_title('X Position Comparison')
    ax4.legend()
    ax4.grid(True)

    # 5. Y Position over time
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(time_latte, latte_data['y'], 'b-o', linewidth=2, label='LaTTe', alpha=0.8)
    ax5.plot(time_ovita, ovita_data['y'], 'r-s', linewidth=2, label='OVITA', alpha=0.8)
    ax5.set_xlabel('Waypoint Index')
    ax5.set_ylabel('Y Position')
    ax5.set_title('Y Position Comparison')
    ax5.legend()
    ax5.grid(True)

    # 6. Z Position over time
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(time_latte, latte_data['z'], 'b-o', linewidth=2, label='LaTTe', alpha=0.8)
    ax6.plot(time_ovita, ovita_data['z'], 'r-s', linewidth=2, label='OVITA', alpha=0.8)
    ax6.set_xlabel('Waypoint Index')
    ax6.set_ylabel('Z Position')
    ax6.set_title('Z Position Comparison')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()

    # Print statistics
    print(f"\n📊 Trajectory Comparison Statistics for {trajectory_name}")
    print("=" * 60)
    print(f"{'Metric':<25} {'LaTTe':<15} {'OVITA':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'Number of waypoints':<25} {len(latte_data['x']):<15} {len(ovita_data['x']):<15} {abs(len(latte_data['x']) - len(ovita_data['x'])):<15}")
    print(f"{'Avg velocity':<25} {np.mean(latte_data['velocity']):<15.4f} {np.mean(ovita_data['velocity']):<15.4f} {abs(np.mean(latte_data['velocity']) - np.mean(ovita_data['velocity'])):<15.4f}")
    print(f"{'Max velocity':<25} {np.max(latte_data['velocity']):<15.4f} {np.max(ovita_data['velocity']):<15.4f} {abs(np.max(latte_data['velocity']) - np.max(ovita_data['velocity'])):<15.4f}")
    print(f"{'Min velocity':<25} {np.min(latte_data['velocity']):<15.4f} {np.min(ovita_data['velocity']):<15.4f} {abs(np.min(latte_data['velocity']) - np.min(ovita_data['velocity'])):<15.4f}")

    # Calculate trajectory length
    def calc_length(data):
        points = np.column_stack([data['x'], data['y'], data['z']])
        diffs = np.diff(points, axis=0)
        lengths = np.sqrt(np.sum(diffs**2, axis=1))
        return np.sum(lengths)

    latte_length = calc_length(latte_data)
    ovita_length = calc_length(ovita_data)
    print(f"{'Total path length':<25} {latte_length:<15.4f} {ovita_length:<15.4f} {abs(latte_length - ovita_length):<15.4f}")

    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Visualization saved to: {save_path}")

    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize LaTTe vs OVITA trajectory comparison')
    parser.add_argument('trajectory_name',
                       help='Base name of trajectory (e.g., latte_10)')
    parser.add_argument('--results_dir', default='comparison_results',
                       help='Directory containing comparison results')
    parser.add_argument('--save', action='store_true',
                       help='Save visualization to file')

    args = parser.parse_args()

    # Construct file paths
    latte_file = os.path.join(args.results_dir, f"latte_adapted_{args.trajectory_name}.json")
    ovita_file = os.path.join(args.results_dir, f"{args.trajectory_name}.json")

    # Check if files exist
    if not os.path.exists(latte_file):
        print(f"❌ LaTTe result file not found: {latte_file}")
        return

    if not os.path.exists(ovita_file):
        print(f"❌ OVITA result file not found: {ovita_file}")
        return

    print(f"📈 Loading trajectory comparison for: {args.trajectory_name}")
    print(f"🧠 LaTTe file: {latte_file}")
    print(f"⚡ OVITA file: {ovita_file}")

    try:
        # Load trajectory data
        latte_data = load_latte_trajectory(latte_file)
        ovita_data = load_ovita_trajectory(ovita_file)

        # Create save path if requested
        save_path = None
        if args.save:
            save_path = os.path.join(args.results_dir, f"{args.trajectory_name}_comparison.png")

        # Create visualization
        fig = create_comparison_visualization(latte_data, ovita_data, args.trajectory_name, save_path)

        print(f"\n✅ Visualization complete! Close the plot window to exit.")
        plt.show()

    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return

if __name__ == "__main__":
    main()