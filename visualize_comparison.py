#!/usr/bin/env python3
"""
Visualize OVITA trajectory comparison results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def load_comparison_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_trajectory_coords(trajectory):
    """Extract x, y, z coordinates from trajectory"""
    if not trajectory:
        return [], [], []

    x_coords = [point['x'] for point in trajectory]
    y_coords = [point['y'] for point in trajectory]
    z_coords = [point['z'] for point in trajectory]
    return x_coords, y_coords, z_coords

def plot_trajectory_comparison(results):
    """Create 3D visualization of trajectory comparison"""
    fig = plt.figure(figsize=(15, 10))

    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Extract OVITA trajectories
    zero_shot = results.get('zero_shot_trajectory', {})
    original_traj = zero_shot.get('trajectory', [])
    modified_traj = zero_shot.get('modified trajectory', [])
    objects = zero_shot.get('objects', [])

    if original_traj:
        x_orig, y_orig, z_orig = extract_trajectory_coords(original_traj)
        ax.plot(x_orig, y_orig, z_orig, 'b-', linewidth=2, label='Original Trajectory', alpha=0.7)
        ax.scatter(x_orig[0], y_orig[0], z_orig[0], color='blue', s=100, marker='o', label='Start')
        ax.scatter(x_orig[-1], y_orig[-1], z_orig[-1], color='blue', s=100, marker='s', label='End')

    if modified_traj:
        x_mod, y_mod, z_mod = extract_trajectory_coords(modified_traj)
        ax.plot(x_mod, y_mod, z_mod, 'r-', linewidth=2, label='OVITA Modified Trajectory', alpha=0.8)
        ax.scatter(x_mod[0], y_mod[0], z_mod[0], color='red', s=100, marker='o')
        ax.scatter(x_mod[-1], y_mod[-1], z_mod[-1], color='red', s=100, marker='s')

    # Plot objects
    for obj in objects:
        x, y, z = obj['x'], obj['y'], obj['z']
        name = obj['name']

        # Color code objects
        color = 'green' if name == 'labyrinth' else 'orange'
        marker = 's' if name == 'labyrinth' else 'o'
        size = 200 if name == 'labyrinth' else 100

        ax.scatter(x, y, z, color=color, s=size, marker=marker,
                  label=f'{name.title()}', alpha=0.8, edgecolors='black')

        # Add object labels
        ax.text(x, y, z + 0.05, name, fontsize=9, ha='center')

    # Formatting
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'OVITA Trajectory Modification\nInstruction: "{zero_shot.get("instruction", "")}"',
                fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add comparison statistics
    if original_traj and modified_traj and objects:
        labyrinth = next((obj for obj in objects if obj['name'] == 'labyrinth'), None)
        if labyrinth:
            # Calculate average distance to labyrinth for both trajectories
            def avg_distance_to_object(traj, obj):
                distances = []
                for point in traj:
                    dist = np.sqrt((point['x'] - obj['x'])**2 +
                                 (point['y'] - obj['y'])**2 +
                                 (point['z'] - obj['z'])**2)
                    distances.append(dist)
                return np.mean(distances)

            orig_dist = avg_distance_to_object(original_traj, labyrinth)
            mod_dist = avg_distance_to_object(modified_traj, labyrinth)
            improvement = ((orig_dist - mod_dist) / orig_dist) * 100

            stats_text = f"""OVITA Performance:
• Original avg distance to labyrinth: {orig_dist:.3f}m
• Modified avg distance to labyrinth: {mod_dist:.3f}m
• Improvement: {improvement:.1f}% closer
• LLM: {results.get('LLM', 'Unknown')}
• Code executable: {results.get('code_executability', False)}"""

            ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    return fig

def main():
    results_file = 'comparison_results/latte_0.json'

    try:
        results = load_comparison_results(results_file)

        print("=== OVITA vs LaTTe Comparison Analysis ===\n")

        # Check what data we have
        has_ovita = bool(results.get('zero_shot_trajectory', {}).get('trajectory'))
        has_latte = bool(results.get('final_trajectory', {}).get('trajectory'))

        print(f"✅ OVITA Data Available: {has_ovita}")
        print(f"❌ LaTTe Data Available: {has_latte}")
        print(f"🤖 LLM Used: {results.get('LLM', 'Unknown')}")
        print(f"⚙️  Code Executable: {results.get('code_executability', False)}")

        if has_ovita:
            zero_shot = results['zero_shot_trajectory']
            print(f"\n📍 Instruction: '{zero_shot.get('instruction', 'N/A')}'")
            print(f"🎯 Objects in environment: {len(zero_shot.get('objects', []))}")
            print(f"📏 Original trajectory points: {len(zero_shot.get('trajectory', []))}")
            print(f"🔄 Modified trajectory points: {len(zero_shot.get('modified trajectory', []))}")

            # Create visualization
            fig = plot_trajectory_comparison(results)

            # Save visualization
            output_file = 'comparison_results/trajectory_visualization.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved: {output_file}")

            # Display the plot
            plt.show()
        else:
            print("\n❌ No trajectory data available for visualization")

        if not has_latte:
            print(f"\n⚠️  Note: LaTTe data is missing from comparison results.")
            print("   This means the LaTTe evaluation didn't complete successfully.")
            print("   Check the comparison script and LaTTe environment setup.")

    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
    except Exception as e:
        print(f"❌ Error processing results: {e}")

if __name__ == "__main__":
    main()