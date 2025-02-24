# This file contains helper functions needed for the agent. 
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import copy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def compare_trajectory(original_trajectory, modified_trajectory, title, points=None, elev=30, azim=45, file_name=None):
    """
    Helper function to visualize the trajectory. Use elev and azim parameters to set the camera view.
    Points is a set of critical points/objects observed in the environment.
    Reds represent modified trajectory.
    """
    # Extract points and velocities
    x1, y1, z1, vel1 = map(list, zip(*original_trajectory))
    x2, y2, z2, vel2 = map(list, zip(*modified_trajectory))
    # Set up a figure with two subplots: one for 3D trajectory and one for velocity profile
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
    plt.tight_layout()
    plt.axis('off')
    
    # 3D Trajectory Plot
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.plot(x1, y1, z1, label='Original Trajectory', color='blue')
    # ax1.scatter(x1, y1, z1, color='blue', marker='o')
    ax1.plot(x2, y2, z2, label='Modified Trajectory', color='red')
    # ax1.scatter(x2, y2, z2, color='red', marker='o')
    ax1.view_init(elev=elev, azim=azim)
    
    # Mark start and end positions for both trajectories
    ax1.scatter(x1[0], y1[0], z1[0], color='blue', marker='o', s=100, label='Original Start')
    ax1.text(x1[0]+0.02, y1[0], z1[0], 'Start', fontsize=18, ha='right', color='blue',font='times new roman')
    
    ax1.scatter(x1[-1], y1[-1], z1[-1], color='blue', marker='^', s=100, label='Original End')
    ax1.text(x1[-1]+0.02, y1[-1], z1[-1], 'End', fontsize=18, ha='right', color='blue',font='times new roman')
    
    ax1.scatter(x2[0], y2[0], z2[0], color='red', marker='o', s=100, label='Modified Start')
    ax1.text(x2[0]+0.02, y2[0], z2[0], 'Start', fontsize=18, ha='right', color='red',font='times new roman')
    
    ax1.scatter(x2[-1], y2[-1], z2[-1], color='red', marker='^', s=100, label='Modified End')
    ax1.text(x2[-1]+0.02, y2[-1], z2[-1], 'End', fontsize=18, ha='right', color='red',font='times new roman')
    
    # Plot the objects present in the environment
    if points is not None:
        for obj in points:
            # Extract object properties
            name = obj["name"]
            name = obj["name"]
            x = obj["x"]
            y = obj["y"]
            z = obj["z"]
            obj_length = obj['dimensions'][0]
            obj_width = obj['dimensions'][1]
            obj_height = obj['dimensions'][2]

            # Define the vertices of the cuboid
            # Correctly define the vertices of a cuboid
            vertices = [
                [x - obj_length / 2, y - obj_width / 2, z - obj_height / 2],  # Bottom-front-left
                [x + obj_length / 2, y - obj_width / 2, z - obj_height / 2],  # Bottom-front-right
                [x + obj_length / 2, y + obj_width / 2, z - obj_height / 2],  # Bottom-back-right
                [x - obj_length / 2, y + obj_width / 2, z - obj_height / 2],  # Bottom-back-left
                [x - obj_length / 2, y - obj_width / 2, z + obj_height / 2],  # Top-front-left
                [x + obj_length / 2, y - obj_width / 2, z + obj_height / 2],  # Top-front-right
                [x + obj_length / 2, y + obj_width / 2, z + obj_height / 2],  # Top-back-right
                [x - obj_length / 2, y + obj_width / 2, z + obj_height / 2],  # Top-back-left
            ]


            # Define the 6 faces of the cuboid using the vertices
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[3], vertices[0], vertices[4], vertices[7]],
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]]
            ]

            # Create a 3D polygon collection
            poly3d = Poly3DCollection(faces, alpha=.6, linewidths=1, edgecolors='grey')
            poly3d.set_facecolor('grey')
            ax1.add_collection3d(poly3d)

            # Add label to the center of the top face of the object
            # Add label to the top of the object
            ax1.text(x + obj_length / 2, y + obj_width / 2, z + obj_height+0.05, name,
            color='black', ha='left', va='top', fontsize=18,font='times new roman')


    # Set labels for the 3D plot
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(title)
    # ax1.legend(loc="upper left")

    # Velocity Profile Plot
    ax2 = fig.add_subplot(212)
    ax2.plot(range(len(vel1)), vel1, label="Original Velocity", color='blue')
    ax2.plot(range(len(vel2)), vel2, label="Modified Velocity", color='red')
    
    # Set labels for the Velocity Profile plot
    ax2.set_xlabel('Position Index')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Velocity Profile')
    # ax2.legend()

    # Save or show plot
    if file_name is not None:
        plt.savefig(file_name)
        plt.close('all')
    else:
        plt.show()



def save_results(trajectory, modified_trajectory_list,instruction, global_feedbacks, local_feedbacks, objects, HLP_list, code_list, file_name,code_executability):
    save_dict={'trajectory':trajectory,'instruction':instruction,'local feedbacks':local_feedbacks, 'global feedbacks': global_feedbacks,'modified_trajectories':modified_trajectory_list,'objects':objects, "code_list": code_list, "HLP_list":HLP_list,"violations":None}
    with open(file_name,'w') as outfile: 
        outfile.write(json.dumps(save_dict))


def get_points(trajectory):
   """
   Helper function to get x,y,z points from trajectory
   """
   x = [point['x'] for point in trajectory]
   y = [point['y'] for point in trajectory]
   z = [point['z'] for point in trajectory if 'z' in point]
   if len(z)==0:
      z=[0.0]*len(trajectory)
   vel = [point['velocity'] for point in trajectory if 'velocity' in point]
   if len(vel)==0:
      vel=[1.0]*len(trajectory)
   return x,y,z,vel

def convert_to_vector(trajectory):
   """
   Trajectory is a json format dict
   """
   return np.vstack([np.array(dim) for dim in get_points(trajectory) ]).T

def convert_from_vector(vector):
   """
   Accepts 4D vector
   Returns the custom format
   """
   trajectory=[]
   for i in range(0, len(vector)):
     trajectory.append({'x':vector[i][0], 'y':vector[i][1], 'z':vector[i][2], 'velocity':vector[i][3] if len(vector[i])==4 else 1.0})
   return trajectory
   
################################################
# optional Trajectory processing functions
################################################

def resample_trajectory(trajectory, n_points):
    trajectory = np.array(trajectory)
    original_points = len(trajectory)
    indices = np.linspace(0, original_points - 1, original_points)
    new_indices = np.linspace(0, original_points - 1, n_points)
    interpolated_x = interp1d(indices, trajectory[:, 0], kind='linear', fill_value="extrapolate")
    interpolated_y = interp1d(indices, trajectory[:, 1], kind='linear', fill_value="extrapolate")
    interpolated_z = interp1d(indices, trajectory[:, 2], kind='linear', fill_value="extrapolate")
    interpolated_velocity = interp1d(indices, trajectory[:, 3], kind='linear', fill_value="extrapolate")
    resampled_trajectory = np.column_stack((
        interpolated_x(new_indices),
        interpolated_y(new_indices),
        interpolated_z(new_indices),
        interpolated_velocity(new_indices)
    ))
    
    return resampled_trajectory


def min_max_normalize_trajectory(trajectory):
    """
    Returns:
    np.array: Min-max normalized trajectory in the range [-1, 1]
    traj_min
    traj_max
    """
    traj=trajectory[:,:3]
    velocity=trajectory[:,3]
    traj_min = np.min(traj, axis=0)
    traj_max = np.max(traj, axis=0)
    
    # Apply min-max normalization to scale to [-1, 1]
    traj_scaled = 2 * (traj - traj_min) / (traj_max - traj_min) - 1
    print(traj_scaled.shape,velocity.shape)
    return np.hstack([traj_scaled,velocity.reshape(-1,1)]), traj_min, traj_max

def reconstruct_trajectory_min_max(traj_scaled,traj_max,traj_min):
   traj_original = (traj_scaled + 1) / 2 * (traj_max - traj_min) + traj_min
   return traj_original


def point_line_distance(point, start, end):
    if np.array_equal(start, end):
        return np.linalg.norm(point - start)
    
    line_vec = end - start
    point_vec = point - start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    projection_length = np.dot(point_vec, line_unitvec)
    
    if projection_length < 0:
        return np.linalg.norm(point - start)
    elif projection_length > line_len:
        return np.linalg.norm(point - end)
    else:
        projection = start + projection_length * line_unitvec
        return np.linalg.norm(point - projection)

def iterative_endpoint_fit(trajectory, tolerance, normalize=True):
    trajectory=convert_to_vector(trajectory)
    # Ignore the velocity component
    if len(trajectory) < 3:
        return trajectory
    
    def simplify_recursive(start_idx, end_idx, points):
        max_dist = 0
        index = start_idx
        for i in range(start_idx + 1, end_idx):
            # Calculate the distance between only the x,y,z and not the vel component
            dist = point_line_distance(points[i][:3], points[start_idx][:3], points[end_idx][:3])
            if dist > max_dist:
                max_dist = dist
                index = i
        
        if max_dist > tolerance:
            left_half = simplify_recursive(start_idx, index, points)
            right_half = simplify_recursive(index, end_idx, points)
            return np.vstack((left_half[:-1], right_half))
        else:
            return np.array([points[start_idx], points[end_idx]])
    
    simplified_trajectory = simplify_recursive(0, len(trajectory) - 1, trajectory)
    return convert_from_vector(simplified_trajectory)

# Return smoothened trajectory using cubic spline function
def smooth_trajectory_spline(trajectory, num_points=80):
   if not trajectory or len(trajectory) < 2:
      return trajectory

   t = np.arange(len(trajectory)) 
   x = np.array([point['x'] for point in trajectory])
   y = np.array([point['y'] for point in trajectory])
   z = np.array([point['z'] for point in trajectory])
   velocity = np.array([point['velocity'] for point in trajectory])
   spline_x = CubicSpline(t, x)
   spline_y = CubicSpline(t, y)
   spline_z = CubicSpline(t, z)
   spline_velocity = CubicSpline(t, velocity)

   t_new = np.linspace(0, len(trajectory) - 1, num=num_points)
   x_smooth = spline_x(t_new)
   y_smooth = spline_y(t_new)
   z_smooth = spline_z(t_new)
   velocity_smooth = spline_velocity(t_new)
   smoothed_trajectory = [
       {'x': x_smooth[i], 'y': y_smooth[i], 'z': z_smooth[i], 'velocity': velocity_smooth[i]}
       for i in range(num_points)
   ]
   return smoothed_trajectory



# Helper function for converting trajectories from LaTTe format to our format
def convert_from_latte(file_path, new_file_path):

    with open(file_path,"r") as file:
        data=json.load(file)

    counter=0
    dataset={}
    # Dataset has fields, input_traj, text, objects
    for key in data.keys():
        traj=[]
        for point in data[key]['input_traj']:
            traj.append(dict({'x':point[0],'y':point[0],'z':point[0],'Velocity':point[3]}))

        instruction=data[key]['text']
        objects=[]
        for i,name in enumerate(data[key]['obj_names']):
            objects.append(dict({'name':name,'x':data[key]['obj_poses'][i][0],'y':data[key]['obj_poses'][i][1],'z':data[key]['obj_poses'][i][2]}))

        dataset[counter]=dict({'trajectory':traj,"instruction":instruction, "objects":objects})
        counter+=1


    with open(new_file_path,"w") as file:
        json.dump(dataset,file)