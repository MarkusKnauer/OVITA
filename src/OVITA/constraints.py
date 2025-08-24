import math
import numpy as np
import copy
from OVITA.util import convert_to_vector, convert_from_vector, resample_trajectory

husky_ws_dimensions=[-4,4,-3,3,0,0]
drone_ws_dimensions=[-10,10,-10,10,0,5]


class TrajectoryOptimizer:
    def __init__(self, safety_distance:float,compute_reachability_fn, lambda_smooth, lambda_obstacle, lambda_original,lambda_reach, epsilon):
        self.safety_distance = safety_distance
        self.lambda_smooth = lambda_smooth
        self.lambda_obstacle = lambda_obstacle
        self.lambda_original = lambda_original
        self.lambda_reach=lambda_reach
        self.epsilon = epsilon
        self.compute_reachability_gradient=compute_reachability_fn

    def optimize(self, trajectory, objects, max_iter=1000, step_size=0.01):
        # Extract velocity and positions from trajectory
        velocity = np.array(trajectory)[:, 3]
        trajectory = np.array(trajectory)[:, :3]
        original_trajectory = trajectory.copy()
        # Prepare obstacles as tuples of position and dimensions
        obstacles = [
            (np.array([obj['x'], obj['y'], obj['z']]), np.array(obj['dimensions']))
            for obj in objects
        ]

        for iteration in range(max_iter):
            # Compute gradients for different components
            smooth_grad = self.compute_smoothness_gradient(trajectory)
            obstacle_grad = self.compute_obstacle_gradient(trajectory, obstacles)
            reachability_grad = self.compute_reachability_gradient(trajectory)
            original_grad = self.compute_original_gradient(trajectory, original_trajectory)
            
            # Combine gradients with respective weights
            total_gradient = (
                self.lambda_smooth * smooth_grad +
                self.lambda_obstacle * obstacle_grad +
                self.lambda_original * original_grad +
                self.lambda_reach * reachability_grad
            )

            # Update trajectory using gradient descent
            trajectory -= step_size * total_gradient

        # Add velocity back to the optimized trajectory
        trajectory = np.hstack((trajectory, velocity.reshape(len(trajectory), 1)))
        return trajectory


    def compute_smoothness_gradient(self, trajectory):
        grad = np.zeros_like(trajectory)
        for i in range(1, len(trajectory) - 1):
            grad[i] = 2 * trajectory[i] - trajectory[i - 1] - trajectory[i + 1]
        return grad


    def compute_obstacle_gradient(self, trajectory, obstacles): 
        grad = np.zeros_like(trajectory)
        for i, point in enumerate(trajectory):
            for obstacle in obstacles:
                centroid, dimensions = obstacle  # centroid and dimensions of the cuboid
                # Compute the distance between the point and the centroid of the obstacle
                distance = np.linalg.norm(point - centroid)

                # If the point is inside the safety envelope, compute the gradient

                if (point[0] < centroid[0] + dimensions[0]/2 + self.safety_distance and 
                    point[0] > centroid[0] - dimensions[0]/2 - self.safety_distance and
                    point[1] < centroid[1] + dimensions[1]/2 + self.safety_distance and 
                    point[1] > centroid[1] - dimensions[1]/2 - self.safety_distance and
                    point[2] < centroid[2] + dimensions[2]/2 + self.safety_distance and 
                    point[2] > centroid[2] - dimensions[2]/2 - self.safety_distance):
                    # Point is inside the expanded cuboid

                    # Compute the direction vector away from the obstacle centroid
                    direction = (point - centroid) / (distance + 1e-6)  # Avoid division by zero
                    # Compute the gradient for this point
                    max_distance=max(dimensions)/2+self.safety_distance
                    grad[i] += -2 * (max_distance-distance) * direction

        return grad

    def compute_original_gradient(self, trajectory, original_trajectory):
        return 2 * (trajectory - original_trajectory)

def compute_reachability_gradient_arm(trajectory):
    min_reach=0.15
    max_reach=1.2
    base_position=[-0.5, 0, 0.6]
    min_z = 0.7  # Table height (minimum z-coordinate)
    # print("#############################")
    grad = np.zeros_like(trajectory)
    for i, point in enumerate(trajectory):
        vector_to_point=point - base_position 
        distance = np.linalg.norm(vector_to_point)
        if distance > max_reach:
            direction = vector_to_point / distance  # Normalize the vector
            grad[i] = 2*(distance - max_reach) * direction
        elif distance < min_reach:
            direction = vector_to_point / distance  # Normalize the vector
            grad[i] = 2*(min_reach - distance) * direction
        # # Handle table height constraint (z > min_z)
        # if point[2] < min_z:
        #     grad[i][2] += 4*(min_z - point[2])  # Push the point upwards in the z-direction
    return grad
   
class RobotArm7DOF_constraint(TrajectoryOptimizer):
    def __init__(self,safety_distance:float,lambda_smooth=1.0, lambda_obstacle=2.0, lambda_original=0.5, lambda_reach=4.0, epsilon=1e-8):
        super().__init__(safety_distance,compute_reachability_gradient_arm, lambda_smooth, lambda_obstacle, lambda_original,lambda_reach, epsilon)

    def satisfy_constraints(self,trajectory:list,objects:dict):
        velocity_max=0.8 #using setspeedoverride of KUKA arm 
        velocity_min=0
        trajectory_satisfied_constraints = copy.deepcopy(trajectory)
        for point in trajectory_satisfied_constraints:
            # Maximum speed constraints
            point['velocity'] = max(min(point['velocity'], velocity_max), velocity_min)
        
        trajectory_final=super().optimize(convert_to_vector(trajectory),objects)
        return convert_from_vector(trajectory_final)
    
    
def compute_reachability_gradient_husky(trajectory):
    min_x, max_x, min_y, max_y, min_z, max_z = husky_ws_dimensions
    grad = np.zeros_like(trajectory)

    for i, point in enumerate(trajectory):
        x, y, z = point
        if x < min_x:
            grad[i][0] = 2 * (min_x - x)
        elif x > max_x:
            grad[i][0] = 2 * (max_x - x)
        if y < min_y:
            grad[i][1] = 2 * (min_y - y)
        elif y > max_y:
            grad[i][1] = 2 * (max_y - y)
        
        # For ground robots, z must remain 0
        grad[i][2] = 2 * z

    return grad


def compute_reachability_gradient_drone(trajectory):
    min_x, max_x, min_y, max_y, min_z, max_z = drone_ws_dimensions
    grad = np.zeros_like(trajectory)

    for i, point in enumerate(trajectory):
        
        x, y, z = point
        if x < min_x:
            grad[i][0] = (min_x - x)
        elif x > max_x:
            grad[i][0] = (max_x - x)
        if y < min_y:
            grad[i][1] = (min_y - y)
        elif y > max_y:
            grad[i][1] = (max_y - y)
        if z < min_z:
            grad[i][2] = (min_z - z)
        elif z > max_z:
            grad[i][2] = (max_z - z)

    return grad
    
class Drone_and_husky_constraints(TrajectoryOptimizer):
    """ 
    max_dimensions is maximum and minimum 
    """
    def __init__(self,safety_distance, is_ground_robot, lambda_smooth=0.5, lambda_obstacle=5.0, lambda_original=0.3,lambda_reach=0, epsilon=1e-8):
        super().__init__(safety_distance,compute_reachability_gradient_husky if is_ground_robot else compute_reachability_gradient_drone,  lambda_smooth, lambda_obstacle, lambda_original,lambda_reach, epsilon)
        # in the format [min_x,max_x,min_y,max_y,min_z,max_z]
        self.max_dimensions=husky_ws_dimensions if is_ground_robot else drone_ws_dimensions

    def calculate_distance(self,point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) **
            2 + (point1[2] - point2[2]) ** 2)
    
    def satisfy_constraints(self, trajectory, objects):
        velocity_max = 3
        velocity_min = 0
        x_min, y_min, z_min = self.max_dimensions[0],self.max_dimensions[2],self.max_dimensions[4]
        x_max, y_max, z_max = self.max_dimensions[1],self.max_dimensions[3],self.max_dimensions[5]
        trajectory_satisfied_constraints = copy.deepcopy(trajectory)
        for point in trajectory_satisfied_constraints:
            # Maximum speed constraints
            point['velocity'] = max(min(point['velocity'], velocity_max), velocity_min)
            point['x'] = max(min(point['x'], x_max), x_min)
            point['y'] = max(min(point['y'], y_max), y_min)
            point['z'] = max(min(point['z'], z_max), z_min)
        
        trajectory_final=super().optimize(convert_to_vector(trajectory_satisfied_constraints),objects)
        return convert_from_vector(trajectory_final)
    

class general_constraints(TrajectoryOptimizer):
    def __init__(self,safety_distance, lambda_smooth=1.0, lambda_obstacle=3.0, lambda_original=0.5, lambda_reach=0, epsilon=1e-8):
        super().__init__(safety_distance,lambda trajectory: 0,  lambda_smooth, lambda_obstacle, lambda_original, lambda_reach, epsilon)

    def satisfy_constraints(self, trajectory, objects):
        trajectory_final=super().optimize(convert_to_vector(trajectory),objects)
        return convert_from_vector(trajectory_final)