import rospy
import json
from geometry_msgs.msg import Twist
from time import sleep
import math

# Function to read the trajectory from a JSON file
def read_trajectory(file_path):
    with open(file_path, 'r') as f:
        trajectory = json.load(f)
    return trajectory

# Function to calculate and publish cmd_vel for the trajectory
def publish_trajectory(trajectory):
    # Create a ROS node
    rospy.init_node('trajectory_publisher', anonymous=True)
    
    # Create a publisher for the cmd_vel topic
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    # Set the rate for publishing (10 Hz)
    rate = rospy.Rate(10)

    # Loop through the trajectory and publish the velocities
    for i in range(len(trajectory) - 1):
        current_point = trajectory[i]
        next_point = trajectory[i + 1]
        
        # Extract positions and velocity
        # x1, y1, v1 = current_point
        # x2, y2, v2 = next_point

        x1, y1 = current_point
        x2, y2= next_point
        v1=1
        
        # Create a Twist message for cmd_vel
        cmd_vel = Twist()
        
        # Linear velocity in x and y direction (based on difference of positions)
        cmd_vel.linear.x = v1  # Assuming the velocity is constant between two points
        cmd_vel.linear.y = 0.0  # Assuming no movement in y direction
        
        # Angular velocity (if there is a change in orientation)
        # Assuming robot needs to turn towards the next point
        delta_x = x2 - x1
        delta_y = y2 - y1
        angle = math.atan2(delta_y, delta_x)
        cmd_vel.angular.z = angle
        
        # Publish the cmd_vel message
        rospy.loginfo(f"y2 {y2}, y1 {y1}")
        rospy.loginfo(f"Publishing cmd_vel: Linear: ({cmd_vel.linear.x}, {cmd_vel.linear.y}), Angular: {cmd_vel.angular.z}")
        cmd_vel_pub.publish(cmd_vel)
        
        # Sleep for the duration of the move (for simplicity, using velocity to determine sleep time)
        move_time = abs(x2 - x1) / v1  # You can refine this based on velocity and distance
        rospy.sleep(move_time)

        if rospy.is_shutdown():
            break

# Main function
if __name__ == '__main__':
    try:
        # Path to the JSON file containing trajectory
        trajectory_file = 'trajectory_1.json'
        
        # Read the trajectory from the file
        trajectory_data = read_trajectory(trajectory_file)['trajectory']
        
        # Publish the trajectory commands
        publish_trajectory(trajectory_data)

    except rospy.ROSInterruptException:
        pass
