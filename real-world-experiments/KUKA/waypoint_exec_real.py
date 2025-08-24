#!/usr/bin/env python

# import rospy
# from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import geometry_msgs
import numpy as np
import ikpy
import pybullet as p
import pybullet_data
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# # Load the KUKA robot model (using ikpy or any other IK library)
from ikpy.chain import Chain

# # Initialize your robot's kinematic chain (URDF file should be correctly specified)  iiwa_link_1
kuka_chain = Chain.from_urdf_file("kuka_ws/src/iiwa_ros/iiwa_description/urdf/iiwa7.urdf",base_elements=["world"])

def compute_ik(pose):
    """Compute the IK for the given pose."""
    # Extract position and orientation from the pose
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [0,0,0,1]
    joint_angles = kuka_chain.inverse_kinematics(position,orientation)
    return list(joint_angles)

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import json
from geometry_msgs.msg import Point,Quaternion
from sensor_msgs.msg import JointState
def get_current_joint_positions():
    """
    Fetch the robot's current joint positions from the `/joint_states` topic.
    """
    try:
        joint_states = rospy.wait_for_message("/joint_states", JointState, timeout=5)
        return list(joint_states.position)
    except rospy.ROSException:
        rospy.logerr("Failed to get current joint states.")
        return [0.0] * 7  # Default positions in case of failure
    
def move_kuka_arm_in_gazebo():
    # Initialize the ROS node
    rospy.init_node('kuka_arm_gazebo_controller')
    # rate = rospy.Rate(10)
    # Create a publisher for the trajectory topic  /position_joint_trajectory_controller/command   /iiwa/PositionTrajectoryController/command
    pub = rospy.Publisher('/iiwa/PositionTrajectoryController/command', JointTrajectory, queue_size=10)
    # Wait for the publisher to be ready
    rospy.sleep(2)

    # Define the waypoints in Cartesian space
    json_file="temp_traj.json"
    joint_traj_vel_file="joint_waypoints_temp_traj_test.json"
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    with open(joint_traj_vel_file, 'r') as file:
        joint_traj_vel = json.load(file)
    data=data[0]

    ###### 
    waypoints = [
        Pose(position=Point(traj["x"]+0.5, traj["y"], traj["z"]-0.6)) for traj in data['trajectory']]
    #######

    # Create a JointTrajectory message
    robot_name="iiwa"
    trajectory_msg = JointTrajectory()
    trajectory_msg.joint_names = [
        f"{robot_name}_joint_1", f"{robot_name}_joint_2", f"{robot_name}_joint_3", 
        f"{robot_name}_joint_4", f"{robot_name}_joint_5", f"{robot_name}_joint_6", f"{robot_name}_joint_7"
    ]
    # Main loop to send joint commands
    for i in range(len(data['trajectory'])):
        # Define your trajectory points
        point = JointTrajectoryPoint()
        point.positions = [-pos for pos in joint_traj_vel["joint_trajectory_pos"][i][:7]]#joint_angles[:7]  # Example joint positions
        point.velocities = [-vel for vel in joint_traj_vel["joint_trajectory_vel"][i][:7]]
        point.time_from_start = rospy.Duration(i+0.01)  # Time to reach this point  5/len(waypoints)
        # Add the points to the trajectory message
        trajectory_msg.points.append(point)
        
    # Publish the trajectory message
    pub.publish(trajectory_msg)
    rospy.loginfo("Trajectory has been sent to the KUKA arm in Gazebo")

    # Keep the node alive until the trajectory is complete
    rospy.sleep(5)
 
if __name__ == '__main__':
    try:
        move_kuka_arm_in_gazebo()
    except rospy.ROSInterruptException:
        pass
