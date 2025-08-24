#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
import json
import math
import time

def send_waypoints(waypoints):
    rospy.init_node('husky_teleoperator')
    
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base action server")
    prev_waypoint=waypoints[0] # it is a tuple of (x,y)

    for i, (x, y) in enumerate(waypoints):
        rospy.loginfo(f"Sending waypoint {i + 1}: ({x}, {y})")

        goal = MoveBaseGoal()

        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position = Point(x, y, 0)
        # Calculating the heading direction
        delta_x=x-prev_waypoint[0]
        delta_y=y-prev_waypoint[1]

        yaw_angle=math.atan2(delta_y,delta_x)
        quaternion = quaternion_from_euler(0,0,yaw_angle)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        # import ipdb; ipdb.set_trace()
        prev_waypoint=(x,y)
        client.send_goal(goal)
        time.sleep(1.5)
        if i==0:
            client.wait_for_result()

        if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
           rospy.loginfo(f"Waypoint {i + 1} reached successfully")
        else:
            rospy.logwarn(f"Failed to reach waypoint {i + 1}")

if __name__ == '__main__':
    try:
        file_name="trajectory_4_modif.json"
        with open(file_name, 'r') as file:
            trajectory = json.load(file)

        waypoints=[]
        prev_waypoint=trajectory['trajectory'][0]
        for i,point in enumerate(trajectory["trajectory"]): 
            waypoints.append(tuple((-(point[0]+0.2), point[1])))

        send_waypoints(waypoints)
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception")

