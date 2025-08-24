#!/usr/bin/env python

import rospy
import moveit_commander
import geometry_msgs.msg
from moveit_commander import MoveGroupCommander
import sys

def main():
    # Initialize the moveit_commander and ROS node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('kuka_arm_waypoint_follower', anonymous=True)

    robot = moveit_commander.RobotCommander()
    # group_names = robot.get_group_names()


    # Instantiate a MoveGroupCommander object for the KUKA arm
    arm = MoveGroupCommander("manipulator", wait_for_servers=20)

    # Set the reference frame for pose targets
    arm.set_pose_reference_frame("iiwa_link_0")

    # Set the allowed planning time
    arm.set_planning_time(10)

    # Set a tolerance on the goal
    arm.set_goal_tolerance(0.01)

    # Define a list of waypoints to follow
    waypoints = []

    # Example waypoint 1
    waypoint1 = geometry_msgs.msg.Pose()
    waypoint1.position.x = 0.5
    waypoint1.position.y = 0.0
    waypoint1.position.z = 0.5
    waypoint1.orientation.x = 0.0
    waypoint1.orientation.y = 0.0
    waypoint1.orientation.z = 0.0
    waypoint1.orientation.w = 1.0
    waypoints.append(waypoint1)

    # Example waypoint 2
    waypoint2 = geometry_msgs.msg.Pose()
    waypoint2.position.x = 0.6
    waypoint2.position.y = 0.1
    waypoint2.position.z = 0.4
    waypoint2.orientation.x = 0.0
    waypoint2.orientation.y = 0.0
    waypoint2.orientation.z = 0.0
    waypoint2.orientation.w = 1.0
    waypoints.append(waypoint2)

    # Example waypoint 3
    waypoint3 = geometry_msgs.msg.Pose()    
    waypoint3.position.x = 0.4
    waypoint3.position.y = -0.1
    waypoint3.position.z = 0.6
    waypoint3.orientation.x = 0.0
    waypoint3.orientation.y = 0.0
    waypoint3.orientation.z = 0.0
    waypoint3.orientation.w = 1.0
    waypoints.append(waypoint3)

    # Plan and execute the trajectory through the waypoints
    (plan, fraction) = arm.compute_cartesian_path(waypoints, 0.01, 0.0)  # waypoints, eef_step, jump_threshold

    if fraction > 0.9:
        rospy.loginfo("Successfully planned the Cartesian path.")
        arm.execute(plan, wait=True)
    else:
        rospy.logwarn("Planning failed with only {:.2f}% of the path planned.".format(fraction * 100))

    # Shut down the MoveIt commander and exit
    moveit_commander.roscpp_shutdown()
    moveit_commander.os._exit(0)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
