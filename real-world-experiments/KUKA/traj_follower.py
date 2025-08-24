
#!/usr/bin/env python
import rospy
from iiwa_msgs.msg import JointPosition
import json
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
def publisher():
    rospy.init_node('iiwa_trajectory_publisher', anonymous=True)
    pub = rospy.Publisher('/iiwa/JointPosition', JointPosition, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz
    # Define the waypoints in Cartesian space
    joint_traj_vel_file="joint_waypoints_temp_traj_test.json"
    # Load JSON data
    with open(joint_traj_vel_file, 'r') as file:
        joint_traj = json.load(file)
    
    rospy.loginfo("Publishing trajectory...")
    threshold=1e-3
    max_steps=500
    while not rospy.is_shutdown():
        for i,point in enumerate(joint_traj["joint_trajectory_pos"]):
            msg = JointPosition()
            msg.position.a1 = point[0]
            msg.position.a2 = point[1]
            msg.position.a3 = point[2]
            msg.position.a4 = point[3]
            msg.position.a5 = point[4]
            msg.position.a6 = point[5]
            msg.position.a7 = point[6]
            
            rospy.loginfo(f"Publishing Trajectory Point {i+1}: {msg}")
            pub.publish(msg)
            for step in range(max_steps):
                current_positions=get_current_joint_positions()
                # Check if we've reached the target within the threshold
                if all(abs(target - current) < threshold for target, current in zip(point, current_positions)):
                    break
            
            
            # Wait for the next point in the trajectory
            rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass

