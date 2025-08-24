#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import json
def publish_trajectory_lines():
    rospy.init_node('trajectory_visualizer')

    # Publisher for trajectory markers
    marker_pub = rospy.Publisher('/trajectory_marker', Marker, queue_size=10)

    rospy.sleep(1)  # Wait for the publisher to be ready

    # Create a Marker message
    marker = Marker()
    marker.header.frame_id = "world"  # Use a fixed frame (e.g., world or base_link)
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.01  # Line width
    marker.color.r = 1.0  # Red
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0  # Alpha (transparency)
    # Define the waypoints in Cartesian space
    json_file="temp_traj.json"
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    data=data[0]
    # Add trajectory points
    trajectory_points = [
        Point(-(traj["x"]+0.5), traj["y"], traj["z"]-0.6) for traj in data['trajectory']
    ]
    marker.points = trajectory_points

    # Publish the marker
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        marker.header.stamp = rospy.Time.now()
        marker_pub.publish(marker)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_trajectory_lines()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python

import rospy
from geometry_msgs.msg import TwistStamped

def publish_twist_stamped():
    # Initialize the ROS node
    rospy.init_node('twist_stamped_publisher', anonymous=True)

    # Create a publisher for TwistStamped messages
    publisher = rospy.Publisher('/cmd_vel_stamped', TwistStamped, queue_size=10)

    # Set the publishing rate
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Create and populate a TwistStamped message
        twist_stamped = TwistStamped()

        # Set header information
        twist_stamped.header.stamp = rospy.Time.now()  # Current timestamp
        twist_stamped.header.frame_id = 'iiwa_link_7'    # Frame ID

        # Set linear velocities (x, y, z)
        twist_stamped.twist.linear.x = 0.5  # Example: Move forward at 0.5 m/s
        twist_stamped.twist.linear.y = 0.0
        twist_stamped.twist.linear.z = 0.0

        # Set angular velocities (x, y, z)
        twist_stamped.twist.angular.x = 0.0
        twist_stamped.twist.angular.y = 0.0
        twist_stamped.twist.angular.z = 0.1  # Example: Rotate around z-axis at 0.1 rad/s

        # Publish the message
        publisher.publish(twist_stamped)

        rospy.loginfo(f"Published TwistStamped: {twist_stamped}")

        # Sleep to maintain the desired publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_twist_stamped()
    except rospy.ROSInterruptException:
        rospy.loginfo("TwistStamped publisher node terminated.")

