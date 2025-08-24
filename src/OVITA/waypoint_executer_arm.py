import pybullet as p
import pybullet_data
import numpy as np
import time
import json
from scipy.spatial.transform import Rotation as R
import pybullet_URDF_models.urdf_models.models_data as md
from PIL import Image
class KUKASIM:
    def __init__(self,data):
        # load urdf data
        obj_name=["book","mug","yellow bowl","box","banana","glue","black marker","green bowl","apple","hammer","square plate","soap","orange cup","potato chip","orange","scissors","red marker","remote controller","spoon","lemon"]
        obj_sim_name=["book_1","mug","yellow_bowl","clear_box_2","plastic_banana","glue_2","black_marker","green_bowl","plastic_apple","two_color_hammer","square_plate_4","soap","orange_cup","potato_chip_3","plastic_orange","scissors","red_marker","remote_controller_2","spoon","plastic_lemon"]
        self.data=data
        objects=data["zero_shot_trajectory"]["objects"]
        for obj in objects:
            obj["name"]=obj_sim_name[obj_name.index(obj["name"])]
        models = md.model_lib()
        # Set up the camera parameters   
        width, height = 1920, 1080
        fov = 100  # Field of view
        aspect = width / height  # Aspect ratio
        near_plane = 0.01  # Near clipping plane
        far_plane = 100  # Far clipping plane

        # Camera position and orientation
        camera_position = [1, 1, 1]  # Camera position in world coordinates
        target_position = [0, 0, 0]  # Target position in world coordinates
        up_vector = [0, 0, 1]  # Up vector of the camera
        # Set the camera parameters
        distance = 2.8  # Camera distance from the target
        yaw = 51.6  # Horizontal rotation (degrees)
        pitch = -51  # Vertical rotation (degrees)
        target_position = [0, 0, 0.5]  # Target position

        # Compute the view and projection matrices
        view_matrix = p.computeViewMatrix(cameraEyePosition=camera_position, cameraTargetPosition=target_position, cameraUpVector=up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near_plane, farVal=far_plane)

        # Desired orientation as roll-pitch-yaw (Euler angles)
        desired_rpy = [0, 0, 0]  # Roll, Pitch, Yaw in radians

        # Convert Euler angles to quaternion
        desired_orientation = R.from_euler('xyz', desired_rpy).as_quat()  # [x, y, z, w]

        def draw_trajectory(trajectory_points,color):
            # Visualize trajectory as a continuous line
            # for point in trajectory_points:
            #     p.addUserDebugLine(
            #         lineFromXYZ=[point["x"],point["y"],point["z"]],
            #         lineToXYZ=[point["x"],point["y"],point["z"]],  # Point to itself to render as a dot
            #         lineColorRGB=color,
            #         lineWidth=10  # Line width for visibility
            # )

            sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[color[0], color[1], color[2], 1])

            for i,point in enumerate(trajectory_points):
                if i%4==0:
                    p.createMultiBody(baseVisualShapeIndex=sphere_visual, basePosition=[point["x"],point["y"],point["z"]])
            for i in range(len(trajectory_points) - 1):
                start_point = [trajectory_points[i]["x"],trajectory_points[i]["y"],trajectory_points[i]["z"]]
                end_point = [trajectory_points[i+1]["x"],trajectory_points[i+1]["y"],trajectory_points[i+1]["z"]]
                p.addUserDebugLine(start_point, end_point, lineColorRGB=color, lineWidth=5)

        # Initialize PyBullet simulation
        p.connect(p.GUI)  # Use p.DIRECT for headless mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For default URDFs
        # p.setPhysicsEngineParameter(numSolverIterations=5000)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable the GUI overlay
        # Apply the camera settings
        p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target_position)
        # Load a plane and a URDF model for context (optional)
        p.loadURDF("plane.urdf")
        # Load the table URDF
        table_position = [0, 0, 0]  # Table's position (x, y, z)
        table_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation

        # Replace 'table.urdf' with the path to your table URDF file
        table_id = p.loadURDF("table/table.urdf", basePosition=table_position, baseOrientation=table_orientation)
        # Load KUKA iiwa 7 DOF robot
        robot_urdf = "kuka_iiwa/model.urdf"  # Update path if needed
        self.robot_id = p.loadURDF(robot_urdf, basePosition=[-0.5, 0, 0.6], useFixedBase=True)  

        # Find the end-effector link index
        self.kuka_end_effector_index = 6  # Adjust based on your KUKA URDF

        # Get the end effector's position and orientation in the world frame
        link_state = p.getLinkState(self.robot_id, self.kuka_end_effector_index)
        end_effector_position = link_state[0]  # [x, y, z] in world frame
        end_effector_orientation = link_state[1]  # Quaternion
        # # Load the left and right gripper URDFs
        # # Replace these with paths to your actual gripper URDF files
        # left_gripper_id = p.loadURDF("gripper/wsg50_one_motor_gripper_left_finger.urdf", basePosition=end_effector_position,baseOrientation=end_effector_orientation)
        # right_gripper_id = p.loadURDF("gripper/wsg50_one_motor_gripper_right_finger.urdf", basePosition=end_effector_position,baseOrientation=end_effector_orientation)



        # # Attach the left gripper
        # p.createConstraint(
        #     parentBodyUniqueId=robot_id,
        #     parentLinkIndex=kuka_end_effector_index,
        #     childBodyUniqueId=left_gripper_id,
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=[0, 0.05, 0],  # Offset the left gripper slightly
        #     childFramePosition=[0, 0, 0]
        # )

        # # Attach the right gripper
        # p.createConstraint(
        #     parentBodyUniqueId=robot_id,
        #     parentLinkIndex=kuka_end_effector_index,
        #     childBodyUniqueId=right_gripper_id,
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=[0, -0.05, 0],  # Offset the right gripper slightly
        #     childFramePosition=[0, 0, 0]  wsg50_with_r2d2_gripper  wsg50_one_motor_gripper_new_free_base
        # )

        self.kuka_gripper_id = p.loadSDF("gripper/wsg50_one_motor_gripper_new_free_base.sdf")[0]
        # Debug: Print visual shape information
        visual_shapes = p.getVisualShapeData(self.kuka_gripper_id)

        # Change the color of all visual shapes in the gripper
        for visual_shape in visual_shapes:
            body_unique_id = visual_shape[0]  # Unique ID of the body
            link_index = visual_shape[1]  # Link index
            p.changeVisualShape(
                objectUniqueId=body_unique_id,
                linkIndex=link_index,
                rgbaColor=[0, 0, 0, 1]  # Green color in RGBA
            )
        p.resetBasePositionAndOrientation(self.kuka_gripper_id, end_effector_position,p.getQuaternionFromEuler([0, 0, 1.57]))
        # jointPositions = [0.000000, -0.011130, -0.206421, 0.205143, -0.009999, 0.000000, -0.010055, 0.000000]
        # for jointIndex in range(p.getNumJoints(kuka_gripper_id)):
        print(p.getNumJoints(self.kuka_gripper_id))
        #     p.resetJointState(kuka_gripper_id, jointIndex, jointPositions[jointIndex])
        #     p.setJointMotorControl2(kuka_gripper_id, jointIndex, p.POSITION_CONTROL, jointPositions[jointIndex], 0)
        # attach gripper to kuka arm
        kuka_cid = p.createConstraint(self.robot_id, 6, self.kuka_gripper_id, 0, p.JOINT_FIXED, end_effector_position, [0, 0, 0.02], [0, 0, 0])
        kuka_cid2 = p.createConstraint(self.kuka_gripper_id, 4, self.kuka_gripper_id, 6, jointType=p.JOINT_GEAR, jointAxis=[1,1,1], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        p.changeConstraint(kuka_cid2, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)

        p.setJointMotorControl2(self.kuka_gripper_id, 2, p.POSITION_CONTROL, targetPosition=1.57, force=100)
        p.setJointMotorControl2(self.kuka_gripper_id, 4, p.POSITION_CONTROL, targetPosition=0.02, force=100)
        p.setJointMotorControl2(self.kuka_gripper_id, 6, p.POSITION_CONTROL, targetPosition=0.02, force=100)

        obj_ids=[]
        for obj in objects:
            obj_id = p.loadURDF(models[obj["name"]], basePosition=[obj["x"],obj["y"],obj["z"]])
            # Adjust friction
            p.changeDynamics(obj_id, -1, lateralFriction=5.0)  # High friction prevents rolling
            obj_ids.append(obj_id)
        # Define the number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        end_effector_index=6
        # Get the joint limits (for safety)
        joint_limits = []
        for joint_idx in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            lower_limit, upper_limit = joint_info[8], joint_info[9]
            joint_limits.append((lower_limit, upper_limit))

        # Load JSON data
        # with open(json_file, 'r') as file:
        #     data = json.load(file)
        # joint_names = [p.getJointInfo(robot_id, i)[1].decode('UTF-8') for i in range(p.getNumJoints(robot_id))]
        # joint_indices = [joint_from_name(robot_id, name) for name in joint_names if "iiwa_joint" in name]  # Adjust joint naming for KUKA

        arm_joint_indices = [0,1,2,3,4,5,6]
        # Extract waypoints and objects
        # waypoints = data["modified_trajectories"][-1]
        self.original_waypoint=data["zero_shot_trajectory"]["trajectory"]
        if 'final_trajectory' in data.keys() and data["final_trajectory"]!={}:
            self.waypoints=data["final_trajectory"]["modified trajectory"]
        else:
            self.waypoints=data["zero_shot_trajectory"]["modified trajectory"]
        print(len(self.waypoints))
        draw_trajectory(self.original_waypoint,[0,0,1])
        draw_trajectory(self.waypoints,[1,0,0])
        self.joint_waypoints=[]
        for wp in self.waypoints:
            full_joint_positions = p.calculateInverseKinematics(self.robot_id, end_effector_index, [wp["x"]-0.1,wp["y"],wp["z"]])  #desired_orientation
            target_joint_positions = [full_joint_positions[joint] for joint in arm_joint_indices]
            self.joint_waypoints.append(target_joint_positions)

    # Define a function to control the robot joints to desired positions
    def move_to_position(self,target_positions,torque, max_steps=500, threshold=1e-3):
        for step in range(max_steps):
            # Get current joint positions
            current_positions = [p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)]
            
            # Check if we've reached the target within the threshold
            if all(abs(target - current) < threshold for target, current in zip(target_positions, current_positions)):
                break
            
            # Control each joint
            for joint_idx in range(self.num_joints):
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_positions[joint_idx],
                    force=torque  # Max force to apply
                )
            
            kuka_end_effector_index = 6  # Adjust based on your KUKA URDF

            # Get the end effector's position and orientation in the world frame
            link_state = p.getLinkState(self.robot_id, kuka_end_effector_index)
            end_effector_position = link_state[0]  # [x, y, z] in world frame
            end_effector_orientation = link_state[1]  # Quaternion
            p.resetBasePositionAndOrientation(self.kuka_gripper_id, end_effector_position,end_effector_orientation)

            # Step simulation
            p.stepSimulation()
            # Slow down the simulation to observe the motion
            time.sleep(0.01)

        # return joint_pos,joint_vel

    # joint_pos_list=[]
    # joint_vel_list=[]
    # Execute the waypoints
    def _exec(self):
        for i,waypoint,owp,wp in zip(range(len(self.joint_waypoints)),self.joint_waypoints,self.original_waypoint,self.waypoints):
            # Extract linear and angular velocities
            # Assuming kuka_id is the robot's unique ID and link 6 is the end-effector
            # ee_state = p.getLinkState(robot_id, linkIndex=6, computeLinkVelocity=True)
            # linear_velocity = np.array(ee_state[6])  # Linear velocity as [vx, vy, vz]
            # angular_velocity = np.array(ee_state[7])  # Angular velocity as [wx, wy, wz]
            # # Record joint states
            # joint_pos=[p.getJointState(robot_id, i)[0] for i in arm_joint_indices]
            # joint_vel=[p.getJointState(robot_id, i)[1] for i in arm_joint_indices]
            # Capture the image
            # if i==15:
            #     image = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            #     # Extract the RGB data
            #     rgb_array = np.array(image[2], dtype=np.uint8)  # Image data is in the 3rd element of the result
            #     rgb_image = rgb_array.reshape((height, width, 4))  # Reshape to (H, W, 4)

            #     # Remove the alpha channel (if needed)
            #     rgb_image = rgb_image[:, :, :3]

            #     # Save the image using Pillow
            #     img = Image.fromarray(rgb_image)
            #     img.save("camera_image.png")

            #     print("Image saved as camera_image.png")

            if owp["velocity"]!=wp["velocity"]:
                torque=600
            else:
                torque=500
            self.move_to_position(waypoint,torque=torque)

        # Disconnect from PyBullet
        p.disconnect()

if __name__ == '__main__':
    # Load JSON data
    file_path=r"trajectory_0.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    kuka_sim=KUKASIM(data)
    kuka_sim._exec()