import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d
import time
import math
import json
from PIL import Image
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from seg import LangSAMPredictor
import urdf_models.models_data as md


class SceneReconstructor:
    def __init__(self, trajectory_file, width=640, height=480, fov=90):
        self.trajectory_file = trajectory_file
        self.width = width
        self.height = height
        self.fov = fov
        self.camera_pos = [0.6, 0.0, 1.0]
        self.target_pos = [0, 0, 0]
        self.up = [0, 0, 1]
        self.models = md.model_lib()
        self.predictor = LangSAMPredictor()
        self.object_pose = []

    def setup_pybullet(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
    
    def load_objects(self):
        with open(self.trajectory_file, 'r') as f:
            data = json.load(f)
        objects = data["zero_shot_trajectory"]["objects"]
        
        obj_name = ["book", "mug", "yellow bowl", "box", "banana", "glue", "black marker",
                    "green bowl", "apple", "hammer", "square plate", "soap", "orange cup", "potato chip",
                    "orange", "scissors", "red marker", "remote controller", "spoon", "lemon"]
        obj_sim_name = ["book_1", "mug", "yellow_bowl", "clear_box_2", "plastic_banana", "glue_2",
                        "black_marker", "green_bowl", "plastic_apple", "two_color_hammer", "square_plate_4",
                        "soap", "orange_cup", "potato_chip_3", "plastic_orange", "scissors", "red_marker",
                        "remote_controller_2", "spoon", "plastic_lemon"]

        prompt = ""
        for obj in objects:
            name = obj["name"].split(" ")[-1] if len(obj["name"].split(" ")) > 1 else obj["name"]
            prompt += name + ". "
        poses=[]
        for obj in objects:
            obj["name"] = obj_sim_name[obj_name.index(obj["name"])]
            obj_id = p.loadURDF(self.models[obj["name"]], basePosition=[obj["x"], obj["y"], obj["z"]])
            poses.append([obj["x"], obj["y"], obj["z"]])
            p.changeDynamics(obj_id, -1, lateralFriction=1.0, restitution=0.0)

        # Let physics settle
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1. / 240.)

        return prompt,poses

    def capture_scene(self):
        self.view_matrix = p.computeViewMatrix(self.camera_pos, self.target_pos, self.up)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=self.fov, aspect=self.width / self.height, nearVal=0.01, farVal=10)

        img = p.getCameraImage(width=self.width, height=self.height,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.proj_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3].astype(np.uint8)
        depth = np.reshape(img[3], (self.height, self.width))
        return rgb, depth

    def process_segmentation(self, rgb, prompt):
        result = self.predictor.predict(rgb, prompt)
        masks = [mask.astype(np.uint8) for mask in result["masks"]]
        boxes = result["boxes"]
        return masks, boxes

    def get_point_cloud(self, depth_buffer, seg_mask, box):
        near, far = 0.01, 10
        depth = far * near / (far - (far - near) * depth_buffer)

        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        fx = self.height / (2 * np.tan(np.radians(self.fov) / 2))
        fy, cx, cy = fx, self.width / 2, self.height / 2

        z = depth
        x3 = (x - cx) * z / fx
        y3 = (y - cy) * z / fy
        pc = np.stack((x3, -y3, -z), axis=-1)

        if box is not None:
            x_min, y_min, x_max, y_max = box
            mask = np.zeros_like(seg_mask, dtype=bool)
            mask[int(y_min):int(y_max), int(x_min):int(x_max)] = True
            combined_mask = (seg_mask > 0) & mask
        else:
            combined_mask = seg_mask > 0

        pc_obj = pc[combined_mask]
        pc_homog = np.concatenate([pc_obj, np.ones((pc_obj.shape[0], 1))], axis=1)
        view_matrix_np = np.array(self.view_matrix).reshape((4, 4), order='F')
        view_matrix_inv = np.linalg.inv(view_matrix_np)
        world_points = (view_matrix_inv @ pc_homog.T).T[:, :3]
        return world_points

    def compute_pose(self, points):
        centroid = np.mean(points, axis=0)
        pca = PCA(n_components=3)
        pca.fit(points - centroid)
        orientation = pca.components_.T
        if np.linalg.det(orientation) < 0:
            orientation[:, -1] *= -1
        dims = np.max((points - centroid) @ orientation, axis=0) - np.min((points - centroid) @ orientation, axis=0)
        quat = R.from_matrix(orientation).as_quat()
        self.draw_axes(centroid, orientation)
        return centroid, quat, dims

    def draw_axes(self, center, orientation, scale=0.2):
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(3):
            p.addUserDebugLine(center, center + orientation[:, i] * scale, colors[i], 3.0)

    def draw_point_cloud(self, points):
        for i in range(0, len(points), 20):
            pt = points[i]
            p.addUserDebugLine(pt, pt + np.array([0, 0, 0.01]), [0, 0, 1], 1)

    def draw_bounding_box(self, position, quat, dims):
        dx, dy, dz = dims / 2.0
        corners = np.array([[dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
                            [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]])
        rot = R.from_quat(quat).as_matrix()
        world_corners = (rot @ corners.T).T + position
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            p.addUserDebugLine(world_corners[i], world_corners[j], [1, 0, 0], 2, lifeTime=5)

    def run(self):
        self.setup_pybullet()
        prompt,original_positions = self.load_objects()
        rgb, depth = self.capture_scene()
        masks, boxes = self.process_segmentation(rgb, prompt)

        pcs_world = [self.get_point_cloud(depth, mask, box) for mask, box in zip(masks, boxes)]
        for pc in pcs_world:
            self.draw_point_cloud(pc)
            position, quat, dims = self.compute_pose(pc)
            self.object_pose.append((position, quat, dims))
        for i,pose in enumerate(self.object_pose):
            print(f"Object {i}\n")
            print(f"Original Position :{original_positions[i]}\n")
            print(f"Estimated Position :{pose[0]}\n")
            print(f"Error : {np.array(original_positions[i])-(pose[0])}")
        try:
            while True:
                for pose in self.object_pose:
                    # print(pose)
                    # print(f"Error : {np.array(original_positions)-(pose[0])}")
                    self.draw_bounding_box(*pose)
                p.stepSimulation()
                time.sleep(1. / 240.)
        except KeyboardInterrupt:
            p.disconnect()
            print("Simulation stopped.")


if __name__ == "__main__":
    file_path = r"trajectory_0.json"
    reconstructor = SceneReconstructor(file_path)
    reconstructor.run()
