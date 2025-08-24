from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lang_sam import LangSAM
import torch
import os
import json
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

class LangSAMPredictor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = LangSAM()
        self.device = device

    def predict(self, image_pil, text_prompt):
        """
        Perform prediction with LangSAM on a given image and text prompt.
        Returns the raw result dictionary.
        """
        image_np = np.array(image_pil).copy()  # Makes it writable
        image_pil = Image.fromarray(image_np)
        results = self.model.predict([image_pil], [text_prompt])
        return results[0]  # Only one image and prompt used

    def save_masks(self, masks, output_prefix="mask"):
        """
        Save each mask as a binary image.
        """
        # for i, mask in enumerate(masks):
        #     img = Image.fromarray((mask * 255).astype(np.uint8))
        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))
        img=Image.fromarray(combined_mask * 255)
        img.save(f"{output_prefix}.png")

    def visualize(self, image, masks, boxes=None, alpha=0.4):
        """
        Visualize masks over the original image (optional: with boxes).
        """
        # image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        for i, mask in enumerate(masks):
            plt.imshow(np.ma.masked_where(mask == 0, mask), cmap='jet', alpha=alpha)
            if boxes is not None:
                box = boxes[i]
                x0, y0, x1, y1 = box
                plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                  edgecolor='lime', facecolor='none', lw=2))
        plt.axis("off")
        plt.show()
if __name__ == "__main__":
    predictor = LangSAMPredictor()
    image_path = "pybullet_image.png"
    #Load Objects
    obj_name=["book","mug","yellow bowl","box","banana","glue","black marker","green bowl","apple","hammer","square plate","soap","orange cup","potato chip","orange","scissors","red marker","remote controller","spoon","lemon"]
    obj_sim_name=["book_1","mug","yellow_bowl","clear_box_2","plastic_banana","glue_2","black_marker","green_bowl","plastic_apple","two_color_hammer","square_plate_4","soap","orange_cup","potato_chip_3","plastic_orange","scissors","red_marker","remote_controller_2","spoon","plastic_lemon"]
    file_path="trajectory.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    objects=data["zero_shot_trajectory"]["objects"]
    # for obj in objects:
    #         obj["name"]=obj_sim_name[obj_name.index(obj["name"])]

    prompt=""
    for obj in objects:
        name=obj["name"].split(" ")[-1] if len(obj["name"].split(" "))>1 else obj["name"]
        prompt+=name + ". "
    print(prompt)
    image_pil = Image.open(image_path).convert("RGB")
    result = predictor.predict(image_pil, prompt)

    masks = result["masks"]         # shape: (N, H, W)
    boxes = result["boxes"]         # optional: (N, 4)
    scores = result["scores"]       # confidence scores
    labels = result["labels"]       # class labels
    print(np.max(masks[0]))
    print(np.min(masks[0]))

    # Save masks
    predictor.save_masks(masks, "wheel_mask")

    # Visualize masks + bounding boxes
    predictor.visualize(image_pil, masks, boxes)
