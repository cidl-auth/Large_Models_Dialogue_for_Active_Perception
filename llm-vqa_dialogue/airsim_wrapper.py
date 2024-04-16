import airsim
import math
import numpy as np
from PIL import Image
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from lavis.common.gradcam import getAttMap
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import io
import torch
import openai


class AirSimWrapper:

    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_vqa, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="pnp_vqa", model_type="large", is_eval=True, device=self.device
            )


    def takeoff(self):
        self.client.takeoffAsync().join()

    def land(self):
        self.client.landAsync().join()

    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def fly_to(self, point):
        if point[2] > 0:
            self.client.moveToPositionAsync(point[0], point[1], -point[2], 5).join()
        else:
            self.client.moveToPositionAsync(point[0], point[1], point[2], 5).join()


    def set_yaw(self, yaw):
        self.client.rotateToYawAsync(yaw, 5).join()

    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw


    def move(self, direction, distance):
        new_position = self.get_drone_position()
        if direction == 'closer':
            new_position[0] += distance
        elif direction == 'back':
            new_position[0] -= distance
        elif direction == 'right':
            new_position[1] += distance
        elif direction == 'left':
            new_position[1] -= distance
        self.fly_to(new_position)


    def screenshot(self):
        client = airsim.MultirotorClient()
        image = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])

        # Extract the image data from the list
        image_data = image[0].image_data_uint8

        image_bytes = bytes(image_data)
        test_image = Image.open(io.BytesIO(image_bytes))
        test_image = test_image.convert("RGB")

        return test_image


    def vqa(self, image, question):

        images = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        question = self.txt_processors["eval"](question)

        samples = {"image": images, "text_input": [question]}
        

        answer_vqa, caption_vqa, gradcam, cap_conf = self.model_vqa.predict_answers(
                    samples=samples,
                    inference_method="generate",
                    num_captions=1,
                    num_patches=20,
                )     

        cap_conf = "{:.4f}".format(cap_conf.item())
        cap_conf = float(cap_conf)

        grad = samples['gradcams']
        answer = f"Answer: {answer_vqa}"
        caption = f"Caption: {caption_vqa} {cap_conf}"

        print(answer)
        print(caption)
        return answer, caption, cap_conf, grad

    def val_test(self, image, caption):

        images = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        caption = self.txt_processors["eval"](caption)

        image_caption_pair = {"image": images, "text_input": [caption]}
        prob = self.model_vqa.validate_confidence(samples=image_caption_pair)
        prob = "{:.4f}".format(prob.item())

        return prob
    
    def attention_map(self, image, gradcam):

        dst_w = 720
        w, h = image.size
        scaling_factor = dst_w / w

        resized_img = image.resize((int(w * scaling_factor), int(h * scaling_factor)))
        norm_img = np.float32(resized_img) / 255
        gradcam_score = gradcam.reshape(24,24)
        gradcam_np = gradcam_score.cpu().numpy().astype(np.float32)
        avg_gradcam = getAttMap(norm_img, gradcam_np, blur=True)

        return avg_gradcam

        

