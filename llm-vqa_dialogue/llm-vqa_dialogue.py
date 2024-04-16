import openai
import re
import argparse
from airsim_wrapper import *
import os
import json
import time
import sys
import random
import torch
from PIL import Image
import cv2
import io

from lavis.models import load_model_and_preprocess
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode 

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="prompts/example_prompt.txt")
parser.add_argument("--sysprompt", type=str, default="system_prompts/rules_prompt.txt")
args = parser.parse_args()

with open("config.json", "r") as f:
    config = json.load(f)

print("Initializing GPT3.5...")
openai.api_key = config["OPENAI_API_KEY"]

with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

chat_history = [
    {
        "role": "system",
        "content": sysprompt
    }
]


def ask(prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )

# ORIGINAL
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]


print(f"Done.")


print(f"Initializing AirSim...")
aw = AirSimWrapper()
print(f"Done.")

with open(args.prompt, "r") as f:
    prompt = f.read()

##==============================================================================================================================
aw.takeoff()
aw.fly_to([aw.get_drone_position()[0] , aw.get_drone_position()[1], aw.get_drone_position()[2] + 6])
baseline_pos = ([aw.get_drone_position()[0] , aw.get_drone_position()[1], aw.get_drone_position()[2]])
# aw.set_yaw(240)
screenshot = aw.screenshot()
position_list  = []
position_index = 0
conf = 0
explanation_image = []
explanation_questions = []
mode = "AP"

while True:    

    question = "What is in the scene?"
    ## Active perception mode
    while mode == "AP": 
        time.sleep(3)
        screenshot = aw.screenshot()
        answer, caption, caption_confidence, _ = aw.vqa(screenshot, question)
        vqa_answer = f"Answer: {answer}\n Caption: [{caption}]"
        time.sleep(5)
        conv = ask(vqa_answer)
        try:
            start_question = conv.find('"')
            end_question = conv.find('"', start_question + 1)
            if start_question != -1 and end_question != -1:
                question = conv[start_question + 1:end_question]
        except (IndexError, ValueError):
            question = conv

        if ("Move closer" in conv):
            aw.move('closer', 10)
        if ("Move back" in conv):
            aw.move('back', 5)
        if "Move right" in conv:
            aw.move('right', 10)
        if "Move left" in conv:
            aw.move('left', 10)
        if "Save position" in conv:
            pos = aw.get_drone_position()
            position_list.append(pos)
            print("Position saved!")

        screenshot = aw.screenshot()
        print(f'Question: {question}')
        if ("I know enough" in conv or "Danger detected" in conv):
            aw.fly_to([0, 0, 0])
            if len(position_list) == 2:
                midpoint = [(position_list[0][i] + position_list[1][i]) / 2 for i in range(3)]
                position_list.append(midpoint)
            print(position_list)
            mode = "V"
            break
    while mode == "V":
        ##Random noise 
        noise_values = len(position_list)
        noise_mean = 0
        noise_std = 0.1
        for i in range (noise_values):
            noise = random.gauss(noise_mean, noise_std)
            position_noise = [(x + noise, y + noise, z + noise) for x,y,z in position_list]

        ## Validation mode
        finished_ap_prompt = "Entering validation mode,output a detailed description of the scene so far and afterwards a general caption for it using logical sense and real world knowledge. Ignore irelevant information."
        time.sleep(5)
        gpt_answer = ask(finished_ap_prompt)
        print(gpt_answer)
        time.sleep(5)
        enter_validation_prompt = "You are entering validation mode. Choose the objects/information inside your caption that you want to validate"
        validation_answer = ask(enter_validation_prompt)
        print(validation_answer)
        numbers_prompt = "Output only the number of objects you want to validate no other words in the sentence. Only the number, Example: 3"
        time.sleep(5)
        number_of_objects = ask(numbers_prompt)
        number_of_objects = int(number_of_objects)
        print(number_of_objects)

        time.sleep(5)
        while position_index < len(position_noise):
            position = position_noise[position_index]
            print(f"We are validating position: {position_index + 1}")
            time.sleep(5)
            print(f"Flying to: {position}")
            aw.fly_to(position)
            time.sleep(5)
            screenshot = aw.screenshot()
            if screenshot not in explanation_image:
                    explanation_image.append(screenshot)
            object_number = 0
            for _ in range(number_of_objects):

                object_number += 1
                validate_prompt = f"You are in position: {position_index + 1}. From the list of objects you want to validate, ask a yes or no question for object number {object_number}. An object is considered validated when it gets the most Yes answers and not validated when it gets the most No answers. Validation is considered completed after you have visited all saved positions so you must keep asking the same questions in all positions."
                print(validate_prompt)
                time.sleep(5)
                validation_question = ask(validate_prompt)

                print(validation_question)
                if validation_question not in explanation_questions:
                    explanation_questions.append(validation_question)
                answer, _, confidence, _ = aw.vqa(screenshot, validation_question)
                validation_answer = f"Answer:{answer}"
                time.sleep(5)
                validate_command = ask(validation_answer)
                time.sleep(5)

            position_index += 1
            print(f"Moving to next position ({position_index + 1})")
        aw.fly_to([0,0,0])
        final_prompt = "You visited all positions provide the previous caption and description and then update them with the validated information, giving a detailed updated description and caption (in one sentence) for the scene. Make sure the captions and descriptions are logical and coherent, filter the parts that aren't logical and remove the parts that werent validated.Afterwards output safety measures for the scene to someone who hasn't seen it as well as which parts of the scene seem safe to be aproached. Keep in mind that the description needs to be long and informative and the caption compact and informative. If you have validated multiple vehicles (car, truck, boat etc..) generalise all of them with the term vehicles. If you validate any possible smoke, or fire or stuck vehicle associated wiith the scene, consider there has been a crash and its a hazard and update the description and caption."
        final_answer = ask(final_prompt)
        print(final_answer)
        mode = "X"
        break
    ## Explanation mode
    while mode == "X":
        print(len(explanation_image))
        print(len(explanation_questions))
        for image in explanation_image:
            for question in explanation_questions:
                answer,_,_,gradcam= aw.vqa(image, question)
                avg_gradcam = aw.attention_map(image, gradcam)
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                axs[0].imshow(image)
                axs[0].set_yticks([])
                axs[0].set_xticks([])
                axs[0].set_title('Original Image', fontsize=10)

                axs[1].imshow(avg_gradcam)
                axs[1].set_yticks([])
                axs[1].set_xticks([])
                axs[1].set_title('GradCam Image', fontsize=10)
                fig.suptitle(f'Question: {question}', fontsize=12)
                plt.tight_layout(rect=[0, 0, 1, 0.10])

                plt.show()
        mode = "END"
        break
    if mode == "END":
        break


## Gradcam visualization code snippet. Replace the question and the position list below to leverage attention maps on specific positions.
#========================================================================================================================================
# for position in position_list:
#     aw.fly_to(position)
#     image = aw.screenshot()
#     question = "Is there a fire burning from flames in the scene?"
#     # conf = aw.val_test(image,question)
#     # conf = "{:.4f}".format(conf.item())
#     # print(conf)
#     answer,_,_,gradcam= aw.vqa(image, question)
#     avg_gradcam = aw.attention_map(image, gradcam)
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#     axs[0].imshow(image)
#     axs[0].set_yticks([])
#     axs[0].set_xticks([])
#     axs[0].set_title('Original Image', fontsize=10)

#     axs[1].imshow(avg_gradcam)
#     axs[1].set_yticks([])
#     axs[1].set_xticks([])
#     axs[1].set_title('GradCam Image', fontsize=10)
#     fig.suptitle(f'Question: {question}', fontsize=12)
#     plt.tight_layout(rect=[0, 0, 1, 0.10])

#     plt.show()
#========================================================================================================================================

