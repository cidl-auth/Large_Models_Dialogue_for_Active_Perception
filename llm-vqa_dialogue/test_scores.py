import openai
import re
import argparse
from airsim_wrapper import *
import math
import numpy as np
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
import lavis

aw = AirSimWrapper()
aw.takeoff()

## Add the position list outputted from the llm-vqa_dialogue.py script.
position_list = []

## Add the first caption outputted from the vqa model and the last caption of the the llm-vqa_dialogue.py script and replace the variables below.
baseline = "a picture looking into the snow - capped mountains with a lake below a rocky surface"
proposed ="A tranquil winter scene with snow-capped mountains and a lake below, devoid  of any man-made structure or train activity."
index = 1
for position in position_list:
    aw.fly_to(position)
    screenshot = aw.screenshot()
    score1 = aw.val_test(screenshot, baseline)
    score2 = aw.val_test(screenshot, proposed)
    print(f"Baseline score {index}: {score1}")
    print(f"Proposed score {index}: {score2}")
    index += 1