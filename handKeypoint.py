import os
from glob import glob

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

base_dir = "./captures"
folders = os.listdir(base_dir)

for folder in folders:
    IMAGE_FILES = glob(f"./captures/{folder}/*.*")
    print(IMAGE_FILES)
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        for idx, file in enumerate(IMAGE_FILES):
            i = idx + 1
            print(i, file)
            ff = np.fromfile(file, np.uint8)
            image = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
            # print(image)
            # print(file)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            xyz = []
            if results.multi_hand_landmarks:
                for landmark in results.multi_hand_landmarks[0].landmark:
                    xyz.append([landmark.x, landmark.y, landmark.z])

                # Print handedness and draw hand landmarks on the image.
                if not os.path.exists(os.path.join("./keypoints", folder)):
                    os.mkdir(os.path.join("./keypoints", folder))
                np.save(f"./keypoints/{folder}/{i}", xyz)
