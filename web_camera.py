from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import glob
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

if __name__ == "__main__":
    model = load_model("./models/ksl-keypoints_epochs-500_batch-10.hdf5")
    categories = [
        "ㄱ",
        "ㄴ",
        "ㄷ",
        "ㄹ",
        "ㅁ",
        "ㅂ",
        "ㅅ",
        "ㅇ",
        "ㅈ",
        "ㅊ",
        "ㅋ",
        "ㅌ",
        "ㅍ",
        "ㅎ",
        "ㅏ",
        "ㅐ",
        "ㅑ",
        "ㅓ",
        "ㅔ",
        "ㅕ",
        "ㅗ",
        "ㅛ",
        "ㅜ",
        "ㅠ",
        "ㅡ",
        "ㅣ",
    ]
    # fig, ax = plt.subplots(2, 5)
    # files = glob.glob("./Sign-Language/Dataset/Examples/*.JPG")
    # for i, f in enumerate(files):
    #     ax = fig.add_subplot(2, 5, i + 1)
    #     pic = Image.open(f)
    #     ax.imshow(pic)
    #     ax.set_title(str(i))
    # plt.show()
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            roi = image[100:300, 200:400].copy()
            image = cv2.rectangle(image, (200, 100), (400, 300), (255, 0, 0), 5, cv2.LINE_8)
            img_to_test = image[100:300, 200:400].copy()
            img_to_test = cv2.resize(img_to_test, (100, 100))

            results = hands.process(img_to_test)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            black = np.zeros((200, 200, 3), np.uint8)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(black, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            xyz = []
            if results.multi_hand_landmarks:
                for landmark in results.multi_hand_landmarks[0].landmark:
                    xyz.append([landmark.x, landmark.y, landmark.z])
                xyz = np.expand_dims(xyz, axis=0)
                xyz = np.asarray(xyz)
                xyz = xyz / 255.0
                print(xyz)
                prob = model.predict_proba(xyz)
                print("Predicted:")
                print(prob)
                print(np.max(prob))
                classes = np.argmax(model.predict(xyz), axis=-1)
                print(classes)
                print(categories[classes[0]])
                temp = Image.fromarray(image)
                draw = ImageDraw.Draw(temp)
                font = ImageFont.truetype("fonts/gulim.ttc", 20)
                draw.text(
                    (image.shape[0] / 7, image.shape[1] / 5),
                    categories[classes[0]] + f"    {np.max(prob)*100:.2f}",
                    font=font,
                    fill=(255, 255, 255),
                )
                image = np.array(temp)
                # cv2.putText(
                #     image,
                #     categories[classes[0]],
                #     (110, 110),
                #     cv2.FONT_HERSHEY_COMPLEX,
                #     2,
                #     (255, 255, 255),
                # )
            cv2.imshow("Hands", image)
            cv2.imshow("keypoints", black)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    # count = 0
    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.flip(frame, 1)
    #     if not ret:
    #         print("error")
    #         break
    #     roi = frame[100:300, 200:400].copy()
    #     frame = cv2.rectangle(frame, (200, 100), (400, 300), (255, 0, 0), 5, cv2.LINE_8)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27:
    #         break

    #     count += 1
    #     if count == 5:
    #         img = frame[100:300, 200:400].copy()
    #         img = cv2.resize(img, (100, 100))
    #         x = image.img_to_array(img)
    #         x = np.expand_dims(x, axis=0)
    #         x = np.asarray(x)
    #         x = x / 255.0

    #         prob = model.predict_proba(x)
    #         print("Predicted:")
    #         print(prob)
    #         print(np.max(prob))
    #         classes = np.argmax(model.predict(x), axis=-1)
    #         print(classes)
    #         print(categories[classes[0]])
    #         cv2.putText(
    #             frame,
    #             categories[classes[0]],
    #             (20, 20),
    #             cv2.FONT_HERSHEY_COMPLEX,
    #             2,
    #             (255, 255, 255),
    #         )
    #         count = 0
    #     cv2.imshow("tensorflow-pi inspector", frame)
    #     cv2.imshow("ROI", roi)

    cap.release()
    cv2.destroyAllWindows()
