import os
import cv2
import uuid
import time
import subprocess
import mediapipe as mp

# labels for images to be collected with number_imgs being the amount of images per label

labels = ['no', 'number0', 'number1', 'okay', 'peace', 'thumbsup', 'thumbsdown']
number_imgs = 6

# Defining of paths

USERS_PATH = "C:/Users/User/CleanProject/pythonProject1"
IMAGES_PATH = os.path.join('Dataset', 'images', 'collectedimages')
LABELIMG_PATH = os.path.join('Dataset', 'labelimg')

# New IMAGE_PATH Directory is made for all labels

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)

# Images are taken with system camera for each label

for label in labels:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = hands.process(frame_rgb)
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         # Draw landmarks on the image
    #         mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks,
    #                                                   mp_hands.HAND_CONNECTIONS)
    print('Collecting images for {}'.format(label))
    time.sleep(6)
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks,
                                                          mp_hands.HAND_CONNECTIONS)
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(4)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

# New Directory is made for the labeling application and the application cloned into the path

if not os.path.exists(LABELIMG_PATH):
    os.makedirs(LABELIMG_PATH)
    subprocess.run(["git", "clone", "https://github.com/tzutalin/labelImg", LABELIMG_PATH])



# Line for Activation of the labeling application and installation of dependencies

if os.name == 'posix':
    subprocess.run(["make", "qt5py3"])
elif os.name == 'nt':
    subprocess.run(["cmd", "/c", f"cd {USERS_PATH} && cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc"])

os.chdir(LABELIMG_PATH)
subprocess.run(["python", "labelImg.py"])

