# Download the dataset

import kagglehub

path = kagglehub.dataset_download(
    "muki2003/yolo-drone-detection-dataset",
    output_dir=r"C:\drone_dataset"
)

print(path)


import numpy as np
import os

# Data directories

train_images_folder = r"C:\drone_project\drone_dataset\train\images"
train_labels_folder = r"C:\drone_project\drone_dataset\train\labels"
valid_images_folder = r"C:\drone_project\drone_dataset\valid\images"
valid_labels_folder = r"C:\drone_project\drone_dataset\valid\labels"
current_folder = train_images_folder
# files = os.listdir(current_folder)

# Image processing software
# install with pip install opencv-python
# install with pip install matplotlib
import cv2
import matplotlib.pyplot as plt
image_path = r"C:\drone_project\drone_dataset\train\images\yoto10818.jpg"
image = cv2.imread(image_path)
# OpenCV uses BGR, matplotlib uses RGB

def display(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Image processing guide
display(image) #display an image
# print(image[100,200]) # BGR data, nott RGB at pixel height = 100, width = 200
# determine the image size
print("Shape:", image.shape) # (height, width, numer of color channels)

# convert to gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Shape:", gray_image.shape)
display(gray_image)

#resize image
resized = cv2.resize(image, (640, 640)) # resize to 640 by 640 pixels
display(resized)

# flip the image
flipped = cv2.flip(image, 1)
display(flipped)

# rotate the image
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
display(rotated)

# draw a bounding box
cv2.rectangle(
    image,          # 1. image to draw on
    (350, 200),     # 2. top-left corner
    (500, 350),     # 3. bottom-right corner
    (255, 0, 0),    # 4. color, remember BGR (blue)
    1               # 5. thickness
)

display(image)

# display the image with its bounding box
# labels: 0 0.501 0.44 0.316 0.357
# class_id = 0
# x_center = 0.501
# y_center = 0.44
# box_width = 0.316
# box_height = 0.357

def display_image_and_box(image_name):
    image_path = os.path.join(train_images_folder, image_name + ".jpg")
    label_path = os.path.join(train_labels_folder, image_name + ".txt")

    image = cv2.imread(image_path)

    with open(label_path, "r") as file:
        data = file.readline().split()

    class_id = int(data[0])
    x_center = float(data[1])
    y_center = float(data[2])
    box_width = float(data[3])
    box_height = float(data[4])

    height, width = image.shape[:2]

    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    display(image)


display_image_and_box("yoto10847")

# %%

# Minimla transfer learning approach
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data=r"C:\drone_dataset\data.yaml",
    epochs=50,
    imgsz=640
)