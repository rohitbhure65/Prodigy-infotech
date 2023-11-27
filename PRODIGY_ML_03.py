import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set the path to the dataset
dataset_path = "/path/to/dogs-vs-cats/train"

# Function to load and preprocess images
def load_images(path):
    images = []
    labels = []
    for filename in os.listdir(path):
        if filename.startswith("cat"):
            label = 0  # 0 for cats
        elif filename.startswith("dog"):
            label = 1  # 1 for dogs
        else:
            continue

        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (64, 64))  # Resize images to a consistent size

