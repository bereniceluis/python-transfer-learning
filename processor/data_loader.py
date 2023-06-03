import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json

class DataLoader:
    def __init__(self, dataset_dir, labels_file):
        self.dataset_dir = dataset_dir
        self.labels_file = labels_file

    def load_images(self, directory):
        images = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
        return images

    def load_labels(self):
        with open(self.labels_file) as file:
            labels = json.load(file)
        return labels

    def load_dataset(self, test_size=0.2, random_state=42):
        X, y = [], []
        labels = self.load_labels()
        for person in os.listdir(self.dataset_dir):
            person_dir = os.path.join(self.dataset_dir, person)
            if not os.path.isdir(person_dir):
                continue
            unmasked_dir = os.path.join(person_dir, 'unmasked_faces')
            masked_dir = os.path.join(person_dir, 'masked_faces')

            unmasked_images = self.load_images(unmasked_dir)
            masked_images = self.load_images(masked_dir)

            if unmasked_images and masked_images:
                X.extend(unmasked_images + masked_images)
                person_label = labels.get(person, 'Unknown')
                y.extend([person_label] * len(unmasked_images) + [person_label] * len(masked_images))
                print(f'Loaded {len(unmasked_images) + len(masked_images)} examples for class: {person_label}')

        # Perform stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)
