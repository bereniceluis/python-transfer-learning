import os
import numpy as np
import uuid
import json
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DatasetPreprocessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def rename_images(self):
        images_path = ('data/dataset_2')
        for person_folder in os.listdir(images_path):
            person_path = os.path.join(images_path, person_folder)
            if os.path.isdir(person_path):
                for face_type_folder in os.listdir(person_path):
                    face_type_path = os.path.join(person_path, face_type_folder)
                    if os.path.isdir(face_type_path):
                        for file_name in os.listdir(face_type_path):
                            file_path = os.path.join(face_type_path, file_name)
                            if os.path.isfile(file_path):
                                file_name_without_ext, extension = os.path.splitext(file_name)
                                if face_type_folder == "unmasked_faces":
                                    new_file_name = f'{person_folder.split("_")[0]}_UNMASKED_{str(uuid.uuid4().hex)[:10]}{extension}'
                                else:
                                    new_file_name = f'{person_folder.split("_")[0]}_FACEMASK_{str(uuid.uuid4().hex)[:10]}{extension}'
                                new_file_path = os.path.join(face_type_path, new_file_name)
                                os.rename(file_path, new_file_path)

    def augment_dataset(self, output_dir):
        dataset_path = ('data/dataset_2')
        
        # Create required directories
        for person in os.listdir(dataset_path):
            person_dir = os.path.join(output_dir, person)
            os.makedirs(person_dir, exist_ok=True)
            os.makedirs(os.path.join(person_dir, "masked_faces"), exist_ok=True)
            os.makedirs(os.path.join(person_dir, "unmasked_faces"), exist_ok=True)

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,  
            fill_mode='nearest'
        )

        for person in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person)
            augmented_person_dir = os.path.join(output_dir, person)

            for class_name in ["masked_faces", "unmasked_faces"]:
                class_dir = os.path.join(person_dir, class_name)
                augmented_class_dir = os.path.join(augmented_person_dir, class_name)

                os.makedirs(augmented_class_dir, exist_ok=True)

                image_paths = [os.path.join(class_dir, file) for file in os.listdir(class_dir)]

                for image_path in image_paths:
                    img = np.array(Image.open(image_path))
                    img = np.expand_dims(img, axis=0)

                    save_dir = augmented_class_dir
                    save_prefix = os.path.splitext(os.path.basename(image_path))[0]

                    datagen.fit(img)
                    for x, val in zip(datagen.flow(img, save_to_dir=save_dir, save_prefix=save_prefix, save_format='jpg'), range(10)):
                        pass

    def generate_labels(self, labels_file):
        labels = {}

        for root, dirs, files in os.walk(self.dataset_dir):
            for subdir in dirs:
                # Extract the sub-directory name
                sub_dir = subdir

                # Extract the person name from the sub-directory name
                person = sub_dir.split('_')
                person_name = person[1].capitalize() + ' ' + person[0].capitalize()

                # Use the sub-directory name as the key and assign the corresponding person name as the value
                labels[sub_dir] = person_name

        # Remove unnecessary labels
        del labels['masked_faces']
        del labels['unmasked_faces']

        # Save the labels to a JSON file
        with open(labels_file, 'w') as f:
            json.dump(labels, f)
