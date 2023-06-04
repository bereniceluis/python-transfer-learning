import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam


class TransferLearningModel:
    def __init__(self, train_dir, val_dir, batch_size=32):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.model = None

    def _create_data_generators(self):
        data_generator = ImageDataGenerator(rescale=1./255)
        custom_classes = os.listdir(self.train_dir)

        train_generator = data_generator.flow_from_directory(
            self.train_dir,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=custom_classes
        )

        validation_generator = data_generator.flow_from_directory(
            self.val_dir,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=custom_classes
        )

        return train_generator, validation_generator

    def _build_model(self, num_classes):
        img_rows, img_cols = 224, 224
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

        for layer in base_model.layers:
            layer.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=num_classes, activation='softmax'))

        return model

    def train(self, num_epochs=20, save_filename='model/mobilenetv2_transferlearning.h5'):
        train_generator, validation_generator = self._create_data_generators()
        num_classes = len(train_generator.class_indices)
        self.model = self._build_model(num_classes)

        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9
        )
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss_fn = 'categorical_crossentropy'

        metrics = [
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top-3'),
            keras.metrics.AUC(name='ROC-AUC', curve='ROC'),
            keras.metrics.AUC(name='PR-AUC', curve='PR'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.TruePositives(name='TP'),
            keras.metrics.TrueNegatives(name='TN'),
            keras.metrics.FalsePositives(name='FP'),
            keras.metrics.FalseNegatives(name='FN')
        ]

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        steps_per_epoch = len(train_generator)
        validation_steps = len(validation_generator)

        history = self.model.fit(
            train_generator,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps
        )

        self.save_model(save_filename)
        return history
    

    def save_model(self, filepath):
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved successfully to {filepath}")
        else:
            print("No model to save. Train the model first.")

