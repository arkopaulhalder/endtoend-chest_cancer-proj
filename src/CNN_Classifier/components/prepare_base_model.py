import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from CNN_Classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepare a full model by freezing layers and adding a softmax prediction layer.

        Args:
            model (tf.keras.Model): The base model to be updated.
            classes (int): Number of classes in the dataset.
            freeze_all (bool): Whether to freeze all layers.
            freeze_till (int): The index of the last layer to be frozen.
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            tf.keras.Model: The full model with the softmax prediction layer.
        """
        if freeze_all:
            # Freeze all layers
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            # Freeze layers up to the specified index
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        # Add a softmax prediction layer
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # Print a summary of the full model
        full_model.summary()
        return full_model
    

    def update_base_model(self) -> None:
        """
        Updates the base model and saves it to the specified path.

        Args:
            self (PrepareBaseModel): The instance of the `PrepareBaseModel` class.

        Returns:
            None
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,  # type: int
            freeze_all=True,  # type: bool
            freeze_till=None,  # type: Optional[int]
            learning_rate=self.config.params_learning_rate,  # type: float
        )

        self.save_model(
            path=self.config.updated_base_model_path,  # type: Path
            model=self.full_model,  # type: tf.keras.Model
        )
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """
        Saves the model to the specified path.

        Args:
            path (Path): The path to save the model.
            model (tf.keras.Model): The model to be saved.

        Raises:
            Exception: If an error occurs while saving the model.
        """
        try:
            # Save the model to the specified path
            model.save(path)
        except Exception as e:
            # Print an error message if an exception occurs
            print(f"Error occurred in loading the model {path} due to {e}")
