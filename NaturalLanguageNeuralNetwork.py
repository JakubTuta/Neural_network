import os
from typing import List, Tuple

import numpy as np

from ActivationFunctions import ActivationFunctions


class NaturalLanguageNeuralNetwork:
    def add_kernel_layer(
        self,
        num_filters: int,
        filter_size: Tuple[int, int],
        weights_range: Tuple[float, float] = (-0.01, 0.01),
        activation_function: str = None,
    ):
        """Adds a kernel layer to the neural network.

        Args:
            - num_filters (int): Number of filters in the layer.
            - filter_size (Tuple[int, int]): Size of each filter in the layer.
            - weights_range (Tuple[float, float], optional): Range for initializing weights. Defaults to (-0.01, 0.01).
            - activation_function (str, optional): Activation function for the layer. Defaults to None.
        """

        if not NaturalLanguageNeuralNetwork.__is_filter_shape_correct(filter_size):
            print(
                "The filter size is incorrect. Filter must consist of 2 the same odd values greater than 0"
            )
            return

        self.kernel_layer = np.random.uniform(
            low=weights_range[0],
            high=weights_range[1],
            size=(num_filters, filter_size[0] ** 2),
        )
        self.kernel_layer_activation_function = activation_function

    def add_output_layer(
        self,
        num_nodes: int,
        example_input: np.ndarray,
        weights_range: Tuple[float, float] = (-0.01, 0.01),
        activation_function: str = None,
    ):
        """Adds an output layer to the neural network.

        Args:
            - num_nodes (int): Number of nodes in the layer.
            - example_input (ndarray): Example input data for determining layer dimensions.
            - weights_range (Tuple[float, float], optional): Range for initializing weights. Defaults to (-0.01, 0.01).
            - activation_function (str, optional): Activation function for the layer. Defaults to None.
        """

        if self.kernel_layer is None:
            print("You have to add a kernel layer before creating an output layer")
            return

        num_filters = len(self.kernel_layer)
        mask_size = len(self.kernel_layer[0])

        num_cut_images = (
            len(NaturalLanguageNeuralNetwork.__cut_image(example_input, mask_size))
            * num_filters
        )

        self.output_layer_weights = np.random.uniform(
            low=weights_range[0],
            high=weights_range[1],
            size=(num_nodes, num_cut_images),
        )
        self.output_layer_activation_function = activation_function

    def fit(
        self,
        input_data: np.ndarray,
        goal_output: np.ndarray,
        alpha: float = 0.01,
        epochs: int = 0,
    ):
        """Fits the neural network to the training data using backpropagation.

        Args:
            - input_data (ndarray): Input data for training.
            - goal_output (ndarray): Desired output data for training.
            - alpha (float, optional): Learning rate. Defaults to 0.01.
            - epochs (int, optional): Number of training epochs. Defaults to 0.
        """

        if self.kernel_layer is None:
            print(
                "You need to add a kernel layer before using fit function. Try using add_kernel_layer()"
            )
            return

        if self.output_layer_weights is None:
            print(
                "You need to add an output layer before using fit function. Try using add_output_layer()"
            )
            return

        goal_output = NaturalLanguageNeuralNetwork.__prepare_output_array(goal_output)

        for _ in range(epochs):
            for series in range(len(input_data)):
                cut_image, kernel_layer, output_layer = self.__calculate_output(
                    input_data[series]
                )
                flatten_kernel_layer = kernel_layer.flatten()[:, np.newaxis]

                output_layer_delta = (
                    2
                    / len(output_layer)
                    * np.subtract(output_layer, goal_output[series, :, np.newaxis])
                )

                kernel_layer_delta = np.matmul(
                    self.output_layer_weights.T, output_layer_delta
                )
                kernel_layer_delta = kernel_layer_delta.reshape(kernel_layer.shape)
                kernel_layer_delta *= ActivationFunctions.derivative_function(
                    kernel_layer, self.kernel_layer_activation_function
                )

                output_layer_weights_delta = np.matmul(
                    output_layer_delta, flatten_kernel_layer.T
                )
                kernel_layer_weights_delta = np.matmul(kernel_layer_delta.T, cut_image)

                self.output_layer_weights -= alpha * output_layer_weights_delta
                self.kernel_layer -= alpha * kernel_layer_weights_delta

    def predict(
        self, input_data: np.ndarray, goal_output: np.ndarray
    ) -> np.ndarray[int]:
        """Predicts output values for input data.

        Args:
            - input_data (ndarray): Input data for prediction.
            - goal_output (ndarray): Desired output data for prediction.

        Returns:
            - np.ndarray[int]: Predicted output values.
        """

        if self.kernel_layer is None:
            print(
                "You need to add a kernel layer before using fit function. Try using add_kernel_layer()"
            )
            return

        if self.output_layer_weights is None:
            print(
                "You need to add an output layer before using fit function. Try using add_output_layer()"
            )
            return

        predictions = np.zeros((len(input_data)))
        goal_output = NaturalLanguageNeuralNetwork.__prepare_output_array(goal_output)

        for index, series in enumerate(input_data):
            output = self.__calculate_output(series)[2]

            predictions[index] = np.argmax(output)

        return predictions

    def guess(self, input_data: np.ndarray) -> List[float]:
        """Makes predictions based on input data.

        Args:
            - input_data (ndarray): Input data for making predictions.

        Returns:
            - List[float]: Predicted output values.
        """

        if self.kernel_layer is None:
            print(
                "You need to add a kernel layer before using fit function. Try using add_kernel_layer()"
            )
            return

        if self.output_layer_weights is None:
            print(
                "You need to add an output layer before using fit function. Try using add_output_layer()"
            )
            return

        output = self.__calculate_output(input_data)[2]
        output = ActivationFunctions.softmax(output.reshape(-1, 10)[0])

        return output

    def save_model(self, filepath: str):
        """Saves the trained model to a file.

        Args:
            - filepath (str): Path to save the model.
        """

        split_filepath = filepath.split("/")
        if len(split_filepath) > 1:
            directory_path = "/".join(split_filepath[:-1])
            os.makedirs(directory_path, exist_ok=True)

        np.savez_compressed(
            filepath,
            kernel_layer=self.kernel_layer,
            kernel_layer_function=self.kernel_layer_activation_function,
            output_weights=self.output_layer_weights,
            output_layer_function=self.output_layer_activation_function,
        )

    def load_model(self, filepath: str):
        """Loads a trained model from a file.

        Args:
            - filepath (str): Path to load the model from.
        """

        if not filepath.endswith(".npz"):
            filepath += ".npz"

        data = np.load(filepath)

        self.kernel_layer = data["kernel_layer"]
        self.kernel_layer_activation_function = data["kernel_layer_function"].item()
        self.output_layer_weights = data["output_weights"]
        self.output_layer_activation_function = data["output_layer_function"].item()

        data.close()

    def __calculate_output(self, input_data):
        cut_image = NaturalLanguageNeuralNetwork.__cut_image(
            input_data, len(self.kernel_layer[0])
        )

        kernel_layer = np.matmul(cut_image, self.kernel_layer.T)
        kernel_layer = ActivationFunctions.neural_network_activation(
            kernel_layer, self.kernel_layer_activation_function
        )

        flatten_kernel_layer = kernel_layer.flatten()[:, np.newaxis]

        output_layer = np.matmul(self.output_layer_weights, flatten_kernel_layer)
        output_layer = ActivationFunctions.neural_network_activation(
            output_layer, self.output_layer_activation_function
        )

        return cut_image, kernel_layer, output_layer

    @staticmethod
    def __cut_image(image, mask_size):
        square_size = int(np.sqrt(mask_size))
        cut_size = square_size // 2

        start_x, end_x = cut_size, image.shape[1] - cut_size
        start_y, end_y = cut_size, image.shape[0] - cut_size

        cut_image = np.array(
            [
                image[
                    row - cut_size : row + cut_size + 1,
                    col - cut_size : col + cut_size + 1,
                ]
                for row in range(start_y, end_y)
                for col in range(start_x, end_x)
            ]
        )
        cut_image = cut_image.reshape(cut_image.shape[0], mask_size)

        return cut_image

    @staticmethod
    def __prepare_output_array(data):
        return np.eye(10)[data]

    @staticmethod
    def __is_filter_shape_correct(data):
        return (
            data is not None
            and (type(data) is tuple or type(data) is list)
            and len(data) == 2
            and data[0] > 0
            and data[0] % 2 == 1
            and data[0] == data[1]
        )
