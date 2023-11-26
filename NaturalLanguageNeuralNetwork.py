from typing import Tuple

import numpy as np

from ActivationFunctions import ActivationFunctions


class NaturalLanguageNeuralNetwork:
    def add_kernel_layer(
        self,
        num_filters: int,
        filter_size: int,
        weights_range: Tuple[float, float] = (-0.01, 0.01),
        activation_function: str = None,
    ):
        self.kernel_layer = np.random.uniform(
            low=weights_range[0],
            high=weights_range[1],
            size=(num_filters, filter_size**2),
        )
        self.kernel_layer_activation_function = activation_function

    def add_output_layer(
        self,
        num_nodes: int,
        example_input: np.ndarray,
        weights_range: Tuple[float, float] = (-0.01, 0.01),
        activation_function: str = None,
    ):
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
        self, input_images: np.ndarray, goal_outputs: np.ndarray, alpha: float = 0.01
    ):
        if self.kernel_layer is None:
            print(
                "You need to add a kernel layer before using fit function. Try using add_kernel_layer()"
            )
            return

        if self.output_layer_weights is None:
            print(
                "You need to add an output layer before using fit function. Try using add_output_layer()"
            )

        for series in range(len(input_images)):
            cut_image, kernel_layer, output_layer = self.__calculate_output(
                input_images[series]
            )
            flatten_kernel_layer = kernel_layer.flatten()[:, np.newaxis]

            output_layer_delta = (
                2
                / len(output_layer)
                * np.subtract(output_layer, goal_outputs[series, :, np.newaxis])
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

    def test_model(self, input_images: np.ndarray, goal_outputs: np.ndarray):
        hit = 0

        for series in range(len(input_images)):
            output_layer = self.__calculate_output(input_images[series])[2]

            if np.argmax(output_layer) == np.argmax(goal_outputs[series]):
                hit += 1

        avg = hit / len(input_images)
        return f"{np.round(avg * 100, 2)}%"

    @staticmethod
    def prepare_numbers(data):
        return data / 255.0

    @staticmethod
    def prepare_number_labels(data):
        return np.eye(10)[data]

    def __calculate_output(self, input_image):
        cut_image = NaturalLanguageNeuralNetwork.__cut_image(
            input_image, len(self.kernel_layer[0])
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
