import os
from typing import List

import numpy as np

from ActivationFunctions import ActivationFunctions
from Layer import Layer


class NeuralNetwork:
    """Implementation of neural network.\n
    A neural network consists of layers and connecting them weights,\n
    each label, except the input layer is computed by multiplying previous layer by connecting them weight layer
    """

    def __init__(self, num_inputs: int, batch_size: int = 1):
        self.batch_size = batch_size
        self.num_inputs = num_inputs

        self.layers = []

    def add_layer(
        self,
        num_nodes: int,
        weight_min_value: float = -0.01,
        weight_max_value: float = 0.01,
        activation_function: str = None,
    ):
        if len(self.layers) == 0:
            num_prev_layer = self.num_inputs
        else:
            num_prev_layer = self.layers[-1].nodes.shape[0]

        new_nodes = np.zeros((num_nodes, self.batch_size))
        new_weights = np.random.uniform(
            low=weight_min_value,
            high=weight_max_value,
            size=(num_nodes, num_prev_layer),
        )

        new_layer = Layer(new_nodes, new_weights, activation_function)
        self.layers.append(new_layer)

    def fit(
        self,
        input_data: np.ndarray,
        goal_output: np.ndarray,
        alpha: float = 0.01,
        dropout_percentage: float = 0,
        epochs: int = 1,
    ):
        if len(self.layers) == 0:
            print("You need to create at least 1 layer using add_layer() function")
            return

        input_data, goal_output = self.__cut_data(input_data, goal_output)
        goal_output = NeuralNetwork.__prepare_output_array(goal_output)

        for _ in range(epochs):
            for batch in range(len(input_data)):
                output, binary_layers = self.__calculate_output(
                    input_data[batch], dropout_percentage
                )

                output_delta = (
                    2 / len(output) * np.subtract(output, goal_output[batch].T)
                ) / self.batch_size

                if len(self.layers) == 1:
                    weights_delta = np.matmul(output_delta, input_data[batch])
                    self.layers[0].weights -= alpha * weights_delta

                elif len(self.layers) == 2:
                    hidden_delta = np.matmul(self.layers[1].weights.T, output_delta)
                    hidden_delta *= ActivationFunctions.derivative_function(
                        self.layers[0].nodes, self.layers[0].function
                    )

                    if len(binary_layers) > 0:
                        hidden_delta *= binary_layers[0]

                    output_weights_delta = np.matmul(
                        output_delta, self.layers[0].nodes.T
                    )
                    hidden_weights_delta = np.matmul(hidden_delta, input_data[batch])

                    self.layers[1].weights -= alpha * output_weights_delta
                    self.layers[0].weights -= alpha * hidden_weights_delta

                else:
                    prev_delta = output_delta
                    for i in range(len(self.layers) - 2, -1, -1):
                        hidden_delta = np.matmul(
                            self.layers[i + 1].weights.T, prev_delta
                        )
                        hidden_delta *= ActivationFunctions.derivative_function(
                            self.layers[i].nodes, self.layers[i].function
                        )

                        if len(binary_layers) > 0:
                            hidden_delta *= binary_layers[i]

                        if i == 0:
                            hidden_weights_delta = np.matmul(
                                hidden_delta, input_data[batch]
                            )
                        else:
                            hidden_weights_delta = np.matmul(
                                hidden_delta, self.layers[i - 1].nodes.T
                            )

                        self.layers[i].weights -= alpha * hidden_weights_delta

                        prev_delta = hidden_delta

                    output_weights_delta = np.matmul(
                        output_delta, self.layers[-2].nodes.T
                    )
                    self.layers[-1].weights -= alpha * output_weights_delta

    def predict(
        self, input_data: np.ndarray, goal_output: np.ndarray
    ) -> np.ndarray[int]:
        if len(self.layers) == 0:
            print("You need to create at least 1 layer using add_layer() function")
            return

        predictions = np.zeros((len(input_data)))
        goal_output = NeuralNetwork.__prepare_output_array(goal_output)

        for index, series in enumerate(input_data):
            output = self.__calculate_output(series)[0]

            predictions[index] = np.argmax(output)

        return predictions

    def guess(self, input_data: np.ndarray) -> List[float]:
        if len(self.layers) == 0:
            print("You need to create at least 1 layer using add_layer() function")
            return

        input_data = np.tile(input_data, (self.batch_size, 1))

        output = self.__calculate_output(input_data)[0][:, 0]
        output = ActivationFunctions.softmax(output)

        return output

    def save_model(self, filepath: str):
        split_filepath = filepath.split("/")
        if len(split_filepath) > 1:
            directory_path = "/".join(split_filepath[:-1])
            os.makedirs(directory_path, exist_ok=True)

        functions = np.array(list([layer.function for layer in self.layers]))

        np.savez_compressed(
            filepath,
            *[layer.weights for layer in self.layers],
            functions=functions,
            batch_size=self.batch_size
        )

    def load_model(self, filepath: str):
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        data = np.load(filepath)

        self.batch_size = data["batch_size"].item()
        data.files.pop(data.files.index("batch_size"))

        functions = data["functions"]
        data.files.pop(data.files.index("functions"))

        self.layers = []
        for index, file in enumerate(data.files):
            weights = data[file]
            nodes = np.zeros((weights.shape[0], self.batch_size))

            self.layers.append(Layer(nodes, weights, functions[index]))

        data.close()

    def __cut_data(self, *data):
        if self.batch_size == 1:
            return data

        return [
            NeuralNetwork.__cut_array_into_batches(array, self.batch_size)
            for array in data
        ]

    def __calculate_output(self, inputs, dropout_percentage=0):
        prev_layer = inputs.T
        prev_weights = self.layers[0].weights
        binary_layers = []

        for index in range(len(self.layers) - 1):
            self.layers[index].nodes = np.matmul(prev_weights, prev_layer)
            self.layers[index].nodes = ActivationFunctions.neural_network_activation(
                self.layers[index].nodes, self.layers[index].function
            )

            if dropout_percentage != 0:
                binary_array = np.random.uniform(0, 1, self.layers[index].nodes.shape)
                binary_array = np.where(binary_array > dropout_percentage, 0, 1)

                self.layers[index].nodes = np.where(
                    binary_array == 1,
                    self.layers[index].nodes / dropout_percentage,
                    0,
                )

                binary_layers.append(binary_array)

            prev_layer = self.layers[index].nodes
            prev_weights = self.layers[index + 1].weights

        self.layers[-1].nodes = np.matmul(prev_weights, prev_layer)

        return self.layers[-1].nodes, binary_layers

    @staticmethod
    def __prepare_output_array(data):
        return np.eye(10)[data]

    @staticmethod
    def __cut_array_into_batches(data, batch_size):
        cut_data = np.array(
            [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        )

        return cut_data
