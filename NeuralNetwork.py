import numpy as np

from ActivationFunctions import ActivationFunctions
from Layer import Layer


class NeuralNetwork:
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
        num_prev_layer = self.num_inputs
        for layer in self.layers:
            num_prev_layer = layer.nodes.shape[0]

        new_nodes = np.zeros((num_nodes, self.batch_size))
        new_weights = np.random.uniform(
            low=weight_min_value,
            high=weight_max_value,
            size=(num_nodes, num_prev_layer),
        )

        new_layer = Layer(new_nodes, new_weights, activation_function)
        self.layers.append(new_layer)

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

    def fit(
        self,
        inputs: np.ndarray,
        goal_outputs: np.ndarray,
        alpha: float = 0.01,
        dropout_percentage: float = 0,
        epochs: int = 1,
    ):
        if len(self.layers) == 0:
            print("You need to create at least 1 layer using add_layer() function")
            return

        for epoch in range(epochs):
            for batch in range(len(inputs)):
                output, binary_layers = self.__calculate_output(
                    inputs[batch], dropout_percentage
                )

                output_delta = (
                    2 / len(output) * np.subtract(output, goal_outputs[batch].T)
                ) / self.batch_size

                if len(self.layers) == 1:
                    weights_delta = np.matmul(output_delta, inputs[batch])
                    self.layers[0].weights -= alpha * weights_delta

                elif len(self.layers) == 2:
                    hidden_delta = np.matmul(self.layers[1].weights.T, output_delta)
                    hidden_delta *= ActivationFunctions.derivative_function(
                        self.layers[0].nodes, self.layers[0].function
                    )

                    if len(binary_layers) == 1:
                        hidden_delta *= binary_layers[0]

                    output_weights_delta = np.matmul(
                        output_delta, self.layers[0].nodes.T
                    )
                    hidden_weights_delta = np.matmul(hidden_delta, inputs[batch])

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

                        if len(binary_layers) != 0:
                            hidden_delta *= binary_layers[i]

                        if i == 0:
                            hidden_weights_delta = np.matmul(
                                hidden_delta, inputs[batch]
                            )
                        else:
                            hidden_weights_delta = np.matmul(
                                hidden_delta, self.layers[i].nodes.T
                            )

                        self.layers[i].weights -= alpha * hidden_weights_delta

                        prev_delta = hidden_delta

                    output_weights_delta = np.matmul(
                        output_delta, self.layers[-2].nodes.T
                    )
                    self.layers[-1].weights -= alpha * output_weights_delta

    def test_model(self, inputs: np.ndarray, goal_outputs: np.ndarray) -> str:
        hit = 0

        for series in range(len(inputs)):
            output, _ = self.__calculate_output(inputs[series])

            if np.argmax(output) == np.argmax(goal_outputs[series]):
                hit += 1

        avg = hit / len(inputs)
        return f"{np.round(avg * 100, 2)}%"

    @staticmethod
    def prepare_numbers(data):
        data = data.reshape(data.shape[0], 784)
        return data / 255.0

    @staticmethod
    def prepare_number_labels(data):
        return np.eye(10)[data]
