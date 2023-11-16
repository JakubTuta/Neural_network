import numpy as np

from ActivationFunctions import ActivationFunctions
from Layer import Layer


class NeuralNetwork:
    def __init__(self, num_inputs):
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
            num_prev_layer = len(layer.nodes)

        new_nodes = np.zeros((num_nodes, 1))
        new_weights = np.random.uniform(
            low=weight_min_value,
            high=weight_max_value,
            size=(num_nodes, num_prev_layer),
        )

        new_layer = Layer(new_nodes, new_weights, activation_function)
        self.layers.append(new_layer)

    def __calculate_output(self, inputs, dropout_percentage):
        prev_layer = inputs.reshape(-1, 1)
        prev_weights = self.layers[0].weights

        for index in range(len(self.layers)):
            self.layers[index].nodes = np.matmul(prev_layer, prev_weights)
            self.layers[index].nodes = ActivationFunctions.neural_network_activation(
                self.layers[index].nodes, self.layers[index].function
            )

            if not dropout_percentage == 0:
                pass

    def fit(
        self,
        inputs: np.ndarray,
        goal_outputs: np.ndarray,
        alpha: float = 0.01,
        dropout_percentage: float = 0,
    ):
        if len(self.layers) == 0:
            print("You need to create at least 1 layer using add_layer() function")
            return

        for series in range(len(inputs)):
            self.__calculate_output(inputs[series], dropout_percentage)
