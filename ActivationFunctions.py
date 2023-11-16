import numpy as np


class ActivationFunctions:
    @staticmethod
    def relu(layer):
        return np.maximum(0, layer)

    @staticmethod
    def sigmoid(layer):
        return 1 / (1 + np.exp(-layer))

    @staticmethod
    def tanh(layer):
        return np.tanh(layer)

    @staticmethod
    def softmax(layer):
        e_x = np.exp(layer - np.max(layer))
        return e_x / e_x.sum()

    @staticmethod
    def softmax_matrix(layer):
        e_x = np.exp(layer - np.max(layer, axis=0))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def neural_network_activation(layer, function):
        match (function):
            case "relu":
                return ActivationFunctions.relu(layer)
            case "sigmoid":
                return ActivationFunctions.sigmoid(layer)
            case "tanh":
                return ActivationFunctions.tanh(layer)
            case "softmax":
                if ActivationFunctions.__is_matrix(layer):
                    return ActivationFunctions.softmax_matrix(layer)
                else:
                    return ActivationFunctions.softmax(layer)
            case _:
                return layer

    @staticmethod
    def derivative_relu(layer):
        output = np.copy(layer)
        output[layer > 0] = 1
        output[layer <= 0] = 0
        return output

    @staticmethod
    def derivative_sigmoid(layer):
        return layer * (1 - layer)

    @staticmethod
    def derivative_tanh(layer):
        return 1 - (layer**2)

    @staticmethod
    def derivative_function(layer, function):
        match (function):
            case "relu":
                return ActivationFunctions.derivative_relu(layer)
            case "sigmoid":
                return ActivationFunctions.derivative_sigmoid(layer)
            case "tanh":
                return ActivationFunctions.derivative_tanh(layer)
            case _:
                return np.ones((len(layer), 1))

    @staticmethod
    def __is_matrix(layer):
        return len(layer.shape) == 2 and not (
            layer.shape[0] == 1 or layer.shape[1] == 1
        )
