import numpy as np


class Network:
    def confusion_matrix(self, y_actual, y_predicted, normalize=False):
        labels = np.unique(np.concatenate((y_actual, y_predicted)))
        matrix = np.zeros((len(labels), len(labels)), dtype=np.uint64)

        for actual, predicted in zip(y_actual, y_predicted):
            matrix[labels == predicted, labels == actual] += 1

        if normalize:
            matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]
            matrix = np.nan_to_num(matrix, nan=0)

        return matrix

    @staticmethod
    def _prepare_output_array(data):
        return np.eye(10)[data]
