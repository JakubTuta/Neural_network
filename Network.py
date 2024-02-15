import typing

import numpy as np


class Network:
    @staticmethod
    def confusion_matrix(
        y_actual: np.ndarray, y_predicted: np.ndarray, normalize: bool = False
    ) -> np.ndarray[int] | np.ndarray[float]:
        """Compute confusion matrix to evaluate the accuracy of a classification

        In the matrix:
            - Rows are actual values
            - Columns are predicted values

        In binary classification:
        matrix[0][0] = True Positive        matrix[0][1] = False Negative
        matrix[1][1] = False Positive       matrix[1][1] = True Negative

        Args:
            y_actual (ndarray): 1 dimensional array of true values
            y_predicted (ndarray): 1 dimensional array of predicted values
            normalize (bool, optional): if set to True, all values are normalized to 1. Defaults to False

        Returns:
            matrix: ndarray of shape (n_classes x n_classes)
        """

        labels = np.unique(np.concatenate((y_actual, y_predicted)))
        matrix = np.zeros((len(labels), len(labels)), dtype=np.uint64)

        for actual, predicted in zip(y_actual, y_predicted):
            matrix[labels == actual, labels == predicted] += 1

        if normalize:
            matrix = matrix.astype("float") / np.sum(matrix)
            matrix = np.nan_to_num(matrix, nan=0)

        return matrix

    @staticmethod
    def get_accuracy(confusion_matrix: np.ndarray) -> typing.Optional[float]:
        """Computes accuracy from the given confusion matrix

        In binary classification:
            matrix[0][0] = True Positive        matrix[0][1] = False Negative
            matrix[1][1] = False Positive       matrix[1][1] = True Negative

            Accuracy = (True Positive + True Negative) / (True Positive + False Negative + False Positive + True Negative)

        Args:
            confusion_matrix (ndarray): 2 dimensional array

        Returns:
            float: returns value from <0, 1> representing the accuracy of the confusion matrix
            NaN: given matrix has incorrect shape
        """

        if not Network.__is_matrix_shape_correct(confusion_matrix):
            return np.nan

        return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    @staticmethod
    def get_precision(
        confusion_matrix: np.ndarray,
    ) -> typing.Union[np.float32, np.ndarray[np.float32], np.nan]:
        """Computes precision from the given confusion matrix

        In binary classification:
            matrix[0][0] = True Positive        matrix[0][1] = False Negative
            matrix[1][1] = False Positive       matrix[1][1] = True Negative

            Accuracy = True Positive / (True Positive + False Positive)

        In multilabel classification:
            function computes precision for each true label

        Args:
            confusion_matrix (ndarray): 2 dimensional array

        Returns:
            float: returns value from <0, 1> representing the accuracy of the confusion matrix,
            ndarray[float]: representing array of precision values for each true label,
            NaN: given matrix has incorrect shape
        """

        if not Network.__is_matrix_shape_correct(confusion_matrix):
            return np.nan

        num_labels = confusion_matrix.shape[0]

        if num_labels == 2:
            column_sum = np.sum(confusion_matrix[:, 0])

            if column_sum == 0:
                return 0

            return np.sum(np.diag(confusion_matrix)) / column_sum

        column_sums = np.sum(confusion_matrix, axis=0)
        array_precisions = np.where(
            column_sums == 0, 0, np.sum(np.diag(confusion_matrix)) / column_sums
        )

        return array_precisions

    @staticmethod
    def get_all_statistics(confusion_matrix: np.ndarray) -> typing.Dict:
        """Computes all statistics from confusion matrix

        Args:
            confusion_matrix (np.ndarray): 2 dimensional array

        Returns:
            dict: a dictionary containing accuracy, precision
        """

        statistics = {}

        statistics["accuracy"] = Network.get_accuracy(confusion_matrix)
        statistics["precision"] = Network.get_precision(confusion_matrix)

        return statistics

    @staticmethod
    def _prepare_output_array(data):
        return np.eye(10)[data]

    @staticmethod
    def __is_matrix_shape_correct(matrix):
        return (
            matrix is not None
            and matrix.shape[0] != 0
            and matrix.shape[1] != 0
            and matrix.shape[0] == matrix.shape[1]
        )
