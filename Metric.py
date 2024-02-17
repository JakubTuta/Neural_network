import typing

import numpy as np


class Metric:
    @staticmethod
    def confusion_matrix(
        y_actual: np.ndarray, y_predicted: np.ndarray, normalize: bool = False
    ) -> np.ndarray[int] | np.ndarray[float]:
        """Compute confusion matrix to evaluate the accuracy of a classification

        In the matrix:
            - Rows are actual values
            - Columns are predicted values

        In binary classification:
            matrix[0][0] = True Positive         matrix[0][1] = False Negative\n
            matrix[1][1] = False Positive        matrix[1][1] = True Negative

        Args:
            - y_actual (ndarray): 1 dimensional array of true values
            - y_predicted (ndarray): 1 dimensional array of predicted values
            - normalize (bool, optional): if set to True, all values are normalized to 1. Defaults to False

        Returns:
            - matrix: ndarray of shape (n_classes x n_classes)
        """

        if not Metric.__are_array_shapes_correct(y_actual, y_predicted):
            print("Arrays have different shape")
            return

        y_actual_unique = np.unique(y_actual)
        y_predicted_unique = np.unique(y_predicted)
        labels = np.unique(np.concatenate((y_actual_unique, y_predicted_unique)))

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

        Accuracy is the proportion of correctly classified instances among all instances

        In binary classification:
            matrix[0][0] = True Positive        matrix[0][1] = False Negative\n
            matrix[1][1] = False Positive       matrix[1][1] = True Negative

            Accuracy = (True Positive + True Negative) / (True Positive + False Negative + False Positive + True Negative)

        Args:
            - confusion_matrix (ndarray): 2 dimensional array

        Returns:
            - float: returns value from <0, 1> representing the accuracy of the confusion matrix
            - NaN: given matrix has incorrect shape
        """

        if not Metric.__is_matrix_shape_correct(confusion_matrix):
            print("The matrix has inconsistent shape")
            return np.nan

        return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    @staticmethod
    def get_precision(
        confusion_matrix: np.ndarray,
    ) -> typing.Union[np.float32, np.ndarray[np.float32], np.nan]:
        """Computes precision from the given confusion matrix

        Precision measures the proportion of True Positive predictions among all positive predictions made by the model

        In binary classification:
            matrix[0][0] = True Positive        matrix[0][1] = False Negative\n
            matrix[1][1] = False Positive       matrix[1][1] = True Negative

            Precision = True Positive / (True Positive + False Positive)

        In multilabel classification:
            The function implements the 'one vs rest' strategy,
            where each label is considered as a positive value
            compared to all other labels, which are treated as negative values

        Args:
            - confusion_matrix (ndarray): 2 dimensional array

        Returns:
            - float: returns value from <0, 1> representing the accuracy of the confusion matrix,
            - ndarray[float]: representing array of precision values for each true label,
            - NaN: given matrix has incorrect shape
        """

        if not Metric.__is_matrix_shape_correct(confusion_matrix):
            print("The matrix has inconsistent shape")
            return np.nan

        num_labels = confusion_matrix.shape[0]

        if num_labels == 2:
            column_sum = np.sum(confusion_matrix[:, 0])

            if column_sum == 0:
                return 0

            return confusion_matrix[0, 0] / column_sum

        column_sums = np.sum(confusion_matrix, axis=0)
        precisions = np.zeros(num_labels)
        diag = np.diag(confusion_matrix)

        for index in range(len(column_sums)):
            if column_sums[index] != 0:
                precisions[index] = diag[index] / column_sums[index]

        return precisions

    @staticmethod
    def get_recall(
        confusion_matrix: np.ndarray,
    ) -> typing.Union[np.float32, np.ndarray[np.float32], np.nan]:
        """Computes recall from the given confusion matrix

        Recall measures the proportion of True Positive predictions among all actual positive instances in the dataset

        In binary classification:
            matrix[0][0] = True Positive        matrix[0][1] = False Negative\n
            matrix[1][1] = False Positive       matrix[1][1] = True Negative

            Recall = True Positive / (True Positive + False Negative)

        In multilabel classification:
            The function implements the 'one vs rest' strategy,
            where each label is considered as a positive value
            compared to all other labels, which are treated as negative values

        Args:
            - confusion_matrix (ndarray): 2 dimensional array

        Returns:
            - float: returns value from <0, 1> representing the recall of the confusion matrix,
            - ndarray[float]: representing array of recall values for each true label,
            - NaN: given matrix has incorrect shape
        """

        if not Metric.__is_matrix_shape_correct(confusion_matrix):
            print("The matrix has inconsistent shape")
            return np.nan

        num_labels = confusion_matrix.shape[0]

        if num_labels == 2:
            row_sum = np.sum(confusion_matrix[0])

            if row_sum == 0:
                return 0

            return confusion_matrix[0, 0] / row_sum

        row_sums = np.sum(confusion_matrix, axis=1)
        recalls = np.zeros(num_labels)
        diag = np.diag(confusion_matrix)

        for index in range(len(row_sums)):
            if row_sums[index] != 0:
                recalls[index] = diag[index] / row_sums[index]

        return recalls

    @staticmethod
    def get_specificity(
        confusion_matrix: np.ndarray,
    ) -> typing.Union[np.float32, np.ndarray[np.float32], np.nan]:
        """Computes specificity from the given confusion matrix

        Specificity is the ration of True Negative predictions to the sum of Negative values

        In binary classification:
            matrix[0][0] = True Positive        matrix[0][1] = False Negative\n
            matrix[1][1] = False Positive       matrix[1][1] = True Negative

            Specificity = True Negative / (False Positive + True Negative)

        In multilabel classification:
            The function implements the 'one vs rest' strategy,
            where each label is considered as a positive value
            compared to all other labels, which are treated as negative values

        Args:
            - confusion_matrix (ndarray): 2 dimensional array

        Returns:
            - float: returns value from <0, 1> representing the specificity of the confusion matrix,
            - ndarray[float]: representing array of specificity values for each true label,
            - NaN: given matrix has incorrect shape
        """

        if not Metric.__is_matrix_shape_correct(confusion_matrix):
            print("The matrix has inconsistent shape")
            return np.nan

        num_labels = confusion_matrix.shape[0]

        if num_labels == 2:
            row_sum = np.sum(confusion_matrix[1])

            if row_sum == 0:
                return 0

            return confusion_matrix[1, 1] / row_sum

        specificities = np.zeros(num_labels)

        for class_index in range(num_labels):
            true_negatives = np.sum(
                np.delete(
                    np.delete(confusion_matrix, class_index, axis=0),
                    class_index,
                    axis=1,
                )
            )

            false_positives = (
                np.sum(confusion_matrix[:, class_index])
                - confusion_matrix[class_index, class_index]
            )

            if (true_negatives + false_positives) != 0:
                specificities[class_index] = true_negatives / (
                    true_negatives + false_positives
                )

        return specificities

    @staticmethod
    def get_fscore(
        confusion_matrix: np.ndarray,
    ) -> typing.Union[np.float32, np.ndarray[np.float32], np.nan]:
        """Computes F1 score from the given confusion matrix

        F1 score is the harmonic mean of precision and recall

        In binary classification:
            matrix[0][0] = True Positive        matrix[0][1] = False Negative\n
            matrix[1][1] = False Positive       matrix[1][1] = True Negative

            F1 score = (2 * True Positive) / (2 * True Positive + False Positive + False Negative)

        In multilabel classification:
            The function implements the 'one vs rest' strategy,
            where each label is considered as a positive value
            compared to all other labels, which are treated as negative values

        Args:
            - confusion_matrix (ndarray): 2 dimensional array

        Returns:
            - float: returns value from <0, 1> representing the F1 score of the confusion matrix,
            - ndarray[float]: representing array of F1 score values for each true label,
            - NaN: given matrix has incorrect shape
        """

        if not Metric.__is_matrix_shape_correct(confusion_matrix):
            print("The matrix has inconsistent shape")
            return np.nan

        num_labels = confusion_matrix.shape[0]

        if num_labels == 2:
            precision = Metric.get_precision(confusion_matrix)
            recall = Metric.get_recall(confusion_matrix)

            if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
                return np.nan

            fscore = 2 * (precision * recall) / (precision + recall)

            return fscore

        precisions = Metric.get_precision(confusion_matrix)
        recalls = Metric.get_recall(confusion_matrix)

        fscores = np.zeros(num_labels)

        for index, (precision, recall) in enumerate(zip(precisions, recalls)):
            if precision + recall != 0:
                fscores[index] = 2 * (precision * recall) / (precision + recall)

        return fscores

    @staticmethod
    def get_all_statistics(confusion_matrix: np.ndarray) -> typing.Dict:
        """Computes all statistics from confusion matrix

        Args:
            - confusion_matrix (np.ndarray): 2 dimensional array

        Returns:
            - dictionary containing accuracy, precision, recall, specificity
        """

        statistics = {}

        statistics["accuracy"] = Metric.get_accuracy(confusion_matrix)
        statistics["precision"] = Metric.get_precision(confusion_matrix)
        statistics["recall"] = Metric.get_recall(confusion_matrix)
        statistics["specificity"] = Metric.get_specificity(confusion_matrix)
        statistics["fscore"] = Metric.get_fscore(confusion_matrix)

        return statistics

    @staticmethod
    def __is_matrix_shape_correct(matrix):
        return (
            matrix is not None
            and matrix.shape[0] != 0
            and matrix.shape[1] != 0
            and matrix.shape[0] == matrix.shape[1]
        )

    @staticmethod
    def __are_array_shapes_correct(array_1, array_2):
        return (
            array_1 is not None
            and array_2 is not None
            and len(array_1) > 0
            and len(array_2) > 0
            and len(array_1) == len(array_2)
        )
