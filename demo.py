import idx2numpy
import numpy as np

from Metric import Metric
from NaturalLanguageNeuralNetwork import NaturalLanguageNeuralNetwork
from NeuralNetwork import NeuralNetwork


def prepare_numbers_nl(data):
    return data / np.max(data)


def prepare_numbers(data):
    data = data.reshape(data.shape[0], 784)
    return data / np.max(data)


def neural_network(train_images, train_labels, test_images, test_labels):
    train_images = prepare_numbers(train_images)
    test_images = prepare_numbers(test_images)

    batch_size = 100

    nn = NeuralNetwork(num_inputs=len(test_images[0]), batch_size=batch_size)
    nn.add_layer(100, activation_function="relu")
    nn.add_layer(10, activation_function="softmax")

    nn.fit(train_images, train_labels, epochs=100)

    predictions = nn.predict(test_images, test_labels)

    cm = Metric.confusion_matrix(test_labels, predictions)
    print(cm)
    statistics = Metric.get_all_statistics(cm)
    print(statistics["accuracy"])

    nn.save_model("models/nn")


def natural_language_neural_network(
    train_images, train_labels, test_images, test_labels
):
    train_images = prepare_numbers_nl(train_images)
    test_images = prepare_numbers_nl(test_images)

    nn = NaturalLanguageNeuralNetwork()
    nn.add_kernel_layer(16, (3, 3), activation_function="relu")
    nn.add_output_layer(10, train_images[0], activation_function="softmax")

    nn.fit(train_images, train_labels, epochs=10)

    predictions = nn.predict(test_images, test_labels)

    cm = Metric.confusion_matrix(test_labels, predictions)
    print(cm)
    statistics = Metric.get_all_statistics(cm)
    print(statistics["accuracy"])

    nn.save_model("models/nlnn.npz")


def main():
    train_images_file = "./MNIST/train-images.idx3-ubyte"
    train_labels_file = "./MNIST/train-labels.idx1-ubyte"
    test_images_file = "./MNIST/t10k-images.idx3-ubyte"
    test_labels_file = "./MNIST/t10k-labels.idx1-ubyte"

    train_images = idx2numpy.convert_from_file(train_images_file)
    train_labels = idx2numpy.convert_from_file(train_labels_file)
    test_images = idx2numpy.convert_from_file(test_images_file)
    test_labels = idx2numpy.convert_from_file(test_labels_file)

    neural_network(train_images, train_labels, test_images, test_labels)
    # natural_language_neural_network(
    #     train_images, train_labels, test_images, test_labels
    # )


if __name__ == "__main__":
    main()
