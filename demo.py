import idx2numpy
import numpy as np

from NaturalLanguageNeuralNetwork import NaturalLanguageNeuralNetwork
from Network import Network
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
    cut_train_images = np.array(
        [
            train_images[i : i + batch_size]
            for i in range(0, len(train_images), batch_size)
        ]
    )
    cut_train_labels = np.array(
        [
            train_labels[i : i + batch_size]
            for i in range(0, len(train_labels), batch_size)
        ]
    )

    nn = NeuralNetwork(num_inputs=len(test_images[0]), batch_size=batch_size)
    nn.add_layer(40, activation_function="relu")
    nn.add_layer(40, activation_function="relu")
    nn.add_layer(10, activation_function="softmax")

    nn.fit(cut_train_images, cut_train_labels, epochs=100)

    predictions = nn.predict(test_images, test_labels)
    print(predictions)


def natural_language_neural_network(
    train_images, train_labels, test_images, test_labels
):
    train_images = prepare_numbers_nl(train_images)
    test_images = prepare_numbers_nl(test_images)

    nn = NaturalLanguageNeuralNetwork()
    nn.add_kernel_layer(16, 9, activation_function="relu")
    nn.add_output_layer(10, train_images[0], activation_function="softmax")

    nn.fit(train_images, train_labels, epochs=2)

    predictions = nn.predict(test_images, test_labels)
    print(predictions)


def main():
    train_images_file = "./MNIST/train-images.idx3-ubyte"
    train_labels_file = "./MNIST/train-labels.idx1-ubyte"
    test_images_file = "./MNIST/t10k-images.idx3-ubyte"
    test_labels_file = "./MNIST/t10k-labels.idx1-ubyte"

    train_images = idx2numpy.convert_from_file(train_images_file)
    train_labels = idx2numpy.convert_from_file(train_labels_file)
    test_images = idx2numpy.convert_from_file(test_images_file)
    test_labels = idx2numpy.convert_from_file(test_labels_file)

    n = Network()
    print(n.confusion_matrix([0, 0, 1, 1, 0], [1, 1, 1, 1, 1], normalize=True))

    # neural_network(train_images, train_labels, test_images, test_labels)
    # natural_language_neural_network(
    #     train_images, train_labels, test_images, test_labels
    # )


if __name__ == "__main__":
    main()
