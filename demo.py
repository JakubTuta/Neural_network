import idx2numpy
import numpy as np

from NeuralNetwork import NeuralNetwork

train_images_file = "./MNIST/train-images.idx3-ubyte"
train_labels_file = "./MNIST/train-labels.idx1-ubyte"
test_images_file = "./MNIST/t10k-images.idx3-ubyte"
test_labels_file = "./MNIST/t10k-labels.idx1-ubyte"

train_images = idx2numpy.convert_from_file(train_images_file)
train_labels = idx2numpy.convert_from_file(train_labels_file)
test_images = idx2numpy.convert_from_file(test_images_file)
test_labels = idx2numpy.convert_from_file(test_labels_file)

train_labels = NeuralNetwork.prepare_number_labels(train_labels)
test_labels = NeuralNetwork.prepare_number_labels(test_labels)

train_images = NeuralNetwork.prepare_numbers(train_images)
test_images = NeuralNetwork.prepare_numbers(test_images)

test_images = test_images[:10000]
test_labels = test_labels[:10000]

batch_size = 100
cut_train_images = np.array(
    [train_images[i : i + batch_size] for i in range(0, len(train_images), batch_size)]
)
cut_train_labels = np.array(
    [train_labels[i : i + batch_size] for i in range(0, len(train_labels), batch_size)]
)


nn = NeuralNetwork(num_inputs=len(test_images[0]), batch_size=batch_size)
nn.add_layer(40, activation_function="relu")
# nn.add_layer(40, activation_function="relu")
nn.add_layer(10, activation_function="softmax")
nn.fit(cut_train_images[:100], cut_train_labels[:100], epochs=500, alpha=0.1)
print(nn.test_model(test_images, test_labels))
