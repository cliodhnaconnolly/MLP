# Cliodhna Connolly - 13434702

import numpy as npy
import random

class MLP:
    def __init__(self, NI, NH, NO):
        self.number_of_inputs = NI
        self.number_of_hidden_units = NH
        self.number_of_outputs = NO
        self.sin = 0
        self.randomise()

    def randomise(self):
        self.lower_layer_weights = npy.array((npy.random.uniform(0.0, 1.0, (self.number_of_inputs, self.number_of_hidden_units))))
        self.upper_layer_weights = npy.array((npy.random.uniform(0.0, 1.0, (self.number_of_hidden_units, self.number_of_outputs))))
        self.lower_weight_changes = npy.array
        self.upper_weight_changes = npy.array

    def forward(self, input_example):
        self.lower_layer_activations = npy.dot(input_example, self.lower_layer_weights)
        self.hidden_neurons = self.sigmoid(self.lower_layer_activations)
        self.upper_layer_activations = npy.dot(self.hidden_neurons, self.upper_layer_weights)
        self.output = self.sigmoid(self.upper_layer_activations)

    def backwards(self, input_examples, target):
        error = target - self.output
        delta_output_sum = npy.multiply(self.der_sigmoid(self.upper_layer_activations), error)
        self.upper_weight_changes = npy.dot(self.hidden_neurons.T, delta_output_sum)
        hidden_neuron_sigmoid = self.der_sigmoid(self.lower_layer_activations)
        delta_hidden_sum = npy.multiply(npy.multiply(delta_output_sum, self.upper_layer_weights.T), hidden_neuron_sigmoid)
        self.lower_weight_changes = npy.dot(input_examples.T, delta_hidden_sum)
        return npy.mean(npy.abs(error))

    def update_weights(self, learning_rate):
        self.lower_layer_weights = npy.add(self.lower_layer_weights, learning_rate * self.lower_weight_changes)
        self.lower_weight_changes = npy.array
        self.upper_layer_weights = npy.add(self.upper_layer_weights, learning_rate * self.upper_weight_changes)
        self.upper_weight_changes = npy.array

    def sigmoid(self, logistic):
        if self.sin == 1:
            return (2 / (1 + npy.exp(-2*logistic))) - 1
        else:
            return 1 / (1 + npy.exp(-logistic))

    def der_sigmoid(self, logistic):
        if self.sin == 1:
            return 1 - (npy.power(self.sigmoid(logistic), 2))
        else:
            return self.sigmoid(logistic) * (1 - self.sigmoid(logistic))

    def isSin(self, sin):
        self.sin = 1


def main():
    # XOR EXAMPLE PART ONE / TWO
    examples = npy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = npy.array([[0], [1], [1], [0]])
    NN = MLP(2,2,1)
    max_epochs = 1000000
    number_of_examples = 4

    print("--------RESULTS XOR--------")
    for i in range(0, number_of_examples):
        NN.forward(examples[i])
        print("BEFORE Output is " + str(NN.output) + " Target: " + str(targets[i]))

    print()

    for e in range(0, max_epochs):
        error = 0
        NN.forward(examples)

        error += NN.backwards(examples, targets)

        NN.update_weights(0.7)
        if e % 50000 == 0:
            print("Epochs are " + str(e) + " Error is " + str(error))
        if e == (max_epochs - 1):
            print()
            print("AFTER Output is " + str(NN.output[0]) + " Target: " + str(targets[0]))
            print("AFTER Output is " + str(NN.output[1]) + " Target: " + str(targets[1]))
            print("AFTER Output is " + str(NN.output[2]) + " Target: " + str(targets[2]))
            print("AFTER Output is " + str(NN.output[3]) + " Target: " + str(targets[3]))
            print("\nFINAL ERROR RATE: " + str(error))

    # 50 VECTORS
    examples_vectors = []
    for i in range (0, 50):
        examples_vectors.append([])
        examples_vectors[i].append(random.uniform(-1, 1))
        examples_vectors[i].append(random.uniform(-1, 1))
        examples_vectors[i].append(random.uniform(-1, 1))
        examples_vectors[i].append(random.uniform(-1, 1))
    targets_vectors = npy.array(npy.random.uniform(0,0, 50))
    targets_vectors = targets_vectors.reshape(50,1)

    for i in range(0, 50):
        temp = npy.sin(examples_vectors[i][0] + examples_vectors[i][1] + examples_vectors[i][2] + examples_vectors[i][3])
        targets_vectors[i] = temp

    training_examples = examples_vectors[:]
    training_targets = npy.copy(targets_vectors)

    index_to_delete = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    training_targets = npy.delete(training_targets, index_to_delete)

    for i in range(40, 50):
        del training_examples[40]

    training_examples = npy.asarray(training_examples)
    training_targets = training_targets.reshape(40,1)

    NN2 = MLP(4, 5, 1)
    NN2.isSin(1)
    max_epochs = 1000000
    print("--------RESULTS 50 VECTORS--------")
    for e in range(0, max_epochs):
        error = 0
        NN2.forward(training_examples)

        error += NN2.backwards(training_examples, training_targets)

        NN2.update_weights(0.007)
        if e % 50000 == 0:
            print("Epochs are " + str(e) + " Error is " + str(error))
        if e == (max_epochs - 1):
            print()
            print("\nFINAL TRAINING ERROR RATE: " + str(error))

    print("--------USING TEST DATA--------")
    for i in range(40, 50):
        NN2.forward(examples_vectors[i])
        print("Output is " + str(NN2.output[0]) + " Target: " + str(targets_vectors[i]))


if __name__ == "__main__":
    main()