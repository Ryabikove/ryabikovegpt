import numpy as np

def sygmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Random init weights:\n",synaptic_weights)

for i in range(2):
    input_layer = training_inputs
    outputs = sygmoid(np.dot(input_layer, synaptic_weights))

    err = training_outputs - outputs
    adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustments

print("Result:\n",outputs)

new_inputs = np.array([[1,1,0],
                       [1,0,0],
                       [0,0,0]])
outputs = sygmoid(np.dot(new_inputs, synaptic_weights))

print("new situantions:\n",outputs)