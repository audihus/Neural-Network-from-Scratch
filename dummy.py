import math
import numpy as np

class NeuralNet:
    def __init__(self, hidden_neuron, input_neuron=2, output_neuron=2):
        self.input_neuron = input_neuron
        self.hidden_neuron = hidden_neuron
        self.output_neuron = output_neuron

        # self.w1 = np.array([[0.1, 0.3],[0.2, 0.4]]) #[[ke h1],[ke h2]]
        # self.b1 = 0.25

        # self.w2 = np.array([[0.5,0.6 ],[0.7, 0.8]]) #[[ke o1], [ke o2]]
        # self.b2 = 0.35

        # Bobot dari input -> hidden (shape: hidden x input)
        self.W1 = np.random.rand(self.hidden_neuron, self.input_neuron)
        # Bias untuk hidden (shape: hidden x 1)
        self.b1 = np.random.rand(self.hidden_neuron, 1)
        # Bobot dari hidden -> output (shape: output x hidden)
        self.W2 = np.random.rand(self.output_neuron, self.hidden_neuron)
        # Bias untuk output (shape: output x 1)
        self.b2 = np.random.rand(self.output_neuron, 1)

    def activation_function(self, z):
        e = math.e
        result = 1/(1 + (e**(-z)))
        return result
    
    def forward(self, x):
        # Hidden Layer
        z1 = np.dot(self.w1, x) + self.b1
        a1 = self.activation_function(z1)

        #Output Layer
        z2 = np.dot(self.w2, a1) + self.b2
        a2 = self.activation_function(z2)

        return a1, a2
    
    def mse(self, y_pred, y_true):
        total = 0.5 * np.mean((y_pred - y_true) ** 2)

        return total

x = np.array([[0.1], [0.5]])
y_true = np.array([[0.05], [0.95]])

nn = NeuralNet(2)
hidden_out, output_out = nn.forward(x)
# print("Hidden output:", hidden_out)
# print("Output output:", output_out)

err = nn.mse(output_out, y_true)
# print("Error:", err)