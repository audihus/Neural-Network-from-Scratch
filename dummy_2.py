import math
import numpy as np

class NeuralNet:
    def __init__(self, input_neuron=2, hidden_neuron=2, output_neuron=2):
        self.input_neuron = input_neuron
        self.hidden_neuron = hidden_neuron
        self.output_neuron = output_neuron

        self.bias_hidden = 0.25
        self.bias_output = 0.35

        self.w_layer1 = [0.1, 0.2, 0.3, 0.4]
        self.w_output = [0.5, 0.7, 0.6, 0.8]

    def activation_function(self, z):
        e = math.e
        result = 1/(1 + (e**(-z)))
        return result
    
    def forward_hidden(self, inputs):
        sumh1 = self.bias_hidden + self.w_layer1[0]*inputs[0] + self.w_layer1[2]*inputs[1]
        sumh2 = self.bias_hidden + self.w_layer1[1]*inputs[0] + self.w_layer1[3]*inputs[1]
        outputh1 = self.activation_function(sumh1)
        outputh2 = self.activation_function(sumh2)
        return [outputh1, outputh2]
    
    def forward_output(self, inputs):
        sumo1 = self.bias_output + self.w_output[0]*inputs[0] + self.w_output[2]*inputs[1]
        sumo2 = self.bias_output + self.w_output[1]*inputs[0] + self.w_output[3]*inputs[1]
        outputo1 = self.activation_function(sumo1)
        outputo2 = self.activation_function(sumo2)
        return [outputo1, outputo2]
    
    def forward(self, x):
        h = self.forward_hidden(x)
        o = self.forward_output(x)
        return o
    
    def mse(self, predict, truth):
        total = 0.0
        length = len(predict)
        for i in range(length):
            total += 0.5 * ((predict[i] - truth[i])**2)

        return total

attribute = [0.1, 0.5]
truth = [0.05, 0.95]

nn = NeuralNet()
h_layer = nn.forward_hidden(attribute)
print(f"Hasil hidden layer: {h_layer}")

o_layer = nn.forward_output(attribute)
print(f"Hasil output layer: {o_layer}")

error = nn.mse(o_layer, truth)
print(f"Error total: {error}")