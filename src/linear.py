from neuron import Neuron


class Linear(Neuron):

    def __init__(self, inputs, weights, bias):
        Neuron.__init__(self, inputs)

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other neurons.
        # The weight and bias values are stored within the
        # respective neurons.
        self.weights = weights
        self.bias = bias

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        self.value = self.bias.value
        for w, x in zip(self.weights, self.inbound_neurons):
            self.value += w.value * x.value
