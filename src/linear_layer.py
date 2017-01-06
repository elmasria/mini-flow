import numpy as np
from layer import Layer


class LinearLayer(Layer):

    def __init__(self, inbound_layer, weights, bias):
        # Notice the ordering of the input layers passed to the
        # Layer constructor.
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        """
        Set the value of this layer to the linear transform output.

        Your code goes here!
        """
        inputs = self.inbound_layers[0].value
        weights = self.inbound_layers[1].value
        bias = self.inbound_layers[2].value
        self.value = np.dot(inputs, weights) + bias
