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

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_layers.
        self.gradients = {
            n: np.zeros_like(n.value) for n in self.inbound_layers}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_layers:
            # Get the partial of the cost with respect to this layer.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this layer's inputs.
            self.gradients[
                self.inbound_layers[0]] += np.dot(
                    grad_cost,
                    self.inbound_layers[1].value.T)
            # Set the partial of the loss with respect to this layer's weights.
            self.gradients[
                self.inbound_layers[1]] += np.dot(
                    self.inbound_layers[0].value.T,
                    grad_cost)
            # Set the partial of the loss with respect to this layer's bias.
            self.gradients[
                self.inbound_layers[2]] += np.sum(grad_cost,
                                                  axis=0,
                                                  keepdims=False)
