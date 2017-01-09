import numpy as np


class Layer:

    def __init__(self, inbound_layers=[]):
        # A list of layers with edges into this layer.
        self.inbound_layers = inbound_layers
        # The eventual value of this layer. Set by running
        # the forward() method.
        self.value = None
        # A list of layers that this layer outputs to.
        self.outbound_layers = []
        # New property! Keys are the inputs to this layer and
        # their values are the partials of this layer with
        # respect to that input.
        self.gradients = {}
        # Sets this layer as an outbound layer for all of
        # this layer's inputs.
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward():
        raise NotImplementedError

    def backward():
        raise NotImplementedError
