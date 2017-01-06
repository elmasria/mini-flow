import numpy as np


class Layer:

    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.value = None
        self.outbound_layers = []
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward():
        raise NotImplementedError

    def backward():
        raise NotImplementedError
