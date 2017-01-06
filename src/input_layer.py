from layer import Layer


class InputLayer(Layer):

    def __init__(self):
        Layer.__init__(self)

    def forward(self, value=None):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input Layer has no inputs so we refer to ourself
        # for the gradient
        self.gradients = {self: 0}
        for n in self.outbound_Layers:
            self.gradients[self] += n.gradients[self]
