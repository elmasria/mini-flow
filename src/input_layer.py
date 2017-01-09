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
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1
