class Neuron:

    def __init__(self, inbound_neurons=[]):
        # Neuron from which this Neuron receives values
        self.inbound_neurons = inbound_neurons
        # Neuron to which this Neuron passes values
        self.outbound_neurons = []
        # A calculated value
        self.value = None
        # For each inbound Neuron here, add this Neuron as an outbound Neuron
        # there.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_neurons` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Backward propagation.

        You'll compute this later.
        """
        raise NotImplemented
