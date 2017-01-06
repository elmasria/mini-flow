from neuron import Neuron


class Input(Neuron):

    def __init__(self):
        # An Input neuron has no inbound neurons,
        # so no need to pass anything to the Neuron instantiator.
        Neuron.__init__(self)

    # NOTE: Input neuron is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other neuron implementations should get the value
    # of the previous neuron from self.inbound_neurons
    #
    # Example:
    # val0 = self.inbound_neurons[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value:
            self.value = value
