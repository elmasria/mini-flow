from neuron import Neuron


class Add(Neuron):
    # You may need to change this...
    def __init__(self, *inputs):
        Neuron.__init__(self, inputs)

    def forward(self):
        """
        For reference, here's the old way from the last
        quiz. You'll want to write code here.
        """
        self.value = 0 if self.value == None else self.value
        for neuron in self.inbound_neurons:
            self.value += neuron.value