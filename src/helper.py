from input import Input

def topological_sort(feed_dict):
    """
    Sort the neurons in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to that Neuron.

    Returns a list of sorted neurons.
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_Neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_Neuron`: A Neuron in the graph, should be the output Neuron (have no outgoing edges).
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output Neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_Neuron.value
