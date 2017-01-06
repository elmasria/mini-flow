from input import Input
from add import Add
from mull import Mull
from linear import Linear
from helper import *

import logging

logging.basicConfig(filename='text.log', level=logging.DEBUG)

# Addition
x, y, z = Input(), Input(), Input()

f = Add(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

result = "{} + {} + {} = {} (according to miniflow)".format(
    feed_dict[x], feed_dict[y], feed_dict[z], output)

logging.info(result)
print(result)

# Multiplication
x, y, z = Input(), Input(), Input()

f = Mull(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

result = "{} * {} * {} = {} (according to miniflow)".format(
    feed_dict[x], feed_dict[y], feed_dict[z], output)
logging.info(result)
print(result)

# Linear
x, y, z = Input(), Input(), Input()
inputs = [x, y, z]

weight_x, weight_y, weight_z = Input(), Input(), Input()
weights = [weight_x, weight_y, weight_z]

bias = Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    x: 6,
    y: 14,
    z: 3,
    weight_x: 0.5,
    weight_y: 0.25,
    weight_z: 1.4,
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
logging.info(output)
print(output)

# Layer
import numpy as np
from input_layer import InputLayer
from linear_layer import LinearLayer
from helper_layer import *

print("Layer Linear Transform")

inputs, weights, bias = InputLayer(), InputLayer(), InputLayer()
f = LinearLayer(inputs, weights, bias)


x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: x, weights: w, bias: b}

graph = topological_sort_layer(feed_dict)
output = forward_pass_layer(f, graph)
logging.info(output)
print(output)

# Sigmoid
from sigmoid import Sigmoid

logging.info("Sigmoid")
print("Sigmoid")

inputs, weights, bias = InputLayer(), InputLayer(), InputLayer()
f = LinearLayer(inputs, weights, bias)
g = Sigmoid(f)

x = np.array([[-1., -2.], [-1, -2]])
w = np.array([[2., -3], [2., -3]])
b = np.array([-3., -5])

feed_dict = {inputs: x, weights: w, bias: b}

graph = topological_sort_layer(feed_dict)
output = forward_pass_layer(g, graph)

logging.info(output)
print(output)