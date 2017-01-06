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

# Cost
from mse import MSE

logging.info("Cost")
print("Cost")

y, a = InputLayer(), InputLayer()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort_layer(feed_dict)
# forward pass
forward_pass_layer_graph(graph)

logging.info(cost.value)
print(cost.value)


# Gradient Descent
import random
def f(x):
    """
    Quadratic function.

    It's easy to see the minimum value of the function
    is 5 when is x=0.
    """
    return x**2 + 5


def df(x):
    """
    Derivative of `f` with respect to `x`.
    """
    return 2*x


# Random number better 0 and 10,000. Feel free to set x whatever you like.
x = random.randint(0, 10000)
# TODO: Set the learning rate
learning_rate = 0.1
epochs = 100

for i in range(epochs+1):
    cost = f(x)
    gradx = df(x)
    print("EPOCH {}: Cost = {:.3f}, x = {:.3f}".format(i, cost, gradx))
    x = gradient_descent_update(x, gradx, learning_rate)

