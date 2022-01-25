#!/usr/bin/env python
import random

import numpy as np

import typing as T

from gym_tictactoe.env import (
    TicTacToeEnv,
    agent_by_mark,
    X_REWARD,
    O_REWARD,
    NO_REWARD,
    next_mark,
)


class Layer:
    def __init__(self):
        self.input: np.ndarray = None
        self.output: np.ndarray = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input: np.ndarray):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error: np.ndarray, learning_rate: float):
        raise NotImplementedError


# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(
        self,
        activation: T.Callable[[np.ndarray], np.ndarray],
        activation_prime: T.Callable[[np.ndarray], np.ndarray],
    ):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data: np.ndarray):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error: np.ndarray, learning_rate: float):
        return self.activation_prime(self.input) * output_error


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.greater(x, 0).astype(int)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.size


class Linear(Layer):
    def __init__(self, input_size: int, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    def backward_propagation(
        self, output_error: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error


class Network:
    def __init__(self):
        self.layers: T.List[Layer] = []
        self.loss: T.Callable[[np.ndarray], np.ndarray] = None
        self.loss_prime: T.Callable[[np.ndarray], np.ndarray] = None

    # add layer to network
    def add(self, layer: Layer):
        self.layers.append(layer)

    # set loss to use
    def use(
        self,
        loss: T.Callable[[np.ndarray], np.ndarray],
        loss_prime: T.Callable[[np.ndarray], np.ndarray],
    ):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data: np.ndarray):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 1e-2,
    ):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print("epoch %d/%d   error=%f" % (i + 1, epochs, err))


class MLP:
    """A simple mlp"""

    def __init__(self):
        self.net = Network()
        self.net.add(Linear(2, 3))
        self.net.add(ActivationLayer(activation=relu, activation_prime=relu_derivative))
        self.net.add(Linear(3, 1))
        self.net.add(ActivationLayer(activation=tanh, activation_prime=tanh_prime))

        self.net.use(loss=mse, loss_prime=mse_prime)

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.net.predict(input)

    def fit(self):
        x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

        self.net.fit(x_train, y_train, learning_rate=0.1)


class NNAgent:
    def __init__(self, mark: str):
        self.mark = mark

        self.net = MLP()
        self.net.fit()

    def act(self, state: T.Tuple[T.Tuple[int], str], legal_actions: T.List[int]) -> int:
        """Returns the action to play"""


class RandomAgent:
    def __init__(self, mark: str):
        self.mark = mark

    def act(self, _, legal_actions: T.List[int]):
        return random.choice(legal_actions)


def play(max_episode=10):
    start_mark = "O"
    env = TicTacToeEnv()

    agents = [RandomAgent("O"), NNAgent("X")]

    for _ in range(max_episode):
        env.set_start_mark(start_mark)
        state = env.reset()
        while not env.done:
            _, mark = state
            env.show_turn(True, mark)

            agent = agent_by_mark(agents, mark)
            legal_actions = env.available_actions()
            action = agent.act(state, legal_actions, env)
            state, reward, _, _ = env.step(action)
            env.render()

        env.show_result(True, mark, reward)

        # rotate start
        start_mark = next_mark(start_mark)


if __name__ == "__main__":
    play()
