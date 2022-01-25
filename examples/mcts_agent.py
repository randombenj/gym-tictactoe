#!/usr/bin/env python
import copy
import math
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


class Node:
    def __init__(
        self,
        env: TicTacToeEnv,
        action: int,
        parent: "Node" = None,
        to_expand: T.List[int] = [],
    ):
        self.action = action
        self.to_expand = to_expand
        self.parent = parent
        self.visits = 1
        self.reward = None
        self.children: T.List["Node"] = []
        self.payoff: np.array = None
        self.env = env

    def select(self) -> "Node":
        """Select next node"""
        node = self
        while len(node.to_expand) == 0:
            if not node.is_leaf():
                node = max(node.children, key=lambda n: n.ucb1())
            else:
                return node
        return node

    def expand(self) -> "Node":
        """Expands the current node and returns the added node"""
        if not self.to_expand:
            return self

        action = self.to_expand.pop()
        env = copy.deepcopy(self.env)
        _, reward, done, _ = env.step(action)

        node = Node(
            action=action,
            parent=self,
            env=env,
            to_expand=copy.deepcopy(self.to_expand),
        )

        if done:
            node.reward = reward

        self.children.append(node)
        return node

    def simulate(self):
        """Simulate game to the end"""
        reward = None
        while not self.env.done:
            _, reward, _, _ = self.env.step(random.choice(self.env.available_actions()))

        if reward:
            self.reward = reward

    def backpropagate(self, mark: str):
        node = self
        while node.parent:
            node.calculate_payoff(mark)
            node.visits += 1
            node = node.parent

    def is_leaf(self) -> bool:
        return len(self.children) == 0 or self.env.done

    def ucb1(self, C: int = 2) -> T.List[float]:
        if not self.parent:
            raise Exception("Can't calculate UCB1 of root node")

        return self.payoff[0] / self.visits + C * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def calculate_payoff(self, mark: str):
        """Calculates the nodes payoff"""
        if self.reward == NO_REWARD:
            self.payoff = np.array([0.5, 0.5])
        else:
            if (mark == "X" and self.reward == X_REWARD) or (
                mark == "O" and self.reward == O_REWARD
            ):
                self.payoff = np.array([1.0, 0.0])
            else:
                self.payoff = np.array([0.0, 1.0])

        for child in self.children:
            self.payoff += child.payoff

    def __repr__(self) -> str:
        return f"Node({self.action}, {self.payoff=}, {self.visits=})"

    def __str__(self, level=0) -> str:
        ret = "    " * level + repr(self) + "\n"
        for i, child in enumerate(self.children):
            ret += child.__str__(level + 1)
        return ret


class MCTSAgent:
    def __init__(self, mark: str, ucb1_c: int = 2, mcts_depth: int = 200):
        self.mark = mark
        self._ucb1_c = ucb1_c
        self._mcts_depth = mcts_depth

    def act(
        self,
        state: T.Tuple[T.Tuple[int], str],
        legal_actions: T.List[int],
        env: TicTacToeEnv,
    ) -> int:
        """Returns the action to play"""
        root_node = Node(
            action=np.argmax(state[0]),
            to_expand=legal_actions,
            env=copy.deepcopy(env),
        )

        for _ in range(self._mcts_depth):
            node = root_node.select()
            node = node.expand()
            node.simulate()
            node.backpropagate(self.mark)

        print(root_node)

        # -- exploit by max visits
        return max(root_node.children, key=lambda n: n.visits).action


class RandomAgent:
    def __init__(self, mark: str):
        self.mark = mark

    def act(self, _, legal_actions: T.List[int], *args):
        return random.choice(legal_actions)


def play(max_episode=10):
    start_mark = "O"
    env = TicTacToeEnv()

    agents = [RandomAgent("O"), MCTSAgent("X")]

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
