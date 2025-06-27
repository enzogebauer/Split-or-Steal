import random
from collections import defaultdict, deque
import numpy as np


class SmartRLAgent:
    def __init__(self):
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.5  # começa explorando mais
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # decai a cada decisão
        self.Q = defaultdict(lambda: [0.0, 0.0])
        self.action_list = ["split", "steal"]
        self.last_opponent_actions = deque(
            [0, 0], maxlen=2
        )  # histórico das 2 últimas ações do oponente
        self.last_your_actions = deque(
            [0, 0], maxlen=2
        )  # histórico das 2 últimas suas ações
        self.last_round = False
        self.current_input = None
        self.current_output = None
        self.mean_amount = 100  # valor médio do prêmio, para normalizar

    def get_name(self):
        return "SmartRL+"

    def discretize_amount(self, amount):
        # Discretiza o valor do prêmio em 3 faixas: baixo, médio, alto
        if amount < 80:
            return 0
        elif amount < 120:
            return 1
        else:
            return 2

    def extract_rl_state(self, state):
        # Estado: (rodadas_restantes, seu_karma, karma_oponente, últimas ações, faixa_prêmio)
        return (
            state[1],  # rounds_left
            np.sign(state[2]),  # seu karma
            np.sign(state[3]),  # karma do oponente
            tuple(state[4]),  # últimas ações do oponente
            tuple(state[5]),  # suas últimas ações
            state[6],  # faixa do prêmio
        )

    def choose_action(self, state):
        state = self.extract_rl_state(state)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_list)
        else:
            return self.action_list[np.argmax(self.Q[state])]

    def update_qtable(self, state, action, reward, next_state):
        alp = self.alpha
        gam = self.gamma
        action_index = self.action_list.index(action)
        state = self.extract_rl_state(state)
        next_state = self.extract_rl_state(next_state)
        self.Q[state][action_index] = (1 - alp) * self.Q[state][action_index] + alp * (
            reward + gam * np.max(self.Q[next_state])
        )

    def decision(self, amount, rounds_left, your_karma, his_karma):
        # Epsilon decrescente
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        amount_bin = self.discretize_amount(amount)
        # Estado inclui histórico das últimas 2 ações do oponente e suas
        novel_input = (
            amount,
            rounds_left,
            your_karma,
            his_karma,
            list(self.last_opponent_actions),
            list(self.last_your_actions),
            amount_bin,
        )
        if self.current_input is not None:
            self.update_qtable(
                self.current_input,
                self.current_output[0],
                self.current_output[-1],
                novel_input,
            )
        self.current_input = novel_input
        action = self.choose_action(self.current_input)
        # Atualiza histórico
        self.last_your_actions.append(1 if action == "steal" else -1)
        return action

    def result(self, your_action, his_action, total_possible, reward):
        # Atualiza histórico do oponente
        self.last_opponent_actions.append(1 if his_action == "steal" else -1)
        # Penalidade para karma negativo
        karma_penalty = -0.2 if self.current_input[2] < 0 else 0
        # Recompensa baseada no valor real ganho, normalizada
        if your_action == "steal" and his_action == "steal":
            reward = 0 + karma_penalty
        elif your_action == "steal" and his_action == "split":
            reward = (total_possible / self.mean_amount) + karma_penalty
        elif your_action == "split" and his_action == "split":
            reward = (
                (total_possible / self.mean_amount) * 0.6 + 0.2 + karma_penalty
            )  # bônus para cooperação
        elif your_action == "split" and his_action == "steal":
            reward = -0.5 + karma_penalty
        self.current_output = (your_action, his_action, total_possible, reward)
        # Zera histórico se for última rodada
        if self.current_input and self.current_input[1] == 0:
            self.last_opponent_actions = deque([0, 0], maxlen=2)
            self.last_your_actions = deque([0, 0], maxlen=2)
