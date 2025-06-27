import random
import numpy as np
from itertools import combinations
import simple_opponents
import your_agent
import rl_agent
import matplotlib.pyplot as plt

mean = 100
variance = 10000  # Large variance


class Game:
    def __init__(self, total_rounds):
        self.rounds_played = 0
        self.total_rounds = total_rounds
        self.current_amount = 0

    def isOver(self):
        return self.rounds_played >= self.total_rounds

    def prepare_round(self):
        self.current_amount = max(mean, np.random.normal(mean, np.sqrt(variance)))

    def play_round(self, left_agent, right_agent, remaining):
        self.rounds_played += 1
        self.prepare_round()
        left_decision = left_agent.decision(
            self.current_amount, remaining, left_agent.karma, right_agent.karma
        )
        assert left_decision in ["split", "steal"]
        right_decision = right_agent.decision(
            self.current_amount, remaining, right_agent.karma, left_agent.karma
        )
        assert right_decision in ["steal", "split"]
        decisions = np.array([left_decision, right_decision])

        if all(decisions == "steal"):
            left_reward = 0
            right_reward = 0
        elif all(decisions == "split"):
            left_reward = self.current_amount / 2
            right_reward = self.current_amount / 2
        elif left_decision == "steal":
            left_reward = self.current_amount
            right_reward = 0
        elif right_decision == "steal":
            right_reward = self.current_amount
            left_reward = 0

        left_agent.total_amount += left_reward
        right_agent.total_amount += right_reward

        left_agent.result(
            left_decision, right_decision, self.current_amount, left_reward
        )
        right_agent.result(
            right_decision, left_decision, self.current_amount, right_reward
        )

        if left_decision == "steal":
            left_agent.add_karma(-1)
        else:
            left_agent.add_karma(+1)

        if right_decision == "steal":
            right_agent.add_karma(-1)
        else:
            right_agent.add_karma(+1)


class Player:
    def __init__(self, agent):
        self.name = agent.get_name()
        self.agent = agent
        self.total_amount = 0
        self.last_decision = "none"
        self.karma = 0

    def reset_karma(self):
        self.karma = 0

    def add_karma(self, value):
        self.karma = min(max(self.karma + value, -5), 5)

    def decision(self, total_amount, rounds_played, your_karma, his_karma):
        self.last_decision = self.agent.decision(
            total_amount, rounds_played, your_karma, his_karma
        )
        return self.last_decision

    def result(self, your_action, his_action, total_possible, reward):
        self.agent.result(your_action, his_action, total_possible, reward)


# --------- AUTOMATIZAÇÃO DE CENÁRIOS E ANÁLISE GRÁFICA ---------

SCENARIOS = {
    "Allgame": [
        Player(simple_opponents.Splitter()),
        Player(simple_opponents.Stealer()),
        Player(simple_opponents.Randy()),
        Player(simple_opponents.Karmine()),
        Player(simple_opponents.Opportunist()),
        Player(simple_opponents.Pretender()),
        Player(your_agent.SmartRLAgent()),
        Player(rl_agent.RLAgent()),
    ],
    "Simple": [
        Player(simple_opponents.Karmine()),
        Player(simple_opponents.Karmine()),
        Player(rl_agent.RLAgent()),
        Player(your_agent.SmartRLAgent()),
    ],
    "Difficult": [
        Player(your_agent.SmartRLAgent()),
        Player(your_agent.SmartRLAgent()),
        Player(rl_agent.RLAgent()),
        Player(your_agent.SmartRLAgent()),
    ],
    "Very difficult": [
        Player(simple_opponents.Pretender()),
        Player(simple_opponents.Pretender()),
        Player(rl_agent.RLAgent()),
        Player(simple_opponents.Karmine()),
    ],
    "Karma-aware": [
        Player(simple_opponents.Karmine()),
        Player(simple_opponents.Karmine()),
        Player(rl_agent.RLAgent()),
        Player(simple_opponents.Stealer()),
    ],
    "Opportunists": [
        Player(simple_opponents.Opportunist()),
        Player(simple_opponents.Opportunist()),
        Player(rl_agent.RLAgent()),
        Player(your_agent.SmartRLAgent()),
    ],
    "3 Karmines": [
        Player(simple_opponents.Karmine()),
        Player(simple_opponents.Karmine()),
        Player(rl_agent.RLAgent()),
        Player(simple_opponents.Karmine()),
    ],
    "RL vs YourAgent": [
        Player(your_agent.SmartRLAgent()),
        Player(rl_agent.RLAgent()),
    ],
}


def run_scenario(agents, nrematches=10, nfullrounds=100):
    total_rounds = int(len(agents) * (len(agents) - 1) * nfullrounds * nrematches / 2)
    game = Game(total_rounds)
    for a in agents:
        a.total_amount = 0
        a.karma = 0
    score_history = {a.name: [0] for a in agents}
    while not game.isOver():
        random.shuffle(agents)
        for a in agents:
            a.reset_karma()
        for player1, player2 in combinations(agents, 2):
            for remaining in reversed(range(0, nrematches)):
                game.play_round(player1, player2, remaining)
                for a in agents:
                    score_history[a.name].append(a.total_amount)
    return score_history


def plot_results(all_histories):
    for scenario, history in all_histories.items():
        plt.figure(figsize=(10, 6))
        for name, scores in history.items():
            plt.plot(scores, label=name)
        plt.title(f"Evolução da pontuação - {scenario}")
        plt.xlabel("Iteração (rodada)")
        plt.ylabel("Pontuação acumulada")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    nrematches = 10
    nfullrounds = 100
    all_histories = {}

    for scenario_name, agent_list in SCENARIOS.items():
        agents = [
            type(a.agent)() if hasattr(a, "agent") else type(a)() for a in agent_list
        ]
        agents = [Player(a) for a in agents]
        print(f"\nRodando cenário: {scenario_name}")
        history = run_scenario(agents, nrematches, nfullrounds)
        all_histories[scenario_name] = history
        final_scores = {name: scores[-1] for name, scores in history.items()}
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        print("Ranking final:")
        for name, score in sorted_results:
            print(f"  {name}: {score:.2f}")
        print(f"Vencedor: {sorted_results[0][0]} com {sorted_results[0][1]:.2f}")

    plot_results(all_histories)


if __name__ == "__main__":
    main()
