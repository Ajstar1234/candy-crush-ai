import csv
import math


def compare_agents(agent_a, agent_b, results):
    a_results = {
        int(row["seed"]): row["won"] == "True"
        for row in results
        if row["agent"] == agent_a
    }

    b_results = {
        int(row["seed"]): row["won"] == "True"
        for row in results
        if row["agent"] == agent_b
    }

    if not a_results or a_results.keys() != b_results.keys():
        raise ValueError("Agents must have results for the same seeds.")

    both_win = 0
    a_only = 0
    b_only = 0
    neither_win = 0

    for seed in a_results:
        a_won = a_results[seed]
        b_won = b_results[seed]

        if a_won and b_won:
            both_win += 1
        elif a_won:
            a_only += 1
        elif b_won:
            b_only += 1
        else:
            neither_win += 1

    print(f"\n{agent_a} vs {agent_b}")
    print(f"Both win: {both_win}")
    print(f"Only {agent_a} wins: {a_only}")
    print(f"Only {agent_b} wins: {b_only}")
    print(f"Neither wins: {neither_win}")
    print(f"Net additional wins: {a_only - b_only}")

    discordant = a_only + b_only

    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(a_only, b_only)

        p_value = min(
            1.0,
            2 * sum(
                math.comb(discordant, k)
                for k in range(smaller + 1)
            ) / (2 ** discordant)
        )

    print(f"Exact McNemar p-value: {p_value:.6g}")


if __name__ == "__main__":
    with open(
        "benchmarks/all_agents_results.csv",
        newline="",
        encoding="utf-8"
    ) as file:
        results = list(csv.DictReader(file))

    for opponent in [
        "Random",
        "Greedy",
        "Objective-Aware",
        "Supervised ML"
    ]:
        compare_agents("DQN - 50 Episodes", opponent, results)