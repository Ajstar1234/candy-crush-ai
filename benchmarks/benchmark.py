import pandas as pd
import matplotlib.pyplot as plt

from simulation import run_simulations

difficulties = {
    "Easy": {"max_turns": 8, "target_candy_count": 10},
    "Medium": {"max_turns": 6, "target_candy_count": 12},
    "Hard": {"max_turns": 5, "target_candy_count": 15}
}

all_results = []

for difficulty, settings in difficulties.items():

    greedy_stats = run_simulations(
        num_games=100,
        max_turns=settings["max_turns"],
        target_candy_count=settings["target_candy_count"],
        objective_aware=False
    )

    objective_stats = run_simulations(
        num_games=100,
        max_turns=settings["max_turns"],
        target_candy_count=settings["target_candy_count"],
        objective_aware=True
    )

    greedy_df = pd.DataFrame(greedy_stats["results"])
    objective_df = pd.DataFrame(objective_stats["results"])

    greedy_df["difficulty"] = difficulty
    objective_df["difficulty"] = difficulty

    greedy_df["agent"] = "Greedy"
    objective_df["agent"] = "Objective-Aware"

    greedy_df["game_id"] = range(len(greedy_df))
    objective_df["game_id"] = range(len(objective_df))

    all_results.append(greedy_df)
    all_results.append(objective_df)    

combined_df = pd.concat(
    all_results,
    ignore_index=True
)

print(combined_df.head())
print(f"Total simulation records: {len(combined_df)}")

combined_df.to_csv("benchmarks/benchmark_results.csv", index=False)

summary = combined_df.groupby(["difficulty", "agent"]).agg({
    "score": "mean",
    "won": "mean",
    "turns_used": "mean",
    "candies_collected": "mean"
})

summary = summary.rename(columns={
    "score": "average_score",
    "won": "win_rate",
    "turns_used": "average_turns",
    "candies_collected": "average_candies"
})

summary["win_rate"] = summary["win_rate"] * 100

summary = summary.sort_values(by="win_rate", ascending=False)

winning_games = combined_df[combined_df["won"] == True]

average_winning_turns = winning_games.groupby(
    ["difficulty", "agent"]
)["turns_used"].mean()

summary["average_winning_turns"] = average_winning_turns

summary.reset_index().to_csv("benchmarks/benchmark_summary.csv", index=False)

win_rate_data = summary["win_rate"]

win_rate_table = win_rate_data.unstack("agent")

win_rate_table = win_rate_table.reindex(
    ["Easy", "Medium", "Hard"]
)

ax = win_rate_table.plot(kind="bar")

plt.title("AI Agent Win Rates by Difficulty")
plt.xlabel("Difficulty")
plt.ylabel("Win Rate (%)")
plt.ylim(0,105)
plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3)

plt.savefig("benchmarks/agent_win_rates.png", dpi=300, bbox_inches="tight")

plt.show()

average_turns_data = summary["average_turns"]

average_turns_table = average_turns_data.unstack("agent")

average_turns_table = average_turns_table.reindex(
    ["Easy", "Medium", "Hard"]
)

ax = average_turns_table.plot(kind="bar")

plt.title("AI Agent Average Turns by Difficulty")
plt.xlabel("Difficulty")
plt.ylabel("Average Turns")
plt.ylim(0, 9)
plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=3)

plt.savefig("benchmarks/agent_average_turns.png", dpi=300, bbox_inches="tight")

plt.show()

comparison = combined_df.pivot(
    index=["difficulty", "game_id"],
    columns="agent",
    values="won"
)

objective_only_wins = comparison[
    (comparison["Greedy"] == False) &
    (comparison["Objective-Aware"] == True)
]

print(len(objective_only_wins))

greedy_only_wins = comparison[
    (comparison["Greedy"] == True) &
    (comparison["Objective-Aware"] == False)
]

print(len(greedy_only_wins))

net_improvement = len(objective_only_wins) - len(greedy_only_wins)

print(f"Net additional wins: {net_improvement}")

print(summary.head())

print(summary.round(2))
print(f"Total records: {len(combined_df)}")