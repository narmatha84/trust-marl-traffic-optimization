
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

filename = "jcmm_comment15_learning_e6.csv"

data = defaultdict(list)

with open(filename, newline="") as f:
    for row in csv.DictReader(f):
        episode = int(row["episode"])
        data[episode].append(row)

episodes = sorted(data)

# Three independent, publication-ready figures.
metrics = [
    ("reward", "Mean training reward", "reward"),
    ("wait", "Mean waiting time (s)", "waiting_time"),
    ("speed", "Mean vehicle speed (m/s)", "speed")
]

for key, ylabel, output in metrics:
    means = []
    deviations = []

    for ep in episodes:
        values = np.array(
            [float(row[key]) for row in data[ep]]
        )
        means.append(values.mean())
        deviations.append(values.std(ddof=1))

    means = np.array(means)
    deviations = np.array(deviations)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        episodes, means,
        marker="o", linewidth=2,
        label="Corrected MAPPO"
    )

    ax.fill_between(
        episodes,
        means - deviations,
        means + deviations,
        alpha=0.2,
        label="±1 standard deviation"
    )

    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_xticks(episodes)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        f"jcmm_comment16_{output}.png",
        dpi=300
    )
    fig.savefig(
        f"jcmm_comment16_{output}.pdf"
    )

    plt.close(fig)

    print(f"\n{ylabel}")
    for ep, mean, sd in zip(
        episodes, means, deviations
    ):
        print(
            f"Episode {ep}: "
            f"{mean:.4f} ± {sd:.4f}"
        )

print("\nAll learning-curve figures saved.")
