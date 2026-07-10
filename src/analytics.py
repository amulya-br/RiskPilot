import matplotlib

# Use non-GUI backend for Flask
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os

LOG_FILE = "logs/risk_log.csv"
GRAPH_FILE = "static/risk_graph.png"


def generate_graph():

    if not os.path.exists(LOG_FILE):
        return

    try:

        df = pd.read_csv(LOG_FILE)

        if len(df) < 2:
            return

        # Keep only latest 30 events
        df = df.tail(30)

        risk_map = {
            "SAFE": 1,
            "MODERATE": 2,
            "HIGH": 3,
            "CRITICAL": 4
        }

        df["RiskValue"] = df["Risk"].map(risk_map)

        # -------- Professional Dark Theme --------

        plt.close("all")
        plt.style.use("dark_background")

        fig, ax = plt.subplots(figsize=(11, 4.8))

        fig.patch.set_facecolor("#1c1c1c")
        ax.set_facecolor("#1c1c1c")

        # Line
        ax.plot(
            range(len(df)),
            df["RiskValue"],
            color="#00F5FF",
            linewidth=3,
            marker="o",
            markersize=8,
            markerfacecolor="#00ff66",
            markeredgecolor="white"
        )

        # Fill below line
        ax.fill_between(
            range(len(df)),
            df["RiskValue"],
            color="#00F5FF",
            alpha=0.20
        )

        # Risk labels
        ax.set_ylim(0.8, 4.2)

        ax.set_yticks([1, 2, 3, 4])

        ax.set_yticklabels(
            ["SAFE", "MODERATE", "HIGH", "CRITICAL"],
            fontsize=11,
            color="white"
        )

        # Titles
        ax.set_title(
            "Live Risk Trend",
            fontsize=18,
            color="#00F5FF",
            fontweight="bold"
        )

        ax.set_xlabel(
            "Recent Events",
            fontsize=12,
            color="white"
        )

        ax.set_ylabel(
            "Risk Level",
            fontsize=12,
            color="white"
        )

        # Grid
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.7,
            alpha=0.35,
            color="white"
        )

        # Cyan border
        for spine in ax.spines.values():
            spine.set_color("#00F5FF")
            spine.set_linewidth(1.2)

        ax.tick_params(colors="white")

        plt.tight_layout()

        plt.savefig(
            GRAPH_FILE,
            dpi=180,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight"
        )

        plt.close(fig)

    except Exception as e:
        print("Graph Error:", e)