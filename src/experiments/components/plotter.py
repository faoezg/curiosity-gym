import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

import pathlib

class Plotter():
    def __init__(self) -> None:
        self.rc = {
            "font.size": 12,
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }

        self.out_dir = pathlib.Path("figures")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def make_reward_figure(self, reward, step, title, reward_prefix, with_marker = False):

        with sns.axes_style("whitegrid", self.rc):
            fig, axe = plt.subplots()
            if (with_marker):
                axe.plot(step, reward, marker="o", color="tab:blue")
            else:
                axe.plot(step, reward, color="tab:blue")
            axe.set_xlabel("Schritt")
            axe.set_ylabel(f"{reward_prefix} Reward")
            axe.set_title(title)

            return fig
        
    def make_reward_figure_with_vertical(self, reward, step, title, reward_prefix, vertical_x_pos, vertical_label, with_marker = False):

        with sns.axes_style("whitegrid", self.rc):
            fig, axe = plt.subplots()
            if (with_marker):
                axe.plot(step, reward, marker="o", color="tab:blue", label=f"{reward_prefix} Reward")
            else:
                axe.plot(step, reward, color="tab:blue", label=f"{reward_prefix} Reward")
            axe.axvline(vertical_x_pos, label=vertical_label, color="gray", linestyle="--")
            axe.set_xlabel("Schritt")
            axe.set_ylabel(f"{reward_prefix} Reward")
            axe.set_title(title)

            self._add_legend(axe)

            return fig



    def make_exploration_figure(self, cells, total_cells, step, title):

        with sns.axes_style("whitegrid", self.rc):
            fig, axe = plt.subplots()
            axe.plot(step, cells, color="tab:blue", label="Besuchte Zellen")
            axe.plot(step, total_cells, color="tab:orange", label="Alle Zellen")
            axe.set_xlabel("Schritt")
            axe.set_ylabel("Besuchte Zellen")
            axe.set_title(title)

            self._add_legend(axe)
            return fig

    def _add_legend(self, axe: Axes):
        lines, labels = axe.get_legend_handles_labels()
        axe.legend(lines, labels)

    def print_figure(self, figure: Figure, filename: str):
        print_path = self.out_dir / f"{filename}.pdf"
        figure.savefig(print_path)
        plt.close(figure)
        print(f"Saved {print_path}")
