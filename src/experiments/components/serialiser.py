import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

import pathlib
import numpy as np
import csv

class Serialiser():
    def __init__(self) -> None:
        self.rc = {
            "font.size": 12,
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "path.simplify": True
        }

        self.out_dir = pathlib.Path("figures")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def make_reward_figure(self, reward, step, title, reward_prefix, with_markers = False):
        sns.set_theme(font_scale=1)
        with sns.axes_style("whitegrid", self.rc):
            fig, axe = plt.subplots()
            if (with_markers):
                reward = np.array(reward)
                reward_mask = np.ma.masked_where(reward == 0, reward)
                axe.plot(step, reward, color="tab:blue", rasterized=True)
                axe.plot(step, reward_mask, marker="o", color="tab:blue", label=f"{reward_prefix} Reward", rasterized=True)
            else:
                axe.plot(step, reward, color="tab:blue", label=f"{reward_prefix} Reward", rasterized=True)
            axe.set_xlabel("Schritt")
            axe.set_ylabel(f"{reward_prefix} Reward")
            axe.set_title(title)

            return fig

    def make_reward_figure_scatter(self, reward, step, title, reward_prefix, with_markers = False):
        sns.set_theme(font_scale=1)
        with sns.axes_style("whitegrid", self.rc):
            fig, axe = plt.subplots()

            axe.scatter(step, reward, color="tab:blue", label=f"{reward_prefix} Reward", linewidths=0.75, rasterized=True)
            axe.set_xlabel("Schritt")
            axe.set_ylabel(f"{reward_prefix} Reward")
            axe.set_title(title)

            return fig
        
        
    def make_reward_figure_with_vertical(self, reward, step, title, reward_prefix, vertical_x_pos, vertical_label, with_markers = False):
        sns.set_theme(font_scale=1)
        with sns.axes_style("whitegrid", self.rc):
            fig, axe = plt.subplots()
            if (with_markers):
                reward = np.array(reward)
                reward_mask = np.ma.masked_where(reward == 0, reward)
                axe.plot(step, reward, color="tab:blue", rasterized=True)
                axe.plot(step, reward_mask, marker="o", color="tab:blue", label=f"{reward_prefix} Reward", rasterized=True)
            else:
                axe.plot(step, reward, color="tab:blue", label=f"{reward_prefix} Reward", rasterized=True)
            axe.axvline(vertical_x_pos, label=vertical_label, color="gray", linestyle="--")
            axe.set_xlabel("Schritt")
            axe.set_ylabel(f"{reward_prefix} Reward")
            axe.set_title(title)

            self._add_legend(axe)


            return fig

    def make_reward_figure_with_vertical_scatter(self, reward, step, title, reward_prefix, vertical_x_pos, vertical_label, with_markers = False):
        sns.set_theme(font_scale=1)
        with sns.axes_style("whitegrid", self.rc):
            fig, axe = plt.subplots()
            axe.scatter(step, reward, color="tab:blue", label=f"{reward_prefix} Reward", linewidths=0.75, rasterized=True)
            axe.axvline(vertical_x_pos, label=vertical_label, color="gray", linestyle="--")
            axe.set_xlabel("Schritt")
            axe.set_ylabel(f"{reward_prefix} Reward")
            axe.set_title(title)

            self._add_legend(axe)


            return fig

    def make_exploration_figure(self, cells, total_cells, step, title):
        sns.set_theme(font_scale=1)
        with sns.axes_style("whitegrid", self.rc):
            fig, axe = plt.subplots()
            axe.plot(step, cells, color="tab:blue", label="Besuchte Zellen", rasterized=True)
            axe.plot(step, total_cells, color="tab:orange", label="Alle Zellen", rasterized=True)
            axe.set_xlabel("Schritt")
            axe.set_ylabel("Besuchte Zellen")
            axe.set_title(title)

            self._add_legend(axe)
            return fig

    def _add_legend(self, axe: Axes):
        lines, labels = axe.get_legend_handles_labels()
        axe.legend(lines, labels)

    def print_figure(self, figure: Figure, filename: str, png = True):
        filetype = "pdf"
        if (png):
            filetype = "png"
        print_path = self.out_dir / f"{filename}.{filetype}"
        figure.savefig(print_path)
        plt.close(figure)
        print(f"Saved {print_path}")

    def write_figure_data_to_csv(self, figure: Figure, filename):
        axes = figure.get_axes()

        x_data = []
        y_data = []
        ax = axes[0]
        line = ax.get_lines()[0]
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        x_np_array = np.array(x_data)
        y_np_array = np.array(y_data)

        stacked_arrays = np.stack((x_np_array, y_np_array))

        print_path = self.out_dir / f"{filename}.csv"
        np.savetxt(print_path, stacked_arrays, delimiter=",")

    def read_data_from_csv(self, csv_path):
        stacked_data = np.genfromtxt(csv_path, delimiter=",")
        x_np_array = np.array(stacked_data[0])
        y_np_array = np.array(stacked_data[1])

        return x_np_array, y_np_array
