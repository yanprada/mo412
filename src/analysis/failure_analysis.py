import pickle
import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FailureAnalysis:
    def __init__(self, pickle_path: str, base_unit: float = 0.6):
        """
        Initialize the FailureAnalysis class with a graph from a pickle file.

        Args:
            pickle_path (str): Path to the pickle file containing the graph.
        """
        self.pickle_path = pickle_path
        self.G = self._load_graph()
        self.base_unit = base_unit

    def _load_graph(self):
        """Load graph from pickle file and extract the largest connected component."""
        with open(self.pickle_path, "rb") as f:
            graph = pickle.load(f)

        # Get the largest connected component
        if graph.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(graph), key=len)
            return graph.subgraph(largest_cc).copy()
        else:
            return graph

    def simulate_failure(self, failure_type: str, num_steps=20):
        """
        Simula a remoção aleatória de nós e rastreia o tamanho do componente gigante.

        Args:
            failure_type (str): Type of failure - "degree" or "random".
            num_steps (int): Número de etapas de remoção a serem simuladas.

        Returns:
            tuple: Duas listas, uma para a fração de nós removidos (f) e outra
                   para a fração do componente gigante (P).
        """
        print(f"Simulando falhas do tipo: {failure_type}")
        G_copy = self.G.copy()
        num_nodes = G_copy.number_of_nodes()
        if num_nodes == 0:
            return [0], [0]

        if failure_type == "degree":
            degrees = dict(G_copy.degree())
            nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)
        else:  # "random"
            nodes_to_remove = list(self.G.nodes())
            random.shuffle(nodes_to_remove)

        f_values = [0]

        if G_copy.number_of_nodes() > 0:
            largest_component = max(nx.connected_components(G_copy), key=len)
            p_values = [len(largest_component) / num_nodes]
        else:
            p_values = [0]

        step_size = num_nodes // num_steps
        if step_size == 0:
            step_size = 1

        for i in range(step_size, num_nodes + 1, step_size):
            # Remove uma fração de nós
            nodes_to_remove_this_step = nodes_to_remove[i - step_size : i]
            G_copy.remove_nodes_from(nodes_to_remove_this_step)

            # Calcula a fração de nós removidos
            f = i / num_nodes

            if G_copy.number_of_nodes() == 0:
                p = 0
            else:
                largest_component = max(nx.connected_components(G_copy), key=len)
                p = len(largest_component) / num_nodes

            f_values.append(f)
            p_values.append(p)

        return f_values, p_values

    def plot_failures(
        self, f_and_p_random, f_and_p_degree, tension, name="CPFL Paulista"
    ):
        """Plota os resultados das simulações de falhas."""

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(16 * self.base_unit, 9 * self.base_unit))

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.set_xlabel("Percentage of Nodes Removed (f)", fontsize=10)
        ax.set_ylabel("Relative Size of the LCC (P∞(f) / P∞(0))", fontsize=10)

        # ax.set_yscale("log")
        # ax.set_xscale("log")

        ax.grid(
            # axis="y",
            alpha=0.65,
            zorder=1,
            which="both",  # Grid for major and minor ticks
            color="lightgrey",
        )

        if f_and_p_random is not None:
            f_random, p_random = f_and_p_random
            plt.plot(
                f_random,
                p_random,
                marker="o",
                markersize=5,
                linestyle="-",
                color="#E84F5E",
                label="Random Failures",
            )

        if f_and_p_degree is not None:
            f_degree, p_degree = f_and_p_degree
            plt.plot(
                f_degree,
                p_degree,
                marker="o",
                markersize=5,
                linestyle="-",
                color="#2772A0",
                label="Targeted attacks",
            )

        # divisiones_x_principales = np.arange(stop=1, step=0.05)
        # ax.set_xticks(divisiones_x_principales)

        plt.title(
            "Robustness analysis against failures and targeted attacks",
            fontsize=18,
            color="black",
            loc="left",
            y=1.07,
        )

        plt.text(
            0.025,
            1.03,
            f"Network: {name}",
            transform=ax.transAxes,
            fontsize=12,
            color="#5B5B5B",
        )

        plt.legend(fontsize=9)
        plt.tight_layout()
        os.makedirs("reports/", exist_ok=True)
        plt.savefig(
            f"reports/robustness_simulation_{tension}.png",
            dpi=500,
            bbox_inches="tight",
        )
        plt.close()
        logger.info(
            "Robustness graph saved in reports/robustness_simulation_%s.png", tension
        )
