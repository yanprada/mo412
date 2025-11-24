import pickle
import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)


class FailureAnalysis:
    def __init__(self, pickle_path: str):
        """
        Initialize the FailureAnalysis class with a graph from a pickle file.

        Args:
            pickle_path (str): Path to the pickle file containing the graph.
        """
        self.pickle_path = pickle_path
        self.G = self._load_graph()

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

    def plot_failures(self, f_and_p_random, f_and_p_degree, tension):
        """Plota os resultados das simulações de falhas."""
        plt.figure(figsize=(10, 6))

        if f_and_p_random is not None:
            f_random, p_random = f_and_p_random
            plt.plot(
                f_random,
                p_random,
                marker="o",
                linestyle="-",
                color="b",
                label="Falhas Aleatórias",
            )

        if f_and_p_degree is not None:
            f_degree, p_degree = f_and_p_degree
            plt.plot(
                f_degree,
                p_degree,
                marker="o",
                linestyle="-",
                color="g",
                label="Ataques Direcionados (Grau)",
            )

        plt.title("Robustez da Rede contra Falhas e Ataques")
        plt.xlabel("Fração de Nós Removidos (f)")
        plt.ylabel("Tamanho Relativo do Componente Gigante (P∞(f) / P∞(0))")
        plt.grid(True)
        plt.legend()
        # Ensure the directory exists
        os.makedirs("reports/", exist_ok=True)
        plt.savefig(f"reports/robustness_simulation_{tension}.png")
        plt.close()
        logger.info(
            "Gráfico de robustez salvo em reports/robustness_simulation_%s.png", tension
        )
