import pickle
import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ComparativeFailureAnalysis:
    def __init__(
        self,
        path_original: str,
        path_redundant: str,
        base_unit: float = 0.6,
        network_subname= 1000,
        view=True
    ):
        """
        Inicializa el análisis comparativo cargando ambos grafos.

        Args:
            path_original (str): Ruta al pickle del grafo original.
            path_redundant (str): Ruta al pickle del grafo con redundancia (optimizado).
        """
        self.base_unit = base_unit

        logger.info("Cargando grafo Original...")
        self.G_orig, self.size_orig, self.n_islas_orig = self._load_graph(path_original)

        logger.info("Cargando grafo con Redundancia...")
        self.G_red, self.size_red, self.n_islas_opt = self._load_graph(path_redundant)

        self.network_subname = f"Inverted meters {network_subname} m"
        self.view=view

    def _load_graph(self, path):
        """Carga el grafo y extrae el Componente Conectado Gigante (LCC)."""
        if not os.path.exists(path):
            logger.error(f"Archivo no encontrado: {path}")
            return nx.Graph(), 0

        with open(path, "rb") as f:
            graph = pickle.load(f)

        # Trabajar solo con el LCC para el análisis de robustez
        if graph.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(graph), key=len)
            G_lcc = graph.subgraph(largest_cc).copy()
            n_islas = nx.number_connected_components(G_lcc)
            return G_lcc, G_lcc.number_of_nodes(), n_islas
        else:
            return graph, 0

    def simulate_failure(self, G, failure_type: str, num_steps=50):
        """
        Simula la remoción de nodos sobre un grafo específico 'G'.
        """
        G_copy = G.copy()
        num_nodes = G_copy.number_of_nodes()

        if num_nodes == 0:
            return [0], [0]

        # Definir el orden de eliminación
        if failure_type == "degree":  # Ataque dirigido
            degrees = dict(G_copy.degree())
            # Ordenar por grado (mayor a menor)
            nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)
        else:  # "random" - Fallo aleatorio
            nodes_to_remove = list(G_copy.nodes())
            random.shuffle(nodes_to_remove)

        # Listas para guardar resultados (f = fracción removida, p = fracción restante)
        f_values = [0]
        p_values = [100]  # Empezamos con el 100% del LCC actual

        # Optimización: No borrar uno por uno si el grafo es enorme, borrar en lotes
        step_size = max(1, num_nodes // num_steps)

        # Bucle de simulación
        # Iteramos hasta el 50% de nodos eliminados o hasta el final si se desea
        # (Generalmente el colapso ocurre antes del 50%)
        limit = num_nodes

        for i in range(step_size, limit + 1, step_size):
            # Obtener lote de nodos a eliminar
            nodes_this_step = nodes_to_remove[i - step_size : i]
            G_copy.remove_nodes_from(nodes_this_step)

            # Calcular métricas
            f = i / num_nodes  # % de nodos eliminados

            if G_copy.number_of_nodes() == 0:
                p = 0
            else:
                # Tamaño del nuevo LCC relativo al tamaño original de ESTE grafo
                largest_component = max(nx.connected_components(G_copy), key=len)
                p = len(largest_component) / num_nodes

            f_values.append(f * 100)
            p_values.append(p * 100)

            # Optimización: Si la red ya colapsó (p < 0.01), paramos
            if p < 0.005:
                break

        return f_values, p_values

    def run_analysis(self, tension_type="MT", num_steps=1):
        """Orquesta las 4 simulaciones y genera el gráfico."""
        logger.info("Iniciando simulaciones para Red Original...")
        orig_rnd_f, orig_rnd_p = self.simulate_failure(self.G_orig, "random", num_steps)
        orig_tgt_f, orig_tgt_p = self.simulate_failure(self.G_orig, "degree", num_steps)

        logger.info("Iniciando simulaciones para Red Redundante...")
        red_rnd_f, red_rnd_p = self.simulate_failure(self.G_red, "random", num_steps)
        red_tgt_f, red_tgt_p = self.simulate_failure(self.G_red, "degree", num_steps)

        # Empaquetar resultados
        results = {
            "orig_random": (orig_rnd_f, orig_rnd_p),
            "orig_target": (orig_tgt_f, orig_tgt_p),
            "red_random": (red_rnd_f, red_rnd_p),
            "red_target": (red_tgt_f, red_tgt_p),
        }

        self.plot_comparison(results, tension_type)

    def plot_comparison(self, results, tension_name):
        """Genera el gráfico comparativo final."""
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(16 * self.base_unit, 9 * self.base_unit))

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --- PLOTEO DE CURVAS ---

        # 1. Red Original (Tonos Rojos/Naranjas)
        f, p = results["orig_random"]
        ax.plot(
            f,
            p,
            color="#E88E4F",
            linestyle="--",
            linewidth=1.5,
            label="Original: Random Failure",
        )

        f, p = results["orig_target"]
        ax.plot(
            f,
            p,
            color="#C0392B",
            linestyle="--",
            linewidth=1.5,
            label="Original: Targeted Attack",
        )

        # 2. Red Redundante (Tonos Verdes/Azules)
        f, p = results["red_random"]
        ax.plot(
            f,
            p,
            color="#5DADE2",
            linestyle="-",
            linewidth=1.5,
            label="Redundant: Random Failure",
        )

        f, p = results["red_target"]
        ax.plot(
            f,
            p,
            color="#117A65",
            linestyle="-",
            linewidth=1.5,
            label="Redundant: Targeted Attack",
        )

        # --- FORMATO ---
        ax.set_xlabel("Fraction of Nodes Removed (f)", fontsize=11)
        ax.set_ylabel("Relative Size of LCC (P)", fontsize=11)
        # ax.set_xlim(0, 0.5) # Generalmente el interés está en el primer 50%
        # ax.set_ylim(0, 1.05)
        # ax.set_yscale("log")

        ax.grid(alpha=0.65, which="both", color="lightgrey")

        # Títulos y Texto
        plt.title(
            f"Robustness Upgrade Validation: {tension_name}",
            fontsize=18,
            color="black",
            loc="left",
            y=1.05,
        )

        # info_text = (
        #     f"NETWORK SCALE CONTEXT:\n"
        #     f"Original Islas Desconectadas: {self.n_islas_orig:<13,}\n"
        #     f"Redundant Islas Desconectadas: {self.n_islas_opt:<13,}\n"
        #     f"Growth Factor: {((self.n_islas_orig - self.n_islas_opt) / self.n_islas_orig) * 100:.3f}%"
        # )
        plt.text(
            0.025,
            1.02,
            self.network_subname,
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )

        plt.legend(fontsize=10, loc="upper right", framealpha=1)
        plt.tight_layout()

        # Guardar

        name_path = self.network_subname.replace(" ", "_")
        tension_name = tension_name.replace(" ", "_")
        out_path = f"reports/robustness_comparison_{tension_name}_{name_path}.png"
        os.makedirs("reports/", exist_ok=True)
        plt.savefig(out_path, dpi=500, bbox_inches="tight")

        if self.view:
            plt.show()
        plt.close()
        logger.info(f"Gráfico comparativo guardado en: {out_path}")



