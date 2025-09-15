"""
Geração e plotagem de redes aleatórias de Erdős-Rényi.
Exercício 1 - Algoritmos em Grafos (MO412)
Disciplina: Redes Complexas
Professor: João Meidanis
Aluno: Yan Prada Moro
Data: 15/09/2025
"""

import logging
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_network(N: int, k_medio: float, nome: str) -> None:
    """Gera uma rede aleatória de Erdős-Rényi e plota a rede."""
    p = k_medio / (N - 1)
    G = nx.erdos_renyi_graph(N, p)
    logger.info("Rede %s: k_medio=%s, p=%.5f", nome, k_medio, p)
    logger.info("Número de arestas geradas: %d", G.number_of_edges())
    logger.info("Grau médio: %.3f\n", sum(dict(G.degree()).values()) / N)
    pos = nx.spring_layout(G, seed=42)
    plot_network(G, pos, nome, N, k_medio)


def plot_network(G, pos, nome, N, k_medio) -> None:
    """Plota a rede gerada."""
    local_path = Path(__file__).parent
    local_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="blue", alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.4)
    plt.axis("off")
    plt.title(f"Rede {nome} (N={N}, k≈{k_medio})")
    plt.savefig(f"{local_path}/rede_{nome}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Função principal para gerar redes com diferentes k_medio."""
    N = 500
    k_values = [0.8, 1.0, 8.0]
    for k in k_values:
        generate_network(N, k, f"rede_{k}")
