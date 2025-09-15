"""
Geração e plotagem de redes aleatórias de Erdős-Rényi.
Exercício 1 - Algoritmos em Grafos (MO412)
Disciplina: Redes Complexas
Professor: João Meidanis
Aluno: Yan Prada Moro
Data: 15/09/2025
"""

from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

def generate_network(N: int, k_medio: float) -> nx.Graph:
    """Gera uma rede aleatória de Erdős-Rényi."""
    p = k_medio / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=42)
    return G


def plot_network(G, N, k_medio) -> None:
    """Plota a rede gerada."""
    pos = nx.spring_layout(G, seed=42)
    local_path = Path(__file__).parent
    local_path.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="blue", alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=0.4, alpha=0.4)
    plt.axis("off")
    plt.title(f"Rede {k_medio} (N={N}, k≈{k_medio})")
    plt.savefig(f"{local_path}/rede_{k_medio}.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Função principal para gerar redes com diferentes k_medio."""
    N = 500
    k_values = [0.8, 1.0, 8.0]
    for k in k_values:
        G = generate_network(N, k)
        plot_network(G, N, k)


if __name__ == "__main__":
    main()
