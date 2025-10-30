import os
import random
import logging
import pandas as pd
import networkx as nx


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(tension: str):
    """Carrega os dados de links e nós a partir de arquivos parquet."""
    df_links = pd.read_parquet(f"data/gold/links_{tension}_tension.parquet")
    df_links = df_links[(df_links["start_id"] != "") & (df_links["end_id"] != "")]
    df_nodes = pd.read_parquet(f"data/gold/nodes_{tension}_tension.parquet")
    df_nodes = df_nodes[df_nodes["PN_CON"] != ""]
    return df_links, df_nodes


def build_network(df_links, df_nodes, tension):
    """
    Constrói um grafo a partir de dataframes de links e nós.

    Args:
        df_links (pd.DataFrame): DataFrame com colunas 'start_id' e 'end_id'.
        df_nodes (pd.DataFrame): DataFrame com a coluna 'PN_CON' para os nós.
        tension (str): Nível de tensão para logging e plotagem.

    Returns:
        nx.MultiGraph: O objeto do grafo.
    """
    G = nx.MultiGraph()

    # Adiciona os nós ao grafo
    nodes = df_nodes["PN_CON"].unique().tolist()
    G.add_nodes_from(nodes)
    # Adiciona as arestas ao grafo
    for _, row in df_links.iterrows():
        G.add_edge(row["start_id"], row["end_id"])
    assert (
        df_links.shape[0] == G.number_of_edges()
    ), "Número de arestas no DataFrame e no grafo não coincidem."
    if df_nodes.shape[0] != G.number_of_nodes():
        logger.info(
            "Existem %d postes com conexões de subestação.",
            df_nodes.shape[0] - G.number_of_nodes(),
        )
    logger.info(
        "Rede criada com %d nós e %d arestas.", G.number_of_nodes(), G.number_of_edges()
    )
    logger.info(
        "Maior componente conectado tem %d nós. Isso representa %.2f%% dos nós.",
        len(max(nx.connected_components(G), key=len)),
        100 * len(max(nx.connected_components(G), key=len)) / G.number_of_nodes(),
    )
    plot_histograms(G.adj, tension, remove_nodes=False)
    return G


def calculate_k_extremes(degrees):
    """
    Calcula o grau mínimo e máximo na lista de graus.

    Args:
        degrees (list): Lista de graus dos nós.

    Returns:
        tuple: Uma tupla contendo (k_min, k_max).
    """
    if not degrees:
        return 0, 0
    return min(degrees), max(degrees)


def calculate_network_moments(degrees):
    """
    Calcula o grau médio (primeiro momento) e o segundo momento do grau.

    Args:
        G (nx.Graph): O grafo da rede.

    Returns:
        tuple: Uma tupla contendo (grau_medio, segundo_momento).
    """
    average_degree = degrees.mean()
    second_moment = (degrees**2).mean()  # média dos quadrados
    return average_degree, second_moment


def calculate_gamma_exponent(degrees):
    """
    Estima o expoente da lei de potência (γ) da distribuição de graus.
    Este método usa regressão linear no gráfico log-log da distribuição de graus.

    Args:
        G (nx.Graph): O grafo da rede.

    Returns:
        float: O valor estimado de γ.
    """
    if not any(degrees):
        return 0.0

    # Contagem da frequência de cada grau
    degree_counts = pd.Series(degrees).value_counts().drop(index=0)

    # Filtra para garantir que temos pelo menos 2 pontos para a regressão
    if len(degree_counts) < 2:
        return 0.0

    # Prepara os dados para o ajuste em escala log-log
    log_degrees = np.log10(degree_counts.index)
    log_counts = np.log10(degree_counts.values)

    # Realiza a regressão linear
    slope, _, _, _, _ = linregress(log_degrees, log_counts)

    # O expoente γ é o negativo do slope
    gamma = -slope

    return gamma


def natural_cutoff_kmax(N: int, gamma: float, kmin: int) -> int:
    """
    Natural cutoff for scale-free networks (heuristic): k_max ~ k_min * N^{1/(gamma-1)} (and <= N-1)
    """
    if gamma <= 1:
        return N - 1
    kmax = int(kmin * N ** (1.0 / (gamma - 1.0)))
    return max(kmin + 1, min(kmax, N - 1))


def calculate_critical_threshold(degrees):
    """
    Calcula o limiar crítico de falha (fc) com base em fórmulas
    para redes de lei de potência.

    Args:
        G (nx.Graph): O grafo da rede.
        gamma (float): O expoente da lei de potência da distribuição de graus.

    Returns:
        float: O valor do limiar crítico (fc).
    """
    k_min, k_max = min(degrees), max(degrees)
    gamma = calculate_gamma_exponent(degrees)
    logger.info("Expoente da distribuição de grau (γ): %.2f", gamma)

    if 2 < gamma < 3:
        gamma_div = (gamma - 2) / (3 - gamma)
        k_min_exp = k_min ** (gamma - 2)
        k_max_exp = k_max ** (3 - gamma)
        kappa = (gamma_div) * k_min_exp * k_max_exp

        fc_molloy_reed = max(0, min(1, (1 - (1 / ((kappa) - 1)))))

    elif gamma > 3:
        gamma_div = (gamma - 2) / (3 - gamma)
        kappa = (gamma_div) * k_min
        fc_molloy_reed = max(0, min(1, (1 - (1 / ((kappa) - 1)))))

    else:
        fc_molloy_reed = 1.0
    logger.info(
        "Limiar crítico teórico (fc) pela fórmula de Molloy-Reed: %.2f", fc_molloy_reed
    )
    return fc_molloy_reed


def simulate_failure(G, failure_type: str, num_steps=20):
    """
    Simula a remoção aleatória de nós e rastreia o tamanho do componente gigante.

    Args:
        G (nx.Graph): O grafo original.
        num_steps (int): Número de etapas de remoção a serem simuladas.

    Returns:
        tuple: Duas listas, uma para a fração de nós removidos (f) e outra
               para a fração do componente gigante (P).
    """
    print(f"Simulando falhas do tipo: {failure_type}")
    G_copy = G.copy()
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return [0], [0]
    if failure_type == "degree":
        degrees = dict(G_copy.degree())
        nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)
    elif failure_type == "betweenness":
        betweenness = nx.betweenness_centrality(G_copy)
        nodes_to_remove = sorted(betweenness, key=betweenness.get, reverse=True)
    else:  # "random"
        nodes_to_remove = list(G.nodes())
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


def kappa_from_moments(mean_k: float, mean_k2: float) -> float:
    """kappa = <k^2>/<k>"""
    if mean_k == 0:
        return 0.0
    return float(mean_k2 / mean_k)


def fc_general(mean_k: float, mean_k2: float) -> float:
    """
    Critical failure threshold under random node removal for an arbitrary degree distribution:
    f_c = 1 - 1 / ( <k^2>/<k> - 1 )  [Eq. 8.7]
    """
    denom = kappa_from_moments(mean_k, mean_k2) - 1.0
    if denom <= 0:
        return 0.0  # No meaningful threshold (already fragmented)
    return 1.0 - (1.0 / denom)


def fc_er(mean_k: float) -> float:
    """Random graph (ER) threshold: f_c = 1 - 1/<k>  [Eq. 8.8]"""
    if mean_k <= 0:
        return 0.0
    return 1.0 - 1.0 / mean_k


def print_log(avg_k, avg_k_squared, kappa):
    """Imprime os principais resultados da análise de rede."""
    logger.info("Grau médio (⟨k⟩): %.2f", avg_k)
    logger.info("Segundo momento do grau (⟨k²⟩): %.2f", avg_k_squared)
    logger.info("Kappa (⟨k²⟩/⟨k⟩): %.2f", kappa)
    logger.info(
        "Critério de Molloy-Reed indica componente gigante? %s",
        "Sim" if kappa > 2 else "Não",
    )
    logger.info(
        "Limiar crítico (fc) para remoção aleatória de nós: %.2f",
        fc_general(avg_k, avg_k_squared),
    )
    logger.info("Limiar crítico (fc) para grafo aleatório (ER): %.2f", fc_er(avg_k))


def plot_failures(f_and_p_random, f_and_p_degree, fc_molloy_reed, tension):
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

    plt.axvline(
        x=fc_molloy_reed,
        color="r",
        linestyle="--",
        label=f"Limiar Crítico Teórico (fc = {fc_molloy_reed:.2f})",
    )
    plt.title("Robustez da Rede contra Falhas e Ataques")
    plt.xlabel("Fração de Nós Removidos (f)")
    plt.ylabel("Tamanho Relativo do Componente Gigante (P∞(f) / P∞(0))")
    plt.grid(True)
    plt.legend()
    # Ensure the directory exists
    os.makedirs(f"visualization/{tension}_tension", exist_ok=True)
    plt.savefig(f"visualization/{tension}_tension/robustness_simulation.png")
    plt.close()
    logger.info("Gráfico de robustez salvo em visualization/%s_tension/", tension)


def calculate_molloy_reed_threshold(G):
    """Calcula o limiar crítico de falha (fc) com base em fórmulas para redes de lei de potência."""
    degrees = np.array([d for _, d in G.degree()])

    max_degree_node, max_degree = max(G.degree(), key=lambda item: item[1])
    nodes_with_degree_2 = [node for node, degree in G.degree() if degree == 2]
    if nodes_with_degree_2:
        sample_size = min(5, len(nodes_with_degree_2))
        sample_nodes = random.sample(nodes_with_degree_2, sample_size)
        logger.info("Amostra de nós com grau 2: %s", sample_nodes)

    logger.info("Nó com maior grau: %s (grau %d)", max_degree_node, max_degree)

    avg_k, avg_k_squared = calculate_network_moments(degrees)
    kappa = kappa_from_moments(avg_k, avg_k_squared)

    print_log(avg_k, avg_k_squared, kappa)

    fc_molloy_reed = calculate_critical_threshold(degrees)
    return fc_molloy_reed


def plot_histograms(
    adj,
    tension_level: str,
    remove_nodes: bool = False,
):
    """Plot degree distribution and log-log degree distribution of the graph."""
    degrees = np.fromiter((len(edges) for edges in adj.values()), dtype=int)
    fstr = ""
    if remove_nodes:
        degrees = degrees[degrees != 2]
        fstr = "_without_degree_2"

    degree_counts = np.bincount(degrees)
    degree_range = np.arange(len(degree_counts))
    percentages = degree_counts / degree_counts.sum() * 100

    # Plot degree distribution
    plt.scatter(degree_range, percentages, color="blue")
    plt.xlabel("Degree")
    plt.ylabel("Percentage (%)")
    plt.title(f"Degree Distribution of Graph - {tension_level}{fstr}")
    plt.grid(True)
    # Ensure the directory exists
    os.makedirs(f"visualization/{tension_level}_tension/", exist_ok=True)
    plt.savefig(f"visualization/{tension_level}_tension/degree_distribution{fstr}.png")
    plt.close()

    # Plot log-log degree distribution
    plt.scatter(degree_range, percentages, color="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree (log scale)")
    plt.ylabel("Percentage (%) (log scale)")
    plt.title(f"Log-Log Degree Distribution of Graph - {tension_level}{fstr}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(f"visualization/{tension_level}/log_degree_distribution_{fstr}.png")
    plt.close()


def main():
    """Função principal para executar a análise de rede."""

    for tension in ["low", "medium", "high"]:
        logger.info("Analisando rede de tensão: %s", tension)
        links, nodes = load_data(tension)

        G = build_network(links, nodes, tension)

        fc_molloy_reed = calculate_molloy_reed_threshold(G)

        f_and_p_random = simulate_failure(G, failure_type="random")
        f_and_p_degree = simulate_failure(G, failure_type="degree")
        plot_failures(f_and_p_random, f_and_p_degree, fc_molloy_reed, tension)


if __name__ == "__main__":
    main()
