import random
import pandas as pd
import networkx as nx
import powerlaw

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def build_network(df_links, df_nodes):
    """
    Constrói um grafo a partir de dataframes de links e nós.

    Args:
        df_links (pd.DataFrame): DataFrame com colunas 'start_id' e 'end_id'.
        df_nodes (pd.DataFrame): DataFrame com a coluna 'PN_CON' para os nós.

    Returns:
        nx.Graph: O objeto do grafo.
    """
    G = nx.Graph()

    # Adiciona os nós ao grafo
    nodes = df_nodes["PN_CON"].unique().tolist()
    G.add_nodes_from(nodes)

    # Adiciona as arestas ao grafo
    for _, row in df_links.iterrows():
        G.add_edge(row["start_id"], row["end_id"])

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
    second_moment = (degrees**2).mean()
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
    degree_counts = pd.Series(degrees).value_counts()

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
    print(f"Expoente da distribuição de grau (γ): {gamma:.2f}")

    if 2 < gamma < 3:
        gamma_div = (gamma - 2) / (3 - gamma)
        k_min_exp = k_min ** (gamma - 2)
        k_max_exp = k_max ** (3 - gamma)
        kappa = (gamma_div) * k_min_exp * k_max_exp

        return max(0, min(1, (1 - (1 / ((kappa) - 1)))))

    elif gamma > 3:
        gamma_div = (gamma - 2) / (3 - gamma)
        kappa = (gamma_div) * k_min
        return max(0, min(1, (1 - (1 / ((kappa) - 1)))))

    else:
        # Para outros valores de gamma (e.g., gamma <= 2 ou gamma == 3),
        # a rede é considerada robusta (fc=1) ou a fórmula não se aplica.
        return 1.0


def simulate_failures(G, failure_type: str, num_steps=20):
    """
    Simula a remoção aleatória de nós e rastreia o tamanho do componente gigante.

    Args:
        G (nx.Graph): O grafo original.
        num_steps (int): Número de etapas de remoção a serem simuladas.

    Returns:
        tuple: Duas listas, uma para a fração de nós removidos (f) e outra
               para a fração do componente gigante (P_inf).
    """
    print(f"Simulando falhas do tipo: {failure_type}")
    G_copy = G.copy()
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return [0], [0]
    if failure_type == "degree":
        # Remove nós em ordem decrescente de grau
        degrees = dict(G_copy.degree())
        nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)
    elif failure_type == "betweenness":
        # Remove nós em ordem decrescente de centralidade de intermediação
        betweenness = nx.betweenness_centrality(G_copy)
        nodes_to_remove = sorted(betweenness, key=betweenness.get, reverse=True)
    else:  # "random"
        # Remove nós aleatoriamente
        nodes_to_remove = list(G.nodes())
        random.shuffle(nodes_to_remove)

    f_values = [0]
    if G_copy.number_of_nodes() > 0:
        largest_component = max(nx.connected_components(G_copy), key=len)
        p_inf_values = [len(largest_component) / num_nodes]
    else:
        p_inf_values = [0]

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
            p_inf = 0
        else:
            largest_component = max(nx.connected_components(G_copy), key=len)
            p_inf = len(largest_component) / num_nodes

        f_values.append(f)
        p_inf_values.append(p_inf)

    return f_values, p_inf_values


def kappa_from_moments(mean_k: float, mean_k2: float) -> float:
    """kappa = <k^2>/<k>"""
    if mean_k == 0:
        return 0.0
    return float(mean_k2 / mean_k)


def molloy_reed_has_giant(mean_k: float, mean_k2: float) -> bool:
    """Molloy–Reed criterion: kappa > 2 indicates a giant component in a random configuration model with degree distribution pk."""
    return kappa_from_moments(mean_k, mean_k2) > 2.0


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


def main():
    tension = "medium"
    df_links = pd.read_parquet(f"data/gold/links_{tension}_tension.parquet")
    df_nodes = pd.read_parquet(f"data/gold/nodes_{tension}_tension.parquet")

    # 1. Constrói o grafo
    G = build_network(df_links, df_nodes)
    print(f"Rede criada com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")

    # 2. Obtém os graus dos nós
    degrees = np.array([d for _, d in G.degree()])

    # 3. Calcula as métricas
    avg_k, avg_k_squared = calculate_network_moments(degrees)
    kappa = kappa_from_moments(avg_k, avg_k_squared)
    print(f"Grau médio (⟨k⟩): {avg_k:.2f}")
    print(f"Segundo momento do grau (⟨k²⟩): {avg_k_squared:.2f}")
    print(f"Kappa (⟨k²⟩/⟨k⟩): {kappa:.2f}")
    print(
        f"Critério de Molloy-Reed indica componente gigante? {'Sim' if molloy_reed_has_giant(avg_k, avg_k_squared) else 'Não'}"
    )
    # 4. Calcula o limiar crítico teórico (fc)
    fc_molloy_reed = calculate_critical_threshold(degrees)
    print(
        f"Limiar crítico teórico (fc) pela fórmula de Molloy-Reed: {fc_molloy_reed:.2f}"
    )

    # 5. Simula falhas
    f_sim_random, p_inf_sim_random = simulate_failures(
        G, failure_type="random", num_steps=20
    )
    f_sim_degree, p_inf_sim_degree = simulate_failures(
        G, failure_type="degree", num_steps=20
    )

    # 6. Plota os resultados da simulação
    plt.figure(figsize=(10, 6))
    plt.plot(
        f_sim_random,
        p_inf_sim_random,
        marker="o",
        linestyle="-",
        color="b",
        label="Falhas Aleatórias",
    )
    plt.plot(
        f_sim_degree,
        p_inf_sim_degree,
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
    plt.show()


if __name__ == "__main__":
    main()
