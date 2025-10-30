import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def view_graph_topology(
    graph: nx, title: str = "Electrical Grid Topology", base_unit: int = 0.5
):
    """
    Generates a clean-style visualization of graph G.

    Args:
    G (nx.Graph): A clean NetworkX graph object.
    title (str): The main title of the graph.
    base_unit (tuple): The size of the figure (width, height) in inches.
    """

    logger.info("Starting visualization of graph")

    # Pastel base colors for nodes and edges
    COLOR__NODES = "#E84F5E"  # noqa: F841
    COLOR_EDGES = "#FCDFC5"

    # Coordinate calculation
    geo = {
        nodo: (graph.nodes[nodo]["latitude"], graph.nodes[nodo]["length"])
        for nodo in graph.nodes()
    }

    # Calculation of Metrics (Degree)
    try:
        node_degrees_values = dict(graph.degree()).values()
        node_degrees = np.array(list(node_degrees_values))

    except Exception as e:
        logger.error(f"Error getting node degrees: {e}")
        return

    if node_degrees.size == 0:
        logger.warning("The graph does not contain any nodes to graph.")
        return

    # COLOR CALCULATION
    max_degree = np.max(node_degrees) if np.max(node_degrees) > 0 else 1
    normalized_degrees = node_degrees / max_degree
    cmap_pastel = plt.cm.get_cmap("PuRd_r")
    colors_by_grade = cmap_pastel(normalized_degrees)

    # Global Style Settings
    plt.style.use("ggplot")

    # Figure Configuration
    fig, ax = plt.subplots(figsize=(16 * base_unit, 9 * base_unit))

    # Clean up the background of the style
    ax.set_facecolor("white")
    ax.grid(color="#CCDDEA", alpha=0.3)
    fig.patch.set_facecolor("white")

    # Graph Drawing
    # Draw Edges (Links)
    nx.draw_networkx_edges(
        graph,
        geo,
        ax=ax,
        alpha=0.65,
        edge_color=COLOR_EDGES,
        width=1.2,
    )

    # Drawing Nodes - Using Degree-Mapped Pastel Colors
    nx.draw_networkx_nodes(
        graph,
        geo,
        ax=ax,
        # node_size=[v * 5 for v in node_degrees],
        node_size=3,
        node_color=COLOR__NODES,
        linewidths=0.05,
        edgecolors=COLOR_EDGES,
        alpha=0.8,
    )

    # Añadir títulos
    plt.title(title, fontsize=18, color="black", loc="left", y=1.05)
    plt.text(
        0.00,
        1.02,
        f"Nodos: {graph.number_of_nodes()} | Links: {graph.number_of_edges()}",
        transform=ax.transAxes,
        fontsize=10,
        color="gray",
    )
    plt.tight_layout()

    # Show Chart
    plt.savefig(
        os.path.join("./reports/", title.replace(" ", "_") + ".png"),
        dpi=500,
        bbox_inches="tight",
    )
    plt.show()
    logger.info("Chart generated and displayed successfully.")
