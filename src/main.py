import logging

import matplotlib.pyplot as plt
import networkx as nx

from analysis.topology_analysis import TopologyAnalysis
from utils.configuration_log import configure_logging_global
from utils.graph_generator import GraphGenerator
from utils.save_load_graph import load_graph
from utils.view_graph_topology import view_graph_topology

configure_logging_global()
logger = logging.getLogger(__name__)

# # # --- 1. Definition of Files ---
NODE_FILE_NAME = "./data/processed/CPFL_Paulista_2023-Nodos.csv"
FILE_NAME_LINKS_BT = "./data/processed/CPFL_Paulista_2023-SSDBT.csv"
FILE_NAME_LINKS_MT = "./data/processed/CPFL_Paulista_2023-SSDMT.csv"
FILE_NAME_LINKS_AT = "./data/processed/CPFL_Paulista_2023-SSDAT.csv"

generate_graphs = False
topological_analysis = True

if generate_graphs:
    gen_bt = GraphGenerator(
        path_links=FILE_NAME_LINKS_BT,
        path_nodes=NODE_FILE_NAME,
        name="CPFL Paulista BT Electrical Network Topology",
    )
    gen_bt.create_graph()
    del gen_bt

    gen_mt = GraphGenerator(
        path_links=FILE_NAME_LINKS_MT,
        path_nodes=NODE_FILE_NAME,
        name="CPFL Paulista MT Electrical Network Topology",
    )
    gen_mt.create_graph()
    del gen_mt

    gen_at = GraphGenerator(
        path_links=FILE_NAME_LINKS_AT,
        path_nodes=NODE_FILE_NAME,
        name="CPFL Paulista AT Electrical Network Topology",
    )
    gen_at.create_graph()
    del gen_at

if topological_analysis:
    logger.info("INICIO del Análisis de Topología CPFL Paulista.")
    FILE_SAVE_GRAPH = (
        "./data/graph/CPFL_Paulista_AT_Electrical_Network_Topology.pickle"
    )

    G = load_graph(path_load=FILE_SAVE_GRAPH)
    # view_graph_topology(graph=G)

    t = TopologyAnalysis(graph=G, report_subname="CPFL Paulista AT")
    t.connectivity_scale()
    t.components_distribution_analysis()
    #t.analizar_criticidad_articulacion_rapida(G, k_muestreo=500)
