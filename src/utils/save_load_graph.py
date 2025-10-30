import logging
import pickle

import networkx as nx

logger = logging.getLogger(__name__)


# --- SAVE THE GRAPH ---
def save_graph(path_save: str, graph: nx):
    try:
        with open(path_save, "wb") as f:
            pickle.dump(graph, f)
        logging.info(f"Graph successfully saved in: {path_save}")
    except Exception as e:
        logging.info(f"Error saving in Pickle: {e}")


# --- LOAD THE GRAPH ---
def load_graph(path_load: str) -> nx:
    try:
        with open(path_load, "rb") as f:
            graph_pickle = pickle.load(f)

        logger.info("Graph successfully loaded from Pickle.")
        logger.info(f"Loaded nodes: {graph_pickle.number_of_nodes()}")

        return graph_pickle
    except FileNotFoundError:
        logger.error(f"Error: The file {path_load} does not exist.")
