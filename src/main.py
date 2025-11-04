"""
Main Orchestration Script for CPFL Network Topology Analysis.

This script serves as the main entry point for the pipeline. It controls
two distinct phases:

1.  Graph Generation (ETL): Loads raw CSV data, cleans it, builds the
    NetworkX graph objects, and saves them to disk as '.pickle' files.
    (Controlled by `generate_graphs`)

2.  Topological Analysis: Loads the pre-built '.pickle' graphs from disk
    and runs the static topological analysis (connectivity, components, etc.)
    on them, generating text and plot reports.
    (Controlled by `topological_analysis`)
"""

import logging

from analysis.graph_generator import GraphGenerator

# --- 3rd Party Imports ---
# import matplotlib.pyplot as plt  # Not used directly here, but in analysis
# import networkx as nx            # Not used directly here, but in analysis
# --- Custom Module Imports ---
from analysis.topology_analysis import TopologyAnalysis
from utils.configuration_log import configure_logging_global
from utils.save_load_graph import load_graph

# --- 1. Global Configuration & Logging ---

# Configure the global logger (e.g., set level, format)
configure_logging_global()
logger = logging.getLogger(__name__)

# Control flags to determine which parts of the pipeline to run.
# This is efficient: run generation once, then set to False
# and run analysis multiple times.
generate_graphs = True
topological_analysis = True


# --- 2. PHASE 1: Graph Generation (ETL) ---
if generate_graphs:
    logger.info("PHASE 1: Starting Graph Generation (ETL)...")
    
    # --- File Definitions ---
    NODE_FILE_NAME = "./data/raw/CPFL_Paulista_2023-Nodos.csv"
    FILE_NAME_LINKS_BT = "./data/raw/CPFL_Paulista_2023-SSDBT.csv" # Low Voltage
    FILE_NAME_LINKS_MT = "./data/raw/CPFL_Paulista_2023-SSDMT.csv" # Medium Voltage
    FILE_NAME_LINKS_AT = "./data/raw/CPFL_Paulista_2023-SSDAT.csv" # High Voltage

    # --- Process Low Voltage (BT) Network ---
    logger.info("Processing Low Voltage (BT) Network...")
    gen_bt = GraphGenerator(
        path_links=FILE_NAME_LINKS_BT,
        path_nodes=NODE_FILE_NAME,
        name="CPFL Paulista BT Electrical Network Topology",
        # Note: BT is mostly radial, a low threshold is appropriate
        threshold_km=1.0 
    )
    gen_bt.create_graph()
    del gen_bt  # Clear memory before processing the next large graph

    # --- Process Medium Voltage (MT) Network ---
    logger.info("Processing Medium Voltage (MT) Network...")
    gen_mt = GraphGenerator(
        path_links=FILE_NAME_LINKS_MT,
        path_nodes=NODE_FILE_NAME,
        name="CPFL Paulista MT Electrical Network Topology",
        threshold_km=2.0 # MT links can be longer
    )
    gen_mt.create_graph()
    del gen_mt  # Clear memory

    # --- Process High Voltage (AT) Network ---
    logger.info("Processing High Voltage (AT) Network...")
    gen_at = GraphGenerator(
        path_links=FILE_NAME_LINKS_AT,
        path_nodes=NODE_FILE_NAME,
        name="CPFL Paulista AT Electrical Network Topology",
        threshold_km=10.0 # AT links are much longer
    )
    gen_at.create_graph()
    del gen_at  # Clear memory

    logger.info("PHASE 1: Graph Generation complete.")

else:
    logger.info("PHASE 1: Graph Generation (ETL) skipped by configuration.")


# --- 3. PHASE 2: Topological Analysis ---
if topological_analysis:
    logger.info("PHASE 2: Starting Topological Analysis...")

    # --- Path definitions for pre-built graphs ---
    FILE_NAME_GRAPH_BT = (
        "./data/graph/CPFL_Paulista_BT_Electrical_Network_Topology.pickle"
    )
    FILE_NAME_GRAPH_MT = (
        "./data/graph/CPFL_Paulista_MT_Electrical_Network_Topology.pickle"
    )
    FILE_NAME_GRAPH_AT = (
        "./data/graph/CPFL_Paulista_AT_Electrical_Network_Topology.pickle"
    )
    
    # --- Analyze Low Voltage (BT) ---
    try:
        logger.info("Loading and analyzing BT Network...")
        G_bt = load_graph(path_load=FILE_NAME_GRAPH_BT)
        analysis_bt = TopologyAnalysis(graph=G_bt, report_subname="CPFL Paulista BT")
        analysis_bt.connectivity_scale()
        analysis_bt.components_distribution_analysis()
        del G_bt, analysis_bt  # Clear memory
        
    except FileNotFoundError:
        logger.error(f"FATAL: BT graph file not found at {FILE_NAME_GRAPH_BT}. "
                     "Run with generate_graphs=True first.")
    except Exception as e:
        logger.error(f"An error occurred during BT analysis: {e}")

    # --- Analyze Medium Voltage (MT) ---
    try:
        logger.info("Loading and analyzing MT Network...")
        G_mt = load_graph(path_load=FILE_NAME_GRAPH_MT)
        analysis_mt = TopologyAnalysis(graph=G_mt, report_subname="CPFL Paulista MT")
        analysis_mt.connectivity_scale()
        analysis_mt.components_distribution_analysis()
        del G_mt, analysis_mt  # Clear memory
        
    except FileNotFoundError:
        logger.error(f"FATAL: MT graph file not found at {FILE_NAME_GRAPH_MT}. "
                     "Run with generate_graphs=True first.")
    except Exception as e:
        logger.error(f"An error occurred during MT analysis: {e}")

    # --- Analyze High Voltage (AT) ---
    try:
        logger.info("Loading and analyzing AT Network...")
        G_at = load_graph(path_load=FILE_NAME_GRAPH_AT)
        analysis_at = TopologyAnalysis(graph=G_at, report_subname="CPFL Paulista AT")
        analysis_at.connectivity_scale()
        analysis_at.components_distribution_analysis()
        del G_at, analysis_at  # Clear memory
        
    except FileNotFoundError:
        logger.error(f"FATAL: AT graph file not found at {FILE_NAME_GRAPH_AT}. "
                     "Run with generate_graphs=True first.")
    except Exception as e:
        logger.error(f"An error occurred during AT analysis: {e}")

    logger.info("PHASE 2: Topological Analysis complete.")

else:
    logger.info("PHASE 2: Topological Analysis skipped by configuration.")

logger.info("CPFL Paulista Network Analysis pipeline finished.")