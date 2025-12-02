"""
Main Orchestration Script for CPFL Network Topology Analysis.

This script acts as the central pipeline controller for analyzing the electrical 
network topology of CPFL Paulista. It manages the workflow through boolean flags,
allowing for modular execution of different analysis phases.

The pipeline consists of six distinct phases:
1.  Graph Generation (ETL): Ingests raw CSV data, cleans it, and serializes NetworkX graphs.
2.  Topological Analysis: Calculates static metrics (degree, components) and generates maps.
3.  Criticality Analysis: Identifies critical nodes using parallelized Betweenness Centrality.
4.  Failure Analysis: Simulates random and targeted attacks on the original networks.
5.  Redundancy Analysis: Identifies strategic "Tie-Lines" to create loops and improve robustness.
6.  Comparative Analysis: Compares the robustness of the original network vs. the optimized network.

Usage:
    Set the boolean flags in the `PIPELINE_FLAGS` dictionary to True/False 
    to enable or disable specific phases.
"""

import logging
import os

# --- Internal Module Imports ---
from utils.configuration_log import configure_logging_global
from utils.save_load_graph import load_graph

# Analysis Modules
from analysis.graph_generator import GraphGenerator
from analysis.topology_analysis import TopologyAnalysis
from analysis.criticality_analysis import CriticalityAnalysis
from analysis.failure_analysis import FailureAnalysis
from analysis.redundancy_analysis import RedundancyAnalysis
from analysis.comparative_failure_analysis import ComparativeFailureAnalysis

# --- 1. Global Configuration & Setup ---

configure_logging_global()
logger = logging.getLogger(__name__)

# Pipeline Control Flags
PIPELINE_FLAGS = {
    "generate_graphs": True,       # Phase 1: ETL from CSV to Pickle
    "topology_analysis": True,      # Phase 2: Static Metrics & Visualization
    "criticality_analysis": True,  # Phase 3: Betweenness & Articulation Points
    "failure_analysis": True,      # Phase 4: Robustness of Original Graph
    "redundancy_analysis": True,    # Phase 5: Propose Investments & Gen. Optimized Graph
    "comparative_analysis": True    # Phase 6: Validate Investment (Original vs Optimized)
}

# Centralized Path Configuration
DATA_PATHS = {
    "raw_nodes": "./data/raw/CPFL_Paulista_2023-Nodos.csv",
    "raw_links": {
        "BT": "./data/raw/CPFL_Paulista_2023-SSDBT.csv",
        "MT": "./data/raw/CPFL_Paulista_2023-SSDMT.csv",
        "AT": "./data/raw/CPFL_Paulista_2023-SSDAT.csv",
    },
    "graphs": {
        "BT": "./data/graph/CPFL_Paulista_BT_Electrical_Network_Topology.pickle",
        "MT": "./data/graph/CPFL_Paulista_MT_Electrical_Network_Topology.pickle",
        "AT": "./data/graph/CPFL_Paulista_AT_Electrical_Network_Topology.pickle",
    }
}


# --- Phase 1: Graph Generation ---
def run_graph_generation():
    logger.info("PHASE 1: Starting Graph Generation (ETL)...")
    
    # Configuration for each voltage level
    configs = [
        {"type": "BT", "thresh": 1.0, "name": "CPFL Paulista BT Electrical Network Topology"},
        {"type": "MT", "thresh": 2.0, "name": "CPFL Paulista MT Electrical Network Topology"},
        {"type": "AT", "thresh": 10.0, "name": "CPFL Paulista AT Electrical Network Topology"},
    ]

    for cfg in configs:
        tension = cfg["type"]
        logger.info(f"Processing {tension} Network...")
        try:
            generator = GraphGenerator(
                path_links=DATA_PATHS["raw_links"][tension],
                path_nodes=DATA_PATHS["raw_nodes"],
                name=cfg["name"],
                threshold_km=cfg["thresh"],
            )
            generator.create_graph(view=False) # View=False for batch processing
            del generator # Force memory release
        except Exception as e:
            logger.error(f"Error generating {tension} graph: {e}")

    logger.info("PHASE 1: Graph Generation complete.")


# --- Phase 2: Topological Analysis ---
def run_topology_analysis():
    logger.info("PHASE 2: Starting Topological Analysis...")

    for tension, path in DATA_PATHS["graphs"].items():
        try:
            logger.info(f"Analyzing {tension} Network...")
            G = load_graph(path_load=path)
            
            analysis = TopologyAnalysis(
                graph=G, 
                report_subname=f"CPFL Paulista {tension}"
            )
            analysis.connectivity_scale()
            analysis.components_distribution_analysis()
            
            del G, analysis
        except FileNotFoundError:
            logger.error(f"Graph file for {tension} not found. Run Generation phase first.")
        except Exception as e:
            logger.error(f"Error in {tension} topology analysis: {e}")

    logger.info("PHASE 2: Topological Analysis complete.")


# --- Phase 3: Criticality Analysis ---
def run_criticality_analysis():
    logger.info("PHASE 3: Starting Criticality Analysis (Parallel)...")

    for tension, path in DATA_PATHS["graphs"].items():
        try:
            logger.info(f"Calculating criticality for {tension} Network...")
            G = load_graph(path_load=path)
            
            # Using 500 samples and 24 cores (adjust n_processes based on machine)
            analysis = CriticalityAnalysis(
                graph=G, 
                k_sample=500, 
                n_processes=16, 
                report_subname=f"CPFL Paulista {tension}"
            )
            analysis.analizar_criticidad_paralela(view=False)
            
            del G, analysis
        except FileNotFoundError:
            logger.error(f"Graph file for {tension} not found.")
        except Exception as e:
            logger.error(f"Error in {tension} criticality analysis: {e}")

    logger.info("PHASE 3: Criticality Analysis complete.")


# --- Phase 4: Failure Analysis (Original Network) ---
def run_failure_analysis():
    logger.info("PHASE 4: Starting Failure Analysis (Original)...")

    for tension, path in DATA_PATHS["graphs"].items():
        try:
            logger.info(f"Simulating failures for {tension} Network...")
            
            failure_analyzer = FailureAnalysis(pickle_path=path)
            
            # 1. Random Failure Simulation
            logger.info(f"[{tension}] Running random failure simulation...")
            f_rand, p_rand = failure_analyzer.simulate_failure("random")

            # 2. Targeted Attack Simulation
            logger.info(f"[{tension}] Running targeted attack simulation...")
            f_deg, p_deg = failure_analyzer.simulate_failure("degree")

            # 3. Plotting
            failure_analyzer.plot_failures(
                f_and_p_random=(f_rand, p_rand),
                f_and_p_degree=(f_deg, p_deg),
                tension=tension,
                name=f"CPFL Paulista {tension}"
            )
            
            del failure_analyzer
        except FileNotFoundError:
            logger.error(f"Graph file for {tension} not found.")
        except Exception as e:
            logger.error(f"Error in {tension} failure analysis: {e}")

    logger.info("PHASE 4: Failure Analysis complete.")


# --- Phase 5: Redundancy Analysis (Investment Proposal) ---
def run_redundancy_analysis():
    logger.info("PHASE 5: Starting Redundancy Analysis (Investment)...")

    # Define scenarios to analyze: (Tension, Zoom Coordinates, Investment Radius)
    # Note: Adding suffix to network_name to ensure unique filenames
    scenarios = [
        # BT Scenarios
        {
            "tension": "BT", "coords": (-46.9575, -22.7694), 
            "dist": 5000, "name": "CPFL Paulista BT Redundancy Map 5000"
        },
        {
            "tension": "BT", "coords": (-46.9575, -22.7694), 
            "dist": 15000, "name": "CPFL Paulista BT Redundancy Map 15000"
        },
        # MT Scenarios
        {
            "tension": "MT", "coords": (-47.96277, -21.119580,), 
            "dist": 5000, "name": "CPFL Paulista MT Redundancy Map 5000"
        },
        {
            "tension": "MT", "coords": (-47.96277, -21.119580), 
            "dist": 15000, "name": "CPFL Paulista MT Redundancy Map 15000"
        },
    ]

    for scen in scenarios:
        try:
            tension = scen["tension"]
            path = DATA_PATHS["graphs"][tension]
            
            logger.info(f"Running optimization for {tension} (Radius: {scen['dist']}m)...")
            G = load_graph(path_load=path)
            
            analysis = RedundancyAnalysis(
                G, 
                zoom_location_manual=scen["coords"], 
                max_distance_m=scen["dist"], 
                zoom_radius_deg=0.001, 
                top_n=500, 
                network_name=scen["name"]
            )
            
            # 1. Find candidates (Tie-Lines logic)
            analysis.find_connection_candidates()
            
            # 2. Generate and Save the Optimized Graph (Used in Phase 6)
            # Using view=False to avoid blocking execution with pop-ups
            analysis.generate_redundancy_graph(budget_limit_meters=scen["dist"], view=False) 
            
            del G, analysis
            
        except FileNotFoundError:
            logger.error(f"Graph file for {tension} not found.")
        except Exception as e:
            logger.error(f"Error in redundancy analysis ({scen['name']}): {e}")

    logger.info("PHASE 5: Redundancy Analysis complete.")


# --- Phase 6: Comparative Analysis (Validation) ---
def run_comparative_analysis():
    logger.info("PHASE 6: Starting Comparative Failure Analysis (Validation)...")

    # Mapping of comparisons: (Original Path, Optimized Path, Label)
    # Note: Optimized paths are derived from the 'network_name' used in Phase 5
    # Standard format: ./data/graph/{name_replaced_spaces}_OPTIMIZED.pickle
    
    comparisons = [
        {
            "orig": DATA_PATHS["graphs"]["BT"],
            "opt": "./data/graph/CPFL_Paulista_BT_Redundancy_Map_5000_5000_OPTIMIZED.pickle",
            "subname": "5000m",
            "type": "CPFL Paulista BT Comparison"
        },
        {
            "orig": DATA_PATHS["graphs"]["BT"],
            "opt": "./data/graph/CPFL_Paulista_BT_Redundancy_Map_15000_15000_OPTIMIZED.pickle",
            "subname": "15000m",
            "type": "CPFL Paulista BT Comparison"
        },
        {
            "orig": DATA_PATHS["graphs"]["MT"],
            "opt": "./data/graph/CPFL_Paulista_MT_Redundancy_Map_5000_5000_OPTIMIZED.pickle",
            "subname": "5000m",
            "type": "CPFL Paulista MT Comparison"
        },
        {
            "orig": DATA_PATHS["graphs"]["MT"],
            "opt": "./data/graph/CPFL_Paulista_MT_Redundancy_Map_15000_15000_OPTIMIZED.pickle",
            "subname": "15000m",
            "type": "CPFL Paulista MT Comparison"
        }
    ]

    for comp in comparisons:
        try:
            logger.info(f"Comparing: {comp['type']} ({comp['subname']})...")
            
            analyzer = ComparativeFailureAnalysis(
                path_original=comp["orig"],
                path_redundant=comp["opt"],
                base_unit=0.6,
                view=False
            )
            
            # Run comparison simulation (random vs targeted for both networks)
            analyzer.run_analysis(
                tension_type=f"{comp['type']} {comp['subname']}", 
                num_steps=1000
            )
            
            del analyzer
            
        except Exception as e:
            logger.error(f"Error in comparative analysis ({comp['subname']}): {e}")

    logger.info("PHASE 6: Comparative Analysis complete.")


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # Execute enabled phases
    if PIPELINE_FLAGS["generate_graphs"]:
        run_graph_generation()
    else:
        logger.info("Skipping Phase 1 (Generation).")

    if PIPELINE_FLAGS["topology_analysis"]:
        run_topology_analysis()
    else:
        logger.info("Skipping Phase 2 (Topology).")

    if PIPELINE_FLAGS["criticality_analysis"]:
        run_criticality_analysis()
    else:
        logger.info("Skipping Phase 3 (Criticality).")
        
    if PIPELINE_FLAGS["failure_analysis"]:
        run_failure_analysis()
    else:
        logger.info("Skipping Phase 4 (Failure Analysis - Original).")

    if PIPELINE_FLAGS["redundancy_analysis"]:
        run_redundancy_analysis()
    else:
        logger.info("Skipping Phase 5 (Redundancy Investment).")

    if PIPELINE_FLAGS["comparative_analysis"]:
        run_comparative_analysis()
    else:
        logger.info("Skipping Phase 6 (Comparative Validation).")
    
    logger.info("Pipeline execution finished.")