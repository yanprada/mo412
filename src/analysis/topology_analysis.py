import datetime
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Set up the logger for this module
logger = logging.getLogger(__name__)


class TopologyAnalysis:
    """
    Performs a descriptive topological analysis on a given NetworkX graph.

    This class calculates fundamental graph metrics to understand the scale,
    connectivity, and structural properties of the network. It generates
    text reports and visualizations.
    """

    def __init__(
        self,
        graph: nx.Graph,
        report_subname: str = "CPFL",
        report_name: str = "Analysis of the Electrical Network Topology",
        base_unit: float = 0.5,
        minimum_node_size: int = 2,
    ):
        """
        Initializes the TopologyAnalysis class.

        Args:
            graph (nx.Graph): The NetworkX graph object to be analyzed.
            report_subname (str): A suffix for report filenames (e.g., "MT_Network").
            report_name (str): The main prefix for report filenames.
            base_unit (float): A multiplier for scaling plot dimensions.
            minimum_node_size (int): The minimum number of nodes for a
                component to be considered a "subnet" (not just an isolated node).
        """
        self.G = graph
        self.report_subname = report_subname
        self.report_name = report_name
        self.base_unit = base_unit
        self.minimum_node_size = minimum_node_size
        
        # Create a standardized base filename for all reports
        self.name_path = (
            report_name.replace(" ", "_") + "_" + self.report_subname.replace(" ", "_")
        )

    def connectivity_scale(self):
        """
        Calculates basic connectivity and scale metrics for the graph G.

        This function computes:
        - Overall scale (nodes, edges)
        - Density (E/V ratio)
        - Component analysis (LCC size, number of components)
        - Degree statistics (max, mean, std)
        - Criticality (articulation points)
        - Redundancy (Gamma index)
        
        Generates a summary text file with expert interpretation.
        """
        logger.info("Starting connectivity and scale analysis...")

        num_nodes = self.G.number_of_nodes()
        num_links = self.G.number_of_edges()

        if num_nodes == 0:
            logger.warning("The graph is empty. Analysis cannot be performed.")
            return

        # 1. Density (E/V ratio)
        # A simple measure of how many links exist per node.
        # Radial (tree-like) networks will have a density very close to 1.
        network_density = num_links / num_nodes

        # 2. Connected Components
        num_components = nx.number_connected_components(self.G)
        # Find the Largest Connected Component (LCC)
        main_component_nodes = max(nx.connected_components(self.G), key=len)
        main_component_size = len(main_component_nodes)
        percentage_main_component = (main_component_size / num_nodes) * 100

        # 3. Degree Metrics
        # Degree quantifies the number of connections for each node.
        node_degrees = np.array(list(dict(self.G.degree()).values()))
        maximum_degree = node_degrees.max()  # The network's biggest "hub"
        average_degree = node_degrees.mean()
        std_dev_degree = node_degrees.std() # Measures network heterogeneity

        # 4. Articulation Points (Cut Vertices)
        # These are *critical* nodes. If an articulation point fails,
        # it splits its component into two or more disconnected pieces.
        # These are the primary single points of failure.
        logger.info("Calculating articulation points... (this may take time)")
        articulation_points = list(nx.articulation_points(self.G))
        num_articulation_points = len(articulation_points)
        percentage_articulation_points = (num_articulation_points / num_nodes) * 100
        logger.info("Articulation points calculated.")

        # 5. Gamma Index (Topological Redundancy)
        # This metric compares the number of actual edges to the
        # maximum *possible* number of edges in a complete graph (clique).
        # It's an academic measure of redundancy.
        
        # We calculate this only for the Main Component (LCC)
        num_nodes_lcc = main_component_size
        num_edges_lcc = self.G.subgraph(main_component_nodes).number_of_edges()

        if num_nodes_lcc > 1:
            # Max possible edges in a graph with V nodes is V*(V-1)/2
            max_possible_edges_lcc = (num_nodes_lcc * (num_nodes_lcc - 1)) / 2
            gamma = num_edges_lcc / max_possible_edges_lcc
        else:
            gamma = 0.0 # Define as 0 if LCC is a single node

        # 6. Generate Report
        report_content = f"""
        =====================================================
        CONNECTIVITY AND SCALE REPORT
        =====================================================
        Analysis Date/Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Network: {self.report_subname}
        -----------------------------------------------------

        [1. SCALE METRICS & GLOBAL TOPOLOGY]
        - Total Nodes (|V|): ---------------- {num_nodes:,}
        - Total Links (|E|): ---------------- {num_links:,}
        - Total Connected Components: ------- {num_components}
        - Average Density (E/V Ratio): ------ {network_density:.3f}

        [2. DEGREE CENTRALITY ANALYSIS]
        - Maximum Degree (Network Hub): ----- {maximum_degree}
        - Average Degree (|k|): ------------- {average_degree:.3f}
        - Std. Deviation of Degree: --------- {std_dev_degree:.3f}
        
        [3. REDUNDANCY AND MAIN COMPONENT]
        - Nodes in LCC (Main Component): ---- {main_component_size:,} nodes
        - LCC Coverage Percentage: ---------- {percentage_main_component:.3f}%
        - Gamma Index (LCC Redundancy): ----- {gamma:.6f}

        [4. VULNERABILITY (SINGLE POINTS OF FAILURE)]
        - Total Articulation Points: -------- {num_articulation_points}
        - Percentage of Nodes as APs: ------- {percentage_articulation_points:.3f}%

        -----------------------------------------------------
        EXPERT INTERPRETATION:
        - LCC Coverage Percentage: If this is < 95%, it implies a significant
          portion of assets are not connected to the main grid. This could
          indicate data quality issues or confirm the existence of
          many independent, isolated systems.
        
        - Average Density: Indicates network redundancy. A value < 1.5
          suggests a topology closer to a tree or radial network (low
          redundancy). MT networks are often slightly > 1.5, while
          BT networks are almost exactly 1.0.

        - Std. Deviation of Degree: A high value implies a heterogeneous
          network (i.e., connectivity is controlled by a few very
          large hubs), which is typical for utility networks.

        - Gamma Index: Measures how "close" the network is to being a
          complete graph. In sparse utility networks (radial or
          branched), this value will be extremely low (near zero).
          
        - Articulation Points: These are the network's critical weak points.
          The lower the percentage of articulation points, the more
          robust (redundant) the network is against node failures. A
          perfect (but non-existent) radial network would have
          almost all non-leaf nodes as articulation points.

        =====================================================
        """

        try:
            # Ensure the ./reports/ directory exists
            os.makedirs("./reports/", exist_ok=True)
            path_report = os.path.join(
                "./reports/", self.name_path + "_connectivity.txt"
            )
            with open(path_report, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"Connectivity report saved to '{path_report}'.")
            logger.info(report_content) # Also print to console

        except Exception as e:
            logger.error(f"Error saving connectivity report: {e}")

    def components_distribution_analysis(self):
        """
        Analyzes the size distribution of all connected components.

        This function identifies all "islands" in the network, separates
        the "real subnets" from "isolated nodes" (based on minimum_node_size),
        and generates a log-log histogram to visualize the distribution.
        This helps identify network fragmentation.
        """
        logger.info("Starting analysis of connected component distribution.")

        # Get a list of all connected components (as sets of nodes)
        components = list(nx.connected_components(self.G))
        
        if not components:
            logger.warning("No components found in the graph.")
            return

        # Store component data
        component_data = []
        for component in components:
            node_size = len(component)
            # Get a sample node ID to represent the component
            id_component = next(iter(component))
            
            component_data.append(
                {"id_component": id_component, "node_size": node_size}
            )

        df_components = pd.DataFrame(component_data)

        # 1. Separate isolated nodes from "real" subnets
        df_isolated = df_components[df_components["node_size"] < self.minimum_node_size]
        
        df_real_subnets = (
            df_components[df_components["node_size"] >= self.minimum_node_size]
            .sort_values(by="node_size", ascending=False)
            .reset_index(drop=True)
        )

        # 2. Calculate metrics
        num_real_components = len(df_real_subnets)
        num_isolated_nodes = len(df_isolated)
        
        if num_real_components > 0:
            maximum_size = df_real_subnets["node_size"].max()
            average_size = df_real_subnets["node_size"].mean()
        else:
            maximum_size = 0
            average_size = 0

        # 3. Generate text report
        report_content = f"""
        =======================================================================
        CONNECTED COMPONENTS (SUBNETS >= {self.minimum_node_size} NODES) REPORT
        =======================================================================
        - Total "Real" Subnets: ----------- {num_real_components}
        - Discarded Isolated Components: -- {num_isolated_nodes}
        - Average Subnet Size: ------------ {average_size:.3f} nodes
        - Maximum Subnet Size (non-LCC): -- {maximum_size} nodes
        
        (Note: The Largest Connected Component (LCC) is reported
         in the main connectivity report.)
        =======================================================================
        """

        try:
            os.makedirs("./reports/", exist_ok=True)
            path_report = os.path.join("./reports/", self.name_path + "_components.txt")
            with open(path_report, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            logger.info(f"Components distribution report saved to '{path_report}'.")
            logger.info(report_content) # Also print to console

        except Exception as e:
            logger.error(f"Error saving components report: {e}")

        # 4. Generate Histogram (only if there are subnets to plot)
        if num_real_components > 0:
            path_report_img = os.path.join("./reports/", self.name_path + "_components_dist.png")

            # --- Plotting ---
            plt.style.use("ggplot")
            fig, ax = plt.subplots(figsize=(16 * self.base_unit, 9 * self.base_unit))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.spines["bottom"].set_color("#D7EAE2")

            ax.set_xlabel("Component Size (Number of Nodes)", fontsize=10)
            ax.set_ylabel("Frequency (Count of Components)", fontsize=10)
            
            # Use a log-log scale. This is CRITICAL for visualizing
            # heavy-tailed distributions, which are common in networks.
            ax.set_yscale("log")
            ax.set_xscale("log")
            
            ax.grid(
                axis="y",
                alpha=0.65,
                zorder=1,
                which="both", # Grid for major and minor ticks
                color="#D7EAE2",
            )

            # Determine a reasonable number of bins
            bins = int(np.sqrt(num_real_components))
            # Use a fixed larger number if sqrt is too small, for better log-binning
            adjusted_bins = bins if bins > 100 else 300 

            df_real_subnets["node_size"].hist(
                bins=adjusted_bins,
                color="#E84F5E",
                edgecolor="#FCDFC5",
                zorder=2,
                ax=ax,
            )

            plt.title(
                "Subnet Size Distribution (Log-Log Scale)",
                fontsize=18,
                color="black",
                loc="left",
                y=1.07,
            )
            plt.text(
                0.025,
                1.03,
                f"Network: {self.report_subname} | Components > {self.minimum_node_size} nodes",
                transform=ax.transAxes,
                fontsize=12,
                color="#5B5B5B",
            )

            plt.tight_layout()
            plt.savefig(path_report_img, dpi=500, bbox_inches="tight")
            plt.show()
            plt.close(fig) # Release memory
            logger.info(f"Component distribution plot saved to '{path_report_img}'")

        else:
            logging.info(
                f"No components of size >= {self.minimum_node_size} found. "
                "Skipping distribution plot."
            )

        logger.info("Subnet analysis completed.")