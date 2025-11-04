import logging
import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from utils.save_load_graph import save_graph

logger = logging.getLogger(__name__)


class GraphGenerator:
    """
    Class for generating a topological graph model of an electrical network from node (pole) and link (cable) files.

    This class handles graph ingestion, cleaning (ETL), and modeling. Cleaning includes geospatial validation of the links to eliminate anomalies based on physical distance.
    """

    def __init__(
        self,
        path_nodes: str,
        path_links: str,
        name: str = "Network Topology",
        threshold_km: float = 1.0,
    ):
        """
        Initializes the graph generator.

        Args:
            path_nodes (str):       Path to the CSV file containing the nodes
                                    (poles).Must contain 'COD_ID', 'GEO_X' (Latitude),'GEO_Y' (Longitude).
            path_links (str):       Path to the CSV file containing the link.
                                    (cables). Must contain 'COD_ID', 'PN_CON_1' (Source Node), 'PN_CON_2' (Destination Node).
            name (str):             Descriptive name for the graph (e.g., "CMPF
                                    Network").
            threshold_km (float):   Maximum distance (in km) that a link can
                                    have to be considered valid. Links longer than this will be considered anomalous and will be removed.
        """

        self.name = name
        self.threshold_km = threshold_km

        # IMPORTANT: Use nx.Graph() (Undirected).
        # This is ideal for Medium Voltage (MV) robustness analysis,
        # as it allows you to find alternative paths and loops.
        # For Low Voltage (LV), consider using nx.DiGraph() to
        # model the unidirectional flow (Transformer -> Customer).

        self.G = nx.Graph()

        logger.info(f"Loading data from {path_nodes} y {path_links}...")
        try:
            # --- Loading of Nodes (Graph Vertices) ---
            df_nodos = pd.read_csv(
                path_nodes,
                usecols=["COD_ID", "GEO_X", "GEO_Y"],
                dtype={"COD_ID": str},
            )

            # Rename columns for clarity (Geo Standard)
            self.df_nodos = df_nodos.rename(
                columns={
                    "COD_ID": "Nodo_ID",
                    "GEO_X": "Latitude",
                    "GEO_Y": "Length",
                }
            )

            # --- Link Loading (Graph Edges) ---
            # It is assumed that the links connect two post IDs (PN_CON_1 and PN_CON_2)
            df_links = pd.read_csv(
                path_links,
                usecols=["COD_ID", "PN_CON_1", "PN_CON_2"],
                dtype={
                    "COD_ID": str,
                    "PN_CON_1": "str",
                    "PN_CON_2": "str",
                },
            )

            self.df_links = df_links.rename(
                columns={
                    "COD_ID": "Link_ID",
                    "PN_CON_1": "Input",
                    "PN_CON_2": "Output",
                }
            )

            logger.info("Data successfully loaded.")

        except FileNotFoundError as e:
            logger.error(f"ERROR: File not found {e.filename}. Check the route")
            exit()
        except Exception as e:
            logger.error(f"Unexpected error during data loading: {e}")
            exit()

    @staticmethod
    def __haversine(lat1: float, len1: float, lat2: float, len2: float) -> float:
        """
        Calculate the geodesic (great circle) distance between two points on the Earth's surface using the Haversine formula.

        Args:
            lat1 (float): Latitude of point 1.
            len1 (float): Longitude of point 1.
            lat2 (float): Latitude of point 2.
            len2 (float): Longitude of point 2.

        Returns:
            float: Distance between the two points in kilometers.

        """

        # Radius of the Earth in kilometers
        R = 6371

        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        len1_rad = math.radians(len1)
        lat2_rad = math.radians(lat2)
        len2_rad = math.radians(len2)

        # Differences
        dlat = lat2_rad - lat1_rad
        dlen = len2_rad - len1_rad

        # Haversine Formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlen / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        return distance

    @staticmethod
    def __view_graph_topology(
        graph: nx,
        title: str = "Electrical Grid Topology",
        base_unit: int = 0.5,
        view=True,
    ):
        """
        Generates a geospatial visualization of the topology of graph G.

        Args:
            graph(nx.Graph):    Clean NetworkX graph object.
            title(str):         Main title of the graph.
            base_unit(tuple):   Multiplier for the size of the figure (width, 
                                height).
            view(bool):         If True, displays the graph (plt.show()).
        """

        logger.info("Starting graph visualization...")

        # Base colors for visualization
        COLOR__NODES = "#E84F5E"  # noqa: F841
        COLOR_EDGES = "#FCDFC5"

        # 1. Position Calculation (Geospatial Layout)
        # Extract the (Lat, Lon) coordinates from the node attributes
        # (Lon, Lat) are inverted for (X, Y) plotting
        # Note: The original code uses (Lat, Lon) as (X, Y). We keep it
        # for consistency with __graph_creating_model, even though (Lon, Lat) is more standard.
        try:
            geo = {
                nodo: (graph.nodes[nodo]["length"], graph.nodes[nodo]["latitude"])
                for nodo in graph.nodes()
            }
        except KeyError:
            logger.error("Error extracting 'latitude'/'length' from nodes. Please check the attributes.")
            # Try again with the original order (Lat, Lon)
            try:
                geo = {
                    nodo: (graph.nodes[nodo]["latitude"], graph.nodes[nodo]["length"])
                    for nodo in graph.nodes()
                }
            except Exception as e:
                logger.error(f"Final failure to extract positions: {e}")
                return

        # 2. Calculation of Metrics (Grade) for Aesthetics
        try:
            node_degrees_values = dict(graph.degree()).values()
            node_degrees = np.array(list(node_degrees_values))

        except Exception as e:
            logger.error(f"Calculation of Metrics (Grade) for Aesthetics: {e}")
            return

        if node_degrees.size == 0:
            logger.warning("The graph contains no nodes to graph.")
            return

        # 3. Color Calculation (Normalization)
        # BUG-FIX: The original code calculates 'colors_by_grade' but doesn't use them.
        # The fixed color (COLOR_NODES) is maintained as in the original.
        max_degree = np.max(node_degrees) if np.max(node_degrees) > 0 else 1
        normalized_degrees = node_degrees / max_degree
        cmap_pastel = plt.cm.get_cmap("PuRd_r")
        colors_by_grade = cmap_pastel(normalized_degrees)  # noqa: F841

        # 4. Global Style Settings
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(16 * base_unit, 9 * base_unit))
        ax.set_facecolor("white")
        ax.grid(color="#CCDDEA", alpha=0.3)
        fig.patch.set_facecolor("white")

        # 5. Graph Drawing
        logger.info("Rendering links...")
        nx.draw_networkx_edges(
            graph,
            geo,
            ax=ax,
            alpha=0.65,
            edge_color=COLOR_EDGES,
            width=1.2,
        )

        logger.info("Rendering nodes...")
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

        # 6. Titles and Annotations
        plt.title(title, fontsize=18, color="black", loc="left", y=1.05)
        plt.text(
            0.00,
            1.02,
            f"Nodos: {graph.number_of_nodes()} | Links: {graph.number_of_edges()}",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )
        
        # Hide axes (since they are geo coordinates)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        # 7. Save and Display
        output_path = os.path.join("./reports/", title.replace(" ", "_") + ".png")
        plt.savefig(
            output_path,
            dpi=500,
            bbox_inches="tight",
        )

        logger.info(f"Graph saved in {output_path}")

        if view:
            plt.show()
            logger.info("Graph generated and displayed successfully.")
        
        # Liberar memoria
        plt.close(fig) 

    def __cleaning_nodes_isolated(self):
        """
        Cleaning Step 1: Remove Isolated Nodes.

        Filter the node DataFrame (self.df_nodes) to keep
        only those nodes that are referenced in the
        link DataFrame (self.df_links).
        """

        logger.info("Starting cleanup of isolated nodes...")

        # Get all the unique node IDs participating in a link
        nodes_in_links_1 = set(self.df_links["Input"]) 
        nodes_in_links_2 = set(self.df_links["Output"])

        # Set of all active nodes
        active_nodes = nodes_in_links_1.union(nodes_in_links_2)
        original_nodes = self.df_nodos.shape[0]

        # Filter the DataFrame of nodes
        self.df_nodos = self.df_nodos[self.df_nodos["Nodo_ID"].isin(active_nodes)]

        filtered_nodes = self.df_nodos.shape[0]

        logger.info(f"Cleaning of isolated nodes completed. Nodes removed: {original_nodes - filtered_nodes}")

    def __cleaning_anomalous_links(self):
        """
        Cleanup Step 2: Remove Anomalous Links (geospatially).

        Identify and remove links whose physical distance (Haversine)
        between their connecting nodes exceeds the threshold (self.threshold_km).

        This is crucial for eliminating data errors.
        """
        logger.info("Starting cleanup of anomalous links...")

        # Ensure that the IDs are of the same type (str) for the merge
        self.df_nodos["Nodo_ID"] = self.df_nodos["Nodo_ID"].astype(str)
        self.df_links["Input"] = self.df_links["Input"].astype(str)
        self.df_links["Output"] = self.df_links["Output"].astype(str)

        # 1. Enrich df_links with source coordinates (Input)
        df_merged = self.df_links.merge(
            self.df_nodos,
            left_on="Input",
            right_on="Nodo_ID",
            how="left",
        )

        df_merged = df_merged.rename(
            columns={
                "Latitude": "Lat_Origin",
                "Length": "Len_Origin",
            }
        ).drop(columns="Nodo_ID")

        # 2. Enrich df_links with destination coordinates (Output)
        df_merged = df_merged.merge(
            self.df_nodos,
            left_on="Output",
            right_on="Nodo_ID",
            how="left",
            suffixes=("_Origin", "_Destination"),
        ).rename(
            columns={
                "Latitude": "Lat_Destination",
                "Length": "Len_Destination",
            }
        ).drop(columns="Nodo_ID")

        # 3. Calculate the Haversine distance for each link
        logger.info("Calculating Haverine distances for all links...")
        
        # Handle links where a coordinate is missing (NaN)
        df_merged_valid = df_merged.dropna(subset=[
            "Lat_Origin", "Len_Origin", 
            "Lat_Destination", "Len_Destination"
        ])
        
        if df_merged_valid.empty:
            logger.warning("There are no links with valid coordinates for calculating distances.")
            return

        df_merged_valid["Distance_KM"] = df_merged_valid.apply(
            lambda row: self.__haversine(
                row["Lat_Origin"],
                row["Len_Origin"],
                row["Lat_Destination"],
                row["Len_Destination"],
            ),
            axis=1,
        )

        # 4. Filter and Remove anomalous links
        df_anomalous_links = df_merged_valid[
            df_merged_valid["Distance_KM"] > self.threshold_km
        ]

        logger.info(f"ANOMALOUS LINKS IDENTIFIED (> {self.threshold_km} KM)")

        if df_anomalous_links.empty:
            logger.info("No links were found that exceeded the threshold.")
        else:
            total_anomalos = len(df_anomalous_links)
            logger.warning(f"They were identified {total_anomalos} anomalous links. Eliminating them.")
            
            # Optional: Log some examples
            if total_anomalos < 20:
                for _idx, row in df_anomalous_links.iterrows():
                    logger.debug(f"  -> Anómalo: {row['Link_ID']} ({row['Input']} a {row['Output']}): {row['Distance_KM']:.2f} km")

            # Get the IDs of the links to delete
            links_to_delete = set(df_anomalous_links["Link_ID"])
            
            # Filter the main link DataFrame
            links_originales = self.df_links.shape[0]
            self.df_links = self.df_links[
                ~self.df_links["Link_ID"].isin(links_to_delete)
            ]
            links_filtrados = self.df_links.shape[0]
            logger.info(f"Link cleanup complete. Links removed: {links_originales - links_filtrados}")

        # Free up memory
        del df_merged
        del df_merged_valid
        del df_anomalous_links

    def __graph_creating_model(self):
        """
        Step 3: Building the Graph Object (NetworkX).
        Iterate over the DataFrames, removing all nodes and links, to build the graph self.G.
        """
        logger.info("Starting graph modeling.")

        # --- Add Nodes ---
        # Save geospatial positions for the display layout
        self.geo = {}
        for _index, row in self.df_nodos.iterrows():
            self.G.add_node(
                row["Nodo_ID"],
                latitude=row["Latitude"],
                length=row["Length"]
            )
            
            # Save position (Lat, Lon) for display
            # (Consistent with the bug/feature of __view_graph_topology)
            self.geo[row["Nodo_ID"]] = (row["Latitude"], row["Length"])

        logger.info(f"Additions {self.G.number_of_nodes()} nodes to the graph.")

        # --- Add Edges (Links) ---
        edges_added = 0
        edges_ignored = 0
        for _index, row in self.df_links.iterrows():
            # Robustness check:
            # Only add the link if both nodes (posts) exist in the graph.
            # This handles links that reference nodes that have been cleaned.
            if row["Input"] in self.G and row["Output"] in self.G:
                self.G.add_edge(row["Input"], row["Output"])
                edges_added += 1
            else:
                # This helps with data cleansing and identifies links
                # to undefined (likely cleaned) nodes
                logger.warning(
                    f"Warning: The link {row['Link_ID']} ignore the connection. "
                    f"Node(s) not found: {row['Input']} o {row['Output']}. Ignoring."
                )
                edges_ignored += 1

        logger.info(f"Added {edges_added} edges (links) to the graph.")
        if edges_ignored > 0:
            logger.warning(f"They were ignored {edges_ignored} edges for missing nodes.")

        logger.info("Graph modeling completed.")

        # --- Basic Metrics ---
        logger.info("--- Basic Graphic Metrics ---")
        logger.info(
            f"Total Nodes (Posts): {self.G.number_of_nodes()}"
        )
        logger.info(f"Total Edges (Cables): {self.G.number_of_edges()}")
        
        # Connectivity analysis can be slow in massive graphs.
        # It's better to analyze connected components separately.
        try:
            num_components = nx.number_connected_components(self.G)
            logger.info(f"Number of connected components (islands): {num_components}")
            if num_components > 1:
                logger.warning("The network is not fully connected. It consists of several separate 'islands'.")
            else:
                logger.info("The network is fully connected.")
        except Exception as e:
            logger.error(f"Connectivity could not be calculated: {e}")


        # --- Save Graph ---
        # Save the graph object (with all attributes) so that
        # you don't have to repeat the ETL process in future analyses.
        path_save = os.path.join(
            "./data/graph", self.name.replace(" ", "_") + ".pickle"
        )
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        
        logger.info(f"Saving graph in {path_save}")
        save_graph(path_save=path_save, graph=self.G)
        logger.info("Graph saved successfully.")

    def create_graph(self, view=True):
        """
        Main public method for orchestrating graph creation.
        Executes the cleaning and modeling sequence.

        Args:
            view(bool): If True, attempts to display a visualization 
            of the resulting graph (not recommended for large graphs).
        """
        # 1. Clean geospatially anomalous links (in. > 1 km)
        self.__cleaning_anomalous_links()
        
        # 2. Clean up nodes that are not participating in any valid links
        self.__cleaning_nodes_isolated()
        
        # 3. Build the graph object in NetworkX
        self.__graph_creating_model()
        
        # 4. Visualize (if requested and the graph is not massive)
        if view:
            self.__view_graph_topology(graph=self.G, title=self.name, view=view)
        else:
            logger.info("Display omitted (view=False or graph too large).")
            
        logger.info(f"Process completed for the graph: {self.name}")
        return self.G
