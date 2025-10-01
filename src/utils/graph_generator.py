import logging

import networkx as nx
import pandas as pd

from .save_load_graph import save_graph
from .haversine import haversine

logger = logging.getLogger(__name__)


class GraphGenerator:
    def __init__(self, path_nodes: str, path_links: str):
        self.G = nx.Graph()

        logger.info(f"Loading data from {path_nodes} y {path_links}...")
        try:
            # Load the node file (Graph vertices)
            df_nodos = pd.read_csv(
                path_nodes,
                usecols=["COD_ID", "GEO_X", "GEO_Y"],
                dtype={"COD_ID": str},
            )

            self.df_nodos = df_nodos.rename(
                columns={
                    "COD_ID": "Nodo_ID",
                    "GEO_X": "Latitude",
                    "GEO_Y": "Length",
                }
            )

            # Load the links file (Graph edges)
            df_links = pd.read_csv(
                path_links,
                usecols=["COD_ID", "PN_CON_1", "PN_CON_2"],
                dtype={"COD_ID": str},
            )

            self.df_links = df_links.rename(
                columns={
                    "COD_ID": "Link_ID",
                    "PN_CON_1": "Input",
                    "PN_CON_2": "Output",
                }
            )

            logger.info("Data loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"ERROR: File not found {e.filename}. Check the route")
            exit()

    def cleaning_anomalous_links(self, threshold_km: int = 90):
        logger.info("Start of anomalous link cleanup")

        # Ensure IDs are of the same type for the merge
        self.df_nodos["Nodo_ID"] = self.df_nodos["Nodo_ID"].astype(str)
        self.df_links["Input"] = self.df_links["Input"].astype(str)
        self.df_links["Output"] = self.df_links["Output"].astype(str)

        # Merge coordinates to the Links
        # source node
        df_bad_links = self.df_links.merge(
            right=self.df_nodos,
            left_on="Input",
            right_on="Nodo_ID",
            suffixes=("_Link", "_Origin"),
        )

        df_bad_links = df_bad_links.rename(
            columns={
                "Nodo_ID": "Nodo_ID_Origin",
                "Latitude": "Lat_Origin",
                "Length": "Len_Origin",
            }
        )

        # destination node
        df_bad_links = df_bad_links.merge(
            self.df_nodos,
            left_on="Output",
            right_on="Nodo_ID",
            suffixes=("_Origin", "_Destination"),
        ).rename(
            columns={
                "Latitude": "Lat_Destination",
                "Length": "Len_Destination",
            }
        )

        # Distance Calculation and Filtering
        logger.info("Distance and cleaning calculation")

        # Apply the Haversine function to obtain the distance of each link
        df_bad_links["Distance_KM"] = df_bad_links.apply(
            lambda row: haversine(
                row["Lat_Origin"],
                row["Len_Origin"],
                row["Lat_Destination"],
                row["Len_Destination"],
            ),
            axis=1,
        )

        # Filter links that exceed the threshold
        df_bad_links = df_bad_links[
            df_bad_links["Distance_KM"] > threshold_km
        ].sort_values(by="Distance_KM", ascending=False)

        logger.info(f"ANOMALOUS LINKS IDENTIFIED (> {threshold_km} KM)")

        if df_bad_links.empty:
            logger.info("No links were found that exceed the threshold")
        else:
            # Select the relevant columns and format the distance
            df_bad_links = df_bad_links[
                [
                    "Link_ID",
                    "Input",
                    "Output",
                    "Distance_KM",
                ]
            ].head(10)

            df_bad_links["Distance_KM"] = (
                df_bad_links["Distance_KM"].round(2).astype(str) + " km"
            )

            logger.info("Anomalous links identified")
            logger.info(df_bad_links.to_string(index=False))
            logger.info("Eliminating anomalous links")

            links_to_delete = df_bad_links["Link_ID"].tolist()

            self.df_links = self.df_links[
                ~self.df_links["Link_ID"].isin(links_to_delete)
            ]

        logger.info("Process of identifying and removing bad links completed.")

        del df_bad_links

    def graph_creating_model(self):
        logger.info("Starting graph modeling.")

        # Add Nodes
        self.geo = {}
        for index, row in self.df_nodos.iterrows():
            self.G.add_node(
                row["Nodo_ID"], latitude=row["Latitude"], length=row["Length"]
            )

            # Save position (Longitude, Latitude) for geographic layout
            self.geo[row["Nodo_ID"]] = (row["Latitude"], row["Length"])
        # Adding Edges (Links)
        for index, row in self.df_links.iterrows():
            # Solo añadir el link si ambos nodos de conexión existen en el grafo
            if row["Input"] in self.G and row["Output"] in self.G:
                self.G.add_edge(row["Input"], row["Output"])
            else:
                # Esto ayuda a la limpieza de datos e identifica links a nodos no definidos
                logger.warning(
                    f"Warning: The link {row['Link_ID']} ignores connection. Node(s) not found: {row['Input']} o {row['Output']}. Ignoring."
                )

        logger.info("Graphics modeling completed.")

        logger.info("--- Basic graphical metrics ---")
        logger.info(
            f"Total number of Nodes (Substations/Points): {self.G.number_of_nodes()}"
        )
        logger.info(f"Total number of Edges (Lines/Links): {self.G.number_of_edges()}")
        logger.info(
            f"Is the network connected? {'Sí' if nx.is_connected(self.G) else 'No'}. (It could be composed of several separate components)."
        )

    def graph_save(self, path_save: str):
        logger.info(f"Saving graph in {path_save}")
        save_graph(path_save=path_save, graph=self.G)
