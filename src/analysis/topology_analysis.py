import datetime
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TopologyAnalysis:
    def __init__(
        self,
        graph: nx,
        report_subname: str = "CPFL",
        report_name: str = "Analysis of the Topology Electrical Network",
        base_unit=0.5,
        minimum_node_size=2,
    ):
        self.G = graph
        self.report_subname = report_subname
        self.report_name = report_name
        self.base_unit = base_unit
        self.minimum_node_size = minimum_node_size
        self.name_path = (
            report_name.replace(" ", "_") + "_" + self.report_subname.replace(" ", "_")
        )

    def connectivity_scale(self):
        """
        Calculates the basic connectivity and scale metrics of the graph G and generates a report in a text file.
        """
        logger.info("Starting connectivity and scale analysis.")

        num_nodes = self.G.number_of_nodes()
        num_links = self.G.number_of_edges()

        if num_nodes == 0:
            logger.warning("The graph is empty. The analysis cannot be performed.")

            return

        # Redundancy or Density (E/V ratio)
        network_density = num_links / num_nodes

        # Componentes Conexos
        num_components = nx.number_connected_components(self.G)
        main_component = max(nx.connected_components(self.G), key=len)
        main_component_size = len(main_component)

        # Degree metrics
        node_degrees = np.array(list(dict(self.G.degree()).values()))
        maximum_degree = node_degrees.max()
        intermediate_grade = node_degrees.mean()
        standard_deviation_degree = node_degrees.std()
        percentage_main_component = (main_component_size / num_nodes) * 100

        # gamma analysis
        # The Gamma Metric (Topological Redundancy)}
        num_nodes_cp = main_component_size
        num_enlaces_cp = self.G.subgraph(main_component).number_of_edges()

        # Articulation points
        joint_points = list(nx.articulation_points(self.G))
        num_joint_points = len(joint_points)
        percentage_joint_points = (num_joint_points / num_nodes) * 100

        if num_nodes_cp > 1:
            # Número máximo de enlaces posibles en el componente principal
            max_enlaces_cp = (num_nodes_cp * (num_nodes_cp - 1)) / 2
            gamma = num_enlaces_cp / max_enlaces_cp
        else:
            gamma = 0.0

        # Report
        report_content = f"""
        =====================================================
        CONNECTIVITY AND SCALE REPORT
        =====================================================
        Date and Time of Analysis: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        -----------------------------------------------------

        [1. SCALE METRICS AND GLOBAL TOPOLOGY]
        - Total Nodes (|V|): -------------- {num_nodes:,}
        - Total Links (|K|): -------------- {num_links:,}
        - Total Connected Components: ----- {num_components}
        - Average Density (E/V Ratio): ---- {network_density:.3f}

        [2. DEGREE CENTRALITY ANALYSIS]
        - Maximum Degree (Network Hubs): -- {maximum_degree}
        - Intermediate Grade (|k|): ------- {intermediate_grade:.3f}
        - Standard Deviation of Grade: ---- {standard_deviation_degree:.3f}
        
        [3. REDUNDANCY AND MAIN COMPONENT]
        - Nodes in the C. P.: ------------- {main_component_size:,} nodes
        - Network Percentage: ------------- {percentage_main_component:.3f}%
        - Gamma (gamma - CP Redundancy): -- {gamma:.6f}

        [4. ARTICULATION POINTS]
        - Total articulation point: ------- {num_joint_points:.3f}
        - Network Percentage: ------------- {percentage_joint_points:.3f}

        -----------------------------------------------------
        INTERPRETACIÓN EXPERTA:
        - If the Principal Component Percentage is < 95%, it means that a significant portion of the delivery points are not connected to the main body, requiring a data quality inspection or confirmation that these are isolated systems.
        - Average Density indicates the redundancy of the network, if a low value <1.5 may suggest a topology closer to a tree or radial network.
        - Standard Deviation: A high value implies a heterogeneous network (few nodes control connectivity).
        - Gamma (gamma): Measures how “close” your network is to being a complete graph. In distributed networks, this value will be extremely low (close to zero) due to the radial or branched topology.
        - The lower the percentage of articulation points in a network, the more robust the network.

        =====================================================
        """

        try:
            path_report = os.path.join(
                "./reports/", self.name_path + "_connectivity.txt"
            )
            with open(path_report, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"Connectivity report successfully saved in '{path_report}'.")
            logger.info(report_content)

        except Exception as e:
            logger.error(f"Error saving report: {e}")

    def components_distribution_analysis(self):
        """
        Calculates the size of each connected component (subnet), filtering out isolated nodes for better histogram clarity.
        """

        logger.info("Starting analysis of connected component distribution.")

        components = list(nx.connected_components(self.G))
        component_data = []

        for component in components:
            node_size = len(component)
            id_component = next(iter(component))

            component_data.append(
                {"id_component": id_component, "node_size": node_size}
            )

        df_components = pd.DataFrame(component_data)

        # Separate isolated nodes from the rest of the subnets
        df_isolated = df_components[df_components["node_size"] < self.minimum_node_size]

        df_real_subnets = (
            df_components[df_components["node_size"] >= self.minimum_node_size]
            .sort_values(by="node_size", ascending=False)
            .reset_index(drop=True)
        )

        # metrics
        num_real_components = len(df_real_subnets)
        num_isolated_nodes = len(df_isolated)

        maximum_size = (
            df_real_subnets["node_size"].max() if num_real_components > 0 else 0
        )

        average_size = (
            df_real_subnets["node_size"].mean() if num_real_components > 0 else 0
        )

        # report
        report_content = f"""
        =======================================================================
        Connected components (subnets >= {self.minimum_node_size} nodes.
        =======================================================================
        - Total Connected Components: ----- {num_real_components:.3f}
        - Discarded Isolated Components: -- {num_isolated_nodes:.3f}
        - Average Size Components: -------- {average_size:.3f}
        - Maximum Size Components: -------- {maximum_size:.3f}
        
        =======================================================================
        """

        try:
            path_report = os.path.join("./reports/", self.name_path + "_components.txt")
            with open(path_report, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(
                f"Components distribution report successfully saved in '{path_report}'."
            )

            logger.info(report_content)

        except Exception as e:
            logger.error(f"Error saving report: {e}")

        # Histogram
        if num_real_components > 0:
            path_report = os.path.join("./reports/", self.name_path + "_components.png")

            # Global Style Settings
            plt.style.use("ggplot")

            fig, ax = plt.subplots(figsize=(16 * self.base_unit, 9 * self.base_unit))

            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            ax.spines["bottom"].set_color("#D7EAE2")

            ax.set_xlabel("Component Size (Number of Nodes)", fontsize=10)
            ax.set_ylabel("Component Frequency", fontsize=10)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.grid(
                axis="y",
                alpha=0.65,
                zorder=1,
                which="both",
                color="#D7EAE2",
            )

            bins = int(np.sqrt(num_real_components))
            adjusted_bins = bins if bins > 100 else 300

            df_real_subnets["node_size"].hist(
                bins=adjusted_bins,
                color="#E84F5E",
                edgecolor="#FCDFC5",
                zorder=2,
            )

            plt.title(
                "Subnet Size Distribution",
                fontsize=18,
                color="black",
                loc="left",
                y=1.07,
            )
            plt.text(
                0.025,
                1.03,
                f"Network {self.report_subname}, Connections > {self.minimum_node_size}",
                transform=ax.transAxes,
                fontsize=12,
                color="#5B5B5B",
            )

            plt.tight_layout()
            plt.savefig(path_report, dpi=500, bbox_inches="tight")
            plt.show()

        else:
            logging.info(
                f"There are no components of size greater than or equal to the {self.minimum_node_size} for distribution analysis."
            )

        logger.info("Subnet analysis completed.")

    def analizar_criticidad_articulacion_rapida(self, G, k_muestreo=1):
        """
        Calcula los Puntos de Articulación y su Centralidad de Intermediación 
        mediante MUESTREO ESTOCÁSTICO (nx.betweenness_centrality con k) para 
        una ejecución rápida en redes grandes.

        Args:
            G (nx.Graph): Objeto grafo NetworkX limpio.
            k_muestreo (int): Número de nodos a usar en el muestreo estocástico 
                            para estimar la Centralidad de Intermediación (k).
        """
        logger.info("Iniciando análisis rápido de Centralidad de Intermediación mediante MUESTREO.")
        
        # 1. PREPARACIÓN DEL GRAFO PARA ANÁLISIS
        
        # Usar solo el Componente Principal (CP) para análisis de Centralidad, 
        # ya que la Centralidad entre subredes es 0 y el CP contiene la mayor parte de la complejidad.
        if nx.is_connected(G):
            G_analisis = G
        else:
            # Extraer el Componente Principal
            cp = max(nx.connected_components(G), key=len)
            G_analisis = G.subgraph(cp)
            logger.warning(f"Centralidad calculada solo en el Componente Principal ({len(cp):,} nodos).")

        num_nodos_analisis = G_analisis.number_of_nodes()

        # Ajustar k_muestreo si es mayor que el número de nodos
        k = min(k_muestreo, num_nodos_analisis)
        
        # 2. CALCULAR CENTRALIDAD DE INTERMEDIACIÓN CON MUESTREO 🌟
        
        logger.info(f"Calculando Centralidad de Intermediación con k={k} nodos muestreados...")
        
        # El uso de 'k' reduce la complejidad a O(k * |E|), lo que acelera dramáticamente el cálculo.
        cb_values = nx.betweenness_centrality(
            G_analisis, 
            k=k, # <--- LA CLAVE DE LA OPTIMIZACIÓN
            normalized=True, 
            seed=42, # Usar una semilla para resultados reproducibles
        )
        
        # 3. CALCULAR PUNTOS DE ARTICULACIÓN
        # Esta es una operación rápida en NetworkX y no necesita muestreo
        puntos_articulacion = list(nx.articulation_points(G_analisis))
        num_articulacion = len(puntos_articulacion)
        
        if num_articulacion == 0:
            print("\nLa red es robusta y no presenta puntos de articulación.")
            return

        # 4. FILTRAR Y CUANTIFICAR LOS PUNTOS CRÍTICOS
        
        datos_criticos = []
        for nodo_id in puntos_articulacion:
            # Solo necesitamos recuperar el valor de C_B para esos puntos de articulación
            cb_valor = cb_values.get(nodo_id, 0.0) 
            datos_criticos.append({
                'COD_ID': nodo_id,
                'Intermediacion_CB': cb_valor
            })

        df_criticos = pd.DataFrame(datos_criticos)

        # 5. REPORTE Y NORMALIZACIÓN
        
        max_cb = df_criticos['Intermediacion_CB'].max()
        df_criticos['Criticidad_Normalizada'] = (df_criticos['Intermediacion_CB'] / max_cb) * 100 if max_cb > 0 else 0
        
        df_criticos_ordenado = df_criticos.sort_values(
            by='Intermediacion_CB', 
            ascending=False
        ).head(10).reset_index(drop=True)

        print("\n" + "="*70)
        print(f"ANÁLISIS DE CRITICIDAD (OPTIMIZADO CON MUESTREO k={k})")
        print(f"Total de Puntos de Articulación Encontrados: {num_articulacion:,}")
        print("="*70)
        print("\n--- TOP 10 Nodos de Articulación MÁS CRÍTICOS (por Intermediación) ---")
        print(df_criticos_ordenado[['COD_ID', 'Intermediacion_CB', 'Criticidad_Normalizada']].to_string(index=False))
        logger.info("Análisis de criticidad (muestreo) completado exitosamente.")

