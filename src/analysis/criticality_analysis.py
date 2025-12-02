import networkx as nx
import pandas as pd
import numpy as np
import os
import time
import logging
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

# --- Configuración ---
logger = logging.getLogger(__name__)


def _centrality_parallelization(graph: nx.Graph, k_sample: int) -> tuple:
    try:
        num_nodos = graph.number_of_nodes()
        if num_nodos < 3:
            return {}, set()

        k = min(k_sample, num_nodos)

        cb_values = nx.betweenness_centrality(
            graph,
            k=k,
            normalized=True,
            seed=42,
        )
        pa_values = set(nx.articulation_points(graph))

        return cb_values, pa_values

    except Exception as e:
        logger.error(f"[Worker Error] Error processing a component: {e}")
        return {}, set()


class CriticalityAnalysis:
    def __init__(
        self,
        graph: nx.Graph,
        k_sample: int = 5000,
        n_processes: int = 8,
        report_name: str = "Analysis of the Electrical Network",
        report_subname: str = "CPFL Paulista",
    ):
        self.graph = graph

        self.report_subname = report_subname
        self.report_name = report_name

        self.k_sample = k_sample
        self.n_processes = n_processes

        ordered_components = sorted(
            nx.connected_components(graph),
            key=len,
            reverse=True,
        )
        sizes = []
        for oc in ordered_components:
            if len(oc) > 10:
                sizes.append(len(oc))
        self.max_components = len(sizes)

        self.name_path = (
            report_name.replace(" ", "_") + "_" + report_subname.replace(" ", "_")
        )

    def __save_results_csv(self, df_critical: pd.DataFrame):
        try:
            os.makedirs("./reports/", exist_ok=True)
            path_report = os.path.join(
                "./reports/", self.name_path + "_criticality.csv"
            )
            df_critical.to_csv(path_report, index=False, encoding="utf-8", sep=";")
            logger.info(f"Criticality results saved in: {path_report}")
        except Exception as e:
            logger.error(f"The CSV file could not be saved: {e}")

    def __plot_criticality_heat_map(
        self,
        df_critical: pd.DataFrame,
        view: bool = True,
        base_unit: int = 0.6,
    ):
        logger.info("Generating a criticality heat map...")
        COLOR__NODES = "#E84F5E"  # noqa: F841
        COLOR_EDGES = "#FCDFC5"

        try:
            pos = {
                node: (
                    self.graph.nodes[node]["length"],
                    self.graph.nodes[node]["latitude"],
                )
                for node in self.graph.nodes()
                if "length" in self.graph.nodes[node]
                and "latitude" in self.graph.nodes[node]
            }

            if not pos:
                logger.warning(
                    "The heat map was not generated: 'length' or 'latitude' attributes are missing from the nodes."
                )
                return

            # plot_data = df_critical.sort_values(
            #     by="Intermediation_CB", ascending=True
            # ).reset_index(drop=True)
            plot_data = df_critical.set_index("COD_ID").to_dict("index")
            critical_nodes = list(plot_data.keys())
            node_colors = [
                plot_data[node]["Intermediation_CB"] for node in critical_nodes
            ]

            plt.style.use("ggplot")
            fig, ax = plt.subplots(figsize=(16 * base_unit, 9 * base_unit))
            ax.set_facecolor("white")
            # ax.grid(color="#CCDDEA", alpha=0.3)
            fig.patch.set_facecolor("white")

            logger.info("Drawing edges...")
            nx.draw_networkx_edges(
                self.graph,
                pos,
                ax=ax,
                alpha=0.45,
                edge_color="whitesmoke",
                width=0.5,
            )

            logger.info("Drawing nodes (nodes)...")
            pos_critics = {node: pos[node] for node in critical_nodes if node in pos}

            nodes = nx.draw_networkx_nodes(
                self.graph,
                pos_critics,
                ax=ax,
                nodelist=list(pos_critics.keys()),
                node_size=15,
                node_color=node_colors,
                cmap=plt.get_cmap("PuRd"),
                alpha=1.0,
            )

            cbar = plt.colorbar(nodes, ax=ax)
            cbar.set_label("Intermediation Centrality (IC)")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="black")

            plt.title(
                "Node Criticality Heat Map",
                fontsize=18,
                color="black",
                loc="left",
                y=1.05,
            )

            plt.text(
                0.025,
                1.02,
                f"Network: {self.report_subname} | Sampling > {self.k_sample} nodes",
                transform=ax.transAxes,
                fontsize=12,
                color="#5B5B5B",
            )
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.grid(
                # axis="y",
                alpha=0.45,
                zorder=1,
                which="both",  # Grid for major and minor ticks
                color="whitesmoke",
            )
            plt.tight_layout()

            try:
                os.makedirs("./reports/", exist_ok=True)
                path_report = os.path.join(
                    "./reports/", self.name_path + "_criticality.png"
                )
                plt.savefig(path_report, dpi=500, bbox_inches="tight")
                logger.info(f"Criticality Heat map saved in: {path_report}")

                if view:
                    plt.show()
                plt.close(fig)
            except Exception as e:
                logger.error(f"The PNG file could not be saved: {e}")

        except KeyError as e:
            logger.warning(
                f"No se generó el mapa de calor. Falta atributo de nodo: {e}"
            )
        except Exception as e:
            logger.error(f"Error generando mapa de calor: {e}")

    def analizar_criticidad_paralela(self, view: bool = True):
        start_time = time.time()

        logger.info(f"START: Parallel Criticality Analysis for: {self.report_subname} ")

        components = list(nx.connected_components(self.graph))
        components.sort(key=len, reverse=True)
        num_total_components = len(components)
        num_analyze = min(num_total_components, self.max_components)
        logger.info(f"The total network has {num_total_components} components.")
        logger.info(f"The following were analyzed: {num_analyze} larger components.")

        subgraphs_analyze = [
            self.graph.subgraph(c).copy() for c in components[:num_analyze]
        ]

        logger.info("Starting multiprocessing pool... (this may take a while)")

        worker_con_k = partial(_centrality_parallelization, k_sample=self.k_sample)

        with Pool(processes=self.n_processes) as pool:
            lista_resultados = pool.map(worker_con_k, subgraphs_analyze)

        logger.info("Parallel calculation complete. Consolidating results...")

        final_cb_results = {}
        final_pa_results = set()
        for cb_dict, pa_set in lista_resultados:
            final_cb_results.update(cb_dict)
            final_pa_results.update(pa_set)

        if not final_cb_results:
            logger.error("Analysis failed: No CB results were obtained.")
            return

        df_critical = pd.DataFrame(
            list(final_cb_results.items()), columns=["COD_ID", "Intermediation_CB"]
        )
        df_critical["is_AP"] = df_critical["COD_ID"].isin(final_pa_results)

        max_cb = df_critical["Intermediation_CB"].max()
        df_critical["Normalized_Criticality"] = (
            (df_critical["Intermediation_CB"] / max_cb) * 100 if max_cb > 0 else 0
        )
        df_critical = df_critical.sort_values(
            by="Intermediation_CB", ascending=True
        ).reset_index(drop=True)

        self.__save_results_csv(df_critical)

        end_time = time.time()
        execution_time = end_time - start_time
        num_nodes_analyzed = len(df_critical)
        num_pa_found = len(final_pa_results)

        report_content = f"""
        CRITICALITY ANALYSIS COMPLETED: {self.report_subname}

        Total Execution Time: {execution_time:.2f} seconds
        Analyzed Nodes (in Top {num_analyze} comp.): {num_nodes_analyzed:,}
        Total Articulation Points Identified: {num_pa_found:,}

        -- TOP 10 MOST CRITICAL Nodes (Highest CB) ---
        {df_critical.tail(10).to_string(index=False)}
        """

        logger.info(report_content)

        self.__plot_criticality_heat_map(df_critical=df_critical, view=view)

        logger.info("Process completed")
