import logging
import os
import random
from math import asin, cos, radians, sin, sqrt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle, ConnectionPatch
from tqdm import tqdm  # Barra de progreso esencial para grandes volúmenes

# Importar función de guardado (asegúrate de que este path sea correcto en tu proyecto)
from utils.save_load_graph import save_graph

logger = logging.getLogger(__name__)


class RedundancyAnalysis:
    def __init__(
        self,
        G: nx.Graph,
        network_name: str = "Redundancy Map",
        top_n=200,
        zoom_location_manual: tuple = None,
        zoom_radius_deg: float = 0.005,
        max_distance_m: float = 200.0,
        base_unit=0.6,
    ):
        self.G = G
        self.candidates_df = None
        self.top_n = top_n
        self.zoom_location_manual = zoom_location_manual
        self.zoom_radius_deg = zoom_radius_deg
        self.max_distance_m = max_distance_m
        self.base_unit = base_unit

        self.network_name = network_name
        self.network_subname = f"Optimization: Tie-Lines (Leaf Nodes) | Radius: {max_distance_m}m"

        self.name_path = network_name.replace(" ", "_")

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Calcula distancia en metros entre dos puntos."""
        R = 6371000  # Radio de la Tierra en metros
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        a = (
            sin(dLat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) ** 2
        )
        c = 2 * asin(sqrt(a))
        return R * c

    def _plot_top_candidates(self, view=True):
        """Genera el gráfico con Zoom de los candidatos."""
        logger.info(f"Generando mapa con zoom de Top {self.top_n} candidatos...")

        if self.candidates_df is None or self.candidates_df.empty:
            logger.warning("No hay candidatos para graficar.")
            return

        top_df = self.candidates_df.head(self.top_n)

        # Configuración del lienzo
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(16 * self.base_unit, 9 * self.base_unit))

        plt.title(self.network_name, fontsize=18, color="black", loc="left", y=1.05)
        plt.text(
            0.025,
            1.02,
            f"{self.network_subname}",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Configuración de geometría del Zoom
        if self.zoom_location_manual:
            zoom_center_lat, zoom_center_lon = self.zoom_location_manual
        else:
            # Zoom automático basado en los puntos más críticos
            high_prio = top_df[top_df["Prioridad_Score"] >= 100]
            if not high_prio.empty:
                zoom_center_lat = np.mean(
                    high_prio["Lat_A"].tolist() + high_prio["Lat_B"].tolist()
                )
                zoom_center_lon = np.mean(
                    high_prio["Lon_A"].tolist() + high_prio["Lon_B"].tolist()
                )
            else:
                zoom_center_lat = np.mean(
                    top_df["Lat_A"].tolist() + top_df["Lat_B"].tolist()
                )
                zoom_center_lon = np.mean(
                    top_df["Lon_A"].tolist() + top_df["Lon_B"].tolist()
                )

        z_lon_min = zoom_center_lon - self.zoom_radius_deg
        z_lon_max = zoom_center_lon + self.zoom_radius_deg
        z_lat_min = zoom_center_lat - self.zoom_radius_deg
        z_lat_max = zoom_center_lat + self.zoom_radius_deg

        # Crear Inset Axes para el Zoom
        ax_zoom = fig.add_axes([0.65, 0.60, 0.30, 0.30])  # Arriba a la derecha
        ax_zoom.set_facecolor("white")
        ax_zoom.set_xlim(z_lon_min, z_lon_max)
        ax_zoom.set_ylim(z_lat_min, z_lat_max)

        # Estilo del recuadro de zoom
        for spine in ax_zoom.spines.values():
            spine.set_color("grey")
            spine.set_linewidth(1)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        ax_zoom.set_title("Detalle Zona Crítica", fontsize=10, color="grey")

        # 3. PREPARAR DATOS COMUNES (Fondo)
        pos_full = {}
        for n, data in self.G.nodes(data=True):
            if "length" in data and "latitude" in data:
                pos_full[n] = (data["length"], data["latitude"])

        all_edges = list(self.G.edges())
        # Muestreo para el fondo global si es muy grande
        if len(all_edges) > 50000:
            import random

            random.seed(42)
            global_edges = random.sample(all_edges, int(len(all_edges) * 0.30))
        else:
            global_edges = all_edges

        def draw_on_axis(target_ax, is_zoom_view):
            # En zoom mostramos todo el detalle, en global la muestra
            edges_to_use = all_edges if is_zoom_view else global_edges

            width_n = 2 if is_zoom_view else 1
            alpha_n = 0.8 if is_zoom_view else 0.2

            width_v = 1.0 if is_zoom_view else 0.5
            alpha_v = 1.0 if is_zoom_view else 0.4

            # A. FONDO (ARISTAS)
            if pos_full:
                try:
                    # Dibujar nodos de fondo (muy tenues)
                    if (
                        is_zoom_view
                    ):  # Solo en zoom pintamos nodos de fondo para contexto
                        nodes_bg = nx.draw_networkx_nodes(
                            self.G,
                            pos_full,
                            ax=target_ax,
                            nodelist=list(pos_full.keys()),
                            node_size=width_n,
                            node_color="#F6F3ED",
                            alpha=alpha_n,
                        )
                        if nodes_bg:
                            nodes_bg.set_zorder(1)

                    # Dibujar aristas de fondo
                    e_art = nx.draw_networkx_edges(
                        self.G,
                        pos_full,
                        ax=target_ax,
                        edgelist=edges_to_use,
                        width=width_v,
                        edge_color="#CCCCCC",
                        alpha=alpha_v,
                        arrows=False,
                    )
                    if e_art:
                        e_art.set_zorder(1)
                except Exception:
                    pass

            # B. NUEVAS CONEXIONES (LÍNEAS)
            for _, row in top_df.iterrows():
                # Filtro de vista para el zoom
                if is_zoom_view:
                    in_lon = (z_lon_min <= row["Lon_A"] <= z_lon_max) or (
                        z_lon_min <= row["Lon_B"] <= z_lon_max
                    )
                    in_lat = (z_lat_min <= row["Lat_A"] <= z_lat_max) or (
                        z_lat_min <= row["Lat_B"] <= z_lat_max
                    )
                    if not (in_lon and in_lat):
                        continue

                color = (
                    "#174E4F" if row["Prioridad_Score"] >= 200 else "#E84F5E"
                )  # Verde oscuro (Anillo) vs Rojo (Isla)

                (line,) = target_ax.plot(
                    [row["Lon_A"], row["Lon_B"]],
                    [row["Lat_A"], row["Lat_B"]],
                    color=color,
                    linewidth=2.5 if is_zoom_view else 1.5,
                    alpha=0.9,
                )
                line.set_zorder(2)

            # C. NODOS (PUNTOS)
            coords_high = []  # Anillos (Priority > 200)
            coords_med = []  # Islas (Priority ~ 100)

            for _, row in top_df.iterrows():
                if is_zoom_view:
                    in_lon = z_lon_min <= row["Lon_A"] <= z_lon_max
                    in_lat = z_lat_min <= row["Lat_A"] <= z_lat_max
                    if not (in_lon and in_lat):
                        continue

                target_list = (
                    coords_high if row["Prioridad_Score"] >= 200 else coords_med
                )
                target_list.append((row["Lon_A"], row["Lat_A"]))
                target_list.append((row["Lon_B"], row["Lat_B"]))

            coords_high = list(set(coords_high))
            coords_med = list(set(coords_med))

            if coords_high:
                sc_h = target_ax.scatter(
                    [c[0] for c in coords_high],
                    [c[1] for c in coords_high],
                    color="#174E4F",
                    s=25 if is_zoom_view else 10,
                    zorder=4,
                )
                if not is_zoom_view:
                    sc_h.set_label("Cierre Anillo (Alta)")

            if coords_med:
                sc_m = target_ax.scatter(
                    [c[0] for c in coords_med],
                    [c[1] for c in coords_med],
                    color="#E84F5E",
                    s=20 if is_zoom_view else 8,
                    zorder=3,
                )
                if not is_zoom_view:
                    sc_m.set_label("Unión Islas (Media)")

        # ==============================================================================
        # 5. EJECUTAR DIBUJADO
        # ==============================================================================
        logger.info("Dibujando vista global...")
        draw_on_axis(ax, is_zoom_view=False)

        logger.info("Dibujando vista zoom...")
        draw_on_axis(ax_zoom, is_zoom_view=True)

        ax.legend(loc="lower right", fontsize=9, facecolor="white", framealpha=1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Conectores visuales del Zoom
        rect = Rectangle(
            (z_lon_min, z_lat_min),
            z_lon_max - z_lon_min,
            z_lat_max - z_lat_min,
            facecolor="none",
            edgecolor="grey",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(rect)

        con1 = ConnectionPatch(
            xyA=(z_lon_min, z_lat_max),
            coordsA=ax.transData,
            xyB=(0, 1),
            coordsB=ax_zoom.transAxes,
            color="grey",
            alpha=0.6,
            linestyle=":",
        )
        fig.add_artist(con1)

        con2 = ConnectionPatch(
            xyA=(z_lon_max, z_lat_min),
            coordsA=ax.transData,
            xyB=(1, 0),
            coordsB=ax_zoom.transAxes,
            color="grey",
            alpha=0.6,
            linestyle=":",
        )
        fig.add_artist(con2)

        # Guardar imagen
        try:
            os.makedirs("./reports/", exist_ok=True)
            path_report = os.path.join("./reports/", self.name_path + "_" + str(self.max_distance_m) +"_redundancy.png")
            plt.savefig(
                path_report, dpi=500, bbox_inches="tight", facecolor=fig.get_facecolor()
            )
            logger.info(f"Redundancy map guardado en: {path_report}")
            if view:
                plt.show()
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error al guardar la imagen: {e}")

    def __export_results(self, view=True):
        """Guarda los resultados en CSV y llama al gráfico."""
        if self.candidates_df is None or self.candidates_df.empty:
            logger.warning("No hay candidatos para exportar.")
            return

        # 1. Exportar CSV
        filename = f"./reports/{self.name_path}_candidatos_cierre_malha.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.candidates_df.to_csv(filename, index=False, sep=";")
        logger.info(f"Reporte de candidatos guardado en: {filename}")

        # 2. Generar Visualización

    def find_connection_candidates(self):
        """
        Versión OPTIMIZADA (Estrategia Tie-Lines):
        1. Busca Nodos Hoja (Finales de línea).
        2. Usa KD-Tree para vecinos cercanos.
        3. Filtra rápido por componentes (Unión de Islas).
        4. Calcula rutas para Cierre de Anillos solo si vale la pena.
        """
        logger.info(f"BUSCANDO TIE-LINES (Optimizado) - Radio {self.max_distance_m}m")

        # 1. Identificar Hojas
        leaf_nodes = [n for n, d in self.G.degree() if d == 1]
        logger.info(f"Identificados {len(leaf_nodes)} finales de línea (hojas).")

        # Preparar coordenadas de hojas
        leaf_nodes_geo = []
        leaf_coords = []
        for n in leaf_nodes:
            data = self.G.nodes[n]
            if "latitude" in data and "length" in data:
                leaf_nodes_geo.append(n)
                leaf_coords.append([data["latitude"], data["length"]])

        leaf_coords = np.array(leaf_coords)

        # 2. Mapa de Componentes (Aceleración x100)
        logger.info("Mapeando componentes conexos...")
        comp_map = {}
        for idx, comp in enumerate(nx.connected_components(self.G)):
            for n in comp:
                comp_map[n] = idx

        # 3. KD-Tree Global (Todos los nodos son destinos potenciales)
        all_nodes_geo = []
        all_coords = []
        for n, data in self.G.nodes(data=True):
            if "latitude" in data and "length" in data:
                all_nodes_geo.append(n)
                all_coords.append([data["latitude"], data["length"]])

        tree_all = cKDTree(np.array(all_coords))

        # 4. Búsqueda de Vecinos (k=3 para velocidad)
        logger.info("Consultando KD-Tree (k=3)...")
        # Radio en grados aprox
        degree_radius = self.max_distance_m / 111000.0
        dists, indices = tree_all.query(
            leaf_coords, k=3, distance_upper_bound=degree_radius
        )

        candidates = []

        logger.info("Evaluando candidatos (Calculando prioridades)...")
        # Iterar con barra de progreso
        for i, (distances_row, indices_row) in tqdm(
            enumerate(zip(dists, indices)),
            total=len(leaf_nodes_geo),
            desc="Procesando Hojas",
        ):
            leaf_node = leaf_nodes_geo[i]
            leaf_comp = comp_map.get(leaf_node)

            # Coordenadas hoja para el DataFrame
            leaf_lat = self.G.nodes[leaf_node]["latitude"]
            leaf_lon = self.G.nodes[leaf_node]["length"]

            for dist_deg, neighbor_idx in zip(distances_row, indices_row):
                if dist_deg == float("inf"):
                    continue

                target_node = all_nodes_geo[neighbor_idx]
                if leaf_node == target_node:
                    continue
                if self.G.has_edge(leaf_node, target_node):
                    continue

                target_comp = comp_map.get(target_node)

                # Coordenadas destino para el DataFrame
                target_lat = self.G.nodes[target_node]["latitude"]
                target_lon = self.G.nodes[target_node]["length"]

                # --- LÓGICA DE PRIORIZACIÓN ---

                # A. INTERCONEXIÓN DE ISLAS (Automática)
                if leaf_comp != target_comp:
                    dist_m = dist_deg * 111000.0
                    candidates.append(
                        {
                            "Nodo_A": leaf_node,
                            "Nodo_B": target_node,
                            "Costo_Metros": dist_m,
                            "Prioridad_Score": 100,  # Base 100
                            "Tipo_Beneficio": "Unión Isla Rápida",
                            "Lat_A": leaf_lat,
                            "Lon_A": leaf_lon,
                            "Lat_B": target_lat,
                            "Lon_B": target_lon,
                        }
                    )
                    continue

                # B. CIERRE DE ANILLOS (Mismo Componente)
                dist_m = self._haversine(leaf_lat, leaf_lon, target_lat, target_lon)
                if dist_m > self.max_distance_m:
                    continue

                try:
                    # Cálculo costoso: Shortest Path
                    length = nx.shortest_path_length(
                        self.G, leaf_node, target_node, weight="length"
                    )

                    # Solo si el anillo es grande (>2km) vale la pena la inversión
                    if length > 2000:
                        # Prioridad basada en ratio (Beneficio / Costo)
                        # Ej: Cerrar 5km con 100m de cable -> Score muy alto
                        score = 200 + (length / dist_m)
                        candidates.append(
                            {
                                "Nodo_A": leaf_node,
                                "Nodo_B": target_node,
                                "Costo_Metros": dist_m,
                                "Prioridad_Score": score,
                                "Tipo_Beneficio": f"Cierre Anillo ({length / 1000:.1f}km)",
                                "Lat_A": leaf_lat,
                                "Lon_A": leaf_lon,
                                "Lat_B": target_lat,
                                "Lon_B": target_lon,
                            }
                        )
                except Exception:
                    pass

        self.candidates_df = pd.DataFrame(candidates)

        if not self.candidates_df.empty:
            # Ordenar: Primero Anillos grandes, luego Islas
            self.candidates_df = self.candidates_df.sort_values(
                "Prioridad_Score", ascending=False
            )
            logger.info(
                f"Finalizado. {len(self.candidates_df)} Tie-Lines estratégicos encontrados."
            )
            self.__export_results()
        else:
            logger.warning("No se encontraron candidatos viables bajo los criterios.")

    def generate_redundancy_graph(self, budget_limit_meters: float = 5000.0, view=True):
        """
        Genera y guarda un nuevo grafo con las inversiones aplicadas.
        """
        budget_limit_meters = self.max_distance_m
        if self.candidates_df is None or self.candidates_df.empty:
            logger.warning(
                "No hay candidatos para generar el grafo. Ejecute find_connection_candidates primero."
            )
            return

        df_proposals = self.candidates_df.copy()  # Usar todos los ordenados

        investments_to_apply = []
        total_cost = 0.0
        count = 0

        logger.info(
            f"Seleccionando inversiones hasta presupuesto de {budget_limit_meters}m..."
        )

        for _, row in df_proposals.iterrows():
            cost = row["Costo_Metros"]

            # Verificar si nos pasamos del presupuesto
            if total_cost + cost > budget_limit_meters:
                break

            # Limite de seguridad por si acaso (ej. max 500 obras)
            if count >= self.top_n:
                break

            investments_to_apply.append((str(row["Nodo_A"]), str(row["Nodo_B"]), cost))
            total_cost += cost
            count += 1

        logger.info(f"--- RESUMEN DE INVERSIÓN APLICADA ---")
        logger.info(f"Obras ejecutadas: {len(investments_to_apply)}")
        logger.info(f"Costo Total (Cable): {total_cost / 1000:.2f} km")
        logger.info(f"-------------------------------------")

        self.top_n= len(investments_to_apply)

        # CREAR EL GRAFO MEJORADO
        G_improved = self.G.copy()

        added_edges = 0
        for u, v, cost in investments_to_apply:
            if u in G_improved and v in G_improved:
                G_improved.add_edge(u, v, length=cost, type="virtual_investment")
                added_edges += 1

        path_save = os.path.join(
            "./data/graph",
            self.name_path + "_" + str(self.max_distance_m) + "_OPTIMIZED.pickle",
        )
        os.makedirs(os.path.dirname(path_save), exist_ok=True)

        logger.info(f"Guardando grafo mejorado en {path_save}")
        save_graph(path_save=path_save, graph=G_improved)
        logger.info("Grafo optimizado guardado exitosamente.")

        self._plot_top_candidates(view=view)
