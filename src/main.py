from utils.graph_generator import GraphGenerator
import logging
from utils.configuration_log import configure_logging_global
from utils.save_load_graph import load_graph
import matplotlib.pyplot as plt
import networkx as nx

configure_logging_global()
logger = logging.getLogger(__name__) 

logger.info("INICIO del Análisis de Topología CPFL Paulista.")

# # --- 1. Definition of Files ---
NODE_FILE_NAME = "./data/processed/CPFL_Paulista_2023-Nodos_R.csv"
FILE_NAME_LINKS = "./data/processed/CPFL_Paulista_2023-SSDBT_R.csv"
FILE_SAVE_GRAPH = "./data/graph/CPFL_Paulista_2023.pickle"

print(f"Loading data from {NODE_FILE_NAME} y {FILE_NAME_LINKS}...")

gen = GraphGenerator(path_links=FILE_NAME_LINKS, path_nodes=NODE_FILE_NAME)
gen.cleaning_anomalous_links()
gen.graph_creating_model()
gen.graph_save(path_save=FILE_SAVE_GRAPH)

# # load grafo
G = load_graph(path_load=FILE_SAVE_GRAPH)
# --- 5. Visualización del Grafo ---
print("\nGenerando visualización del grafo...")

# 5.1 Preparar las posiciones (Layout Geográfico)
# Usamos las coordenadas GEO_X y GEO_Y para posicionar los nodos en el mapa (Layout_Pos)
# Esto da una visualización que simula la distribución geográfica real de la red.
pos = {
    nodo: (G.nodes[nodo]["latitude"], G.nodes[nodo]["length"]) for nodo in G.nodes()
}

# 5.2 Configuración de la Figura
plt.figure(figsize=(15, 15))  # Tamaño grande para una red geográfica

# 5.3 Dibujar Nodos
# Coloración basada en el grado del nodo (cuántas conexiones tiene)
# node_degrees = [G.degree(n) for n in G.nodes()]
# nx.draw_networkx_nodes(
#     G,
#     pos,
#     node_size=[v * 0.1 for v in node_degrees],  # Tamaño proporcional al grado
#     node_color=node_degrees,
#     # cmap=plt.cm.Wistia, # Paleta de colores para el grado
#     alpha=0.6,
# )

# 5.4 Dibujar Aristas
nx.draw_networkx_edges(
    G,
    pos,
    # alpha=0.9,
    edge_color="gray",
    width=1.5,
    # style='solid'
)

# 5.5 Configuración Final del Gráfico
plt.title(
    f"Topología de la Red de Distribución CPFL Paulista\n(Nodos: {G.number_of_nodes()}, Links: {G.number_of_edges()})",
    fontsize=16,
)
plt.xlabel("Longitud (GEO_Y)")
plt.ylabel("Latitud (GEO_X)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

print("Visualización completada. Se ha generado la ventana del gráfico.")

# --- 6. Próximos Pasos Analíticos Sugeridos ---
print("\n--- Próximos Pasos ---")
print("El grafo ha sido modelado. Sugerencias para el análisis experto:")
print(
    "1. **Identificar Puntos Críticos (Grado):** Analizar los nodos con alto grado (mayor número de conexiones)."
)
print(
    "2. **Análisis de Componentes Conexos:** Si no es conexo, analizar cada subred por separado."
)
print(
    "3. **Centralidad:** Calcular métricas como *Betweenness Centrality* para identificar puentes de tráfico de flujo de potencia."
)