import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import logging
import os
from utils.save_load_graph import load_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_networks_absolute(path_original, path_optimized):
    # 1. CARGAR GRAFOS
    logger.info("Cargando grafos...")
    try:
        G_orig = load_graph(path_original)
        G_opt = load_graph(path_optimized)
    except FileNotFoundError:
        logger.error("No se encontraron los archivos .pickle")
        return

    # 2. ANÁLISIS ESTÁTICO (KPIs Inmediatos)
    lcc_orig = len(max(nx.connected_components(G_orig), key=len))
    lcc_opt = len(max(nx.connected_components(G_opt), key=len))
    
    n_islas_orig = nx.number_connected_components(G_orig)
    n_islas_opt = nx.number_connected_components(G_opt)
    
    print("\n" + "="*50)
    print("COMPARATIVA DE IMPACTO INMEDIATO")
    print("="*50)
    print(f"METRICA               | ORIGINAL      | OPTIMIZADO    | MEJORA")
    print(f"----------------------|---------------|---------------|-------")
    print(f"Nodos en Red Princ.   | {lcc_orig:<13,} | {lcc_opt:<13,} | +{((lcc_opt/lcc_orig)-1)*100:.3f}%")
    print(f"Islas Desconectadas   | {n_islas_orig:<13,} | {n_islas_opt:<13,} | -{((n_islas_orig-n_islas_opt)/n_islas_orig)*100:.3f}%")
    print("="*50 + "\n")

    # 3. SIMULACIÓN DE ROBUSTEZ (ABSOLUTA)
    # Atacamos ambos grafos y medimos cuántos nodos QUEDAN VIVOS (número real, no %)
    logger.info("Ejecutando simulación comparativa (Ataque Dirigido)...")
    
    curve_orig = _simulate_absolute_attack(G_orig)
    curve_opt = _simulate_absolute_attack(G_opt)
    
    # 4. GRAFICAR
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 7))
    
    # Eje X: Número de nodos eliminados (hasta 1000 para ver detalle)
    steps = min(len(curve_orig), len(curve_opt), 1000) 
    x = range(steps)
    
    plt.plot(x, curve_orig[:steps], color='red', label='Red Original (Actual)', linewidth=2, linestyle='--')
    plt.plot(x, curve_opt[:steps], color='green', label='Red Optimizada (Con Inversión)', linewidth=3)
    
    # Rellenar el área entre curvas (Esa área es el BENEFICIO TANGIBLE)
    plt.fill_between(x, curve_orig[:steps], curve_opt[:steps], color='green', alpha=0.1, label='Clientes "Rescatados"')
    
    plt.title("Comparativa Real de Robustez: Nodos Conectados", fontsize=16)
    plt.xlabel("Número de Postes Críticos Eliminados (Ataque)", fontsize=12)
    plt.ylabel("Total de Nodos Operativos (Conectados)", fontsize=12)
    # plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True)
    
    output = "./reports/COMPARATIVA_FINAL_ABSOLUTA.png"
    plt.savefig(output, dpi=300)
    logger.info(f"Gráfico guardado en {output}")
    plt.show()

def _simulate_absolute_attack(G):
    # Calculamos Centralidad Rápida para definir el ataque
    # Usamos degree centrality porque es rápido y correlaciona bien con "hubs" locales para esta prueba
    # O betweenness si tienes tiempo, pero degree es suficiente para ver la tendencia
    centrality = nx.degree_centrality(G)
    attack_plan = sorted(centrality, key=centrality.get, reverse=True)
    
    G_sim = G.copy()
    
    # Obtener LCC inicial
    if nx.is_empty(G_sim): return [0]
    lcc_current = max(nx.connected_components(G_sim), key=len)
    history = [len(lcc_current)]
    
    # Simular los primeros 1000 fallos
    limit = min(10, len(attack_plan))
    
    for i in range(limit):
        node = attack_plan[i]
        if node in G_sim:
            G_sim.remove_node(node)
        
        if G_sim.number_of_nodes() > 0:
            # Medir LCC absoluto
            try:
                largest_cc = max(nx.connected_components(G_sim), key=len)
                history.append(len(largest_cc))
            except ValueError:
                history.append(0)
        else:
            history.append(0)
            
    return history

# if __name__ == "__main__":
    # RUTAS A TUS ARCHIVOS PICKLE
    # Ajusta los nombres según tus archivos generados
ORIGINAL = "./data/graph/CPFL_Paulista_MT_Electrical_Network_Topology.pickle"

# Este debe ser el archivo que generó generate_redundancy_graph
OPTIMIZED = "./data/graph/CPFL_Paulista_MT_Redundancy_Map_OPTIMIZED.pickle" 

compare_networks_absolute(ORIGINAL, OPTIMIZED)