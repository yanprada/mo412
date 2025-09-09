"""
links merging utility for electrical network links.
"""

from __future__ import annotations

from typing import Iterable, Optional

from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from shapely.ops import linemerge, unary_union

from utils.data_source import DataSource
import os


# ---------------------------------------------------------------------------
# Graph / links Merging
# ---------------------------------------------------------------------------
class LinksMerger:
    """Merge line links based on connectivity and points of interest."""

    def __init__(self, poi_ids: set, pk: str):
        self.poi_ids = frozenset(poi_ids)
        self.pk = pk
        self.pk_a = f"{pk}_1"
        self.pk_b = f"{pk}_2"

    def _graph(self, cand: pd.DataFrame):
        adj = defaultdict(list)

        indices = cand.index.to_numpy()
        a_values = cand[self.pk_a].to_numpy()
        b_values = cand[self.pk_b].to_numpy()

        endpoints = dict(zip(indices, zip(a_values, b_values)))

        for idx, a, b in zip(indices, a_values, b_values):
            adj[a].append(idx)
            adj[b].append(idx)

        return adj, endpoints

    def _remove_irrelevant_nodes(self, adj) -> frozenset:
        return frozenset(
            node
            for node, edges in adj.items()
            if node in self.poi_ids and len(edges) != 2
        )

    @staticmethod
    def _flatten(geoms: Iterable) -> list:
        out = []
        for g in geoms:
            if not g or g.is_empty:
                continue
            gt = g.geom_type
            if gt == "LineString":
                out.append(g)
            elif gt == "MultiLineString":
                out.extend([lg for lg in g.geoms if not lg.is_empty])
            else:
                b = getattr(g, "boundary", None)
                if b is not None and not b.is_empty:
                    if b.geom_type == "LineString":
                        out.append(b)
                    elif b.geom_type == "MultiLineString":
                        out.extend([lg for lg in b.geoms if not lg.is_empty])
        return out

    def _walk(
        self, start_edge: int, start_nodes: frozenset, adj, endpoints
    ) -> Optional[dict]:
        visited_edges = {start_edge}
        path = [start_edge]
        a0, b0 = endpoints[start_edge]

        def extend(node, prev_edge):
            curr = node
            last = prev_edge
            while True:
                if curr in start_nodes and curr not in (a0, b0):
                    break
                next_edges = [
                    e for e in adj[curr] if e != last and e not in visited_edges
                ]
                if len(next_edges) != 1:
                    break
                ne = next_edges[0]
                visited_edges.add(ne)
                path.append(ne)
                na, nb = endpoints[ne]
                curr = na if curr != na else nb
                last = ne

        extend(a0, start_edge)
        extend(b0, start_edge)

        path_array = np.array(path)
        node_list = []
        for e in path_array:
            na, nb = endpoints[e]
            node_list.extend([na, nb])

        unique_nodes, counts = np.unique(node_list, return_counts=True)
        endpoints_chain = unique_nodes[counts == 1]

        if len(endpoints_chain) != 2:
            return None
        return {"edges": path, "start": endpoints_chain[0], "end": endpoints_chain[1]}

    def _merge_geometry(self, lines: list):
        flat = self._flatten(lines)
        if not flat:
            return None
        merged = linemerge(flat)
        if merged.geom_type == "MultiLineString":
            merged = linemerge(unary_union(flat))
        return merged

    def plot_histograms(
        self,
        adj,
        tension_level: str,
        remove_nodes: bool = False,
    ):
        """Plot degree distribution and log-log degree distribution of the graph."""
        degrees = np.fromiter((len(edges) for edges in adj.values()), dtype=int)
        fstr = ""
        if remove_nodes:
            degrees = degrees[degrees != 2]
            fstr = "_without_degree_2"

        degree_counts = np.bincount(degrees)
        degree_range = np.arange(len(degree_counts))
        percentages = degree_counts / degree_counts.sum() * 100

        # Plot degree distribution
        plt.scatter(degree_range, percentages, color="blue")
        plt.xlabel("Degree")
        plt.ylabel("Percentage (%)")
        plt.title(f"Degree Distribution of Graph - {tension_level}{fstr}")
        plt.grid(True)
        # Ensure the directory exists
        os.makedirs(f"visualization/{tension_level}/", exist_ok=True)
        plt.savefig(f"visualization/{tension_level}/degree_distribution{fstr}.png")
        plt.close()

        # Plot log-log degree distribution
        plt.scatter(degree_range, percentages, color="blue")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Degree (log scale)")
        plt.ylabel("Percentage (%) (log scale)")
        plt.title(f"Log-Log Degree Distribution of Graph - {tension_level}{fstr}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(f"visualization/{tension_level}/log_degree_distribution_{fstr}.png")
        plt.close()

    def merge(self, df_links: pd.DataFrame, tension_level: str) -> gpd.GeoDataFrame:
        """
        Merge line links based on connectivity and points of interest.
        """
        cand = df_links.copy()
        if cand.empty:
            return cand

        adj, endpoints = self._graph(cand)
        self.plot_histograms(adj, tension_level)
        self.plot_histograms(adj, tension_level, remove_nodes=True)

        start_nodes = self._remove_irrelevant_nodes(adj)

        n_edges = len(cand)
        edge_indices = np.arange(n_edges)

        geoms = cand["geometry"].to_numpy()
        comps = cand["COMP"].to_numpy()

        seen = set()

        starts, ends, n_edges_list, lengths, geoms_out = [], [], [], [], []

        for edge_idx in tqdm(
            edge_indices, desc="Merging links", disable=n_edges < 1000
        ):
            if edge_idx in seen:
                continue

            a, b = endpoints[edge_idx]
            if a not in start_nodes and b not in start_nodes:
                continue

            walked = self._walk(edge_idx, start_nodes, adj, endpoints)
            if not walked:
                continue

            seen.update(walked["edges"])
            edges = walked["edges"]

            geom = self._merge_geometry(geoms[edges])
            if geom is None or geom.is_empty:
                continue

            starts.append(walked["start"])
            ends.append(walked["end"])
            n_edges_list.append(len(edges))
            lengths.append(comps[edges].sum())
            geoms_out.append(geom)

        result_df = gpd.GeoDataFrame(
            {
                "start_id": starts,
                "end_id": ends,
                "n_edges": n_edges_list,
                "lenth_meters": lengths,
                "geometry": geoms_out,
            },
            geometry="geometry",
            crs=DataSource.get("links_medium_tension").crs,
        )
        return result_df
