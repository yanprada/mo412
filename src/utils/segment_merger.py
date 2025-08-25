"""
Segment merging utility for electrical network segments.
"""

from __future__ import annotations

from typing import Iterable, Optional

from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import numpy as np

from shapely.ops import linemerge, unary_union

from utils.data_source import DataSource


# ---------------------------------------------------------------------------
# Graph / Segment Merging
# ---------------------------------------------------------------------------
class SegmentMerger:
    """Merge line segments based on connectivity and points of interest."""

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

    def merge(self, df_segments: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Merge line segments based on connectivity and points of interest.
        """
        cand = df_segments.copy()
        if cand.empty:
            return cand

        adj, endpoints = self._graph(cand)
        start_nodes = self._remove_irrelevant_nodes(adj)
        seen = set()
        records = []

        n_edges = len(cand)
        edge_indices = np.arange(n_edges)

        for edge_idx in tqdm(
            edge_indices, desc="Merging segments", disable=n_edges < 1000
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

            geom_series = cand.iloc[walked["edges"]]["geometry"]
            geom = self._merge_geometry(geom_series.tolist())

            if geom is None or geom.is_empty:
                continue
            records.append(
                {
                    "start_id": walked["start"],
                    "end_id": walked["end"],
                    "n_edges": len(walked["edges"]),
                    "geometry": geom,
                }
            )

        if not records:
            return gpd.GeoDataFrame(
                columns=["start_id", "end_id", "n_edges", "geometry"],
                geometry="geometry",
                crs=getattr(cand, "crs", None),
            )

        result_df = gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=DataSource.get("segments_medium_tension").crs,
        )

        return result_df
