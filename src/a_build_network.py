"""
This script merges links of medium and high tension electrical networks
based on points of interest (nodes) and substations. It calculates coverage
percentages and saves the merged links and relevant nodes to CSV files for
visualization.
"""

import os
from typing import Union
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
import shapely
from utils.data_source import DataSource
from utils.links_merger import LinksMerger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PATHS = {
    "low_tension": {
        "nodes": "data/vizualization/nodes_low_tension",
        "links": "data/vizualization/links_low_tension",
    },
    "medium_tension": {
        "nodes": "data/vizualization/nodes_medium_tension",
        "links": "data/vizualization/links_medium_tension",
    },
    "high_tension": {
        "nodes": "data/vizualization/nodes_high_tension",
        "links": "data/vizualization/links_high_tension",
    },
}

# ---------------------------------------------------------------------------
# Extra Functions
# ---------------------------------------------------------------------------


def load_nodes(pk: str) -> dict:
    """Load all required datasets and return them in a dictionary."""
    return {
        "nodes": DataSource.read("points").rename(columns={"COD_ID": pk}),
        "substations": DataSource.read("substations").rename(columns={"COD_ID": pk}),
    }


def append_links(data: dict, tension_level: str) -> dict:
    """Append links, consumers, and suppliers to the data dictionary."""
    data[f"links_{tension_level}"] = DataSource.read(f"links_{tension_level}")
    data[f"consumers_{tension_level}"] = DataSource.read(f"consumers_{tension_level}")
    data[f"suppliers_{tension_level}"] = DataSource.read(f"suppliers_{tension_level}")
    return data


def extract_ids(data: dict, pk: str, tension_level: str) -> dict:
    """Extract unique IDs from all datasets."""
    return {
        "nodes_ids": frozenset(data["nodes"][pk].unique()),
        "substation_ids": frozenset(data["substations"][pk].unique()),
        f"suppliers_{tension_level}_ids": frozenset(
            data[f"suppliers_{tension_level}"][pk].unique()
        ),
        f"consumers_{tension_level}_ids": frozenset(
            data[f"consumers_{tension_level}"][pk].unique()
        ),
        f"links_{tension_level}_ids": frozenset(
            np.concatenate(
                [
                    data[f"links_{tension_level}"][f"{pk}_1"].values,
                    data[f"links_{tension_level}"][f"{pk}_2"].values,
                ]
            )
        ),
    }


def calculate_coverage(ids: dict, tension_level) -> None:
    """Calculate and print coverage percentages."""

    def pct(a: frozenset, b: frozenset) -> float:
        return len(a & b) / len(b) if b else 0.0

    nodes_ids = ids["nodes_ids"]
    substation_ids = ids["substation_ids"]

    pct_nodes = pct(nodes_ids, ids[f"links_{tension_level}_ids"])
    pct_sup = pct(nodes_ids, ids[f"suppliers_{tension_level}_ids"]) + pct(
        substation_ids, ids[f"suppliers_{tension_level}_ids"]
    )
    pct_con = pct(nodes_ids, ids[f"consumers_{tension_level}_ids"]) + pct(
        substation_ids, ids[f"consumers_{tension_level}_ids"]
    )

    logger.info("Nodes coverage in links_%s_tension: %.2f", tension_level, pct_nodes)
    logger.info("Nodes coverage in suppliers_%s_tension: %.2f", tension_level, pct_sup)
    logger.info("Nodes coverage in consumers_%s_tension: %.2f", tension_level, pct_con)


def append_extra_columns(data: dict, pk: str, tension_level: str) -> pd.DataFrame:
    """Append extra columns from consumers to the links DataFrame."""
    nodes_in_links = set(
        np.concatenate(
            [
                data[f"links_processed_{tension_level}"]["start_id"].values,
                data[f"links_processed_{tension_level}"]["end_id"].values,
            ]
        )
    )
    df_links_substations = data[f"links_{tension_level}"]
    if "SUB" in df_links_substations.columns:
        link_substation_1 = dict(
            zip(df_links_substations[f"{pk}_1"], df_links_substations["SUB"])
        )
        link_substation_2 = dict(
            zip(df_links_substations[f"{pk}_2"], df_links_substations["SUB"])
        )
        link_substation = {**link_substation_1, **link_substation_2}
        link_substation = {
            k: v for k, v in link_substation.items() if k in nodes_in_links
        }
        data[f"links_processed_{tension_level}"]["SUB_start"] = data[
            f"links_processed_{tension_level}"
        ]["start_id"].map(link_substation)
        data[f"links_processed_{tension_level}"]["SUB_end"] = data[
            f"links_processed_{tension_level}"
        ]["end_id"].map(link_substation)
    return data


def check_geometry(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> pd.DataFrame:
    """Check if the DataFrame has a geometry column."""
    if "geometry" not in df.columns:
        return df

    if isinstance(df, gpd.GeoDataFrame):
        return df

    if hasattr(df["geometry"].iloc[0], "__geo_interface__"):
        return df

    df = df.copy()  # Avoid modifying original
    try:
        df["geometry"] = shapely.from_wkb(df["geometry"].values)
    except:
        df["geometry"] = shapely.from_wkt(df["geometry"].values)
    return df


def merge_links(data: dict, ids: dict, pk: str, tension_level: str) -> tuple:
    """Merge links for both medium and high tension."""
    nodes_ids = ids["nodes_ids"]
    path_links = ".".join([PATHS[tension_level]["links"], "parquet"])
    path_nodes = ".".join([PATHS[tension_level]["nodes"], "parquet"])
    links_merger = LinksMerger(nodes_ids, pk)
    if not os.path.exists(path_links) or not os.path.exists(path_nodes):
        return links_merger.merge(data[f"links_{tension_level}"])
    df = pd.read_parquet(path_links)
    return check_geometry(df)


def concat_data(data, pk, type, tensions):
    df = pd.DataFrame()
    for tension_level in tensions:
        df_temp = data[f"{type}_{tension_level}"]
        df_temp["DIC"] = df_temp.filter(like="DIC_").sum(axis=1)
        df_temp = df_temp[["DIC", "SUB", pk]]
        df = pd.concat([df, df_temp], ignore_index=True)
    return df.groupby([pk, "SUB"], as_index=False).sum()


def get_endpoints(geom):
    if geom.is_empty:
        return None, None

    if isinstance(geom, LineString):
        return Point(geom.coords[0]), Point(geom.coords[-1])

    if isinstance(geom, MultiLineString):
        # use first and last LineString in the collection
        first = list(geom.geoms)[0]
        last = list(geom.geoms)[-1]
        return Point(first.coords[0]), Point(last.coords[-1])

    return None, None


def save_data(data: dict, pk: str, tensions: str) -> None:
    """
    Save a single DataFrame to a CSV file.
    """
    cols_node = [pk, "MUN", "ARE_LOC", "DIC", "SUB", "geometry"]
    df_consumers = concat_data(data, pk, "consumers", tensions)
    df_suppliers = concat_data(data, pk, "suppliers", tensions)
    nodes_suppliers_not_in_consumers = set(df_suppliers[pk].unique()).difference(
        df_consumers[pk].unique()
    )
    df_suppliers = df_suppliers[df_suppliers[pk].isin(nodes_suppliers_not_in_consumers)]
    df_consumers = pd.concat([df_consumers, df_suppliers])
    for tension_level in tensions:
        df_links = data[f"links_processed_{tension_level}"]
        df_links[["start_point", "end_point"]] = df_links["geometry"].apply(
            lambda g: pd.Series(get_endpoints(g))
        )
        map_1 = dict(zip(df_links.start_id, df_links.start_point))
        map_2 = dict(zip(df_links.end_id, df_links.end_point))
        map_points = {**map_1, **map_2}
        df_nodes = data[f"nodes"]

        nodes_links_clean = set(
            np.concatenate([df_links["start_id"].values, df_links["end_id"].values])
        )
        df_nodes_links = df_nodes[df_nodes[pk].isin(nodes_links_clean)]
        df_nodes_links = pd.merge(
            df_nodes_links,
            df_consumers,
            on=pk,
            how="left",
        )
        df_nodes_links["geometry"] = (
            df_nodes_links[pk].apply(map_points.__getitem__)
        ).astype(str)

        df_links = df_links.drop(columns=["start_point", "end_point"]).astype(
            {"geometry": str}
        )
        df_links.to_parquet(".".join([PATHS[tension_level]["links"], "parquet"]))
        df_nodes_links[cols_node].to_parquet(
            ".".join([PATHS[tension_level]["nodes"], "parquet"])
        )
        df_links.to_csv(
            ".".join([PATHS[tension_level]["links"], "csv"]), index=False, sep=";"
        )
        df_nodes_links[cols_node].to_csv(
            ".".join([PATHS[tension_level]["nodes"], "csv"]), index=False, sep=";"
        )


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------


def main():
    """Main function to execute the merging process."""
    tensions = ["high_tension", "medium_tension", "low_tension"]
    pk = "PN_CON"
    data = load_nodes(pk)
    for tension_level in tensions:
        logger.info("Processing %s", tension_level)
        data = append_links(data, tension_level)
        ids = extract_ids(data, pk, tension_level)
        calculate_coverage(ids, tension_level)
        df_links = merge_links(data, ids, pk, tension_level)
        data[f"links_processed_{tension_level}"] = df_links
        data = append_extra_columns(data, pk, tension_level)
    save_data(data, pk, tensions)


if __name__ == "__main__":
    main()
