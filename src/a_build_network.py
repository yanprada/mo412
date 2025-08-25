"""
This script merges segments of medium and high tension electrical networks
based on points of interest (POIs) and substations. It calculates coverage
percentages and saves the merged segments and relevant POIs to CSV files for
visualization.
"""

import logging
import numpy as np

from utils.data_source import DataSource
from utils.segment_merger import SegmentMerger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extra Functions
# ---------------------------------------------------------------------------


def load_data(pk: str) -> dict:
    """Load all required datasets and return them in a dictionary."""
    return {
        "poi": DataSource.read("points").rename(columns={"COD_ID": pk}),
        "sub": DataSource.read("substations").rename(columns={"COD_ID": pk}),
        "seg_md": DataSource.read("segments_medium_tension"),
        "con_md": DataSource.read("consumers_medium_tension"),
        "sup_md": DataSource.read("suppliers_medium_tension"),
        "seg_hg": DataSource.read("segments_high_tension"),
        "sup_hg": DataSource.read("suppliers_high_tension"),
        "con_hg": DataSource.read("consumers_high_tension"),
    }


def extract_ids(data: dict, pk: str) -> dict:
    """Extract unique IDs from all datasets."""
    return {
        "poi_ids": frozenset(data["poi"][pk].unique()),
        "sub_ids": frozenset(data["sub"][pk].unique()),
        "sup_md_ids": frozenset(data["sup_md"][pk].unique()),
        "con_md_ids": frozenset(data["con_md"][pk].unique()),
        "seg_md_ids": frozenset(
            np.concatenate(
                [data["seg_md"][f"{pk}_1"].values, data["seg_md"][f"{pk}_2"].values]
            )
        ),
        "sup_hg_ids": frozenset(data["sup_hg"][pk].unique()),
        "con_hg_ids": frozenset(data["con_hg"][pk].unique()),
        "seg_hg_ids": frozenset(
            np.concatenate(
                [data["seg_hg"][f"{pk}_1"].values, data["seg_hg"][f"{pk}_2"].values]
            )
        ),
    }


def calculate_coverage(ids: dict) -> None:
    """Calculate and print coverage percentages."""

    def pct(a: frozenset, b: frozenset) -> float:
        return len(a & b) / len(b) if b else 0.0

    poi_ids = ids["poi_ids"]
    sub_ids = ids["sub_ids"]

    pct_poi_md = pct(poi_ids, ids["seg_md_ids"])
    pct_sup_md = pct(poi_ids, ids["sup_md_ids"]) + pct(sub_ids, ids["sup_md_ids"])
    pct_con_md = pct(poi_ids, ids["con_md_ids"]) + pct(sub_ids, ids["con_md_ids"])

    pct_poi_hg = pct(poi_ids, ids["seg_hg_ids"])
    pct_sup_hg = pct(poi_ids, ids["sup_hg_ids"]) + pct(sub_ids, ids["sup_hg_ids"])
    pct_con_hg = pct(poi_ids, ids["con_hg_ids"]) + pct(sub_ids, ids["con_hg_ids"])

    logger.info("POI coverage in segments_medium_tension: %.2f", pct_poi_md)
    logger.info("POI coverage in suppliers_medium_tension: %.2f", pct_sup_md)
    logger.info("POI coverage in consumers_medium_tension: %.2f", pct_con_md)
    logger.info("POI coverage in segments_high_tension: %.2f", pct_poi_hg)
    logger.info("POI coverage in suppliers_high_tension: %.2f", pct_sup_hg)
    logger.info("POI coverage in consumers_high_tension: %.2f", pct_con_hg)


def merge_segments(data: dict, ids: dict, pk: str) -> tuple:
    """Merge segments for both medium and high tension."""
    poi_ids = ids["poi_ids"]
    segment_merger = SegmentMerger(poi_ids, pk)
    df_seg_md_merged = segment_merger.merge(data["seg_md"])
    df_seg_hg_merged = segment_merger.merge(data["seg_hg"])

    return df_seg_md_merged, df_seg_hg_merged


def save_single_data(df_seg, df_pois, path, tension_level):
    """
    Save a single DataFrame to a CSV file.
    """
    pois_seg_clean = set(
        np.concatenate([df_seg["start_id"].values, df_seg["end_id"].values])
    )
    df_pois_seg = df_pois[df_pois["PN_CON"].isin(pois_seg_clean)]

    df_seg.to_csv(f"{path}/links_{tension_level}.csv", index=False, sep=";")
    df_pois_seg.to_csv(f"{path}/nodes_{tension_level}.csv", index=False, sep=";")


def save_data(df_seg_md_merged, df_seg_hg_merged, df_poi):
    """
    Save merged segment and point of interest data to CSV files.
    """
    save_single_data(df_seg_md_merged, df_poi, "data/vizualization", "medium_tension")
    save_single_data(df_seg_hg_merged, df_poi, "data/vizualization", "high_tension")


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------


def main():
    """Main function to execute the merging process."""

    pk = "PN_CON"
    data = load_data(pk)
    ids = extract_ids(data, pk)
    calculate_coverage(ids)
    df_seg_md_merged, df_seg_hg_merged = merge_segments(data, ids, pk)
    save_data(df_seg_md_merged, df_seg_hg_merged, data["poi"])


if __name__ == "__main__":
    main()
