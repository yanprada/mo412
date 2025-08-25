"""Data source management module for handling geographic and tabular data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import logging
import pandas as pd
import shapely
import geopandas as gpd
import pyogrio

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSource:
    """Data source configuration for reading and caching datasets."""

    name: str
    path: str
    layer: Optional[str]
    crs: Optional[str]
    query: Optional[str] = None

    def materialized_path(self) -> Path:
        """Return the materialized path as a Path object."""
        return Path(self.path)

    @classmethod
    def get(cls, name: str) -> "DataSource":
        """Get a DataSource by name from the DATABASES registry."""
        try:
            return DATABASES[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown database '{name}'. Available: {list(DATABASES)}"
            ) from exc

    @staticmethod
    def _build_sql(layer: str | None, query: str | None) -> str:
        if not layer:
            raise ValueError("Layer not defined.")
        base = f"SELECT * FROM {layer}"
        return f"{base} {query}".strip() if query else base

    @staticmethod
    def _read_materialized(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_parquet(path) if path.exists() else None

    def read_from_raw(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Read the dataset from the raw data source."""
        raw = self.get("raw_data")
        sql = self._build_sql(self.layer, self.query)
        df = pyogrio.read_dataframe(f"zip://{raw.path}", sql=sql)
        if isinstance(df, gpd.GeoDataFrame):
            if df.crs is None:
                df.set_crs(raw.crs, inplace=True)
            if self.crs and df.crs != self.crs:
                df = df.to_crs(self.crs)
        return df

    @staticmethod
    def _check_geometry(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> pd.DataFrame:
        """Check if the DataFrame has a geometry column."""
        if "geometry" not in df.columns:
            return df

        if isinstance(df, gpd.GeoDataFrame):
            return df

        if hasattr(df["geometry"].iloc[0], "__geo_interface__"):
            return df

        df = df.copy()  # Avoid modifying original
        df["geometry"] = shapely.from_wkb(df["geometry"].values)
        return df

    @classmethod
    def read(cls, name: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Read a dataset by name, using cached version if available or creating it from raw data."""
        logger.info("Reading dataset '%s'", name)
        src = cls.get(name)
        path = src.materialized_path()
        if (df := cls._read_materialized(path)) is not None:
            return cls._check_geometry(df)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = src.read_from_raw()
        df.to_parquet(path)
        return cls._check_geometry(df)


DATABASES: dict[str, DataSource] = {
    "segments_high_tension": DataSource(
        name="segments_high_tension",
        path="data/processed/segments_high_tension.parquet",
        layer="SSDAT",
        crs="EPSG:4326",
    ),
    "segments_medium_tension": DataSource(
        name="segments_medium_tension",
        path="data/processed/segments_medium_tension.parquet",
        layer="SSDMT",
        crs="EPSG:4326",
    ),
    "segments_low_tension": DataSource(
        name="segments_low_tension",
        path="data/processed/segments_low_tension.parquet",
        layer="SSDBT",
        crs="EPSG:4326",
    ),
    "points": DataSource(
        name="points",
        path="data/processed/points.parquet",
        layer="PONNOT",
        crs="EPSG:4326",
    ),
    "consumers_medium_tension": DataSource(
        name="consumers_medium_tension",
        path="data/processed/consumers_medium_tension.parquet",
        layer="UCMT_tab",
        crs="EPSG:4326",
    ),
    "suppliers_medium_tension": DataSource(
        name="suppliers_medium_tension",
        path="data/processed/suppliers_medium_tension.parquet",
        layer="UGMT_tab",
        crs="EPSG:4326",
    ),
    "consumers_high_tension": DataSource(
        name="consumers_high_tension",
        path="data/processed/consumers_high_tension.parquet",
        layer="UCAT_tab",
        crs="EPSG:4326",
    ),
    "suppliers_high_tension": DataSource(
        name="suppliers_high_tension",
        path="data/processed/suppliers_high_tension.parquet",
        layer="UGAT_tab",
        crs="EPSG:4326",
    ),
    "substations": DataSource(
        name="substations",
        path="data/processed/substations.parquet",
        layer="SUB",
        crs="EPSG:4326",
    ),
    "raw_data": DataSource(
        name="raw_data",
        path="data/raw/CPFL_PAULISTA - 2023-12-31.gdb.zip",
        layer=None,
        crs="EPSG:4674",
    ),
}
