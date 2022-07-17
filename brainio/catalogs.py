from pathlib import Path

import pandas as pd
from pandas import DataFrame


SOURCE_CATALOG = "source_catalog"


class Catalog(DataFrame):
    # http://pandas.pydata.org/pandas-docs/stable/development/extending.html#subclassing-pandas-data-structures
    _metadata = pd.DataFrame._metadata + ["identifier", "source_path", "url", "get_loader_class", "from_files"]

    @property
    def _constructor(self):
        return Catalog

    @classmethod
    def get_loader_class(cls):
        return CatalogLoader

    @classmethod
    def from_files(cls, identifier, csv_path, url=None):
        loader_class = cls.get_loader_class()
        loader = loader_class(
            cls=cls,
            identifier=identifier,
            csv_path=csv_path,
            url=url
        )
        return loader.load()


class CatalogLoader:
    def __init__(self, cls, identifier, csv_path, url=None):
        self.cls = cls
        self.identifier = identifier
        self.csv_path = Path(csv_path)
        self.url = url

    def load(self):
        catalog = pd.read_csv(self.csv_path)
        catalog = self.cls(catalog)
        catalog.identifier = self.identifier
        catalog.source_path = self.csv_path
        catalog.url = self.url
        return catalog


