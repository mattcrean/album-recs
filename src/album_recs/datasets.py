from pathlib import Path
from kedro.io import AbstractDataset
import pandas as pd


class RawFilePathDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def _load(self) -> Path:
        return self._filepath

    def _save(self, data) -> None:
        raise NotImplementedError("Raw input file is read-only.")

    def _describe(self) -> dict:
        return {"filepath": str(self._filepath)}


class ParquetFileDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def _load(self) -> pd.DataFrame:
        return pd.read_parquet(self._filepath)

    def _save(self, data: Path) -> None:
        if not self._filepath.exists():
            raise FileNotFoundError(f"Expected Parquet output not found: {data}")

    def _describe(self) -> dict:
        return {"filepath": str(self._filepath)}
