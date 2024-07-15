from pathlib import Path
from typing import Tuple, Optional, Any

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class DataLoader(ABC):

    @abstractmethod
    def load_data(
            self
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_splits(
            self,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        pass


class PejorativeLoader(DataLoader):

    def __init__(
            self,
            data_path: Path,
            filename: str = 'dataset.csv'
    ):
        self.filename = filename
        self.data_path = data_path.joinpath(self.filename)

    def load_data(
            self
    ) -> pd.DataFrame:
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True)

        data = pd.read_csv(self.data_path)
        data = data.dropna()
        data['pejorative'] = data['pejorative'].astype(int)
        data['sample_id'] = np.arange(len(data))

        data.rename(columns={'word': 'search_word'}, inplace=True)

        return data

    def get_splits(
            self,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        data = self.load_data()

        return data, None, None
