from __future__ import annotations
from qpython.qconnection import QConnection
import pandas as pd

from datasets.dataset_loader import DatasetLoader


class BitmexLoader(DatasetLoader):

    def __init__(self, port=5000):
        self.port = port
        self.q = None

    def __enter__(self) -> BitmexLoader:
        self.q = QConnection(host='localhost', port=self.port, username="crm", password="crm")
        try:
            self.q.open()
            self._load_bitmex_data()
            self._aggregate()
            return self
        except Exception as e:
            self.q.close()
            raise Exception(f"Could not connect, load data or aggregate data\nError: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.q.close()

    def _load_bitmex_data(self) -> None:
        self.q('''
        if[not `trades in tables[];
            system "l D:/datasets/q-datasets/Bitcoin2/Bitcoin2/bitmexDb/bitmexDb"];
        
        pivot:{[t]
            u:`$string asc distinct last f:flip key t;
            pf:{x#(`$string y)!z};
            p:?[t;();g!g:-1_ k;(pf;`u;last k:key f;last key flip value t)];
            p};
            
        dates: exec distinct date from (select distinct date from trades) where date >= 2018.01.01;
        holdoutRatio: `train`validation`test!0.6 0.2 0.2;
        ''')

    def _aggregate(self, bin_length: int = 10) -> None:
        self.q('''
        toBin: {[table; targetDate; interval]
            aggregated :select sum size, vwap: size wavg price, nTrades: count i
                           by date, timeBin: interval xbar `minute$timestamp
                           from table where date=targetDate;
            :update cumSize: sums size, cumVWAP: sums vwap from aggregated
        };
        
        if[not `aggByDate in tables[];
            aggByDate: raze toBin[trades; ;%s] peach dates];
            
        pivoted: {:pivot[2!select date, timeBin, cumSize from x]};
        ''' % bin_length)

    def get_table(self, train_validation_test: str, pivoted: bool = False) -> pd.DataFrame:
        self.q('''// By bins over dates - predicting by bin
        getRange: {:`int$holdoutRatio[x]*count dates};
        
        trainSet: select from aggByDate where date in dates[til getRange[`train]];
        validationSet: select from aggByDate where date in dates[getRange[`train] + til getRange[`validation]];
        testSet: select from aggByDate where date in dates[getRange[`train]+getRange[`validation]+til getRange[`test]];
        ''')
        return self.q(("pivoted " if pivoted else "") +
                      {"train": "trainSet", "validation": "validationSet", "test": "testSet"}[train_validation_test],
                      pandas=True)

    @property
    def train_set(self) -> pd.DataFrame:
        return self.get_table("train", pivoted=True)

    @property
    def val_set(self) -> pd.DataFrame:
        return self.get_table("validation", pivoted=True)

    @property
    def test_set(self) -> pd.DataFrame:
        return self.get_table("test", pivoted=True)


if __name__ == "__main__":
    import numpy as np
    with BitmexLoader() as bitmex_loader:
        arr = np.array(bitmex_loader.get_table("train", True).iloc[:, 0])
        print(np.mean((arr - arr) ** 2))
        print(bitmex_loader.get_table("train", True).iloc[:, 0])
