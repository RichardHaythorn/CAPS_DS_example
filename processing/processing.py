"""Main processing functions and classes"""
import datetime

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_els_figure(dataframe):
    """Creates a spectrogram of the given dataframe"""
    x = dataframe.index.to_numpy()
    y = range(63)
    z = dataframe.to_numpy().transpose()

    fig, ax = plt.subplots(figsize=(20, 10))
    minmax = (1, 6e5)
    ax.pcolormesh(
        x, y, z, shading="nearest", norm=LogNorm(vmin=minmax[0], vmax=minmax[1])
    )
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Energy Level")
    fig.autofmt_xdate()
    return fig, ax


def make_x_y(dataframe: pd.DataFrame):
    """Split a dataframe into X and y"""
    X = dataframe.drop(columns="Rammed")
    y = dataframe["Rammed"]
    return X, y


def format_times(
    df: pl.DataFrame, start_ram_time: datetime.datetime, end_ram_time: datetime.datetime
):
    """Label certain times as ram"""
    df = df.with_columns(
        (
            pl.when(
                pl.col("Time").is_between(start_ram_time, end_ram_time, closed="none")
            )
            .then(1)
            .otherwise(0)
        ).alias("Rammed")
    )

    return df


def get_df(
    filename: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    scale: bool = True,
) -> pd.DataFrame:
    """Get and format a dataframe from a file"""

    dataframe = pl.read_parquet(filename).drop("Unnamed: 0")

    dataframe = dataframe.with_columns(
        pl.col("Time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
    )

    dataframe = format_times(dataframe, start_time, end_time)

    dataframe = dataframe.to_pandas()

    if scale:
        dataframe.iloc[:, :63] = StandardScaler().fit_transform(dataframe.iloc[:, :63])

    return dataframe


def join_y(
    X_test: pd.DataFrame, y_pred: np.ndarray, y_true: pd.Series = None
) -> pd.DataFrame:
    """Join the predictions to the truth, add times"""
    joint_y = y_true.to_frame()
    joint_y["predicted"] = y_pred
    joint_y = X_test[["Time"]].join(joint_y)
    return joint_y


class RamModel:
    """Class for holding the sklearn model"""

    def __init__(self, train_flybys: list, max_iter: int = 100) -> None:
        self.train_flybys = train_flybys
        self.clf = LogisticRegression(class_weight="balanced", max_iter=max_iter)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_train_data(self, flyby_info: dict, scale: bool = True) -> None:
        """Load data from flybys"""
        train_dfs = []
        for flyby in self.train_flybys:
            for _, anode in flyby_info[flyby].anodes.items():
                train_dfs.append(
                    get_df(
                        anode.filepath,
                        anode.start_ram_time,
                        anode.end_ram_time,
                        scale,
                    )
                )

        dataframe = pd.concat(train_dfs)
        X, y = make_x_y(dataframe)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=42
        )

    def fit(self):
        """Wrapper for fit"""
        self.clf.fit(self.X_train.drop(columns="Time"), self.y_train)

    def predict(self, df: pd.DataFrame = None):
        """Wrapper for predict"""
        return (
            self.clf.predict(self.X_test.drop(columns="Time"))
            if df is None
            else self.clf.predict(df.drop(columns="Time"))
        )
