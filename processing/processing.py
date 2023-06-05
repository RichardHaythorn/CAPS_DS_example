import pandas as pd
import polars as pl
import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def get_els_figure(df):
    x = df.index.to_numpy()
    y = range(63)
    z = df.to_numpy().transpose()

    fig, ax = plt.subplots(figsize=(20, 10))
    minmax = (1, 6e5)
    ax.pcolormesh(
        x, y, z, shading="nearest", norm=LogNorm(vmin=minmax[0], vmax=minmax[1])
    )
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Energy Level")
    fig.autofmt_xdate()
    return fig, ax


def make_X_y(df: pd.DataFrame):
    X = df.drop(columns="Rammed")
    y = df["Rammed"]
    return X, y


def format_times(
    df: pl.DataFrame, start_ram_time: datetime.datetime, end_ram_time: datetime.datetime
):
    df = df.with_columns(
        (
            pl.when(pl.col("Time").is_between(start_ram_time,end_ram_time,closed="none")
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
    
    df = pl.read_csv(filename).drop("")

    df = df.with_columns(
        pl.col("Time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
    )

    df = format_times(df, start_time, end_time)

    df = df.to_pandas()

    if scale:
        df.iloc[:, :63] = StandardScaler().fit_transform(df.iloc[:, :63])

    return df

def incorrect_preds(X_test, y_preds: np.ndarray, y_true: pd.Series = None) -> pd.DataFrame:
    def join_y( y_pred: np.ndarray, y_true: pd.Series = None) -> pd.DataFrame:
        joint_y = y_true.to_frame()
        joint_y["predicted"] = y_pred
        return joint_y
    joint_y = join_y(y_preds, y_true)
    wrong_preds = joint_y.query("Rammed != predicted")
    wrong_preds = X_test[["Time"]].join(wrong_preds, how="right")
    return wrong_preds

class RamModel:
    def __init__(self, train_flybys: list) -> None:
        self.train_flybys = train_flybys
        self.pipe = make_pipeline(LogisticRegression(class_weight="balanced"))
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_train_data(self, flyby_info: dict, scale: bool = True) -> None:
        
        train_dfs = []        
        for flyby in self.train_flybys:
            for anode_num, anode in flyby_info[flyby].anodes.items():
                train_dfs.append(get_df(
                    anode.filepath,
                    anode.start_ram_time,
                    anode.end_ram_time,
                    scale,
                ))


        df = pd.concat(train_dfs)
        X, y = make_X_y(df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=42
        )

    def fit(self):
        self.pipe.fit(self.X_train.drop(columns="Time"), self.y_train)

    def predict(self, df: pd.DataFrame = None):
        return (
            self.pipe.predict(self.X_test.drop(columns="Time"))
            if df is None
            else self.pipe.predict(df.drop(columns="Time"))
        )

    def f1_score(self, y_pred: pd.DataFrame = None):
        return f1_score(self.y_test, y_pred)
    
    
    def precision_score(self, y_pred: pd.DataFrame = None):
        return precision_score(self.y_test, y_pred)

