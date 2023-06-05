import pandas as pd
import polars as pl
from datetime import datetime

def make_X_y(df: pl.DataFrame):
    X = df.select(pl.col("*").exclude("Rammed")).to_pandas()
    y = df.select(pl.col("Rammed")).to_series().to_pandas()
    return X, y


def format_times(
    df: pl.DataFrame, start_ram_time: datetime.time, end_ram_time: datetime.time
):  
    df = df.with_columns(
        (
            pl.when(
                (pl.col("Time").dt.hour() >= start_ram_time.hour)
                & (pl.col("Time").dt.hour() <= end_ram_time.hour)
                & (pl.col("Time").dt.minute() >= start_ram_time.minute)
                & (pl.col("Time").dt.minute() <= end_ram_time.minute)
            )
            .then(1)
            .otherwise(0)
        ).alias("Rammed")
    )

    return df


def get_df(
    filename: str, start_time: datetime.datetime, end_time: datetime.datetime
) -> pl.DataFrame:
    df = pl.read_csv(filename).drop("")
    df = df.with_columns(
        pl.col("Time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
    )
    df = format_times(df,start_time,end_time)

    return df