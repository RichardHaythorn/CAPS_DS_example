"""Place for holding data about flybys"""

from datetime import datetime
from dataclasses import dataclass


@dataclass
class Anode:
    """Class for holding information about a flyby"""

    filepath: dict
    start_ram_time: dict
    end_ram_time: dict


@dataclass
class Flyby:
    """Class for holding information about a flyby"""

    anodes: dict
    start_time: datetime
    end_time: datetime


flyby_info = {
    "T55": Flyby(
        anodes={
            "3": Anode(
                "data/ELS_data_T55_a3.parquet",
                datetime(2009, 5, 21, 21, 26, 0),
                datetime(2009, 5, 21, 21, 29, 0),
            ),
            "4": Anode(
                "data/ELS_data_T55_a4.parquet",
                datetime(2009, 5, 21, 21, 24, 0),
                datetime(2009, 5, 21, 21, 28, 0),
            ),
        },
        start_time=datetime(2009, 5, 21, 21, 20, 0),
        end_time=datetime(2009, 5, 21, 21, 35, 0),
    ),
    "T56": Flyby(
        anodes={
            "3": Anode(
                "data/ELS_data_T56_a3.parquet",
                datetime(2009, 6, 6, 19, 58, 0),
                datetime(2009, 6, 6, 20, 3, 0),
            ),
            "4": Anode(
                "data/ELS_data_T56_a4.parquet",
                datetime(2009, 6, 6, 19, 58, 0),
                datetime(2009, 6, 6, 20, 3, 0),
            ),
        },
        start_time=datetime(2009, 6, 6, 19, 57, 0),
        end_time=datetime(2009, 6, 6, 20, 7, 0),
    ),
    "T57": Flyby(
        anodes={
            "3": Anode(
                "data/ELS_data_T57_a3.parquet",
                datetime(2009, 6, 22, 18, 30, 0),
                datetime(2009, 6, 22, 18, 35, 0),
            ),
            "4": Anode(
                "data/ELS_data_T57_a4.parquet",
                datetime(2009, 6, 22, 18, 30, 0),
                datetime(2009, 6, 22, 18, 35, 0),
            ),
        },
        start_time=datetime(2009, 6, 22, 18, 25, 0),
        end_time=datetime(2009, 6, 22, 18, 40, 0),
    ),
    "T58": Flyby(
        anodes={
            "3": Anode(
                "data/ELS_data_T58_a3.parquet",
                datetime(2009, 7, 8, 17, 4, 0),
                datetime(2009, 7, 8, 17, 7, 0),
            ),
            "4": Anode(
                "data/ELS_data_T58_a4.parquet",
                datetime(2009, 7, 8, 17, 1, 0),
                datetime(2009, 7, 8, 17, 4, 0),
            ),
        },
        start_time=datetime(2009, 7, 8, 17, 0, 0),
        end_time=datetime(2009, 7, 8, 17, 10, 0),
    ),
    "T59": Flyby(
        anodes={
            "3": Anode(
                "data/ELS_data_T59_a3.parquet",
                datetime(2009, 7, 24, 15, 32, 0),
                datetime(2009, 7, 24, 15, 36, 0),
            ),
            "4": Anode(
                "data/ELS_data_T59_a4.parquet",
                datetime(2009, 7, 24, 15, 33, 0),
                datetime(2009, 7, 24, 15, 36, 0),
            ),
        },
        start_time=datetime(2009, 7, 24, 15, 20, 0),
        end_time=datetime(2009, 7, 24, 15, 35, 0),
    ),
}
