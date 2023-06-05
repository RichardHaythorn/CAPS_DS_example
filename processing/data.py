
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
                "data/ELS_data_T55_a3.csv",
                datetime(2009, 5, 21, 21, 25, 0),
                datetime(2009, 5, 21, 21, 30, 0),
            ),
            "4": Anode(
                "data/ELS_data_T55_a4.csv",
                datetime(2009, 5, 21, 21, 25, 0),
                datetime(2009, 5, 21, 21, 30, 0),
            ),
        },
        start_time=datetime(2009, 5, 21, 21, 22, 0),
        end_time=datetime(2009, 5, 21, 21, 35, 0),
    ),
    "T56": Flyby(
        anodes={
            "3": Anode(
                "data/ELS_data_T56_a3.csv",
                datetime(2009, 6, 6, 19, 59, 0),
                datetime(2009, 6, 6, 20, 3, 0),
            ),
            "4": Anode(
                "data/ELS_data_T56_a4.csv",
                datetime(2009, 6, 6, 19, 59, 0),
                datetime(2009, 6, 6, 20, 3, 0),
            ),
        },
        start_time=datetime(2009, 6, 6, 19, 57, 0),
        end_time=datetime(2009, 6, 6, 20, 7, 0),
    )
}