from typing import List, Tuple
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

DATA_FOLDER = "./dataset/raw_data/EEM"
MEASUREMENT_SHEETNAME = "Data(2018-2020)"
FEATURE_COLUMNS = [
    "W.T.",
    "pH",
    "DO",
    "EC",
    "BOD5",
    "CODMn",
    "SS",
    "TN",
    "TP",
    "TOC",
    "DOC",
    "Chl-a",
    "TN,",
    "NH3-N",
    "NO3-N",
    "DTP",
    "PO4-P",
    "Water Depth",
]


def get_file_list(folder_path: str) -> List[str]:
    def date_time_parser(s: str) -> datetime:
        return datetime.strptime(s[:-5], "%Y %b%d")

    file_list = os.listdir(path=folder_path)
    filtered_validation_files = filter(
        lambda s: s[:2] != "~$" and s[:4] != ".DS_", file_list
    )
    sorted_by_datetime = sorted(filtered_validation_files, key=date_time_parser)
    return sorted_by_datetime


def get_eem_matrix(base_path: str, file_list: List[str]) -> np.ndarray:
    heatmap = []
    for file in file_list:
        fullpath = os.path.join(base_path, file)
        df = pd.read_excel(io=fullpath, sheet_name=None, header=None)
        for k, v in df.items():
            if k == "18b":
                continue
            heatmap.append(v.to_numpy())
    return np.array(heatmap)


def get_measurement_result(base_path: str, file_name: str) -> np.ndarray:
    fullpath = os.path.join(base_path, file_name)
    data = pd.read_excel(io=fullpath, sheet_name=MEASUREMENT_SHEETNAME)
    return data.to_numpy()

def prepare_dataset(base_path: str, eem_path: str, measurement_path: str) -> Tuple[np.ndarray, np.ndarray]:
    eem_fullpath = os.path.join()
    eem_matrix = get_eem_matrix(base_path=base_path, file_list=[])
