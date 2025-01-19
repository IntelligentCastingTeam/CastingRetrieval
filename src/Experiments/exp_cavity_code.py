import os
import sys
from pathlib import Path
import json

import numpy as np
import dvc
import dvc.api
from tqdm import tqdm

sys.path.append("./src")
from signatures import heat_signature, cal_centroid, cal_volume


def exp_cavity_code():
    params = dvc.api.params_show()

    hull_path = params["cavity_code"]["hull_path"]
    cavity_path = params["cavity_code"]["cavity_path"]
    save_path = Path(params["cavity_code"]["save_path"])
    save_path.mkdir(exist_ok=True)

    for path, _, files in os.walk(hull_path):
        for file in tqdm(files):
            fn, ext = os.path.splitext(file)
            if not os.path.exists(os.path.join(cavity_path, file)):
                continue
            hull = np.load(os.path.join(hull_path, file))
            cavity = np.load(os.path.join(cavity_path, file))

            hull_code = {
                "centroid": cal_centroid(hull).tolist(),
                "volume": cal_volume(hull),
                "code": heat_signature(
                    hull,
                    dim=params["cavity_code"]["hull_dim"],
                    steps=params["code_extraction"]["steps"],
                    alpha=params["code_extraction"]["alpha"],
                ).tolist(),
            }

            cavities_values = set(cavity.flatten().tolist())
            cavity_codes = []

            for value in cavities_values:
                if value == 0:
                    continue
                cav = np.zeros_like(cavity)
                cav[cavity == value] = 1
                if np.sum(cav) == 1:
                    continue
                cav_code = {
                    "centroid": cal_centroid(cav).tolist(),
                    "volume": cal_volume(cav),
                    "code": heat_signature(
                        cav,
                        dim=params["cavity_code"]["cavity_dim"],
                        steps=params["code_extraction"]["steps"],
                        alpha=params["code_extraction"]["alpha"],
                    ).tolist(),
                }
                cavity_codes.append(cav_code)
            hull_code["cavity_codes"] = cavity_codes

            with open(os.path.join(save_path, f"{fn}.json"), "w") as fp:
                json.dump(hull_code, fp)


if __name__ == "__main__":
    exp_cavity_code()
