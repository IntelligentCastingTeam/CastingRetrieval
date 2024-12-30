import os
import sys
import numpy as np
import json

from tqdm import tqdm
from pathlib import Path
from dvc.api import params_show

sys.path.append("./src")
from signatures import cal_simulation


def exp_compare_code():
    params = params_show()
    save_path = Path(params["cavity_comparison"]["save_path"])
    save_path.mkdir(exist_ok=True)

    code_path = params["cavity_comparison"]["code_path"]

    codes = dict()

    for path, _, files in os.walk(params["cavity_comparison"]["code_path"]):
        for file in tqdm(files):
            name, ext = os.path.splitext(file)
            if ext.lower() != ".json":
                continue

            with open(os.path.join(path, file), "r") as fp:
                code = json.load(fp)
                codes[name] = code

    for name, code in tqdm(codes.items()):
        sims = dict()
        for target_name, target_code in codes.items():
            if name == target_name:
                continue

            sim = cal_simulation(code, target_code)
            sims[target_name] = sim

        with open(os.path.join(save_path, f"{name}.json"), "w") as fp:
            json.dump(sims, fp)


if __name__ == "__main__":
    exp_compare_code()
