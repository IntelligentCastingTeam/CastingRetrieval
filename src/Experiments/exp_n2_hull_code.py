import numpy as np
import json
import sys
import os

from pathlib import Path
from tqdm import tqdm
from dvc.api import params_show

sys.path.append("./src\\")
from signatures import sample_voxel_N2


def exp_n2_hull_code():
    params = params_show()

    model_path = params["n2_hull_code"]["model_path"]
    save_path = Path(params["n2_hull_code"]["save_path"])
    save_path.mkdir(exist_ok=True)

    codes = {}

    for path, _, files in os.walk(model_path):
        for file in tqdm(files):
            name, ext = os.path.splitext(file)
            if ext.lower() != ".npy":
                continue

            model = np.load(os.path.join(path, file))

            code = sample_voxel_N2(
                model=model,
                output_dimension=params["n2_hull_code"]["dim"],
                shape=np.array(params["n2_hull_code"]["shape"]),
            )
            codes[name] = code.tolist()

    with open(os.path.join(save_path, "n2_hull_code.json"), "w") as fp:
        json.dump(codes, fp)
    pass


if __name__ == "__main__":
    exp_n2_hull_code()
