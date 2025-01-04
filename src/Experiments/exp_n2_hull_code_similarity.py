import json
import os
from pathlib import Path

import numpy as np
from dvc.api import params_show
from tqdm import tqdm


def exp_n2_hull_code_similarity():
    params = params_show()
    save_path = Path(params["n2_hull_code_similarity"]["save_path"])
    save_path.mkdir(exist_ok=True)

    with open(params["n2_hull_code_similarity"]["code_path"], "r") as fp:
        codes = json.load(fp)

    for name, code in tqdm(codes.items()):
        sims = {}
        code = np.array(code)
        for target_name, target_code in codes.items():
            if name == target_name:
                continue
            target_code = np.array((target_code))
            distance = np.linalg.norm(code - target_code)
            sim = 1 / (1 + distance)
            sims[target_name] = sim

        with open(os.path.join(save_path, f"{name}.json"), "w") as fp:
            json.dump(sims, fp)
    pass


if __name__ == "__main__":
    exp_n2_hull_code_similarity()
