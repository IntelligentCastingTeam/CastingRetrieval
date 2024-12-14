import json
import logging
import os
import traceback
from pathlib import Path

import numpy as np
from dvc.api import params_show
from tqdm import tqdm


def exp_compare_heat_code():
    params = params_show()
    save_path = Path(params["code_comparison"]["save_path"])
    save_path.mkdir(exist_ok=True)

    with open(params["code_comparison"]["code_path"]) as fd:
        codes = json.load(fd)

    for name, code in tqdm(codes.items()):
        try:
            distances = {}
            code = np.array(code)
            for target_name, target_code in codes.items():
                if name == target_name:
                    continue
                target_code = np.array(target_code)
                distance = np.linalg.norm(code - target_code)
                distances[target_name] = distance

            with open(os.path.join(save_path, f"{name}.json"), "w") as fp:
                json.dump(distances, fp)

            logging.info(f"{name} heat code comparsion completed.")
        except Exception:
            logging.error(
                f"{name} heat code comparsion error: {traceback.format_exc()}"
            )


if __name__ == "__main__":
    exp_compare_heat_code()
