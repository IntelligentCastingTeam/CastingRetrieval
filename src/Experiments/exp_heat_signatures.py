import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from dvc.api import params_show
from tqdm import tqdm

sys.path.append("./src")
from signatures import heat_signature


def extract_heat_signatures():
    params = params_show()

    save_path = Path(params["code_extraction"]["save_path"])
    save_path.mkdir(exist_ok=True)

    codes = {}

    for path, dirs, files in os.walk(params["code_extraction"]["model_path"]):
        for file in tqdm(files):
            name, ext = os.path.splitext(file)
            if ext.lower() != ".npy":
                continue

            try:
                model = np.load(os.path.join(path, file))
                code = heat_signature(
                    model=model,
                    dim=params["code_extraction"]["dim"],
                    steps=params["code_extraction"]["steps"],
                    alpha=params["code_extraction"]["alpha"],
                )

                codes[name] = code.tolist()

                logging.info(f"{name} code extraction completed.")

            except Exception:
                logging.error(f"{name} code extraction error: {traceback.format_exc()}")

    with open(os.path.join(save_path, "heat_codes.json"), "w") as f:
        json.dump(codes, f)
    pass


if __name__ == "__main__":
    extract_heat_signatures()
