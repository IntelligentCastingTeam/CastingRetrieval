import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo
from bson import ObjectId
from sympy import true
from tqdm import tqdm

sys.path.append("./src")
from dvc.api import params_show

from dvclive import Live


def retrieval_pr():
    params = params_show()

    similarity_path = params["statistic"]["similarity_path"]
    save_path = Path(params["statistic"]["save_path"])
    save_path.mkdir(exist_ok=True)
    truth_path = params["statistic"]["truth_path"]

    volume_thres = params["statistic"]["volume_thres"]
    ssa_thres = params["statistic"]["ssa_thres"]

    client = pymongo.MongoClient("mongodb://localhost")
    db = client["castingprocesses"]
    parts = db["parts"]

    if not os.path.exists(truth_path):
        sims_dict = defaultdict(list)

        docs = parts.find({"sim_level": {"$exists": True}})

        for doc in tqdm(docs):
            id = str(doc["_id"])
            sim_level = doc["sim_level"]
            sim_parts = []
            for key in sim_level:
                sim_part = parts.find_one({"_id": ObjectId(key)})
                volume_sim = 1 - abs(doc["volume"] - sim_part["volume"]) / (
                    doc["volume"] + sim_part["volume"]
                )
                ssa_sim = 1 - abs(doc["ssa"] - sim_part["ssa"]) / (
                    doc["ssa"] + sim_part["ssa"]
                )
                if volume_sim < volume_thres or ssa_sim < ssa_thres:
                    continue
                sim_parts.append(key)

            if len(sim_parts) == 0:
                continue
            sims_dict[id] = sim_parts

        with open(truth_path, "w") as fp:
            json.dump(sims_dict, fp)
    else:
        with open(truth_path, "r") as fp:
            sims_dict = json.load(fp)

    with Live() as live:
        data = pd.DataFrame()
        for name in tqdm(sims_dict):
            file = f"{name}.json"
            part_first = parts.find_one({"_id": ObjectId(name)})
            with open(os.path.join(similarity_path, file), "r") as fp:
                distances = json.load(fp)
            for thres in np.arange(start=0.5, stop=1, step=0.05):
                tp_count = 0
                t_count = 0

                for id in distances:
                    part_second = parts.find_one({"_id": ObjectId(id)})
                    volume_sim = 1 - abs(
                        part_first["volume"] - part_second["volume"]
                    ) / (part_first["volume"] + part_second["volume"])
                    ssa_sim = 1 - abs(part_first["ssa"] - part_second["ssa"]) / (
                        part_first["ssa"] + part_second["ssa"]
                    )
                    if volume_sim < volume_thres or ssa_sim < ssa_thres:
                        continue
                    sim = 1 / (1 + distances[id])
                    if sim > thres:
                        t_count += 1
                        if id in sims_dict[name]:
                            tp_count += 1

                if t_count == 0:
                    precision = 0
                else:
                    precision = tp_count / t_count

                recall = tp_count / len(sims_dict[name])

                data = pd.concat(
                    [
                        data,
                        pd.DataFrame(
                            {
                                "name": str(name),
                                "thres": thres,
                                "precision": precision,
                                "recall": recall,
                            },
                            index=[
                                0,
                            ],
                        ),
                    ],
                    ignore_index=True,
                )
        data.to_csv(os.path.join(save_path, "statistic.csv"), index=False)
        data = data.groupby("thres")[["precision", "recall"]].mean()
        live.log_plot(
            name="PR Curve",
            datapoints=data,
            x="recall",
            y="precision",
            template="linear",
            x_label="Recall",
            y_label="Precision",
        )

    pass


if __name__ == "__main__":
    retrieval_pr()
