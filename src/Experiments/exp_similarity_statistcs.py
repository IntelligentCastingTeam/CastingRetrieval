import json
import logging
import os
import sys
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo
from bson import ObjectId
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

    with Live("./dvclive/pr") as live:
        data = pd.DataFrame()
        for name in tqdm(sims_dict):
            if len(sims_dict[name]) == 0:
                continue
            file = f"{name}.json"
            doc = parts.find_one({"_id": ObjectId(name)})
            with open(os.path.join(similarity_path, file), "r") as fp:
                distances = json.load(fp)
            distances = sorted(distances.items(), key=lambda k: (k[1]), reverse=False)
            tp_count = 0
            t_count = 0
            for index in range(len(distances)):
                id, dis = distances[index]
                sim_part = parts.find_one({"_id": ObjectId(id)})
                volume_sim = 1 - abs(doc["volume"] - sim_part["volume"]) / (
                    doc["volume"] + sim_part["volume"]
                )
                ssa_sim = 1 - abs(doc["ssa"] - sim_part["ssa"]) / (
                    doc["ssa"] + sim_part["ssa"]
                )
                if volume_sim < volume_thres or ssa_sim < ssa_thres:
                    continue

                sim = math.floor(1 / (1 + dis) * 100) / 100.0
                t_count += 1
                if id in sims_dict[name]:
                    tp_count += 1

                precision = tp_count / t_count
                recall = tp_count / len(sims_dict[name])

                data = pd.concat(
                    [
                        data,
                        pd.DataFrame(
                            {
                                "name": str(name),
                                "count": t_count,
                                "thres": sim,
                                "precision": precision,
                                "recall": recall,
                                "FScore": (
                                    2 * precision * recall / (precision + recall)
                                    if recall != 0
                                    else 0
                                ),
                            },
                            index=[
                                0,
                            ],
                        ),
                    ],
                    ignore_index=True,
                )
        data.to_csv(os.path.join(save_path, "statistic.csv"), index=False)
        pr_data = data.groupby("thres")[["precision", "recall"]].mean()
        pr_data = pr_data.reset_index()
        count_data = data.groupby("count")[["precision", "recall"]].mean()
        count_data = count_data.reset_index()
        live.log_plot(
            name="PR Curve",
            datapoints=pr_data,
            x="recall",
            y="precision",
            template="linear",
            x_label="Recall",
            y_label="Precision",
        )
        live.log_plot(
            name="Count pr curve",
            datapoints=count_data,
            x="recall",
            y="precision",
            x_label="Recall",
            y_label="Precision",
        )

        thres_F = data.groupby("thres")[["FScore"]].mean()
        thres_F = thres_F.reset_index()
        count_F = data.groupby("count")[["FScore"]].mean()
        count_F = count_F.reset_index()
        live.log_plot(
            name="Thres FScore",
            datapoints=thres_F,
            x="thres",
            y="FScore",
            x_label="Threshold",
            y_label="FScore",
        )
        live.log_plot(
            name="Count FScore",
            datapoints=count_F,
            x="count",
            y="FScore",
            x_label="Count",
            y_label="FScore",
        )

    pass


def retrieval_err():
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

    docs = parts.find({"sim_level": {"$exists": True}})
    doc_ids = [doc["_id"] for doc in docs]

    R = []
    n = 4
    for i in range(n):
        R.append((2**i - 1) / (2 ** (n - 1)))

    errs = pd.DataFrame()

    for doc_id in tqdm(doc_ids):
        id = str(doc_id)
        doc = parts.find_one({"_id": doc_id})
        with open(os.path.join(similarity_path, f"{id}.json"), "r") as fp:
            simlist = json.load(fp)

        simlist = sorted(simlist.items(), key=lambda k: (k[1]), reverse=False)

        R_list = []
        err = 0
        if len(simlist) == 0:
            continue
        for part_id, _ in simlist:
            sim_part = parts.find_one({"_id": ObjectId(part_id)})
            volume_sim = 1 - abs(doc["volume"] - sim_part["volume"]) / (
                doc["volume"] + sim_part["volume"]
            )
            ssa_sim = 1 - abs(doc["ssa"] - sim_part["ssa"]) / (
                doc["ssa"] + sim_part["ssa"]
            )
            if volume_sim < volume_thres or ssa_sim < ssa_thres:
                continue

            if part_id in doc["sim_level"]:
                level = doc["sim_level"][part_id]
            else:
                level = 0

            if len(R_list) == 0:
                err = R[level]
            else:
                err += R[level] * np.prod(1 - np.array(R_list)) / (len(R_list) + 1)

            R_list.append(R[level])

        if err == 0:
            continue
        errs = pd.concat(
            [
                errs,
                pd.DataFrame(
                    {"name": id, "err": err},
                    index=[0],
                ),
            ],
            ignore_index=True,
        )
    errs.to_csv(os.path.join(save_path, "err.csv"), index=False)
    mean_err = errs["err"].mean()
    with Live("./dvclive\\err") as live:
        live.log_metric("err", mean_err, plot=False)


if __name__ == "__main__":
    retrieval_pr()
    retrieval_err()
