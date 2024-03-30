#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import json
import numpy as np
import os

ONE_KEYS = ["fold", "uid", "dataset"]
ALL_KEYS = [
    "fold",
    "uid",
    "questions",
    "concepts",
    "responses",
    "timestamps",
    "usetimes",
    "selectmasks",
    "is_repeat",
    "qidxs",
    "rest",
    "orirow",
    "cidxs",
    "dataset",
]


def get_sub_dataset(data_config, train_ratio=1.0):
    df = pd.read_csv(
        os.path.join(data_config["dpath"], f"train_valid_quelevel.csv")
        )
    ins, ss, qs, cs, seqnum = calStatistics(
        df=df, stares=[], key="original train+valid question level"
    )
    print(
        f"origin interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}"
    )
    if train_ratio < 1.0:
        sub_data_path = os.path.join(
            data_config["dpath"], f"train_valid_quelevel_{train_ratio}.csv"
        )
        if not os.path.exists(sub_data_path):
            finaldf = extract_sub_data(df, train_ratio)
            finaldf.to_csv(sub_data_path)
        else:
            finaldf = pd.read_csv(sub_data_path)
        ins, ss, qs, cs, seqnum = calStatistics(
            df=finaldf, stares=[], key="original train+valid question level"
        )   
        print(
            f"after extract interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}"
        )
    
    sub_sequence_df = generate_sequences(
        finaldf,
        effective_keys={
            "uid",
            "questions",
            "concepts",
            "responses",
            "fold"
        },
        min_seq_len=3,
        maxlen=200,
        pad_val=-1
    )
    dpath = data_config["dpath"]
    if train_ratio < 1.0:
        sub_sequence_df.to_csv(
            f"{dpath}/train_valid_sequences_quelevel_{train_ratio}.csv",
            index=None,
        )
    else:
        print("do not have ratio")
    
    ins, ss, qs, cs, seqnum = calStatistics(
        df=sub_sequence_df, stares=[], key="train+valid sequences question level"
    )
    print(
        f"after extract  sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}"
    )
    
    return 


def extract_sub_data(df, train_ratio):
    final_sub_df = pd.DataFrame()
    # 对每个fold 按比例抽取
    for fold in range(5):
        sub_df = df[df["fold"] == fold]
        sub_df = sub_df.sample(frac=train_ratio,random_state=1024)
        final_sub_df = pd.concat([final_sub_df, sub_df],ignore_index=True)
    print(
        f"extract_sub_origin_data...original_stu_nums:{df.shape}, extract_nums:{final_sub_df.shape}"
    )
    return final_sub_df


def save_dcur(row, effective_keys):
    dcur = dict()
    for key in effective_keys:
        if key not in ONE_KEYS:
            dcur[key] = row[key].split(",")  # [int(i) for i in row[key].split(",")]
        else:
            dcur[key] = row[key]
    return dcur


def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    # 判断df中是否有timestamps列，如果有，则effective_keys中加入timestamps
    if "timestamps" in df.columns:
        effective_keys.add("timestamps")
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(
                        ",".join(dcur[key][j : j + maxlen])
                    )  # [str(k) for k in dcur[key][j: j + maxlen]]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            j += maxlen
        if rest < min_seq_len:  # delete sequence len less than min_seq_len
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate(
                    [dcur[key][j:], np.array([pad_val] * pad_dim)]
                )
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    # after preprocess data, report
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf


def calStatistics(df, stares, key):
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = list()
            for c in cs:
                cc = c.split("_")
                fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
    stares.append(",".join([str(s) for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]

