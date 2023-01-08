#!/usr/bin/env python
import numpy as np
import pandas as pd
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from tqdm.auto import tqdm
from collections import defaultdict


def generate_descriptors(infile_name, descriptor_type_list):
    df = pd.read_csv(infile_name)
    desc_dict = {}
    for desc_type in descriptor_type_list:
        desc_dict[desc_type] = MakeGenerator([desc_type])

    res_dict = defaultdict(list)
    for smi in tqdm(df.smiles):
        for desc_type in descriptor_type_list:
            res = desc_dict[desc_type].process(smi)
            if not res[0]:
                print("oops")
            res_dict[desc_type].append(res[1:])

    for k, v in res_dict.items():
        df[k] = v
    return df


if __name__ == "__main__":
    descriptor_type_list = ["Morgan3", "Morgan3Counts"]
    desc_df = generate_descriptors("data/CHEMBL204_Ki.csv", descriptor_type_list)

