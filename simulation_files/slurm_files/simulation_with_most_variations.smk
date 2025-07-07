## Convert from fst to parquet files

import os
import pandas as pd
import datetime
import numpy as np


## If Cluster


output_directory = os.path.join("/", "scratch", "sj4461", "covid_output", "EON_simulation")

file_directory = os.path.join("/", "home", "sj4461", "network_amplification",\
"simulation_files")

py_file = os.path.join(file_directory, "cloud_simulation.py")

output_file = os.path.join(output_directory, "SEIR_grocery_discount_speed_{speed}_ei_{beta_ei}_rs_{beta_rs}_N_{population}_group_{group}.parquet")

group_list = list(np.arange(1,61))

population_list = [10000]

speed_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.33, 0.5]

ei_list = [0.2, 0.5]

rs_list = [0, 1/60, 1/200]

###################################################
# Snakemake rules
###################################################


rule all:
    input:
        expand(output_file, speed = speed_list, population = population_list, \
            group = group_list, beta_ei = ei_list, beta_rs = rs_list)

rule regression:
    params:
        "{group}",
        "{speed}",
        "{beta_ei}",
        "{beta_rs}",
        "{population}"
    output:
        output_file
    script:
        py_file
