import numpy as np
import pandas as pd



def collect_loading_time(log_fpath):
    pasitive_times = 0
    zero_times = 0
    pasitive_time = 0
    with open(log_fpath, 'r') as f:
        for row in f:
            if "load data cost time " in row:
                loading_time = row.split("load data cost time ")[-1].split(',')[0]
                loading_time = int(loading_time)
                if loading_time!=0:
                    pasitive_times += 1
                    pasitive_time += loading_time
                    print(f"loading time: {loading_time}")
                else:
                    zero_times += 1
    print(f"pasitive_times: {pasitive_times}")
    print(f"zero_times: {zero_times}")
    print(f"positive time: {pasitive_time}")

    return None


if __name__ == '__main__':
    log_fpath = '/data/jjia/multi_task/mt/scripts/results/slurmlogs/slurm-134141_0.out'
    collect_loading_time(log_fpath)
