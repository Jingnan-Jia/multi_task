from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(ex_dt):
    path_before = "/data/jjia/multi_task/mt/scripts/results/lobe/"
    path_after = "/infer_pred/lola11/metrics_on_fissure.csv"  # _on_corrected_gdth
    data_pd_list = []
    for name, ex_id in ex_dt.items():
        metric_file = path_before + ex_id + path_after
        df = pd.read_csv(metric_file)
        df_dt = {name: df}
        data_pd_list.append(df_dt)
    return data_pd_list


def comparison_plot(data_pd_list: List[pd.DataFrame], metrics: List[str]):
    """
    metrics: dice	jaccard	precision	recall	fpr	fnr	vs	hd	msd	mdsd	stdsd	hd95

    """

    for metric in metrics:
        if metric not in list(data_pd_list[0].values())[0].columns.to_list():
            raise Exception(f"the metrics {metric} is not in data column {list(data_pd_list[0].values())[0].columns.to_list()}")
        name_ls = [list(df_dt.keys())[0] for df_dt in data_pd_list]
        df_ls = [list(df_dt.values())[0] for df_dt in data_pd_list]
        data = [d[metric].to_numpy() for d in df_ls]
        fig, ax = plt.subplots()
        ax.set_title(f'{metric} comparison')
        ax.boxplot(data, showmeans=False)
        # ax.set_yscale('log')

        plt.xticks(list(range(1, 1+len(name_ls))), name_ls, rotation=90)
        plt.show()


def merge_pd_by(data_df: pd.DataFrame, by: str):
    grouped_df = data_df.groupby([by]).mean()
    return grouped_df

if __name__ == "__main__":
    ex_dt = {
        # 'baseline_old': '16',
             'baseline': '1635031568_650',
             'lb+vs': '1635031142_572',
             'lb+rc': '1635031142_162',
             'lb+vs ad_lr=0.1': '1635031115_575',
             'lb+rc ad_lr=0.1': '1635031115_596',
             'lb+vs ad_gd=0.1': '1635031365_299',
             'lb+rc ad_gd=0.1': '1635031365_967',
             # 'lb+vs+rc ad_lr=0.1 eat': '1635031434_34',
             # 'lb+vs+rc ad_gd=0.1 eat' : '1635031434_249',
             # 'lb+vs+rc ad_lr=0.1 fat': '1635031521_727',
             # 'lb+vs+rc ad_gd=0.1 fat': '1635031521_103'
             }
    data_pd_list = load_data(ex_dt)


    # df_baseline = pd.read_csv(
    #     "/data/jjia/multi_task/mt/scripts/results/lobe/1635031568_650/infer_pred/GLUCOLD/metrics_corrected_gdth.csv")
    #
    # df_baseline_old_tr = pd.read_csv(
    #     "/data/jjia/multi_task/mt/scripts/results/lobe/16/infer_pred/GLUCOLD/metrics_corrected_gdth.csv")
    #
    # data_pd_list = [{'baseline_old': df_baseline_old_tr},
    #                 {'baseline':df_baseline}]
    new_data_pd_list = []
    for df_dt in data_pd_list:
        for k, v in df_dt.items():
            new_v = merge_pd_by(v, 'filename')
            new_data_pd_list.append({k: new_v})

    comparison_plot(new_data_pd_list, metrics=['hd95','hd', 'msd'])