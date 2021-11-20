from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(ex_dt, fissure_or_lobe='lobe', task='lobe', dataset='GLUCOLD', largest_connected=True):
    path_before = "/data/jjia/multi_task/mt/scripts/results/" + task + "/"
    if task == 'av':
        path_after = "/infer_pred/" + dataset + "/metrics_on_av.csv"

    elif task == 'lobe':
        if largest_connected:
            if dataset=='lola11':
                path_after = "/infer_pred/" + dataset + "/largest_connected/points/metrics_on_fissure_points.csv"
            else:
                path_after = "/infer_pred/" + dataset + "/largest_connected/metrics_on_" + fissure_or_lobe + ".csv"
        else:
            if dataset=='lola11':
                raise Exception("All fissure points for lola11 are based on largest connected method.")
            else:
                if fissure_or_lobe == 'lobe':
                    path_after = "/infer_pred/" + dataset + "/metrics_on_lobe.csv"  # _on_corrected_gdth
                else:
                    path_after = "/infer_pred/" + dataset + "/fissure/metrics_on_fissure.csv"

    data_pd_list = []
    for name, ex_id in ex_dt.items():
        metric_file = path_before + ex_id + path_after
        df = pd.read_csv(metric_file)
        df_dt = {name: df}
        data_pd_list.append(df_dt)
    return data_pd_list


def comparison_plot(data_pd_list: List[pd.DataFrame], metrics: List[str], title_prefix: str):
    """
    metrics: dice	jaccard	precision	recall	fpr	fnr	vs	hd	msd	mdsd	stdsd	hd95

    """

    for metric in metrics:
        if metric not in list(data_pd_list[0].values())[0].columns.to_list():
            raise Exception(
                f"the metrics {metric} is not in data column {list(data_pd_list[0].values())[0].columns.to_list()}")
        name_ls = [list(df_dt.keys())[0] for df_dt in data_pd_list]
        df_ls = [list(df_dt.values())[0] for df_dt in data_pd_list]
        data = [d[metric].to_numpy() for d in df_ls]
        fig, ax = plt.subplots()
        ax.set_title(f'{title_prefix}: {metric}')
        ax.boxplot(data, showmeans=True)
        # ax.set_yscale('log')

        plt.xticks(list(range(1, 1 + len(name_ls))), name_ls, rotation=90)
        plt.show()


def merge_pd_by(data_df: pd.DataFrame, by: str):
    grouped_df = data_df.groupby([by]).mean()
    return grouped_df


if __name__ == "__main__":
    task = 'av'  # av
    dataset = 'SYSU' # 'lola11'
    fissure_or_lobe = 'lobe'  # 'fissure', 'lobe', 'fissure_points'
    largest_connected = False
    different_tr_nb = False
    if task == 'av':
        ex_dt = {
            'av, tr_nb=7, av+rc': '21',
            'av, tr_nb=1, av+rc': '18',
            'av, tr_nb=1, av+rc+aw': '17',
            # 'av, tr_nb=3, av+rc+aw': '16',
            # 'av, tr_nb=3, av+aw': '20',
            # 'av, tr_nb=3, av': '19',
        }
    else:
        if different_tr_nb:
            ex_dt = {'lb, tr_nb=1': '64',
                     'lb, tr_nb=3': '61',
                     'lb, tr_nb=8': '66',
                     'lb, tr_nb=16': '65'}
        else:
            ex_dt = {
                # 'baseline_old': '16',
                'lb': '66',
                'lb+vs': '60',
                'lb+rc': '59',
                'lb+vs, ad_lr=0.1': '57',
                'lb+rc, ad_lr=0.1': '58',
                'lb+vs, ad_gd=0.1': '62',
                'lb+rc, ad_gd=0.1': '63',
                'lb+vs+rc, ad_lr=0.1': '67',
                'lb+vs+rc, ad_gd=0.1': '68',
            }
        # ex_dt = {
        #     'lb, tr_nb=16, coarse gdth': '88',
        #     'lb, tr_nb=16, fine gdth': '65',
        # }
            # 'lb+rc ad_lr=0.1': '1635031115_596',
            # 'lb+vs ad_gd=0.1': '1635031365_299',
            # 'lb+rc ad_gd=0.1': '1635031365_967',
            # 'lb+vs+rc ad_lr=0.1 eat': '1635031434_34',
            # 'lb+vs+rc ad_gd=0.1 eat' : '1635031434_249',
            # 'lb+vs+rc ad_lr=0.1 fat': '1635031521_727',
            # 'lb+vs+rc ad_gd=0.1 fat': '1635031521_103'


    data_pd_list = load_data(ex_dt, fissure_or_lobe=fissure_or_lobe, task=task, dataset=dataset, largest_connected=largest_connected)

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
    # 'dice', 'fpr', 'fnr', for lobe
    if task=='lobe':
        title_prefix = task + ' segmentation (tr_nb=16) ' + '_'.join([dataset, fissure_or_lobe])
    elif task=='av':
        title_prefix = task + ' segmentation ' + '_'.join([dataset])

    comparison_plot(new_data_pd_list, metrics=['dice', 'fpr', 'fnr', 'hd95', 'msd'],
                    title_prefix=title_prefix)  # 'hd95','hd', 'msd' for fissure metrics
