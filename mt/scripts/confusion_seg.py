import os
import seg_metrics.seg_metrics as sg
from medutils.medutils import get_all_ct_names, load_itk, save_itk, get_intersection_files
import pandas as pd
import numpy as np


def confusion_seg(gdth_path=None, pred_path=None, csv_file = None):
    gdth_files = get_all_ct_names(gdth_path)
    pred_files = get_all_ct_names(pred_path)
    gdth_files, pred_files = get_intersection_files(gdth_files, pred_files)
    confu_dt = {}
    for gdth_fpath, pred_fpath in zip(gdth_files, pred_files):
        gdth = load_itk(gdth_fpath)
        pred = load_itk(pred_fpath)

        gdth = pd.Series(gdth.flatten(), name="gdth")
        pred = pd.Series(pred.flatten(), name="pred")
        df_confusion = pd.crosstab(index=gdth, columns= pred,
                                   margins=True, normalize="columns")
        csv_file_ = csv_file.split('.csv')[0] + "_" + pred_fpath.split("Pat_")[-1].split('_')[0] + '.csv'
        df_confusion.to_csv(csv_file_)
        confu_dt[csv_file_] = df_confusion

    return confu_dt


dir1 = "/data/jjia/multi_task/mt/scripts/results/av/"
dir2 = "/infer_pred/SYSU/av_masked_by_lung"
ex_id_ls = ["16", "17", "18", "19", "20", "21"]

gdth_path = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/av/av_masked_by_lung"
print(ex_id_ls)
for ex_id in ex_id_ls:
    pred_path = dir1 + ex_id + dir2
    csv_file = os.path.join(pred_path, 'confusion_on_av_masked_by_lung.csv')

    confusion_dt = confusion_seg(gdth_path, pred_path, csv_file)
    print(f'confusion: {confusion_dt}')