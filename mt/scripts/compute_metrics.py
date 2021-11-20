import os
import seg_metrics.seg_metrics as sg


dir1 = "/data/jjia/multi_task/mt/scripts/results/av/"
dir2 = "/infer_pred/SYSU"
ex_id_ls = [
    # "57",
    # "58",
    # "59",
    # "60",
    # "61",
    # "62",
    # "63",
    # "64",
    # "18",
    # "19",
    "16",
    # "21",
             ]

gdth_path = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/av"
print(ex_id_ls)
for ex_id in ex_id_ls:

    pred_path = dir1 + ex_id + dir2
    csv_file = os.path.join(pred_path, 'metrics_on_av.csv')

    metrics = sg.write_metrics(labels=[1,2],  # exclude background
                               gdth_path=gdth_path,
                               pred_path=pred_path,
                               csv_file=csv_file,
                               metrics=['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'hd95', 'msd', 'mdsd',
                         'stdsd'])
    print('metrics: ', metrics)