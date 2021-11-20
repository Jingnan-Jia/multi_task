import os
import seg_metrics.seg_metrics as sg


dir1 = "/data/jjia/multi_task/mt/scripts/results/av/"
dir2 = "/infer_pred/SYSU/av_masked_by_lung"
ex_id_ls = [
"16", "17", "18", "19", "20", "21"
             ]

gdth_path = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/av/av_masked_by_lung"
print(ex_id_ls)
for ex_id in ex_id_ls:

    pred_path = dir1 + ex_id + dir2
    csv_file = os.path.join(pred_path, 'metrics_on_av_masked_by_lung.csv')

    metrics = sg.write_metrics(labels=[1,2],  # exclude background
                               gdth_path=gdth_path,
                               pred_path=pred_path,
                               csv_file=csv_file,
                               metrics=['dice', 'jaccard', 'precision', 'recall', 'fpr', 'fnr', 'vs', 'hd', 'hd95', 'msd', 'mdsd',
                         'stdsd'])
    print('metrics: ', metrics)