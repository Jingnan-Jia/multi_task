import os
import seg_metrics.seg_metrics as sg


dir1 = "/data/jjia/multi_task/mt/scripts/results/lobe/"
dir2 = "/infer_pred/lola11"
ex_id_ls = [
    # "1635031568_650",
    #          "1635031142_572",
    #          "1635031142_162",
    #          "1635031115_575",
    #          "1635031115_596",
             "1635031365_299",
             "1635031365_967"
             ]

gdth_path = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/lola11"

for ex_id in ex_id_ls:
    pred_path = dir1 + ex_id + dir2
    csv_file = os.path.join(pred_path, 'metrics_on_fissure.csv')

    metrics = sg.write_metrics(labels=[1],  # exclude background
                               gdth_path=gdth_path,
                               pred_path=pred_path,
                               csv_file=csv_file,
                               metrics=['hd', 'hd95', 'msd'])
    print('metrics: ', metrics)