# -*- coding: utf-8 -*-
# @Time    : 1/24/21 1:36 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import seg_metrics.seg_metrics as sg


labels = [0, 1, 2, 3, 4, 5]
ex_id = '1631188707_500'

pred_file = '/data/jjia/multi_task/mt/scripts/results/lobe/'+ ex_id + '/infer_pred/valid/to_submit'

gdth_file = '/data/jjia/multi_task/mt/scripts/data/data_ori_space/lobe/valid'
csv_file = '/data/jjia/multi_task/mt/scripts/results/lobe/'+ ex_id + '/infer_pred/valid/lobe_metrics_all.csv'

metrics = sg.write_metrics(labels=labels[:],  # exclude background
                  gdth_path=gdth_file,
                  pred_path=pred_file,
                  csv_file=csv_file)

