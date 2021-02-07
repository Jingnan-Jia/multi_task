# -*- coding: utf-8 -*-
# @Time    : 1/24/21 1:36 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import seg_metrics.seg_metrics as sg


labels = [0, 1, 2, 3, 4, 5]

gdth_file = '/data/jjia/monai/models/lobe/1610983656_838/infer_pred/lobe/to_submit'
pred_file = '/data/jjia/monai/data_ori_space/lobe/valid'
csv_file = 'lobe_metrics_all.csv'

metrics = sg.write_metrics(labels=labels[:],  # exclude background
                  gdth_path=gdth_file,
                  pred_path=pred_file,
                  csv_file=csv_file)

