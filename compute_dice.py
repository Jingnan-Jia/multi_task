# -*- coding: utf-8 -*-
# @Time    : 11/25/20 10:16 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import seg_metrics.seg_metrics as sg

labels = [0, 1]
gdth_path = '/data/jjia/monai/COVID-19-20_v2/Train/seg'
pred_path = '/data/jjia/monai/output_train/to_submit'
csv_file = '/data/jjia/monai/output_train/to_submit/metrics.csv'

metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                  gdth_path=gdth_path,
                  pred_path=pred_path,
                  csv_file=csv_file)
print(metrics)