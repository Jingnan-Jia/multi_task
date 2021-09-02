# -*- coding: utf-8 -*-
# @Time    : 11/25/20 10:16 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import seg_metrics.seg_metrics as sg

labels = [0, 1]
    # '/data/jjia/monai/COVID-19-20_v2/Train/seg'
# pred_path = '/data/jjia/monai/output_train/to_submit'
gdth_path = "/data/jjia/monai/data_ori_space/lola11"
pred_path_first = "/data/jjia/monai/models/lobe/"
pred_path_middles = ["1610983656_838",
                     "1610983655_386",
                     "1610983655_203",
                     "1610983655_428",
                     "1611151945_187",
                     "1611151945_114",
                     "1611151945_179",
                     "1611151945_514",
                     "1611152191_525",
                     "1611152191_177",
                     "1611152190_335",
                     "1611152191_299"

                     ]
pred_path_last = "/infer_pred/lola11/to_submit"
for middle in pred_path_middles:
    pred_path = pred_path_first + middle + pred_path_last

    for postfix in ['lung_seg', 'fissure_1_seg']:
        csv_file = pred_path + '/' + postfix + '_metrics.csv'

        sg.write_metrics(labels=labels[1:],  # exclude background
                          gdth_path=gdth_path,
                          pred_path=pred_path,
                          csv_file=csv_file, prefix=None, postfix=postfix)
    # print(metrics)