import sys

sys.path.append("../..")
import argparse

from mt.mymodules.generate_fissure_from_masks import gntFissure, gntLung

parser = argparse.ArgumentParser(description="Run multi-task UNet segmentation.")
parser.add_argument('--ex', help='ex id ', type=str, default='57')
args = parser.parse_args()

# lobe_abs_dir = "/data/jjia/monai/models/lobe/1611152191_299/infer_pred/lola11/to_submit"
dir_1 = "/data/jjia/multi_task/mt/scripts/results/lobe/"
dir_2 = "/infer_pred/GLUCOLD"
id_list = [
    # args.ex
            # "57",
            # "58",
            # "59",
            # "60",
            # "61",
            # "62",
            # "63",
            # "64",
            # "65",
            # "66",
            # "67",
            "88",
            ]
dir_list = [dir_1 + id + dir_2 for id in id_list]

# dir_list = ["/data/jjia/multi_task/mt/scripts/data/data_ori_space/lobe/test_data"]
for dir in dir_list:
    print(dir)
    gntFissure(dir, radiusValue=1, workers=5, labels=[1,2,3,4,5])
    # gntLung(dir, workers=5)
