import sys

sys.path.append("../..")

from mt.mymodules.generate_fissure_from_masks import gntFissure, gntLung

# lobe_abs_dir = "/data/jjia/monai/models/lobe/1611152191_299/infer_pred/lola11/to_submit"
dir_1 = "/data/jjia/multi_task/mt/scripts/results/lobe/"
dir_2 = "/infer_pred/lola11"
id_list = [
            "1635031568_650",
            "1635031142_572",
            "1635031142_162",
            "1635031115_575",
            "1635031115_596",
            "1635031365_299",
            "1635031365_967",
            # "1635031434_249",
            # "1635031434_34",
            # "1635031521_103",
            # "1635031521_727",
            # "16",
            ]
dir_list = [dir_1 + id + dir_2 for id in id_list]
for dir in dir_list:
    gntFissure(dir, radiusValue=1, workers=5, labels=[1,2,3,4,5])
    # gntLung(dir, workers=5)
