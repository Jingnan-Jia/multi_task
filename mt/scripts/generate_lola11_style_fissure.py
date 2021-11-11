import glob
import sys
sys.path.append("../..")
from tqdm import tqdm
import os
from mt.mymodules.generate_fissure_from_masks import gntFissure, gntLung
from medutils.medutils import get_all_ct_names, load_itk, save_itk
import numpy as np

def gnt_lola11_style_fissure(gdth_folder, pred_folder):
    gdth_files = sorted(glob.glob(gdth_folder + "/*fissure_1_seg.nii.gz"))
    pred_files = sorted(glob.glob(pred_folder + "/*fissure_1_seg.nii.gz"))
    for gdth_file, pred_file in tqdm(zip(gdth_files, pred_files), total=len(gdth_files)):
        gdth, ori, sp = load_itk(gdth_file, require_ori_sp=True)
        pred, ori, sp = load_itk(pred_file, require_ori_sp=True)
        gdth = np.rollaxis(gdth, 1, 0)
        pred = np.rollaxis(pred, 1, 0)
        fissure_points = np.zeros_like(gdth)
        for idx, (gdth_slice, pred_slice) in enumerate(zip(gdth, pred)):
            if np.any(gdth_slice):
                fissure_points[idx] = pred_slice
        fissure_points = np.rollaxis(fissure_points, 1, 0)  # roll axis back
        save_itk(pred_folder + "/" + os.path.basename(gdth_file), fissure_points, ori, sp)




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
gdth_folder = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/lola11"
for pred_folder in dir_list:
    print(pred_folder)
    # gntFissure(dir, radiusValue=1, workers=5, labels=[1,2,3,4,5])
    gnt_lola11_style_fissure(gdth_folder, pred_folder)
    # gntLung(dir, workers=5)
