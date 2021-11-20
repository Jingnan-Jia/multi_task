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
        save_dir = pred_folder + "/points"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_itk(save_dir + '/' + os.path.basename(gdth_file), fissure_points, ori, sp)




    # lobe_abs_dir = "/data/jjia/monai/models/lobe/1611152191_299/infer_pred/lola11/to_submit"
dir_1 = "/data/jjia/multi_task/mt/scripts/results/lobe/"
dir_2 = "/infer_pred/lola11/largest_connected"
id_list = [
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
gdth_folder = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/lola11"
for pred_folder in dir_list:
    print(pred_folder)
    # gntFissure(dir, radiusValue=1, workers=5, labels=[1,2,3,4,5])
    gnt_lola11_style_fissure(gdth_folder, pred_folder)
    # gntLung(dir, workers=5)
