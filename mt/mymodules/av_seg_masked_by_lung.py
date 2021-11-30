import glob
import numpy as np
import csv
from medutils.medutils import get_all_ct_names, load_itk, save_itk
import os


def intersection(lung_files, av_files):
    lung_pat_nb = [os.path.basename(i).split('Pat_')[-1].split('_')[0] for i in lung_files]
    av_pat_nb = [os.path.basename(i).split('Pat_')[-1].split('_')[0] for i in av_files]
    out_ls = []
    for pat, lung_file in zip(lung_pat_nb, lung_files):
        if pat in av_pat_nb:
            out_ls.append(lung_file)

    return out_ls


def av_seg_masked_by_lung(ex_id):


    lung_dirname = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/av/binary_lung_masks"
    lung_files = get_all_ct_names(lung_dirname)


    dir1 = "/data/jjia/multi_task/mt/scripts/results/av/"
    dir2_av = "/infer_pred/SYSU"
    dir2_av_masked_by_lung = dir2_av + "/av_masked_by_lung"

    pred_path = dir1 + ex_id + dir2_av
    save_path = dir1 + ex_id + dir2_av_masked_by_lung

    av_files = get_all_ct_names(pred_path)
    lung_files = intersection(lung_files, av_files)

    for lung_file, av_file in zip(lung_files, av_files):
        # if '28.nrrd' in scan_file:
        #     print(scan_file)
        # ct_scan.shape: (717,, 512, 512), spacing: 0.5, 0.741, 0.741
        lung, origin, spacing = load_itk (filename=lung_file, require_ori_sp=True)
        av, origin, spacing = load_itk (filename=av_file, require_ori_sp=True)
        out = lung * av
        # ct_scan[ct_scan==6] = 0

        # ct_scan[ct_scan==7] = 4
        # ct_scan[ct_scan==8] = 5
        # print(np.max(ct_scan), np.min(ct_scan))

        # print(f"saved at {scan_file.split('.mha')[0] + '.nii.gz'}")
        target_fpath = save_path + "/" + os.path.basename(av_file)
        save_itk(target_fpath, out, origin, spacing)
        print(f"successfully save av masked by lung to {target_fpath}")