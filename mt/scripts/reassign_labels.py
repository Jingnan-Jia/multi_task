import glob
import numpy as np
import csv
from medutils.medutils import get_all_ct_names, load_itk, save_itk


dirname = "/data/jjia/multi_task/mt/scripts/data/data_ori_space/av"
scan_files = get_all_ct_names(dirname, name_suffix='_seg')
print(scan_files)
for scan_file in scan_files:
    # if '28.nrrd' in scan_file:
    #     print(scan_file)
    # ct_scan.shape: (717,, 512, 512), spacing: 0.5, 0.741, 0.741
    ct_scan, origin, spacing = load_itk (filename=scan_file, require_ori_sp=True)
    # ct_scan[ct_scan==4] = 0
    # ct_scan[ct_scan==5] = 0
    # ct_scan[ct_scan==6] = 0

    # ct_scan[ct_scan==7] = 4
    # ct_scan[ct_scan==8] = 5
    print(np.max(ct_scan), np.min(ct_scan))

    # print(f"saved at {scan_file.split('.mha')[0] + '.nii.gz'}")
    # save_itk(scan_file.split('.mha')[0] + '.nii.gz', ct_scan, origin, spacing)