
import glob
import numpy as np
import csv
from medutils.medutils import get_all_ct_names, load_itk, save_itk


dirname = "/data/jjia/mt/data/lobe/train/gdth_ct/GLUCOLD"
scan_files = get_all_ct_names(dirname)
print(scan_files)
for scan_file in scan_files:
    print(scan_file)
    # ct_scan.shape: (717,, 512, 512), spacing: 0.5, 0.741, 0.741
    ct_scan, origin, spacing = load_itk (filename=scan_file, require_ori_sp=True)

    ct_scan[ct_scan==32] = 0
    ct_scan[ct_scan==64] = 0

    save_itk(scan_file.split('.nrrd')[0] + '.nii.gz', ct_scan, origin, spacing)



