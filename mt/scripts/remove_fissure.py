import glob
import sys
sys.path.append("../..")
from tqdm import tqdm
import os
from mt.mymodules.generate_fissure_from_masks import gntFissure, gntLung
from medutils.medutils import get_all_ct_names, load_itk, save_itk
import numpy as np
import glob
import copy
import random
from scipy.interpolate import NearestNDInterpolator

from scipy import ndimage as nd


def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]



pat_ls = ['03', '17', '21', '22', '25']
for pat in pat_ls:
    ct_fpath = '/data/jjia/multi_task/mt/scripts/data/data_ori_space/lobe/GLUCOLD_patients_' + pat + '_ct.nii.gz'
    fissure_fpath = '/data/jjia/multi_task/mt/scripts/data/data_ori_space/lobe/test_data_fissure/GLUCOLD_patients_' + pat + '_fissure_1_seg.nii.gz'
    ct, ori, sp = load_itk(ct_fpath, require_ori_sp=True)
    fissure, _, __ = load_itk(fissure_fpath, require_ori_sp=True)

    tmp_ct = copy.deepcopy(ct)
    # value, counts = np.unique(tmp_ct, return_counts=True)
    # count_sort_ind = np.argsort(-counts)
    # peak_values = value[count_sort_ind]
    #
    # fill_value = 0
    # for peak in peak_values:
    #     if peak > -1200 and peak < -500:
    #         fill_value = peak
    #         break
    #
    # if fill_value == 0:
    #     raise Exception(f"fill value is still 0 !")
    # replace_array = np.random.randint(fill_value - 20, fill_value + 20, size=tmp_ct.shape)
    # new_array = np.where(fissure==0, tmp_ct, replace_array)


    # interp = NearestNDInterpolator(np.transpose(fissure), tmp_ct[fissure])
    # new_array = interp(*np.indices(tmp_ct.shape))
    new_array = fill(tmp_ct, fissure)

    new_fpath = os.path.dirname(ct_fpath) + "/remove_fissure/" + os.path.basename(ct_fpath)
    save_itk(new_fpath, new_array, ori, sp)
    print(f"yes !")