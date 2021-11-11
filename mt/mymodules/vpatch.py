
import numpy as np
import random
import sys
from functools import wraps
import time


def get_ct_for_patching( idx):
    """
    a_hgh and b_hgh must exist. but:
    if trgt_sz and trgt_sp are None and not self.mtscale, a_low, b_low, c_low would be None
    if not self.aux, c_low, c_hgh would be None
    :param idx: index of ct scan
    :return: a_low, a_hgh, b_low, b_hgh, c_low, c_hgh
    """

    # with padding, shape of a,b_ori is (z,y,x,1), spacing is (z,y,x)
    a_ori, b_ori, spacing = self._load_img_pair(idx)
    if self.task != 'no_label':  # encode first, downsample next. Otherwise we need to encode down and ori ct.
        b_ori = one_hot_encode_3d(b_ori, self.labels)  # shape: (z,y,x,chn)
        c_ori = one_hot_encode_3d(c_ori, [0, 1]) if self.aux else None  # shape: (z,y,x,2)

    if self.data_argum:
        if self.aux:
            a_ori, b_ori, c_ori = random_transform(a_ori, b_ori, c_ori)  # shape: (z,y,x,chn)
        else:
            a_ori, b_ori = random_transform(a_ori, b_ori)  # shape: (z,y,x,chn)

    if self.io != "1_in_hgh_1_out_hgh" and not (any(self.tspzyx) or any(self.tszzyx)):
        raise Exception("io is: " + str(self.io) + " but did not set trgt_space_list or trgt_sz_list")

    a_low, a_hgh, b_low, b_hgh, c_low, c_hgh = None, None, None, None, None, None
    if "in_low" in self.io or "2_in" in self.io:
        a_low = downsample(a_ori, is_mask=False,
                           ori_space=spacing, trgt_space=self.tspzyx,
                           ori_sz=a_ori.shape, trgt_sz=self.tszzyx,
                           order=1, labels=self.labels)  # shape: (z,y,x,chn)
    if "out_low" in self.io or "2_out" in self.io:
        b_low = downsample(b_ori, is_mask=True,
                           ori_space=spacing, trgt_space=self.tspzyx,
                           ori_sz=b_ori.shape, trgt_sz=self.tszzyx,
                           order=0, labels=self.labels)  # shape: (z,y,x,chn)
        if self.aux:
            c_low = downsample(c_ori, is_mask=True,
                               ori_space=spacing, trgt_space=self.tspzyx,
                               ori_sz=c_ori.shape, trgt_sz=self.tszzyx,
                               order=0, labels=self.labels)  # shape: (z,y,x,2)

    if "in_hgh" in self.io or "2_in" in self.io:
        a_hgh = a_ori  # shape: (z,y,x,chn)
    if "out_hgh" in self.io or "2_out" in self.io:
        b_hgh = b_ori  # shape: (z,y,x,chn)
        if self.aux:
            c_hgh = c_ori

    return a_low, a_hgh, b_low, b_hgh, c_low, c_hgh  # shape (z,y,x,chn)


def get_a2_patch_origin_finish(a, origin, p_sh, a2):
    """
    :param a:
    :param origin:
    :param p_sh:
    :param a2:
    :return: origin_a2, finish_a2, numpy array, shape (3, )
    """
    p_sh = np.array(p_sh)
    origin = np.array(origin)
    scale_ratio = np.array(a2.shape) / np.array(a.shape)  # shape: (z, x,y,1)
    center_idx_a = origin + p_sh // 2  # (z, y, x)
    center_idx_a2 = center_idx_a * scale_ratio[:-1]  # (z, y, x)
    center_idx_a2 = center_idx_a2.astype(int)  # (z, y, x)
    origin_a2 = center_idx_a2 - p_sh // 2  # the patch size in a2 is still p_sh
    finish_a2 = center_idx_a2 + p_sh // 2  # (z, y, x)
    return origin_a2, finish_a2


def get_a2_patch(a2, patch_shape, ref, ref_origin):
    """
    :param a2: shape (z, y, x, chn)
    :param patch_shape: (96, 144, 144)
    :param ref: lower or higher resolution array with same dimentions with a2
    :param ref_origin: 3 dimensions (z,y,x)
    :return:
    """
    # get a2_patch according a and its ori, patchshape
    # get the origin list and finish list of a2 patch
    origin_a2, finish_a2 = get_a2_patch_origin_finish(ref, ref_origin, patch_shape, a2)  # np array (z,y,x) shape (3, )

    if all(i >= 0 for i in origin_a2) and all(m < n for m, n in zip(finish_a2, a2.shape)):
        # original_a2 is positive, and finish_a2 is smaller than a2.shape
        idx_a2 = [np.arange(o_, f_) for o_, f_ in zip(origin_a2, finish_a2)]
        a2_patch = a2[np.ix_(idx_a2[0], idx_a2[1], idx_a2[2])]

    else:  # origin_a2 is negative or finish_a2 is greater than a2.shape
        print("origin_a2 or finish_a2 is out of the a2 shape, prepare to do padding")
        pad_origin = np.zeros_like(origin_a2)
        pad_finish = np.zeros_like(finish_a2)
        for i in range(len(origin_a2)):
            if origin_a2[i] < 0:  # patch is out of the left or top of a
                pad_origin[i] = abs(origin_a2[i])
                origin_a2[i] = 0

            if finish_a2[i] > a2.shape[i]:  # patch is out of the right or bottom of a
                pad_finish[i] = finish_a2[i] - a2.shape[i]
                finish_a2[i] = a2.shape[i]

        idx_a2 = [np.arange(o_, f_) for o_, f_ in zip(origin_a2, finish_a2)]
        a2_patch = a2[np.ix_(idx_a2[0], idx_a2[1], idx_a2[2])]  # (z, y, x, chn)

        pad_origin, pad_finish = np.append(pad_origin, 0), np.append(pad_finish, 0)
        pad_width = tuple([(i, j) for i, j in zip(pad_origin, pad_finish)])
        a2_patch = np.pad(a2_patch, pad_width, mode='minimum')  # (z, y, x)

    return a2_patch, idx_a2


def random_patch_(scan, patch_shape, p_middle, needorigin=0, ptch_seed=None):
    random.seed(ptch_seed)
    if ptch_seed:
        print("ptch_seed for this patch is " + str(ptch_seed))
    sh = np.array(scan.shape)  # (z, y, x, chn)
    p_sh = np.array(patch_shape)  # (z, y, x)

    range_vals = sh[0:3] - p_sh  # (z, y, x)
    if any(range_vals <= 0):  # patch size is smaller than image shape
        raise Exception("patch size is bigger than image size. patch size is ", p_sh, " image size is", sh)

    origin = []
    if p_middle:  # set sampling specific probability on central part
        tmp_nb = random.random()
        if ptch_seed:
            print("p_middle random float number for this patch is " + str(tmp_nb))
        if tmp_nb < p_middle:
            range_vals_low = list(map(int, (sh[0:3] / 3 - p_sh // 2)))
            range_vals_high = list(map(int, (sh[0:3] * 2 / 3 - p_sh // 2)))
            # assert range_vals_low > 0 and range_vals_high > 0, means if one patch cannot include center area
            if all(i > 0 for i in range_vals_low) and all(j > 0 for j in range_vals_high):
                for low, high in zip(range_vals_low, range_vals_high):
                    origin.append(random.randint(low, high))
                # print('p_middle, select more big vessels!')
    if len(origin) == 0:
        origin = [random.randint(0, x) for x in range_vals]
        if ptch_seed:
            print("origin for this patch is " + str(origin))

    finish = origin + p_sh
    for finish_voxel, a_size in zip(finish, sh):
        if finish_voxel > a_size:
            print('warning!:patch size is bigger than a size', file=sys.stderr)

    idx = [np.arange(o_, f_) for o_, f_ in zip(origin, finish)]
    patch = scan[np.ix_(idx[0], idx[1], idx[2])]
    if needorigin:
        return patch, idx, origin
    else:
        return patch, idx


def random_patch(a_low, a_hgh, b_low, b_hgh, patch_shape=(64, 128, 128),
                 p_middle=None, task=None, io=None, ptch_seed=None):
    """
    get ramdom patches from the given ct.
    :param a: one ct array, dimensions: 4, shape order: (z, y, x, chn) todo: check the shape order
    :param b: ground truth, binary masks of object, shape order: (z, y, x, 2)
    :param c: auxiliary a, binary boundary of object, shape order: (z, y, x, 2)
    :param patch_shape: patch shape, shape order: (z, y, x) todo: check the shape order
    :param p_middle: float number between 0~1, probability of patches in the middle parts of ct a
    :return: one patch of ct a (with one patch of gdth and aux if gdth and aux are not None), dimensions: 4,shape oreders are the same as their correspoonding input as
    """
    if task != "vessel":
        p_middle = None
    if io == "1_in_low_1_out_low":  # appllied for lobe_only
        a_low_patch, idx_low = random_patch_(a_low, patch_shape, p_middle, ptch_seed=ptch_seed)
        b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2])]
        return [a_low_patch, b_low_patch]
    elif io == "1_in_hgh_1_out_hgh":  # appllied for vessel_only
        a_hgh_patch, idx_hgh = random_patch_(a_hgh, patch_shape, p_middle, ptch_seed=ptch_seed)
        b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2])]
        return [a_hgh_patch, b_hgh_patch]
    else:  # 2_in_...
        if task == "lobe":  # get idx_low at first
            a_low_patch, idx_low, origin = random_patch_(a_low, patch_shape, p_middle, needorigin=True,
                                                         ptch_seed=ptch_seed)
            a_hgh_patch, idx_hgh = get_a2_patch(a_hgh, patch_shape, ref=a_low, ref_origin=origin)
        else:
            a_hgh_patch, idx_hgh, origin = random_patch_(a_hgh, patch_shape, p_middle, needorigin=True,
                                                         ptch_seed=ptch_seed)
            a_low_patch, idx_low = get_a2_patch(a_low, patch_shape, ref=a_hgh, ref_origin=origin)
        if io == "2_in_1_out_low":  # get idx_hgh at first
            b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2])]
            return [[a_low_patch, a_hgh_patch], b_low_patch]
        elif io == "2_in_1_out_hgh":
            b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2])]
            return [[a_low_patch, a_hgh_patch], b_hgh_patch]
        else:
            b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2])]
            b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2])]
            return [[a_low_patch, a_hgh_patch], [b_low_patch, b_hgh_patch]]