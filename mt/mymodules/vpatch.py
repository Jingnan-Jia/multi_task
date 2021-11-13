from monai.transforms import LoadImaged, AddChanneld, Orientationd, Spacingd, ScaleIntensityRanged, SpatialPadd, Spacing

import numpy as np
import random
import sys
from functools import wraps
import time
from monai.transforms.transform import MapTransform
from monai.config import DtypeLike, KeysCollection



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
    :param a2: shape (chn, x,y,z)
    :param patch_shape: (144, 144, 96)
    :param ref: lower or higher resolution array with same dimentions with a2
    :param ref_origin: 3 dimensions (x,y,z)
    :return:
    """
    # get a2_patch according a and its ori, patchshape
    # get the origin list and finish list of a2 patch
    patch_shape = [0, patch_shape[0], patch_shape[1], patch_shape[2]]

    origin_a2, finish_a2 = get_a2_patch_origin_finish(ref, ref_origin, patch_shape, a2)  # np array (x,y,z) shape (3, )

    if all(i >= 0 for i in origin_a2) and all(m < n for m, n in zip(finish_a2, a2.shape)):
        # original_a2 is positive, and finish_a2 is smaller than a2.shape
        idx_a2 = [np.arange(o_, f_) for o_, f_ in zip(origin_a2, finish_a2)]
        a2_patch = a2[np.ix_(idx_a2[0], idx_a2[1], idx_a2[2], idx_a2[3])]

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
        a2_patch = a2[np.ix_(idx_a2[0], idx_a2[1], idx_a2[2], idx_a2[3])]  # (z, y, x, chn)

        pad_origin, pad_finish = np.append(pad_origin, 0), np.append(pad_finish, 0)
        pad_width = tuple([(i, j) for i, j in zip(pad_origin, pad_finish)])
        a2_patch = np.pad(a2_patch, pad_width, mode='minimum')  # (z, y, x)

    return a2_patch, idx_a2


def random_patch_(scan, patch_shape, p_middle, needorigin=0, ptch_seed: int=None):
    random.seed(ptch_seed)
    if ptch_seed:
        print("ptch_seed for this patch is " + str(ptch_seed))
    sh = np.array(scan.shape)  # (chn, x, y, z)
    p_sh = np.array(0, patch_shape[0], patch_shape[1], patch_shape[2])  # (chn, x, y, z)

    range_vals = sh - p_sh  # (chn, x, y, z)
    if any(range_vals <= 0):  # patch size is smaller than image shape
        raise Exception("patch size is bigger than image size. patch size is ", p_sh, " image size is", sh)

    origin = [0]
    if p_middle:  # set sampling specific probability on central part
        tmp_nb = random.random()
        if ptch_seed:
            print("p_middle random float number for this patch is " + str(tmp_nb))
        if tmp_nb < p_middle:
            range_vals_low = list(map(int, (sh[-3:] / 3 - p_sh[-3:] // 2)))
            range_vals_high = list(map(int, (sh[-3:] * 2 / 3 - p_sh[-3:] // 2)))
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
    patch = scan[np.ix_(idx[0], idx[1], idx[2], idx[3])]
    if needorigin:
        return patch, idx, origin
    else:
        return patch, idx


def random_patch(a_low, a_hgh, b_low, b_hgh, patch_shape=(128, 128, 64),
                 p_middle=None, task=None, io=None):
    """
    get ramdom patches from the given ct.
    :param a: one ct array, dimensions: 4, shape order: (chn, x,y,z)
    :param b: ground truth, binary masks of object, shape order: (chn, x,y,z)
    :param patch_shape: patch shape, shape order: (x,y,z) r
    :param p_middle: float number between 0~1, probability of patches in the middle parts of ct a
    :return: one patch of ct a (with one patch of gdth and aux if gdth and aux are not None), dimensions: 4,shape oreders are the same as their correspoonding input as
    """
    ptch_seed = random.randint(1, 10000) # set a new start of  fixed random seed to crop a pair of patches for image and gdth.
    if task != "vessel" or 'av':
        p_middle = None
    a_low_patch, b_low_patch, a_hgh_patch, b_hgh_patch = None, None, None, None
    if io == "1_in_low_1_out_low":  # appllied for lobe_only
        a_low_patch, idx_low = random_patch_(a_low, patch_shape, p_middle, ptch_seed=ptch_seed)
        b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2], idx_low[3])]
        # return [a_low_patch, b_low_patch]
    elif io == "1_in_hgh_1_out_hgh":  # appllied for vessel_only
        a_hgh_patch, idx_hgh = random_patch_(a_hgh, patch_shape, p_middle, ptch_seed=ptch_seed)
        b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2], idx_hgh[3])]
        # return [a_hgh_patch, b_hgh_patch]
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
            b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2], idx_low[3])]
            # return [[a_low_patch, a_hgh_patch], b_low_patch]
        elif io == "2_in_1_out_hgh":
            b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2], idx_hgh[3])]
            # return [[a_low_patch, a_hgh_patch], b_hgh_patch]
        else:
            b_low_patch = b_low[np.ix_(idx_low[0], idx_low[1], idx_low[2], idx_low[3])]
            b_hgh_patch = b_hgh[np.ix_(idx_hgh[0], idx_hgh[1], idx_hgh[2], idx_hgh[3])]
            # return [[a_low_patch, a_hgh_patch], [b_low_patch, b_hgh_patch]]

    return a_low_patch, b_low_patch, a_hgh_patch, b_hgh_patch



class MultiScaled(MapTransform):
    def __init__(self,
                 image_key: str,
                 label_key: str,
                 io: str,
                 tsp_xy: float,
                 tsp_z: float
                 ):
        self.image_key = image_key
        self.label_key = label_key

        self.io = io
        self.tsp_xy = tsp_xy
        self.tsp_z = tsp_z
        self.spacing_image = Spacing(pixdim=(self.tsp_xy, self.tsp_xy, self.tsp_z), mode="bilinear")
        self.spacing_gdth = Spacing(pixdim=(self.tsp_xy, self.tsp_xy, self.tsp_z), mode="nearest")


    def __call__(self, data):
        """
                a_hgh and b_hgh must exist. but:
                if trgt_sp are None and not self.mtscale, a_low, b_low would be None
                :param idx: index of ct scan
                :return: a_low, a_hgh, b_low, b_hgh,
                """
        d = dict(data)

        if (self.io != "1_in_hgh_1_out_hgh") and not any(self.tspzyx):
            # low resolution require target space zyx or target size zyx
            raise Exception("io is: " + str(self.io) + " but did not set trgt_space_list or trgt_sz_list")

        image_low, image_hgh, gdth_low, gdth_hgh = None, None, None, None
        if "in_low" in self.io or "2_in" in self.io:
            image_low = self.spacing_image(d[self.image_key])
        if "out_low" in self.io or "2_out" in self.io:
            gdth_low = self.spacing_gdth(d[self.label_key])

        if "in_hgh" in self.io or "2_in" in self.io:
            image_hgh = d[self.image_key]  # shape: (z,y,x,chn)
        if "out_hgh" in self.io or "2_out" in self.io:
            gdth_hgh = d[self.label_key]  # shape: (z,y,x,chn)

        d['image_low_patch'] = image_low
        d['image_hgh_patch'] = image_hgh

        d['gdth_low_patch'] = gdth_low
        d['gdth_hgh_patch'] = gdth_hgh

        return d


class RandMultiScaleCropd(MapTransform):
    """
    The shape of data['image_key'] should be (chn, x, y, z, ...).
    So this transform must be after "addchannel'.

    1. Down-sample image and gdth to low-spacing.
    2. Get a pair of patches from image and gdth. The two patches have the same center points coordinates.
    3.


    """
    def __init__(self,
                 io: str,
                 p_middle: float,
                 task: str,
                 ptch_sz,
    ptch_z_sz):
        """
        The keys must be ('image_key', 'label_key')!
        """
        self.task = task
        self.io = io
        self.p_middle = p_middle

        self.ptch_sz = ptch_sz
        self.ptch_z_sz = ptch_z_sz



    def __call__(self, data):
        d = dict(data)

        a_low, a_hgh, b_low, b_hgh = d['image_low_patch'],d['image_hgh_patch'] ,d['gdth_low_patch'] ,d['gdth_hgh_patch']

        a_low_patch, b_low_patch, a_hgh_patch, b_hgh_patch = random_patch(a_low, a_hgh, b_low, b_hgh,
                                    patch_shape=(self.ptch_sz, self.ptch_sz, self.ptch_z_sz),
                                    p_middle=self.p_middle, task=self.task, io=self.io)

        d['image_low_patch'] = a_low_patch
        d['image_hgh_patch'] = a_hgh_patch

        d['gdth_low_patch'] = b_low_patch
        d['gdth_hgh_patch'] = b_hgh_patch

        return d


