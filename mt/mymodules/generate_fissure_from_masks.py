import medutils.medutils as futil
import numpy as np
import SimpleITK as sitk
import copy
import threading
import time
from medutils.medutils import get_all_ct_names, execute_the_function_multi_thread, load_itk, save_itk
import glob
from tqdm import tqdm
import os

def get_fissure(scan, radiusValue=3, labels=None):
    lobe_4 = copy.deepcopy(scan)
    lobe_4[lobe_4 != labels[0]] = 0
    lobe_4[lobe_4 == labels[0]] = 1
    lobe_5 = copy.deepcopy(scan)
    lobe_5[lobe_5 != labels[1]] = 0
    lobe_5[lobe_5 == labels[1]] = 1
    lobe_6 = copy.deepcopy(scan)
    lobe_6[lobe_6 != labels[2]] = 0
    lobe_6[lobe_6 == labels[2]] = 1
    lobe_7 = copy.deepcopy(scan)
    lobe_7[lobe_7 != labels[3]] = 0
    lobe_7[lobe_7 == labels[3]] = 1
    lobe_8 = copy.deepcopy(scan)
    lobe_8[lobe_8 != labels[4]] = 0
    lobe_8[lobe_8 == labels[4]] = 1
    right_lung = lobe_4 + lobe_5 + lobe_6
    left_lung = lobe_7 + lobe_8

    f_dilate = sitk.BinaryDilateImageFilter()
    f_dilate.SetKernelRadius(radiusValue)
    f_dilate.SetForegroundValue(1)
    f_subtract = sitk.SubtractImageFilter()

    right_lung = sitk.GetImageFromArray(right_lung.astype('int16'))
    right_lung_diated = f_dilate.Execute(right_lung)
    rightlungBorder = f_subtract.Execute(right_lung_diated, right_lung)
    rightlungBorder = sitk.GetArrayFromImage(rightlungBorder)

    left_lung = sitk.GetImageFromArray(left_lung.astype('int16'))
    left_lung_diated = f_dilate.Execute(left_lung)
    leftlungBorder = f_subtract.Execute(left_lung_diated, left_lung)
    leftlungBorder = sitk.GetArrayFromImage(leftlungBorder)

    border = np.zeros((scan.shape))
    for lobe in [lobe_4, lobe_5, lobe_6, lobe_7, lobe_8]:
        lobe = sitk.GetImageFromArray(lobe.astype('int16'))
        lobe_dilated = f_dilate.Execute(lobe)
        lobe_border = f_subtract.Execute(lobe_dilated, lobe)
        lobe_border = sitk.GetArrayFromImage(lobe_border)
        border += lobe_border
    fissure_left = border - leftlungBorder - leftlungBorder
    fissure_right = border - rightlungBorder - rightlungBorder
    fissure = fissure_left + fissure_right
    fissure[fissure <1] = 0
    fissure[fissure >=1] = 1

    return fissure


def writeLung(ctFpath, lungFpath):
    scan, origin, spacing = futil.load_itk(ctFpath, require_ori_sp=True)
    scan[scan>0]=1
    lung = scan
    futil.save_itk(lungFpath, lung, origin, spacing)
    print('save ct mask at', lungFpath)


def writeFissure(ctFpath, fissureFpath, radiusValue=3, labels=None):
    scan, origin, spacing = futil.load_itk(ctFpath, require_ori_sp=True)
    fissure = get_fissure(scan, radiusValue=radiusValue, labels=labels)
    futil.save_itk(fissureFpath, fissure, origin, spacing)
    print('save ct mask at', fissureFpath)

    # f_dilate = sitk.BinaryDilateImageFilter()
    # f_dilate.SetKernelRadius(radiusValue)
    # f_dilate.SetForegroundValue(1)
    # f_subtract = sitk.SubtractImageFilter()
    #
    # scan_list = []
    # for label in [4,5,6,7,8]:  #exclude background, right lung, 3 lobes
    #     # Threshold the value [label, label+1), results in values inside the range 1, 0 otherwise
    #     # itkimageOneLabel = sitk.BinaryThreshold(itkimage, float(label), float(label+1), 1, 0)
    #     scanOneLabel = copy.deepcopy(scan)
    #     scanOneLabel[scanOneLabel != label] = 0  # note the order of the two lines
    #     scanOneLabel[scanOneLabel==label] = 1
    #     itkimageOneLabel = sitk.GetImageFromArray(scanOneLabel.astype('int16'))
    #     dilatedOneLabel = f_dilate.Execute(itkimageOneLabel)
    #     image = f_subtract.Execute(dilatedOneLabel, itkimageOneLabel)
    #     scanOneLabel = sitk.GetArrayFromImage(image)
    #     scan_list.append(scanOneLabel)
    # scanOneLung = np.array(scan_list) # shape (6/2, 144, 144, 600)
    #
    # for i in range(scanOneLung.shape[0]):
    #     scanOneLung[i] -= lungBorder_array
    #     scanOneLung[i][scanOneLung[i] < 1] = 0  # avoid -1 values
    #     scanOneLung[i][scanOneLung[i] >= 1] = 1
    #
    # scanOneLung = np.rollaxis(scanOneLung, 0, 4)  # shape (144, 144, 600, 6/2)
    # scanOne = np.sum(scanOneLung, axis=-1) # shape (144, 144, 600)
    # scanOne [scanOne < 1] = 0 # note the order of the two lines!
    # scanOne [scanOne >= 1] = 1


def gntFissure(Absdir, radiusValue=3, workers=10, number=None, labels=None):
    scan_files = get_all_ct_names(Absdir)
    if number:
        scan_files = scan_files[:number]
    def consumer(mylock):  # neural network inference needs GPU which can not be computed by multi threads, so the
        # consumer is just the upsampling only.
        while True:
            with mylock:
                ct_fpath = None
                if len(scan_files):  # if scan_files are empty, then threads should not wait any more
                    print(threading.current_thread().name + " gets the lock, thread id: " + str(
                        threading.get_ident()) + " prepare to compute fissure , waiting for the data from queue")
                    ct_fpath = scan_files.pop()  # wait up to 1 minutes
                    print(threading.current_thread().name + " gets the data, thread id: " + str(
                        threading.get_ident()) + " prepare to release the lock.")

            if ct_fpath is not None:
                t1 = time.time()
                print(threading.current_thread().name + "is computing fissure") # lola11-49_ct.nii.gz
                fissureFpath = Absdir + '/' + ct_fpath.split('/')[-1].split('_seg')[0] + "_fissure_"+ \
                               str(radiusValue) + '_seg.nii.gz'
                writeFissure(ct_fpath, fissureFpath, radiusValue, labels)
                t3 = time.time()
                print("it costs tis seconds to compute the fissure of the data " + str(t3 - t1))
            else:
                print(threading.current_thread().name + "scan_files are empty, finish the thread")
                return None

    execute_the_function_multi_thread(consumer, workers=workers)


def gnt_lola11_style_fissure(gdth_folder, pred_folder):
    """
    gdth_folder and pred_folder must include fissure segmentation files with the name *fissure_1_seg.nii.gz
    """
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



def gntLung(Absdir, workers=10, number=None):
    scan_files = get_all_ct_names(Absdir)
    if number:
        scan_files = scan_files[:number]
    def consumer(mylock):  # neural network inference needs GPU which can not be computed by multi threads, so the
        # consumer is just the upsampling only.
        while True:
            with mylock:
                ct_fpath = None
                if len(scan_files):  # if scan_files are empty, then threads should not wait any more
                    print(threading.current_thread().name + " gets the lock, thread id: " + str(
                        threading.get_ident()) + " prepare to compute lung , waiting for the data from queue")
                    ct_fpath = scan_files.pop()  # wait up to 1 minutes
                    print(threading.current_thread().name + " gets the data, thread id: " + str(
                        threading.get_ident()) + " prepare to release the lock.")

            if ct_fpath is not None:
                t1 = time.time()
                print(threading.current_thread().name + "is computing lung")
                lungFpath = Absdir + '/lung' + '_' + ct_fpath.split('/')[-1]
                writeLung(ct_fpath, lungFpath)
                t3 = time.time()
                print("it costs tis seconds to compute the lung of the data " + str(t3 - t1))
            else:
                print(threading.current_thread().name + "scan_files are empty, finish the thread")
                return None

    execute_the_function_multi_thread(consumer, workers=workers)



'''
'1599475109_302_lrlb0.0001lrvs1e-05mtscale1netnol-nnlpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599428838_623_lrlb0.0001lrvs1e-05mtscale0netnolpm0.5nldLUNA16ao0ds0tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599479049_663_lrlb0.0001lrvs1e-05mtscale0netnolpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599479049_59_lrlb0.0001lrvs1e-05mtscale1netnolpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599475109_771_lrlb0.0001lrvs1e-05mtscale1netnol-novpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '1599475109_302_lrlb0.0001lrvs1e-05mtscale1netnol-nnlpm0.5nldLUNA16ao1ds2tsp1.4z2.5pps100lbnb17vsnb50nlnb400ptsz144ptzsz96',
                '''
def main():
    pass

if __name__=="__main__":
    main()