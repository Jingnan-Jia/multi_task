"""
Expected Calibration Error for semantic segmentation tasks
"""
import pdb
import nrrd  # pip install pynrrd
import traceback
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import jjnutils.util as cu
import torch
import torch.nn.functional as F
from scipy.interpolate import interpn
from skimage.transform import resize

FILE_DIR = Path(__file__).parent.absolute()
PLOT_DIR = Path(FILE_DIR).joinpath('_tmp')
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

nan_value = -0.1


def get_ece_patient(y_true, y_predict, patient_id, res_global, verbose=False, show=False):
    """
    Params
    ------
    y_true    : [H,W,D,C], np.array, binary
    y_predict : [H,W,D,C], np.array, with probability values
    patient_id: str
    res_global: dict

    Output
    ------
    res_global: dict
    Reference
    ---------
     - On Calibration of Modern Neural Networks
       - Non author implementation: https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.py
    """

    if verbose: print('\n - [get_ece()] patient_id: ', patient_id)

    try:

        # Step 0 - Init
        res = {}
        label_count = y_true.shape[-1]

        # Step 1 - Calculate outputs (in terms of label_id)
        o_true = np.argmax(y_true, axis=-1)
        o_predict = np.argmax(y_predict, axis=-1)

        # Step 2 - Loop over different classes
        for label_id in range(label_count):
            if verbose: print(' --- [get_ece()] label_id: ', label_id)

            # Step 2.1 - Make res_global for that label_id
            if label_id not in res_global:
                res_global[label_id] = {'o_predict_label': [], 'y_predict_label': [], 'o_true_label': []}

                # Step 2.2 - Get o_predict_label(label_ids), o_true_label(label_ids), y_predict_label(probs) [and append to global list]
            o_true_label = o_true[o_predict == label_id]
            o_predict_label = o_predict[o_predict == label_id]
            y_predict_label = y_predict[:, :, :, label_id][o_predict == label_id]
            res_global[label_id]['o_true_label'].extend(o_true_label.flatten().tolist())
            res_global[label_id]['o_predict_label'].extend(o_predict_label.flatten().tolist())
            res_global[label_id]['y_predict_label'].extend(y_predict_label.flatten().tolist())

            if len(o_true_label) and len(y_predict_label):

                # Step 2.3 - Bin the probs and calculate their mean
                y_predict_label_bin_ids = np.digitize(y_predict_label, np.array(
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]), right=False) - 1
                y_predict_binned_vals = [y_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in
                                         range(label_count)]
                y_predict_bins_mean = [np.mean(vals) if len(vals) else nan_value for vals in y_predict_binned_vals]

                # Step 2.4 - Calculate the accuracy of each bin
                o_predict_label_bins = [o_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in
                                        range(label_count)]
                o_true_label_bins = [o_true_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
                y_predict_bins_accuracy = [np.sum(o_predict_label_bins[bin_id] == o_true_label_bins[bin_id]) / len(
                    o_predict_label_bins[bin_id]) if len(o_predict_label_bins[bin_id]) else nan_value for bin_id in
                                           range(label_count)]
                y_predict_bins_len = [len(o_predict_label_bins[bin_id]) for bin_id in range(label_count)]

                # Step 2.5 - Wrapup
                N = np.prod(y_predict_label.shape)
                ce = np.array((np.array(y_predict_bins_len) / N) * (
                            np.array(y_predict_bins_accuracy) - np.array(y_predict_bins_mean)))
                ce[
                    ce == 0] = nan_value  # i.e. y_predict_bins_accuracy[bin_id] == y_predict_bins_mean[bind_id] = nan_value
                res[label_id] = ce

            else:
                res[label_id] = -1

            # Plot patient-wise and labelwise plots
            if show:
                print(' --- [get_ece()] y_predict_bins_accuracy: ',
                      ['%.4f' % (each) for each in np.array(y_predict_bins_accuracy)])
                print(' --- [get_ece()] CE : ', ['%.4f' % (each) for each in np.array(res[label_id])])
                print(' --- [get_ece()] ECE: ', np.sum(np.abs(res[label_id][res[label_id] != nan_value])))

                # GT Probs (sorted) in plt.plot (with equally-sized bins)
                if 0:

                    tmp = np.sort(y_predict_label)
                    tmp_len = len(tmp)
                    plt.plot(range(len(tmp)), tmp, color='orange')
                    for boundary in np.arange(0, tmp_len, int(tmp_len // 10)): plt.plot([boundary, boundary],
                                                                                        [0.0, 1.0], color='black',
                                                                                        alpha=0.5, linestyle='dashed')
                    plt.plot([0, 0], [0, 0], color='black', alpha=0.5, linestyle='dashed', label='Bins(equally-sized)')
                    plt.title('Sorted Softmax Probs (GT) (label={})\nPatient:{}'.format(label_id, patient_id))
                    plt.legend()
                    # plt.show()
                    plt.savefig(
                        str(Path(PLOT_DIR).joinpath('ECE_SortedProbs_label_{}_{}.png'.format(label_id, patient_id))),
                        bbox_inches='tight');
                    plt.close()

                # ECE plot
                if 1:

                    plt.plot(np.arange(11), np.arange(11) / 10.0, linestyle='dashed', color='black', alpha=0.8)
                    plt.scatter(np.arange(len(y_predict_bins_mean)) + 0.5, y_predict_bins_mean, alpha=0.5, color='g',
                                marker='s', label='Mean Pred')
                    plt.scatter(np.arange(len(y_predict_bins_accuracy)) + 0.5, y_predict_bins_accuracy, alpha=0.5,
                                color='b', marker='x', label='Accuracy')
                    diff = np.array(y_predict_bins_accuracy) - np.array(y_predict_bins_mean)
                    for bin_id in range(len(y_predict_bins_accuracy)): plt.plot([bin_id + 0.5, bin_id + 0.5],
                                                                                [y_predict_bins_accuracy[bin_id],
                                                                                 y_predict_bins_mean[bin_id]],
                                                                                color='pink')
                    plt.plot([bin_id + 0.5, bin_id + 0.5],
                             [y_predict_bins_accuracy[bin_id], y_predict_bins_mean[bin_id]], color='pink', label='CE')
                    plt.xticks(ticks=np.arange(11), labels=np.arange(11) / 10.0)
                    plt.title('CE (label={})\nPatient:{}'.format(label_id, patient_id))
                    plt.xlabel('Probability')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    # plt.show()
                    plt.savefig(str(Path(PLOT_DIR).joinpath('ECE_label_{}_{}.png'.format(label_id, patient_id))),
                                bbox_inches='tight');
                    plt.close()

    except:
        traceback.print_exc()
        pdb.set_trace()

    return res_global


def get_ece_global(ece_global):
    try:

        # Step 0 - Init
        ece_labels_obj = {}
        ece_labels = []
        label_count = len(ece_global)
        ece_global_obj_keys = list(ece_global.keys())

        # Step 1 - Loop over all labelids (across all patients)
        for label_id in ece_global_obj_keys:

            o_true_label = np.array(ece_global[label_id]['o_true_label'])
            o_predict_label = np.array(ece_global[label_id]['o_predict_label'])
            y_predict_label = np.array(ece_global[label_id]['y_predict_label'])
            if label_id in ece_global: del ece_global[label_id]

            # Step 1.1 - Bin the probs and calculate their mean
            y_predict_label_bin_ids = np.digitize(y_predict_label,
                                                  np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]),
                                                  right=False) - 1
            y_predict_binned_vals = [y_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in
                                     range(label_count)]
            y_predict_bins_mean = [np.mean(vals) if len(vals) else nan_value for vals in y_predict_binned_vals]

            # Step 1.2 - Calculate the accuracy of each bin
            o_predict_label_bins = [o_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
            o_true_label_bins = [o_true_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
            y_predict_bins_accuracy = [np.sum(o_predict_label_bins[bin_id] == o_true_label_bins[bin_id]) / len(
                o_predict_label_bins[bin_id]) if len(o_predict_label_bins[bin_id]) else nan_value for bin_id in
                                       range(label_count)]
            y_predict_bins_len = [len(o_predict_label_bins[bin_id]) for bin_id in range(label_count)]

            # Step 1.3 - Wrapup
            N = np.prod(y_predict_label.shape)
            ce = np.array((np.array(y_predict_bins_len) / N) * (
                        np.array(y_predict_bins_accuracy) - np.array(y_predict_bins_mean)))
            ce[ce == 0] = nan_value
            ece_label = np.sum(np.abs(ce[ce != nan_value]))
            ece_labels.append(ece_label)
            ece_labels_obj[label_id] = {'y_predict_bins_mean': y_predict_bins_mean,
                                        'y_predict_bins_accuracy': y_predict_bins_accuracy, 'ce': ce, 'ece': ece_label}

        print('\n')
        print(' - ece_labels   : ', ['%.4f' % each for each in ece_labels])
        print(' - ece          : %.4f' % np.mean(ece_labels))
        print(' - ece (w/o bgd): %.4f' % np.mean(ece_labels[1:]))
        print(' - ece (w/o bgd, w/o chiasm): %.4f' % np.mean(ece_labels[1:2] + ece_labels[3:]))

        # Step 2 - Plot ECE
        for label_id in ece_labels_obj:
            y_predict_bins_mean = ece_labels_obj[label_id]['y_predict_bins_mean']
            y_predict_bins_accuracy = ece_labels_obj[label_id]['y_predict_bins_accuracy']
            ece = ece_labels_obj[label_id]['ece']

            plt.plot(np.arange(11), np.arange(11) / 10.0, linestyle='dashed', color='black', alpha=0.8)
            plt.scatter(np.arange(len(y_predict_bins_mean)) + 0.5, y_predict_bins_mean, alpha=0.5, color='g',
                        marker='s', label='Mean Pred')
            plt.scatter(np.arange(len(y_predict_bins_accuracy)) + 0.5, y_predict_bins_accuracy, alpha=0.5, color='b',
                        marker='x', label='Accuracy')
            for bin_id in range(len(y_predict_bins_accuracy)): plt.plot([bin_id + 0.5, bin_id + 0.5],
                                                                        [y_predict_bins_accuracy[bin_id],
                                                                         y_predict_bins_mean[bin_id]], color='pink')
            plt.plot([bin_id + 0.5, bin_id + 0.5], [y_predict_bins_accuracy[bin_id], y_predict_bins_mean[bin_id]],
                     color='pink', label='CE')
            plt.xticks(ticks=np.arange(11), labels=np.arange(11) / 10.0)
            plt.title('CE (label={})\nECE: {}'.format(label_id, '%.5f' % (ece)))
            plt.xlabel('Probability')
            plt.ylabel('Accuracy')
            plt.ylim([-0.15, 1.05])
            plt.legend()

            # plt.show()
            path_results = str(Path(PLOT_DIR).joinpath('results_ece_label{}.png'.format(label_id)))
            plt.savefig(str(path_results), bbox_inches='tight')
            plt.close()

    except:
        traceback.print_exc()
        pdb.set_trace()


if __name__ == "__main__":

    ex_id = "1630962918_874"
    patient_ids = ['26', '27', '29', '30']

    ece_global = {}
    for patient_id in patient_ids:
        seg_fpath = "/data/jjia/multi_task/mt/scripts/results/lobe/" + ex_id + "/infer_pred/valid/pbb_maps" +\
                    "/GLUCOLD_patients_" + patient_id + "_ct.npy"

        gdt_fpath = "/data/jjia/multi_task/mt/scripts/results/lobe/" + ex_id + "/infer_pred/lobe/valid_gdth" +\
                    "/GLUCOLD_patients_" + patient_id + "_seg.nii.gz"

        y_true = cu.load_itk(gdt_fpath)
        y_true_ts = torch.tensor(y_true)
        y_true_ts = F.one_hot(y_true_ts.to(torch.int64), num_classes=6)
        y_true = y_true_ts.numpy()


        y_predict = np.load(seg_fpath)
        y_predict = y_predict[0]
        y_predict = y_predict.transpose()

        y_predict = resize(y_predict, y_true.shape)
        print('start save predict')
        cu.save_itk('predict.mha', y_predict, [1,1,1], [1,1,1], dtype='float')
        print('finish save predict')

        # y_true, _ = nrrd.read(str(Path(PLOT_DIR).joinpath('{}_true.nrrd'.format(patient_id))))
        # y_predict, _ = nrrd.read(str(Path(PLOT_DIR).joinpath('{}_predict.nrrd'.format(patient_id))))
        patient_id = patient_id
        ece_global = get_ece_patient(y_true, y_predict, patient_id, ece_global, verbose=True, show=False)

    get_ece_global(ece_global)