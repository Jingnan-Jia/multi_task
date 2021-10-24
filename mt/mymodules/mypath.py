# -*- coding: utf-8 -*-
# @Time    : 11/21/20 12:10 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import os
import time
import numpy as np
from functools import wraps
from mt.mymodules.set_args_mtnet import get_args
args = get_args()

def mkdir_dcrt(fun):  # decorator to create directory if not exist
    """
    A decorator to make directory output from function if not exist.

    :param fun: a function which outputs a directory
    :return: decorated function
    """

    @wraps(fun)
    def decorated(*args, **kwargs):
        output = fun(*args, **kwargs)
        if '.' in output.split('/')[-1]:
            output = os.path.dirname(output)
            if not os.path.exists(output):
                os.makedirs(output)
                print('successfully create directory:', output)
        else:
            if not os.path.exists(output):
                os.makedirs(output)
                print('successfully create directory:', output)

        return fun(*args, **kwargs)

    return decorated

class PathInit():
    def __init__(self):
        self.record_file = 'records.csv'


class Mypath(PathInit):
    """
    Here, I use 'fpath' to indicatate full file path and name, 'path' to respresent the directory,
    'file_name' to respresent the file name in the parent directory.

    The new path structure looks like:

    - data_ori_space                            -> data_path
        - lobe                                  -> data_task_dir
            - LOLA11                            -> data_task_sub_dir
        - vessel

    - results                                   -> results_dir
        - lobe
            - 12345_234
                - LOLA11
                    - figures                   -> model_figure_path
                    - infer_pred                -> infer_pred_dir
                    - dices.csv                 -> dices_fpath
                    - all_metrics.csv           -> all_metrics_fpath
                    - train.pt                  -> model_fpath_best_patch
                    - train_patch.pt            -> model_fpath_best_whole
                    - set_args.py

        - vessel

        - slurmlogs                             -> log_path
            - 12345_234
    =====================================================================
    Old path look like:

        - data_ori_space                        -> data_path
        - lobe                                  -> data_dir
            - train
                - ori_ct
                    - LOLA11                    -> ori_ct_path
                - gdth_ct
                    - LOLA11                    -> gdth_path
        - vessel

    - results                                   -> results_dir
        - lobe
            - train
                - pred
                    - LOLA11                    -> pred_path
                        - dices.csv             -> dices_fpath
                        - all_metrics.csv       -> all_metrics_fpath

        - vessel
        - figures                               -> model_figure_path
        - models                                -> model_path
            - lobe
                - 12345_234                     -> id_dir
                    - infer_pred                -> infer_pred_dir
                    - 12345_234_args.py
                    - 12345_234_patch_train.pt  -> model_fpath_best_patch
                    - 12345_234_train.pt        -> model_fpath_best_whole
            - vessel
        - slurmlogs                             -> log_path
            - lobe                              -> task_log_dir
                - 12345_234_train.csv           -> train_log_fpath
            - vessel
    """

    def __init__(self, id, task, data_path='data_ori_space',  check_id_dir=False):

        """
        initial valuables.

        :param task: task name, e.g. 'lobe'
        :param current_time: a string represent the current time. It is used to name models and log files. If None, the time will be generated automatically.
        """
        super().__init__()
        # two top level directories
        self.data_dir = 'data'
        self.results_dir = 'results'

        self.task = task
        self.task_dir = os.path.join(self.results_dir, self.task)
        # self.module_dir = os.path.dirname(__file__)
        self.data_path = os.path.join('data', data_path)  # data_xy77_z5 or data_ori_space
        self.log_dir = os.path.join(self.results_dir, 'slurmlogs')

        if isinstance(id, (int, float)):
            self.id = str(int(id))
        else:
            self.id = id  # id should be string

        self.id_dir = os.path.join(self.self.task_dir, str(id))  # results/lobe/12
        if check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for directory in [self.log_dir, self.results_dir, self.task_dir, self.id_dir]:
            if not os.path.isdir(directory):
                os.makedirs(directory)
                print('successfully create directory:', directory)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')
        self.model_wt_structure_fpath = os.path.join(self.id_dir, 'model_wt_structure.pt')


    def data_sub_dir(self):
        """
        Sub directory of tasks. It is used to choose different datasets (like 'GLUCOLD', 'SSc').

        :return: sub directory name
        """

        if self.task == 'lesion':
            sub_dir = args.sub_dir_ls
        elif self.task == 'lobe':
            sub_dir = args.sub_dir_lb
        elif self.task == 'vessel':
            sub_dir = args.sub_dir_vs
        elif self.task == "lung":
            sub_dir = args.sub_dir_lu
        elif self.task == "airway":
            sub_dir = args.sub_dir_aw
        elif self.task == 'recon':
            sub_dir = args.sub_dir_rc
        else:
            raise Exception("task is not valid: ", self.task)
        return sub_dir

    @mkdir_dcrt
    def data_dir(self):
        """data directory.

        Returns: dataset directory for a specific task

        """

        data_dir = os.path.join(self.data_path, self.task)
        return data_dir

    @mkdir_dcrt
    def task_log_dir(self):
        """log directory for the specific task."""
        task_log_dir = os.path.join(self.log_dir, self.task)
        return task_log_dir

    @mkdir_dcrt
    def metrics_fpath(self, phase='train'):
        """log full path to save training measurements during training."""
        return os.path.join(self.id_dir, phase + '_metrics.csv')

    def infer_pred_dir(self):
        infer_pred_dir = os.path.join(self.id_dir, "infer_pred")
        print(f"infer results are saved at {infer_pred_dir}")
        return infer_pred_dir

    @mkdir_dcrt
    def model_figure_path(self):
        """
        Directory where to save figures of model architecture.

        :return: model figure directory
        """
        model_figure_path = os.path.join(self.results_dir, 'figures')
        return model_figure_path

    @mkdir_dcrt
    def model_fpath_best_patch(self, phase, id=None):
        """Full path to save best model according to training loss. """
        if ex_id is None:
            return os.path.join(self.id_dir, self.id + '_patch_' + phase + '.pt')
        else:
            return os.path.join(self.id_dir, id + '_patch_' + phase + '.pt')

    @mkdir_dcrt
    def model_fpath_best_whole(self, phase='train', id=None):
        """Full path to save best model according to training loss. """
        if self.task == "recon":
            if id is None:
                return os.path.join(self.id_dir, self.id + '_patch_' + phase + '.pt')
            else:
                return os.path.join(self.id_dir, id + '_patch_' + phase + '.pt')
        else:
            if id is None:
                return os.path.join(self.id_dir, self.id + '_' + phase + '.pt')
            else:
                return os.path.join(self.id_dir, id + '_' + phase + '.pt')

    @mkdir_dcrt
    def ori_ct_path(self, phase, sub_dir=None):
        """Absolute directory of the original ct for training dataset
        :param sub_dir:
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        if sub_dir is None:
            data_path = os.path.join(self.data_path, self.task, phase, 'ori_ct', self.data_sub_dir())
        else:
            data_path = os.path.join(self.data_path, self.task, phase, 'ori_ct', sub_dir)
        return data_path

    @mkdir_dcrt
    def gdth_path(self, phase, sub_dir=None):
        """
        absolute directory of the ground truth of ct for training dataset
        :param sub_dir:
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        if sub_dir is None:
            gdth_path = os.path.join(self.data_path, self.task, phase, 'gdth_ct', self.data_sub_dir())
        else:
            gdth_path = os.path.join(self.data_path, self.task, phase, 'gdth_ct', sub_dir)
        return gdth_path

    @mkdir_dcrt
    def pred_path(self, phase, sub_dir=None, cntd_pts=False):
        """
        absolute directory of the prediction results of ct for training dataset
        :param cntd_pts:
        :param sub_dir:
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        if sub_dir is None:
            pred_path = os.path.join(self.results_dir, self.task, phase, 'pred', self.data_sub_dir(), self.id)
        else:
            pred_path = os.path.join(self.results_dir, self.task, phase, 'pred', sub_dir, self.id)
        if cntd_pts:
            pred_path += "/cntd_pts"

        return pred_path

    @mkdir_dcrt
    def dices_fpath(self, phase):
        """Full path of the saved dice."""
        pred_path = self.pred_path(phase)
        return os.path.join(pred_path, 'dices.csv')

    @mkdir_dcrt
    def all_metrics_fpath(self, phase, fissure=False, sub_dir=data_sub_dir):
        """Full path of the saved dice.

        Args:
            phase: 'train' or 'valid'
            fissure:
            sub_dir:

        Returns:

        """
        pred_path = self.pred_path(phase, sub_dir=sub_dir)
        if fissure:
            return os.path.join(pred_path, 'all_metrics_fissure.csv')
        else:
            return os.path.join(pred_path, 'all_metrics.csv')
