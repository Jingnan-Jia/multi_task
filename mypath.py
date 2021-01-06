# -*- coding: utf-8 -*-
# @Time    : 11/21/20 12:10 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from set_args_mtnet import args

import os
import time
import numpy as np
from functools import wraps


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


class Mypath(object):
    """
    Here, I use 'fpath' to indicatate full file path and name, 'path' to respresent the directory,
    'file_name' to respresent the file name in the parent directory.
    """

    def __init__(self, task, current_time=None):

        """
        initial valuables.

        :param task: task name, e.g. 'lobe'
        :param current_time: a string represent the current time. It is used to name models and log files. If None, the time will be generated automatically.
        """

        self.task = task
        self.dir_path = os.path.dirname(os.path.realpath(__file__))  # abosolute path of the current .py file
        self.model_path = os.path.join(self.dir_path, 'models')
        self.log_path = os.path.join(self.dir_path, 'logs')
        self.data_path = os.path.join(self.dir_path, 'data_xy77_z5')
        self.results_path = os.path.join(self.dir_path, 'results')

        if current_time:
            self.str_name = current_time
        else:
            self.str_name = str(int(time.time())) + '_' + str(np.random.randint(1000))

    def sub_dir(self):
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
        """
        data directory.
        :return: data dataset directory for a specific task
        """
        data_dir = self.data_path + '/' + self.task
        return data_dir

    @mkdir_dcrt
    def task_log_dir(self):
        """
        log directory.
        :return: directory to save logs
        """
        task_log_dir = self.log_path + '/' + self.task
        return task_log_dir


    @mkdir_dcrt
    def train_log_fpath(self):
        """
        log full path to save training  measuremets during training.

        :return: log full path with suffix .csv
        """
        task_log_dir = self.task_model_dir()
        return task_log_dir + '/' + self.str_name + 'train.csv'
    
    
    @mkdir_dcrt
    def task_model_dir(self, current_time=None):
        """
        model directory.
        :return: directory to save models
        """
        if current_time:
            task_model_dir = self.model_path + '/' + self.task + "/" + current_time
        else:
            task_model_dir = self.model_path + '/' + self.task + "/" + self.str_name
        return task_model_dir

    def infer_pred_dir(self, current_time=None):
        task_model_dir = self.task_model_dir(current_time)
        print(f"infer results are saved at {task_model_dir+'/infer_pred'}")
        return task_model_dir+"/infer_pred"
        


    @mkdir_dcrt
    def model_figure_path(self):
        """
        Directory where to save figures of model architecture.

        :return: model figure directory
        """
        model_figure_path = self.dir_path + '/figures'
        return model_figure_path


    @mkdir_dcrt
    def args_fpath(self):
        """
                full path of model arguments.

                :return: model arguments full path
                """
        task_model_path = self.task_model_dir()
        return task_model_path + '/' + self.str_name + '_args.py'


    @mkdir_dcrt
    def model_fpath_best_patch(self, phase, str_name=None):
        """
        full path to save best model according to training loss.
        :return: full path
        """
        task_model_path = self.task_model_dir()
        if str_name is None:
            return task_model_path + '/' + self.str_name + '_patch_' + phase + '.pt'
        else:
            return task_model_path + '/' + str_name + '_patch_' + phase + '.pt'

    @mkdir_dcrt
    def model_fpath_best_whole(self, phase='train', str_name=None):
        """
        full path to save best model according to training loss.
        :return: full path
        """
        task_model_path = self.task_model_dir()

        if self.task == "recon":
            if str_name is None:
                return task_model_path + '/' + self.str_name + '_patch_' + phase + '.pt'
            else:
                return task_model_path + '/' + str_name + '_patch_' + phase + '.pt'
        else:
            if str_name is None:
                return task_model_path + '/' + self.str_name + '_' + phase + '.pt'
            else:
                return task_model_path + '/' + str_name + '_' + phase + '.pt'


    @mkdir_dcrt
    def ori_ct_path(self, phase, sub_dir=None):
        """
        absolute directory of the original ct for training dataset
        :param sub_dir:
        :param phase: 'train' or 'valid'
        :return: directory name
        """
        if sub_dir is None:
            data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct/' + self.sub_dir()
        else:
            data_path = self.data_path + '/' + self.task + '/' + phase + '/ori_ct/' + sub_dir
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
            gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct/' + self.sub_dir()
        else:
            gdth_path = self.data_path + '/' + self.task + '/' + phase + '/gdth_ct/' + sub_dir
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
            pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/' + self.sub_dir() + '/' + self.str_name
        else:
            pred_path = self.results_path + '/' + self.task + '/' + phase + '/pred/' + sub_dir + '/' + self.str_name
        if cntd_pts:
            pred_path += "/cntd_pts"

        return pred_path

    @mkdir_dcrt
    def dices_fpath(self, phase):
        """
        full path of the saved dice
        :param phase: 'train' or 'valid'
        :return: file name to save dice
        """
        pred_path = self.pred_path(phase)
        return pred_path + '/dices.csv'

    @mkdir_dcrt
    def all_metrics_fpath(self, phase, fissure=False, sub_dir=sub_dir):
        """
        full path of the saved dice
        :param sub_dir:
        :param fissure:
        :param phase: 'train' or 'valid'
        :return: file name to save dice
        """
        pred_path = self.pred_path(phase, sub_dir=sub_dir)
        if fissure:
            return pred_path + '/all_metrics_fissure.csv'
        else:
            return pred_path + '/all_metrics.csv'
