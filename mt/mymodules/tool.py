# -*- coding: utf-8 -*-
# @Time    : 7/5/21 5:23 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import argparse
import datetime
import os
import shutil
import time
from typing import Union, Tuple

import myutil.myutil as futil
import numpy as np
import nvidia_smi
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import WeightedRandomSampler

from mt.mymodules.mypath import Mypath as Path
from mt.mymodules.mypath import PathInit

class Tracker():
    def __init__(self, task_name, data_path='data_ori_space', ld_name=None):
        self.task_name = task_name
        self.data_path = data_path
        self.log_dict = {"task_name": task_name}
        self.current_id = None
        self.args = None
        self.record_file = PathInit(self.task_name).record_file
        self.ld_name = ld_name
    def record(self, key, values):
        self.log_dict[key] = values

    def record_1st(self, args: argparse.Namespace) -> int:
        """First record in this experiment.

        Args:
            task: 'score' or 'pos' for score and position prediction respectively.
            args: arguments.

        Returns:
            new_id

        Examples:
            :func:`ssc_scoring.run` and :func:`ssc_scoring.run_pos`

        """
        self.args = args
        lock = FileLock(self.record_file + ".lock")  # lock the file, avoid other processes write other things
        with lock:  # with this lock,  open a file for exclusive access
            with open(self.record_file, 'a'):
                df, new_id = get_df_id(self.record_file)
                self.current_id = new_id
                if args.mode == 'train':
                    mypath = Path(new_id,task=self.task_name, data_path='data_ori_space',  check_id_dir=True)  # to check if id_dir already exist
                else:
                    mypath = Path(new_id,task=self.task_name, data_path='data_ori_space',  check_id_dir=False)

                start_date = datetime.date.today().strftime("%Y-%m-%d")
                start_time = datetime.datetime.now().time().strftime("%H:%M:%S")
                # start record by id, date,time row = [new_id, date, time, ]
                idatime = {'ID': new_id, 'start_date': start_date, 'start_time': start_time}
                args_dict = vars(args)
                idatime.update(args_dict)  # followed by super parameters

                if len(df) == 0:  # empty file
                    df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
                else:
                    index = df.index.to_list()[-1]  # last index
                    for key, value in idatime.items():  # write new line
                        try:
                            df.at[index + 1, key] = value  #
                        except:
                            df[key] = df[key].astype('object')
                            df.at[index + 1, key] = value  #

                df = fill_running(df)  # fill the state information for other experiments
                df = correct_type(df)  # aviod annoying thing like: ID=1.00
                write_and_backup(df, self.record_file, mypath)
        return new_id

    def record_2nd(self) -> None:
        """Second time to save logs.

        Args:
        Returns:
            None. log_dict saved to disk.

        Examples:
            :func:`ssc_scoring.run` and :func:`ssc_scoring.run_pos`

        """
        lock = FileLock(self.record_file + ".lock")
        with lock:  # with this lock,  open a file for exclusive access
            print("record path", os.path.realpath(self.record_file))
            df = pd.read_csv(os.path.realpath(self.record_file), delimiter=',')
            print('df', df)
            index = df.index[df['ID'] == self.current_id].to_list()
            if len(index) > 1:
                raise Exception("over 1 row has the same id", id)
            elif len(index) == 0:  # only one line,
                index = 0
            else:
                index = index[0]

            end_date = datetime.date.today().strftime("%Y-%m-%d")
            end_time = datetime.datetime.now().time().strftime("%H:%M:%S")
            df.at[index, 'end_date'] = end_date
            df.at[index, 'end_time'] = end_time

            # usage
            f = "%Y-%m-%d %H:%M:%S"
            t1 = datetime.datetime.strptime(df['start_date'][index] + ' ' + df['start_time'][index], f)
            t2 = datetime.datetime.strptime(df['end_date'][index] + ' ' + df['end_time'][index], f)
            elapsed_time = time_diff(t1, t2)
            df.at[index, 'elapsed_time'] = elapsed_time

            mypath = Path(self.current_id, task=self.task_name, data_path=self.data_path, check_id_dir=False)  # evaluate old model
            mypath2 = Path(self.ld_name, task=self.task_name, data_path=self.data_path, check_id_dir=False)  # evaluate old model

            df = add_best_metrics(df, mypath, mypath2, index)

            for key, value in self.log_dict.items():  # convert numpy to str before writing all self.log_dict to csv file
                if type(value) in [np.ndarray, list]:
                    str_v = ''
                    for v in value:
                        str_v += str(v)
                        str_v += '_'
                    value = str_v
                df.loc[index, key] = value
                if type(value) is int:
                    df[key] = df[key].astype('Int64')

            for column in df:
                if type(df[column].to_list()[-1]) is int:
                    df[column] = df[column].astype('Int64')  # correct type again, avoid None/1.00/NAN, etc.

            args_dict = vars(self.args)
            args_dict.update({'ID': self.current_id})
            for column in df:
                if column in args_dict.keys() and type(args_dict[column]) is int:
                    df[column] = df[column].astype(float).astype('Int64')  # correct str to float and then int
            write_and_backup(df, self.record_file, mypath)


def get_loss_min(fpath: str) -> float:
    """Get minimum loss from fpath.

    Args:
        fpath: A csv file in which the loss at each epoch is recorded

    Returns:
        Minimum loss value

    Examples:
        :func:`mt.mymodules.tool.get_loss_min('1635031365_299train.csv')`

    """
    loss = pd.read_csv(fpath)
    mae = min(loss['ave_tr_loss'].to_list())
    return mae


def eval_net_mae(mypath: Path, mypath2: Path) -> float:
    """Copy trained model and loss log to new directory and get its valid_mae_best.

    Args:
        mypath: Current experiment Path instance
        mypath2: Trained experiment Path instance, if mypath is empty, copy files from mypath2 to mypath

    Returns:
        valid_mae_minimum

    Examples:
        :func:`ssc_scoring.run.train` and :func:`ssc_scoring.run_pos.train`

    """
    shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
    for mo in ['train', 'valid', 'test']:
        try:
            shutil.copy(mypath2.metrics_fpath(mo), mypath.metrics_fpath(mo))  # make sure there is at least one model
        except FileNotFoundError:
            print(f'Cannot find the metrics of this mode: {mo}, pass it')
            pass
    valid_mae_best = get_loss_min(mypath2.metrics_fpath('valid'))
    print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
    return valid_mae_best


def add_best_metrics(df: pd.DataFrame,
                     mypath: Path,
                     mypath2: Path,
                     index: int) -> pd.DataFrame:
    """Add best metrics: loss, mae (and mae_end5 if possible) to `df` in-place.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)
        mypath: Current Path instance
        mypath2: Old Path instance, if the loss file can not be find in `mypath`, copy it from `mypath2`
        index: Which row the metrics should be writen in `df`

    Returns:
        `df`

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    modes = ['train', 'valid', 'test']

    metrics_max = 'dice_ex_bg'
    df.at[index, 'metrics_max'] = metrics_max

    for mode in modes:
        lock2 = FileLock(mypath.metrics_fpath(mode) + ".lock")
        # when evaluating/inference old models, those files would be copied to new the folder
        with lock2:
            try:
                loss_df = pd.read_csv(mypath.metrics_fpath(mode))
            except FileNotFoundError:  # copy loss files from old directory to here
                shutil.copy(mypath2.metrics_fpath(mode), mypath.metrics_fpath(mode))
                try:
                    loss_df = pd.read_csv(mypath.metrics_fpath(mode))
                except FileNotFoundError:  # still cannot find the loss file in old directory, pass this mode
                    print(f'Cannot find the metrics of this mode: {mode}, pass it')
                    continue

            best_index = loss_df[metrics_max].idxmax()
            if mode == 'train':
                metrics_ls = ['loss', 'dice_ex_bg', 'dice_inc_bg']
            elif mode == 'valid':
                metrics_ls = ['loss', 'dice_ex_bg', 'dice_inc_bg']

            for metric in metrics_ls:
                df.at[index, mode + '_' + metric] = round( loss_df[metric][best_index], 3)

    return df


def write_and_backup(df: pd.DataFrame, record_file: str, mypath: Path) -> None:
    """Write `df` to `record_file` and backup it to `mypath`.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)
        record_file: A file in hard disk saving df
        mypath: Path instance

    Returns:
        None. Results are saved to disk.

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st` and :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    df.to_csv(record_file, index=False)
    shutil.copy(record_file, os.path.join(os.path.dirname(record_file),
                                          'cp_' + os.path.basename(record_file)))
    df_lastrow = df.iloc[[-1]]
    df_lastrow.to_csv(os.path.join(mypath.id_dir, os.path.basename(record_file)),
                      index=False)  # save the record of the current ex


def fill_running(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the old record of completed experiments if the state of them are still 'running'.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)

    Returns:
        df itself

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st`

    """
    for index, row in df.iterrows():
        if 'State' not in list(row.index) or row['State'] in [None, np.nan, 'RUNNING']:
            try:
                jobid = row['outfile'].split('-')[-1].split('_')[0]  # extract job id from outfile name
                seff = os.popen('seff ' + jobid)  # get job information
                for line in seff.readlines():
                    line = line.split(
                        ': ')  # must have space to be differentiated from time format 00:12:34
                    if len(line) == 2:
                        key, value = line
                        key = '_'.join(key.split(' '))  # change 'CPU utilized' to 'CPU_utilized'
                        value = value.split('\n')[0]
                        df.at[index, key] = value
            except:
                pass
    return df


def correct_type(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the type of values in `df`. to avoid the ID or other int valuables become float number.

        Args:
            df: A DataFrame saving metrics (and other super-parameters)

        Returns:
            df itself

        Examples:
            :func:`ssc_scoring.mymodules.tool.record_1st`

        """
    for column in df:
        ori_type = type(df[column].to_list()[-1])  # find the type of the last valuable in this column
        if ori_type is int:
            df[column] = df[column].astype('Int64')  # correct type
    return df


def get_df_id(record_file: str) -> Tuple[pd.DataFrame, int]:
    """Get the current experiment ID. It equals to the latest experiment ID + 1.

    Args:
        record_file: A file to record experiments details (super-parameters and metrics).

    Returns:
        dataframe and new_id

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st`

    """
    if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
        new_id = 1
        df = pd.DataFrame()
    else:
        df = pd.read_csv(record_file)  # read the record file,
        last_id = df['ID'].to_list()[-1]  # find the last ID
        new_id = int(last_id) + 1
    return df, new_id




def time_diff(t1: datetime, t2: datetime) -> str:
    """Time difference.

    Args:
        t1: time 1
        t2: time 2

    Returns:
        Elapsed time

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    # t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    # t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2 - t1

    return str(t_elapsed).split('.')[0]  # drop out microseconds


def _bytes_to_megabytes(value_bytes: int) -> float:
    """Convert bytes to megabytes.

    Args:
        value_bytes: bytes number

    Returns:
        megabytes

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_gpu_info`

    """
    return round((value_bytes / 1024) / 1024, 2)


def record_mem_info() -> int:
    """

    Returns:
        Memory usage in kB

    .. warning::

        This function is not tested. Please double check its code before using it.

    """

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    print('int(memusage.strip())')

    return int(memusage.strip())


def record_gpu_info(outfile) -> Tuple:
    """Record GPU information to `outfile`.

    Args:
        outfile: The format of `outfile` is: slurm-[JOB_ID].out

    Returns:
        gpu_name, gpu_usage, gpu_util

    Examples:

        >>> record_gpu_info('slurm-98234.out')

        or

        :func:`ssc_scoring.run.gpu_info` and :func:`ssc_scoring.run_pos.gpu_info`

    """

    if outfile:
        jobid_gpuid = outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        # self.log_dict['gpuname'] = gpuname
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_usage = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total)) + ' MB'
        # self.log_dict['gpu_mem_usage'] = gpu_mem_usage
        gpu_util = 0
        for i in range(5):
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_util += res.gpu
            time.sleep(1)
        gpu_util = gpu_util / 5
        # self.log_dict['gpu_util'] = str(gpu_util) + '%'
        return gpuname, gpu_mem_usage, str(gpu_util) + '%'
    else:
        print('outfile is None, can not show GPU memory info')
        return None, None, None
