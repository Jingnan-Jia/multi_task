# -*- coding: utf-8 -*-
# @Time    : 11/20/20 11:58 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import os
import torch
from typing import Dict, List
import csv
import sys

sys.path.append("../..")

from mt.mymodules.task import TaskArgs
from mt.mymodules.set_args_mtnet import get_args
from mt.mymodules.task_supply import mt_netnames, mt_netname_ta, mt_tr_ta_list

# from find_connect_parts import write_connected_lobes
# from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler

def train_mtnet(args):
    net_names: List[str] = mt_netnames(args)
    net_ta_dict: Dict[str, TaskArgs] = mt_netname_ta(net_names, args)

    if args.mode == "train":
        if args.train_mode == "stepbystep":  # alternative training
            for idx_ in range(args.step_nb):
                print('step number: ', idx_)
                tr_tas: List[TaskArgs] = mt_tr_ta_list(net_ta_dict, idx_, args.main_net_name, args.fat)
                for ta in tr_tas:
                    ta.run_one_step(net_ta_dict, idx_)
                    ta.do_validation_if_need(net_ta_dict, idx_)

                    if args.save_w:
                        net_w = ta.main_task.net.enc.down_3.convs.conv_1.conv.weight
                        net_grad = net_w.grad
                        try:
                            norm_grad_csv = ta.mypath.task_model_dir() + '/' + ta.mypath.str_name + "grad.csv"
                            if not os.path.isfile(norm_grad_csv):
                                with open(norm_grad_csv, "a") as f:
                                    writer = csv.writer(f, delimiter=',')
                                    l = ["net_name", "net_w", "net_grad", "net_w_norm", "net_grad_norm"]
                                    writer.writerow(l)
                            with open(norm_grad_csv, "a") as f:
                                writer = csv.writer(f, delimiter=',')
                                l = [ta.net_name, net_w[0][0][0][0], net_grad[0][0][0][0], torch.norm(net_w).item(), torch.norm(net_grad).item()]
                                writer.writerow(l)
                        except:
                            pass

        else:
            tr_tas: List[TaskArgs] = mt_tr_ta_list(net_ta_dict, 0)
            for ta in tr_tas:
                ta.run_all_epochs()
    else:
        for net_name, ta in net_ta_dict.items():
            ta.infer()


if __name__ == '__main__':
    args = get_args()
    train_mtnet(args)
