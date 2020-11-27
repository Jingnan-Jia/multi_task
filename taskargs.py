# -*- coding: utf-8 -*-
# @Time    : 11/20/20 11:16 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from abc import ABC
from abc import abstractmethod
import torch


class CommonTask(ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = True
        
    @abstractmethod
    def run_one_step(self):
        raise NotImplementedError

    @abstractmethod
    def run_all_epochs(self):
        raise NotImplementedError

    @abstractmethod
    def do_vilidation_if_need(self):
        raise NotImplementedError