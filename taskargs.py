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
    def run_one_step(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_all_epochs(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def do_validation_if_need(self, *args, **kwargs):
        raise NotImplementedError