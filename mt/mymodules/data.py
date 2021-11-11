# -*- coding: utf-8 -*-
# @Time    : 9/2/21 10:35 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import threading
from queue import Queue


class FluentDataloader():
    def __init__(self, dataloader, dataloader2, keys):
        self.dataloader = dataloader
        self.dataloader2 = dataloader2
        self.keys = keys
        self.data_q = Queue(maxsize=5)
        self.data_q2 = Queue(maxsize=5)


    def _start_dataloader(self, dl, q):
        # dl = iter(dl)
        while True:
            # print("start true")
            # # for data in dl:
            # try:
            #     data = next(dl)
            # except StopIteration:
            #     dl = iter(dl)
            #     data = next(dl)
            for data in dl:
                x_pps = data[self.keys[0]]
                y_pps = data[self.keys[1]]
                for x, y in zip(x_pps, y_pps):
                    # print("start put")
                    x = x[None, ...]
                    y = y[None, ...]
                    data_single = (x, y)
                    if q == 1:
                        self.data_q.put(data_single)
                        # print(f"data_q's valuable data: {self.data_q.qsize()}")
                    else:
                        self.data_q2.put(data_single)
                        # print(f"data_q2's valuable data: {self.data_q2.qsize()}")
                    # print("self.data_q in subprocess", self.data_q)
                    # print("self.data_q2 in subprocess", self.data_q2)
                    # time.sleep(60)

    def run(self):
        p1 = threading.Thread(target=self._start_dataloader, args=(self.dataloader, 1,))
        p2 = threading.Thread(target=self._start_dataloader, args=(self.dataloader, 2,))


        # p1 = multiprocessing.Process(target=self._start_dataloader, args=(self._dataloader, 1,))
        # p2 = multiprocessing.Process(target=self._start_dataloader, args=(self.dataloader2, 2,))
        p1.start()
        p2.start()

        use_q2 = False
        print("self.data_q", self.data_q)
        print("self.data_q2", self.data_q2)
        count = 0
        while True:
            # time.sleep(10)
            if len(self.keys) == 2:
                # if count<15:
                #     print(f"data_q.size: {self.data_q.qsize()}")
                #     print(f"data_q2.size: {self.data_q2.qsize()}")
                #     count+=1
                if self.data_q.qsize() > 0 or self.data_q2.qsize() > 0:
                    # print('self.data_q.size', self.data_q.qsize())
                    if self.data_q.empty() or use_q2:
                        q = self.data_q2
                        use_q2 = True
                    else:
                        q = self.data_q
                        use_q2 = False
                else:
                    continue
                # if data_q.empty() and data_q2.empty():
                #     # print('empty')
                #     continue
                # else:

                data = q.get(timeout=1000)
                # print('get data successfully')
                yield data


def singlethread_ds(dl):
    keys = ("image", "label")
    while True:
        for data in dl:
            x_pps = data[keys[0]]
            y_pps = data[keys[1]]
            for x, y in zip(x_pps, y_pps):
                # print("start put by single thread, not fluent thread")
                x = x[None, ...]
                y = y[None, ...]
                data_single = (x, y)
                yield data_single


def inifite_generator(dataloader, dataloader2=None, keys = ("image", "label")):
    if dataloader2 != None:
        fluent_dl = FluentDataloader(dataloader, dataloader2, keys)
        return fluent_dl.run()
    else:
        return singlethread_ds(dataloader)