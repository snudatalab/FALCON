"""
FALCON: FAst and Lightweight CONvolution

Authors:
 - Chun Quan (quanchun@snu.ac.kr)
 - U Kang (ukang@snu.ac.kr)
 - Data Mining Lab. at Seoul National University.

File: utils/timer.py
 - Contain source code for a timer.
 - Code is got from https://dev.tencent.com/u/zzpu/p/yolov2/git/raw/4b2c6c7df1876363aba3bbd600aa68e4deeb4487/utils/timer.py

Version: 1.0

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

"""

import time


class Timer(object):
    """A simple timer."""
    def __init__(self):
        """
        Initialize timer.
        """
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        """
        Start timer.
        """
        self.start_time = time.time()

    def toc(self, average=True):
        """
        Stop timer.
        :param average: whether to calculate the average time
        """
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff