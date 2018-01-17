#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/12/13 下午5:03
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : multiprocess_ex.py
@desc :
"""
import os
from multiprocessing import Pool, Process


class DataItem(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


def pro_func(x):
    return x.x ** x.y


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(name):
    info('function f')
    print('hello', name)


def main_process():
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()


def main_pool():
    with Pool(5) as p:
        print(p.map(pro_func, (DataItem(1, 2), DataItem(2, 2), DataItem(3, 2))))


if __name__ == '__main__':
    # main_pool()
    main_process()