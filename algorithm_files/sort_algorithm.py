#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/10/31 下午3:54
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : sort_algorithm.py
@desc :
"""

import time
import random


def bubble_sort(in_lists):
    """
    冒泡排序
    """
    count = len(in_lists)
    for i in range(count):
        for j in range(i+1, count):
            if in_lists[j] < in_lists[i]:
                in_lists[i], in_lists[j] = in_lists[j], in_lists[i]
    return in_lists


def select_sort(in_lists):
    """
    选择排序
    """
    count = len(in_lists)
    for i in range(count):
        min_i = i
        for j in range(i+1, count):
            if in_lists[j] < in_lists[min_i]:
                min_i = j
        in_lists[min_i], in_lists[i] = in_lists[i], in_lists[min_i]
    return in_lists


def insert_sort(in_lists):
    """
    插入排序
    """
    for i in range(1, len(in_lists)):
        key = in_lists[i]
        j = i - 1
        while j >= 0:
            if in_lists[j] > key:
                in_lists[j+1], in_lists[j] = in_lists[j], key
            j -= 1
    return in_lists


def shell_sort(in_lists):
    """
    希尔排序
    """
    count = len(in_lists)
    step = 2
    group = count // step
    while group > 0:
        for i in range(group):
            j = i + group
            while j < count:
                k = j - group
                key = in_lists[j]
                while k >= 0:
                    if in_lists[k] > key:
                        in_lists[k + group], in_lists[k] = in_lists[k], key
                    k -= group
                j += group
        group //= step
    return in_lists


def quick_sort(in_lists, l, r):
    if l < r:
        q = partition(in_lists, l, r)
        quick_sort(in_lists, l, q - 1)
        quick_sort(in_lists, q + 1, r)
    return in_lists


def partition(array, l, r):
    x = array[r]
    i = l - 1
    for j in range(l, r):
        if array[j] <= x:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[r] = array[r], array[i + 1]
    return i + 1


def quick_sort_python(in_lists):
    if len(in_lists) < 2:
        return in_lists
    else:
        par_pos = len(in_lists)//2
        pk = in_lists[par_pos]
        less = [i for i in in_lists[:par_pos] + in_lists[par_pos+1:] if i <= pk]
        greater = [i for i in in_lists[:par_pos] + in_lists[par_pos+1:] if i > pk]
        return quick_sort_python(less) + [pk] + quick_sort_python(greater)


def sum_recursion(in_lists):
    if len(in_lists) == 1:
        return in_lists[0]
    else:
        return in_lists[0] + sum_recursion(in_lists[1:])


def merge_sort(l):
    """
    归并排序
    """
    if len(l) > 1:
        t = len(l)//2
        it1 = iter(merge_sort(l[:t]))
        x1 = next(it1)
        it2 = iter(merge_sort(l[t:]))
        x2 = next(it2)
        l = []
        try:
            while True:
                if x1 <= x2:
                    l.append(x1)
                    x1 = next(it1)
                else:
                    l.append(x2)
                    x2 = next(it2)
        except StopIteration:
            if x1 <= x2:
                l.append(x2)
                l.extend(it2)
            else:
                l.append(x1)
                l.extend(it1)
    return l


if __name__ == '__main__':
    test_int_list = [random.randint(0, 200) for i in range(10)]
    start_t = time.clock()
    print(quick_sort(test_int_list, 0, len(test_int_list)-1))
    # print(quick_sort_python(test_int_list))
    # print(merge_sort(test_int_list))
    end_t = time.clock()
    print('<{}> using time <{}>'.format(quick_sort_python.__name__, end_t - start_t))


