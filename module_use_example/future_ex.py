#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/12/14 上午9:56
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : future_ex.py
@desc :
"""
import os
import math
import urllib.request
import concurrent
from concurrent.futures import ThreadPoolExecutor

URLS = ['http://www.foxnews.com/',
        'http://www.baidu.com/',
        'http://down.7po.com/',
        'http://www.bbc.co.uk/']

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]


def thread_exe():
    # Retrieve a single page and report the URL and contents
    def load_url(url, timeout):
        with urllib.request.urlopen(url, timeout=timeout) as conn:
            return conn.read()

    # We can use a with statement to ensure threads are cleaned up promptly
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
            else:
                print('%r page is %d bytes' % (url, len(data)))


def is_prime(n):
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


def process_exe():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f_list = [executor.submit(is_prime, i) for i in PRIMES]
        for i in f_list:
            print(i.running())
        print('all result is following:')
        for i in concurrent.futures.as_completed(f_list):
            print(i.result())


if __name__ == '__main__':
    # thread_exe()
    process_exe()
