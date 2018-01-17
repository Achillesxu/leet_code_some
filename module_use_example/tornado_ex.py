#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/12/13 下午3:41
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : tornado_ex.py
@desc :
"""
import os
import math
from functools import partial
from tornado import gen, ioloop
from tornado import httpclient
from tornado import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


@gen.coroutine
def get_url(in_url=None):
    httpclient.AsyncHTTPClient.configure('tornado.curl_httpclient.CurlAsyncHTTPClient')
    http_client = httpclient.AsyncHTTPClient()
    try:
        ret = yield http_client.fetch(in_url if in_url else 'http://www.google.com/')
        print(ret.code)
        print(ret.body)
    except httpclient.HTTPError as e:
        print(e)
    except Exception as e:
        print(e)
    http_client.close()


class RunExecutor(object):
    def __init__(self):
        self.io_loop = ioloop.IOLoop.current()
        self.executor = ThreadPoolExecutor(os.cpu_count())

    @concurrent.run_on_executor
    def run_http_client(self, in_url):
        h_c = httpclient.HTTPClient()
        try:
            ret = h_c.fetch(in_url)
        except httpclient.HTTPError as e:
            print('http error:', e)
            return None
        except Exception as e:
            print('exception:', e)
            return None
        else:
            return ret.code, ret.body

    def destroy_executor(self):
        try:
            self.executor.shutdown(True)
        except Exception as e:
            print('shutdown error', e)


@gen.coroutine
def get_url_run():
    run_ex = RunExecutor()
    fu_run = run_ex.run_http_client('http://www.google.com/')
    try:
        ret = yield fu_run
    except Exception as e:
        print('get_url_run', fu_run.exception())
    else:
        if ret is not None:
            print(ret)
        else:
            print('error')
    finally:
        run_ex.destroy_executor()


def is_prime(n, callback):
    if n % 2 == 0:
        return callback(False)

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return callback(False)
    return callback(True)


def is_prime_i(n):
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


@gen.coroutine
def gen_task(num):
    is_true = yield gen.Task(is_prime, num)
    print(is_true)


PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]


@gen.coroutine
def mul_process():
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())
    f_list = [executor.submit(is_prime_i, i) for i in PRIMES]
    res_list = yield gen.Multi(f_list)
    print(res_list)
    executor.shutdown()


if __name__ == '__main__':
    try:
        ioloop.IOLoop.current().run_sync(mul_process)
    except ioloop.TimeoutError as e:
        print(e)

