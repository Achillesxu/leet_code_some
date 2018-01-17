#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/12/15 下午5:44
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : tornado_websocket_client.py
@desc :
"""
import time
from functools import partial
from tornado import ioloop
from tornado import websocket
from tornado import gen


@gen.coroutine
def connect_server(url):
    conn = yield websocket.websocket_connect(url)
    over_cnt = 5
    while True:
        message_1 = 'Easy doesnt enter into grown-up life'
        conn.write_message(message_1, binary=False)
        msg = yield conn.read_message()
        if msg is None:
            break
        else:
            print(msg)
            over_cnt -= 1
        if over_cnt <= 0:
            break
    conn.close(14991, 'cnt coming!!!')


if __name__ == '__main__':
    in_url = 'ws://localhost:14991/con_1'
    ioloop.IOLoop.current().run_sync(partial(connect_server, in_url))
