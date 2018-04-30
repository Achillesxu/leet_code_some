#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : leet_code_some
@Time : 2018/3/12 下午4:39
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : tornado_graceful_stop.py
@desc :
"""
import asyncio
import logging
import signal

from tornado import httpserver
from tornado import ioloop
from tornado import options
from tornado import web


class Application(web.Application):
    pass


def main_entrance():
    options.parse_command_line()
    http_server = httpserver.HTTPServer(Application, xheaders=True)
    http_server.listen(options.port if options.port else 12345)

    MAX_WAIT_SECONDS_BEFORE_STOP = 10

    def sig_handler(sig, frame):
        logging.warning('Caught Signal: {!r}'.format(sig))
        ioloop.IOLoop.current().add_callback(shutdown())

    def shutdown():
        logging.info('Stopping http server at port {}'.format(1))
        http_server.stop()
        logging.info('IOLoop will be stopped in {} seconds'.format(MAX_WAIT_SECONDS_BEFORE_STOP))

        deadline = ioloop.IOLoop.current().time() + MAX_WAIT_SECONDS_BEFORE_STOP

        cur_io_loop = ioloop.IOLoop.current()

        def stop_loop():
            now = ioloop.IOLoop.current().time()
            if now < deadline:
                # only wait for some time
                cur_io_loop.add_timeout(now + 1, stop_loop)
            else:
                cur_io_loop.stop()
                logging.info('Shutdown')

        stop_loop()

    signal.signal(signal.SIGQUIT, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    ioloop.IOLoop.current().start()
    logging.info('Exiting the server!')


if __name__ == '__main__':
    # main_entrance()
    io_cur = ioloop.IOLoop.current()
    if isinstance(io_cur, asyncio.AbstractEventLoop):
        print('yes')
    else:
        print(io_cur)
        print('no')
