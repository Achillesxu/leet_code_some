#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/12/15 下午5:43
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : tornado_websocket_server.py
@desc :
"""
from tornado import ioloop
from tornado import httpserver
from tornado.web import Application
from tornado import websocket


class WebSocHandler(websocket.WebSocketHandler):
    def open(self):
        print(self.request.uri, self.request.remote_ip)

    def on_message(self, message):
        print('we get message: ', message)
        self.write_message('server get meg: ' + message)
        if 'close right now' in message:
            self.close(code=100, reason='client send msg to close!!!')

    def on_close(self):
        print(self.close_code)
        print(self.close_reason)


if __name__ == '__main__':
    app = Application([(r"/con_1", WebSocHandler)])
    app.listen(14991)
    print('start server')
    ioloop.IOLoop.current().start()
