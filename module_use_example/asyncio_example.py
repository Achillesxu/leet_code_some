#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : leet_code_some
@Time : 2018/3/7 上午9:31
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : asyncio_example.py
@desc : compare the threading and asyncio
"""
import asyncio
import datetime
import itertools
import sys
import threading
import time


class Signal:
    go = True


def spin(msg, signal):
    write, flush = sys.stdout.write, sys.stdout.flush
    for char in itertools.cycle('|/-\\'):
        status = char + ' ' + msg
        write(status)
        flush()
        write('\x08' * len(status))
        time.sleep(.1)
        if not signal.go:
            break
    write(' ' * len(status) + '\x08' * len(status))


def slow_function():
    time.sleep(3)
    return 42


def supervisor():
    signal = Signal()
    spinner = threading.Thread(target=spin, args=('thinking', signal))
    print('spinner object:', spinner)
    spinner.start()
    result = slow_function()
    signal.go = False
    spinner.join()
    return result


def main_threading():
    result = supervisor()
    print('Answer:', result)


@asyncio.coroutine
def async_spin(msg):
    write, flush = sys.stdout.write, sys.stdout.flush
    for char in itertools.cycle('|/-\\'):
        status = char + ' ' + msg
        write(status)
        flush()
        write('\x08' * len(status))
        time.sleep(.1)
        try:
            yield from asyncio.sleep(1)
        except asyncio.CancelledError:
            break
    write(' ' * len(status) + '\x08' * len(status))


@asyncio.coroutine
def async_slow_func():
    yield from asyncio.sleep(3)
    return 42


@asyncio.coroutine
def async_supervisor():
    spinner = asyncio.async(async_spin('thinking!'))
    print('spinner object:', spinner)
    result = yield from async_slow_func()
    spinner.cancel()
    return result


def main_async():
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(supervisor())
    loop.close()
    print('Answer:', result)


async def display_datetime(loop):
    end_time = loop.time() + 5.0
    while True:
        print(datetime.datetime.now())
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(1)


def test_display_datetime():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(display_datetime(loop))
    loop.close()


if __name__ == '__main__':
    # main_threading()
    test_display_datetime()
