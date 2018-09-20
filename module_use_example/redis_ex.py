#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/12/12 下午4:06
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : redis_ex.py
@desc :
"""
import json
import redis

r_pool = redis.ConnectionPool(host='localhost', port=6379, db=0,
                              decode_responses=True, connection_class=redis.Connection)
r_ins = redis.Redis(connection_pool=r_pool)


if __name__ == '__main__':
    # print(r_ins.set('record_xushiyin', 1))
    # print(r_ins.get('record_xushiyin'))
    # print(r_ins.incr('record_xushiyin', 1))
    # print(r_ins.get('record_xushiyin'))
    # r_ins.set('record_xushiyin_1', 10)
    # r_ins.set('record_xushiyin_2', 10)
    # r_ins.set('record_xushiyin_3', 10)
    # r_ins.set('record_xushiyin_4', 10)
    # r_ins.set('record_xushiyin_5', 10)
    r_ins.hset('xushiyin', 'second', json.dumps({'content': '我的短信'}))
    pp = r_ins.hget('xushiyin', 'second')
    print(pp)
    print(json.loads(pp))
    # p_ps = r_ins.pubsub()
    # p_ps.subscribe('achilles_xushy')
    # p_ps.subscribe('achilles_xushy1')
    # r_ins.publish('achilles_xushy', 'ready to out?')
    # r_ins.publish('achilles_xushy1', 'ready to out???')
    # print(p_ps.get_message())
    # print(p_ps.get_message())
    # p_ps.unsubscribe('achilles_xushy')
    # print(p_ps.get_message())
    # print(p_ps.get_message())
    # print(p_ps.get_message())
    # for k, v in r_ins.config_get().items():
    #     print(k, v)





