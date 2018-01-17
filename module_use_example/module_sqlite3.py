#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/11/14 下午5:26
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : module_sqlite3.py
@desc : some skills from http://charlesleifer.com/blog/going-fast-with-sqlite-and-python/
"""
import sqlite3

# Open database in autocommit mode by setting isolation_level to None.
conn = sqlite3.connect('app.db', isolation_level=None)

