#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : leet_code_some
@Time : 2018/4/30 下午4:54
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : python_attr_property.py
@desc : dict without attribute, so the code is wrong, find AttrDict at pypi
"""
import json
import warnings
from collections import abc
from keyword import iskeyword
from pathlib import Path
from urllib.request import urlopen

URL = 'http://www.oreilly.com/pub/sc/osconfeed'
JSON = 'osconfeed.json'


def load():
    p_name = Path(JSON)
    print(p_name.absolute())
    if not p_name.exists():
        msg = f'downloading {URL} to {JSON}'
        warnings.warn(msg)
        with urlopen(URL) as remote, p_name.open(mode='wb') as local:
            local.write(remote.read())

    with p_name.open(encoding='utf-8') as fp:
        return json.load(fp)


def is_keyword(k):
    """check k whether is python keyword, need some code"""
    if k.isidentifier():
        return iskeyword(k)
    else:
        raise TypeError


class FrozenJSON:
    def __init__(self, mapping):
        """check attribute name whether is validate"""
        self.__data = {}
        for k, v in mapping.items():
            if is_keyword(k):
                k += '_'
            self.__data[k] = v
        print(self.__data)

    def __new__(cls, arg):
        if isinstance(arg, abc.Mapping):
            return super().__new__(cls)
        elif isinstance(arg, abc.MutableSequence):
            return [cls.__new__(cls, x) for x in arg]
        else:
            return arg

    def __getattr__(self, name):
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return FrozenJSON(self.__data[name])

    # @classmethod
    # def build(cls, obj):
    #     """the function is same with __new__()"""
    #     if isinstance(obj, abc.Mapping):
    #         return cls(obj)
    #     elif isinstance(obj, abc.MutableSequence):
    #         return [cls.build(x) for x in obj]
    #     else:
    #         return obj


if __name__ == '__main__':
    fj = FrozenJSON({'pp': 12, 'dd': [{'pq': 'od'}, {'kl': 'av'}]})
    print(fj.pp)
    print(fj.dd[1].pq)
