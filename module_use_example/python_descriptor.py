#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : leet_code_some
@Time : 2018/4/30 上午11:06
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : python_descriptor.py
@desc : python 3.6 new descriptor
"""
import abc
import logging
from functools import wraps, partial

import collections


class AutoStorage:
    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name


class Validated(abc.ABC, AutoStorage):
    def __set__(self, instance, value):
        value = self.validate(instance, value)
        super().__set__(instance, value)

    @abc.abstractmethod
    def validate(self, instance, value):
        """return validated value or raise ValueError"""


class Quantity(Validated):
    """a number greater than zero"""

    def validate(self, instance, value):
        if value <= 0:
            raise ValueError('value must be > 0')
        return value


class NonBlank(Validated):
    """a string with at least one non-space character"""

    def validate(self, instance, value):
        value = value.strip()
        if len(value) == 0:
            raise ValueError('value cant be empty or blank')
        return value


class LineItem:
    description = NonBlank()
    weight = Quantity()
    price = Quantity()

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price


def quantity():
    try:
        quantity.counter += 1
    except AttributeError:
        quantity.counter = 0

    storage_name = f'_quantity:{quantity.counter}'

    def qty_getter(instance):
        return getattr(instance, storage_name)

    def qty_setter(instance, value):
        if value > 0:
            setattr(instance, storage_name, value)
        else:
            raise ValueError('value must be > 0')

    return property(fget=qty_getter, fset=qty_setter)


class LineItemFuc:
    weight = quantity()
    price = quantity()

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price


def cls_name(obj_or_cls):
    cls = type(obj_or_cls)
    if cls is type:
        cls = obj_or_cls
    return cls.__name__.split('.')[-1]


def display(obj):
    cls = type(obj)
    if cls is type:
        return f'<class {obj.__name__}>'
    elif cls in [type(None), int]:
        return repr(obj)
    else:
        return f'<{cls_name(obj)} object>'


def print_args(name, *args):
    pseudo_args = ', '.join(display(x) for x in args)
    print(f'-> {cls_name(args[0])}.__{name}__({pseudo_args})')


class Overriding:
    """数据描述符或强制描述符"""

    def __get__(self, instance, owner):
        print_args('get', self, instance, owner)

    def __set__(self, instance, value):
        print_args('set', self, instance, value)


class OverridingNoGet:
    """没有__get__的覆盖性描述符"""

    def __set__(self, instance, value):
        print_args('set', self, instance, value)


class NonOverriding:
    """非数据描述符或遮盖型描述符"""

    def __get__(self, instance, owner):
        print_args('get', self, instance, owner)


class Managed:
    over = Overriding()
    over_not_get = OverridingNoGet()
    non_over = NonOverriding()

    def spam(self):
        print(f'-> Managed.spam({display(self)})')


class Text(collections.UserString):
    def __repr__(self):
        return f'Text({self.data!r})'

    def reverse(self):
        return self[::-1]


def logged(level, name=None, message=None):
    """定义一个可接受参数的装饰器"""

    def decorate(func):
        log_name = name if name else func.__module__
        log = logging.getLogger(log_name)
        log_msg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, log_msg)
            # print(level, log_msg, log_name)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def attach_wrapper(obj, func=None):
    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func


def logged_re(level, name=None, message=None):
    """定义用户可修改的属性的装饰器"""

    def decorate(func):
        log_name = name if name else func.__module__
        log = logging.getLogger(log_name)
        log_msg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log.log(level, log_msg)
            # print(level, log_msg, log_name)
            return func(*args, **kwargs)

        @attach_wrapper(wrapper)
        def set_level(new_level):
            nonlocal level
            level = new_level

        @attach_wrapper(wrapper)
        def set_message(new_msg):
            nonlocal log_msg
            log_msg = new_msg

        return wrapper

    return decorate


def logged_opt(func=None, level=logging.INFO, name=None, message=None):
    if func is None:
        return partial(logged_opt, level=level, name=name, message=message)
    log_name = name if name else func.__module__
    log = logging.getLogger(log_name)
    log_msg = message if message else func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        log.log(level, log_msg)
        # print(level, log_msg, log_name)
        return func(*args, **kwargs)

    return wrapper


@logged(logging.INFO)
def add(x, y):
    return x + y


if __name__ == '__main__':
    # try:
    #     a_item = LineItem('apple sum of price', 0, 8)
    # except ValueError as e:
    #     print(e)
    # else:
    #     print(a_item.description)
    #     print(a_item.subtotal())
    # try:
    #     a_item = LineItemFuc('apple sum of price', 0, 8)
    # except ValueError as e:
    #     print(e)
    # else:
    #     print(a_item.description)
    #     print(a_item.subtotal())
    # mm = Managed()
    # print(mm.over)
    # print(Managed.over)
    # print('\n')
    # mm.over = 7
    # print(mm.over)
    print(add(3, 4))
