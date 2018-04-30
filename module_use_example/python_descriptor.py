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


if __name__ == '__main__':
    # try:
    #     a_item = LineItem('apple sum of price', 0, 8)
    # except ValueError as e:
    #     print(e)
    # else:
    #     print(a_item.description)
    #     print(a_item.subtotal())
    try:
        a_item = LineItemFuc('apple sum of price', 0, 8)
    except ValueError as e:
        print(e)
    else:
        print(a_item.description)
        print(a_item.subtotal())
