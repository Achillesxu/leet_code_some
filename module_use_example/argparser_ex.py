#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/12/15 上午11:16
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : argparser_ex.py
@desc :
"""
import decimal

IMPORT_TYPE_LIST = ['file', 'interface']

SA_LEVEL = {
    (0, 1500): {'rate': 3, 'add': 0},
    (1500, 4500): {'rate': 10, 'add': 105},
    (4500, 9000): {'rate': 20, 'add': 555},
    (9000, 35000): {'rate': 25, 'add': 1005},
    (35000, 55000): {'rate': 30, 'add': 2755},
    (55000, 80000): {'rate': 35, 'add': 5505},
    (80000, float('inf')): {'rate': 45, 'add': 13505},
}


def arg_parse():
    pass


def func_1(i_file_name):
    print(i_file_name)


def func_2():
    print('use', )


def get_number(in_num):
    mid_val = decimal.Decimal(in_num - 3500)
    if mid_val < decimal.Decimal(0):
        mid_val = decimal.Decimal(0)

    for i in SA_LEVEL.keys():
        if decimal.Decimal(i[0]) <= mid_val < decimal.Decimal(i[1]):
            tax = mid_val * decimal.Decimal(SA_LEVEL[i]['rate']) / decimal.Decimal(100) \
                  - decimal.Decimal(SA_LEVEL[i]['add'])
            return decimal.Decimal(in_num) - tax, tax
    else:
        return -1, -1


def get_name_pw_info(in_name):
    pass


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="""import nan gua media info from file or from nangua interface""")
    # parser.add_argument('type', nargs=1, type=str, choices=IMPORT_TYPE_LIST,
    #                     help='select media info import type')
    # parser.add_argument('--json_file', nargs=1, type=str, default=None,
    #                     help='need media json file if select to import media info from file')
    # name_arg = parser.parse_args()
    # if name_arg.type[0] == 'file':
    #     if os.path.exists(name_arg.json_file[0]) and os.path.isfile(name_arg.json_file[0]):
    #         func_1(name_arg.json_file[0])
    # else:
    #     func_2()
    # print('import finish, datetime <{}>'.format(datetime.datetime.now()))
    input_salary = 11222.71
    a, b = get_number(input_salary)
    if a == -1:
        print('cal error')
    else:
        print('all salary is <{:.2f}>, real salary <{}>, in_tax <{}>'.format(input_salary, a, b))
