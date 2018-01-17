#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/11/15 上午11:07
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : request_get_download_url.py
@desc :
"""
import sys
import time
import requests
import lxml.etree


q_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0',
    "Content-Type": "text/xml",
    "Connection": "close"
}

xpath_current_tab = '//div[@class="o_cn2"]//div[@class="tabs-list current-tab"]//@href'


def main(in_url):
    res = requests.get(in_url, headers=q_headers)
    if res.status_code == 200:
        root_node = lxml.etree.HTML(res.content)
        down_url_list = root_node.xpath(xpath_current_tab)
        for i in down_url_list:
            print(i)
    else:
        print('request <{}>, return code <{}>'.format(in_url, res.status_code))


if __name__ == '__main__':
    start_t = time.time()
    print('request <{}> .................'.format(sys.argv[1]))
    main(sys.argv[1])
    end_t = time.time()
    print('{} using time <{}>'.format(__file__, end_t - start_t))
