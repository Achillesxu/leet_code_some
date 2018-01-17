#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/11/13 下午5:24
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : string_find_algorothm.py
@desc :
"""
# boyer-moore algorithm
# https://github.com/TWal/ENS_boyermoore/blob/master/python/better.py


def bm_search(in_src, in_pat):
    def pre_process(i_in_pat):
        pat_len = len(i_in_pat)
        # bad char
        bad_char = [-1] * 256
        for i_ in range(pat_len):
            bad_char[ord(i_in_pat[i_])] = i_

        # good suffix
        kmp = [-1] * (pat_len + 1)
        good_suffix = [0] * (pat_len + 1)
        i_ = pat_len
        j_ = pat_len + 1
        kmp[i_] = j_
        while i_ > 0:
            while j_ <= pat_len and i_in_pat[i_ - 1] != i_in_pat[j_ - 1]:
                if good_suffix[j_] == 0:
                    good_suffix[j_] = j_ - i_
                j_ = kmp[j_]
            i_ -= 1
            j_ -= 1
            kmp[i_] = j_

        j_ = kmp[0]
        for i_ in range(pat_len + 1):
            if good_suffix[i_] == 0:
                good_suffix[i_] = j_
            if i_ == j_:
                j_ = kmp[j_]
        return in_pat, bad_char, good_suffix

    i = 0
    n = len(in_src)
    pat, bc, gs = pre_process(in_pat)
    m = len(pat)
    while i <= n - m:
        j = m-1
        while j >= 0 and pat[j] == in_src[i+j]:
            j -= 1
        if j < 0:
            return i
        i += max(gs[j+1], j-bc[ord(in_src[i+j])])
        # i += j-bc[ord(in_src[i+j])]
    return -1


# kmp algorithm
# http://jakeboxer.com/blog/2009/12/13/the-knuth-morris-pratt-algorithm-in-my-own-words/
def kmp_search(in_src, in_pat):
    src_len = len(in_src)
    pat_len = len(in_pat)
    res_search = -1
    if pat_len > src_len:
        return -1

    def partial_match_table(i_in_pat):
        pmt_list = [0 for i in i_in_pat]

        def get_longest_match_str(i_in_str):
            longest_m_num = 0
            for zi in range(1, len(i_in_str)):
                if i_in_str[:zi] == i_in_str[-zi:]:
                    longest_m_num = zi
            return longest_m_num

        for d in range(2, len(i_in_pat)+1):
            # find longest match string
            pmt_list[d-1] = get_longest_match_str(i_in_pat[:d])
        return pmt_list
    par_m_t = partial_match_table(in_pat)
    j = 0
    while j + pat_len <= src_len:
        m = 0
        while True:
            if m == pat_len:
                return j
            elif in_src[m+j] == in_pat[m]:
                m += 1
            else:
                break
        if m == 0:
            j += 1
        else:
            j = m - par_m_t[m - 1]

    return res_search


# Manacher‘s Algorithm
def manacher_search(in_str):
    """
    come from http://www.jianshu.com/p/799bc53d4e3d
     #  a  #  b  #  a  #  b  #  a  #
    [1, 2, 1, 4, 1, 6, 1, 4, 1, 2, 1]
     0  1  2  3  4  5  6  7  8  9  10
    :param in_str:
    :return:
    """
    s_str = '#' + '#'.join(in_str) + '#'
    r_l = [0] * len(s_str)
    max_right = 0
    pos = 0

    for i in range(len(s_str)):
        if i < max_right:
            r_l[i] = min(r_l[2*pos-i], max_right - i)
        else:
            r_l[i] = 1
        # 尝试扩展，注意处理边界, 前两个条件，判断是否数组到边界，第三个条件判断字符是否相等
        while i - r_l[i] >= 0 and i + r_l[i] < len(s_str) and s_str[i - r_l[i]] == s_str[i + r_l[i]]:
            r_l[i] += 1
        # 更新MaxRight,pos
        if r_l[i] + i - 1 > max_right:
            max_right = r_l[i] + i - 1
            pos = i
        print(r_l)
    return [i-1 for i in r_l if i-1 > 0]


if __name__ == '__main__':
    # print(bm_search('xushiyin86', 'yin'))
    # abababca
    test_str = 'abababca'
    # print(kmp_search('bbbbabababcq', 'abababcaddddddddd'))
    # print(kmp_search('bbbbabababca', 'abababca'))
    # print(bm_search('bbbbabababca', 'abababca'))
    print(manacher_search('ababa'))

