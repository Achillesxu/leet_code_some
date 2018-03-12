#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/11/3 下午1:47
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : leetcode1.py
@desc : 1---100
"""
import logging
import time
from ctypes import *
from functools import wraps

r_log = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def decorate_time(func):
    @wraps(func)
    def wrapper(*argv, **kwargv):
        start_t = time.clock()
        res_val = func(*argv, **kwargv)
        end_t = time.clock()
        r_log.info('<{}> using time <{}>'.format(func.__name__, end_t - start_t))
        return res_val
    return wrapper


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def produce_random_list(in_list_node_num, dou):
    """
    produce list
    :param in_list_node_num:
    :param dou: 1, or 2
    :return: linked list
    """
    if dou % 2:
        root_node = ListNode(1)
    else:
        root_node = ListNode(2)
    p_node = root_node
    for i in range(3 if dou == 1 else 4, in_list_node_num, 2):
        temp_node = ListNode(i)
        p_node.next = temp_node
        p_node = temp_node
    return root_node


def produce_node_list(in_list):
    """
    produce node list
    :param in_list:
    :return:
    """
    root_n = ListNode(in_list[0])
    temp_n = root_n
    for i in in_list[1:]:
        temp = ListNode(i)
        temp_n.next = temp
        temp_n = temp
    return root_n


def loop_print_linked_value(in_linked_list):
    root_h = in_linked_list
    p_node = root_h
    while 1:
        print(p_node.val)
        p_node = p_node.next
        if p_node is None:
            break


class TreeNode(object):
    def __init__(self, i_val):
        self.val = i_val
        self.left = None
        self.right = None


def yield_nodes_tree(i_rev=0):
    if i_rev:
        root = TreeNode(1)
        l1 = TreeNode(2)
        r1 = TreeNode(3)
    else:
        root = TreeNode(1)
        l1 = TreeNode(2)
        r1 = TreeNode(3)

    root.left = l1
    root.right = r1
    l1_l1 = TreeNode(4)
    l1_r1 = TreeNode(5)
    l1.left = l1_l1
    l1.right = l1_r1

    r1_l1 = TreeNode(6)
    r1_r1 = TreeNode(7)
    r1.left = r1_l1
    r1.right = r1_r1

    r2_r1 = TreeNode(8)
    l1_l1.right = r2_r1

    return root


def loop_binary_tree_nodes(i_root):
    if i_root:
        print(i_root.val)
        loop_binary_tree_nodes(i_root.left)
        loop_binary_tree_nodes(i_root.right)


def level_output_tree(in_root):
    stack = []
    node = in_root
    if in_root:
        stack.append(node)
    while stack:
        node = stack.pop(0)
        yield node.val
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)


class Employee:
    def __init__(self, i_id, i_importance, i_subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = i_id
        # the importance value of this employee
        self.importance = i_importance
        # the id of direct subordinates
        self.subordinates = i_subordinates


class Solution1(object):
    @decorate_time
    def hamming_distance(self, x, y):
        """
        hamming distance
        :param x: int
        :param y: int
        :return: int
        """
        res = 0
        temp = x ^ y
        while c_int(temp).value:
            res += 1
            temp &= temp - 1
        return res

    @decorate_time
    def judge_route_circle(self, moves):
        """
        Initially, there is a Robot at position (0, 0). Given a sequence of its moves, judge if this robot makes
        a circle, which means it moves back to the original place.
        The move sequence is represented by a string. And each move is represent by a character.
        The valid robot moves are R (Right), L (Left), U (Up) and D (down).
        The output should be true or false representing whether the robot makes a circle.
        :param moves: str
        :return: bool
        """
        if moves:
            move_list = list(moves)
            count_u = move_list.count('U')
            count_d = move_list.count('D')
            count_r = move_list.count('R')
            count_l = move_list.count('L')
            if count_u - count_d + count_r - count_l:
                return False
            else:
                return True
        else:
            return True

    @decorate_time
    def two_sum_add_to_target(self, nums, target):
        """
        Given an array of integers, return indices of the two numbers such that they add up to a specific target.
        You may assume that each input would have exactly one solution, and you may not use the same element twice.
        :param nums: list
        :param target: int
        :return: list
        """
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j and nums[i] + nums[j] == target:
                    return [i, j]
        else:
            return []

    @decorate_time
    def reverse_integer(self, x):
        """
        Given a 32-bit signed integer, reverse digits of an integer.
        Example 1:

        Input: 123
        Output:  321
        Example 2:

        Input: -123
        Output: -321
        Example 3:

        Input: 120
        Output: 21
        Note:
        Assume we are dealing with an environment which could only hold integers within the 32-bit signed integer range. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.
        :param x: int
        :return: int
        """
        reverse_int = 0
        if x == 0:
            return reverse_int
        neg = 1
        if x < 0:
            neg, x = -1, -x
        while x:
            reverse_int = reverse_int * 10 + x % 10
            x //= 10

        reverse_int = reverse_int * neg
        if reverse_int < -(1 << 31) or reverse_int > (1 << 31):
            return 0
        return reverse_int

    @decorate_time
    def is_palindrome(self, x):
        """
        If you are thinking of converting the integer to string, note the restriction of using extra space.
        You could also try reversing an integer. However, if you have solved the problem "Reverse Integer",
        you know that the reversed integer might overflow. How would you handle such case?
        There is a more generic way of solving this problem.
        :param x:
        :return:
        """
        if x == 0:
            return True
        pal_int = self.reverse_integer(x)
        if x + pal_int == x:
            return False
        elif x == pal_int:
            return True
        else:
            return False

    @decorate_time
    def roman_to_integer(self, in_s):
        """
        Given a roman numeral, convert it to an integer.
        Input is guaranteed to be within the range from 1 to 3999.
        :param in_s: str
        :return: int
        """
        ROMAN = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        if in_s:
            index = len(in_s) - 2
            sum_int = ROMAN[in_s[-1]]
            while index >= 0:
                if ROMAN[in_s[index]] < ROMAN[in_s[index + 1]]:
                    sum_int -= ROMAN[in_s[index]]
                else:
                    sum_int += ROMAN[in_s[index]]
                index -= 1
            return sum_int
        else:
            return 0

    @decorate_time
    def integer_to_roman(self, num):
        """
        integer range 1-3999
        :param num:
        :return:
        """
        def parse(i_digit, i_index):
            NUMS = {
                1: 'I',
                2: 'II',
                3: 'III',
                4: 'IV',
                5: 'V',
                6: 'VI',
                7: 'VII',
                8: 'VIII',
                9: 'IX',
            }
            ROMAN = {
                'I': ['I', 'X', 'C', 'M'],
                'V': ['V', 'L', 'D', '?'],
                'X': ['X', 'C', 'M', '?']
            }

            in_s = NUMS[i_digit]
            return in_s.replace('X', ROMAN['X'][i_index]).replace('I', ROMAN['I'][i_index]).\
                replace('V', ROMAN['V'][i_index])

        s = ''
        index = 0
        while num != 0:
            digit = num % 10
            if digit != 0:
                s = parse(digit, index) + s
            num = num // 10
            index += 1
        return s

    @decorate_time
    def longest_common_prefix(self, in_str_list):
        """
        Write a function to find the longest common prefix string amongst an array of strings.
        :param in_str_list: list[str]
        :return: str
        """
        min_len = min([len(i_str) for i_str in in_str_list])
        min_str = ''
        for i in in_str_list:
            if len(i) == min_len:
                min_str = i
                break
        while min_str:
            if all([min_str in i for i in in_str_list]):
                return min_str
            else:
                min_str = min_str[:-1]
        else:
            return None

    @decorate_time
    def valid_parentheses(self, in_str):
        """
        Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is
        valid.
        The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
        :param in_str:
        :return: bool
        """
        pa_list = ['(', ')', '{', '}', '[', ']']
        pa_dict = {'(': ')', '{': '}', '[': ']'}
        clear_str = ''.join([i for i in in_str if i in pa_list])
        if clear_str:
            check_stack = list()
            check_stack.append(clear_str[0])
            if len(clear_str) % 2 == 0:
                for pi in clear_str[1:]:
                    check_stack.append(pi)
                    if pa_dict.get(check_stack[-2]) == check_stack[-1]:
                        check_stack.pop()
                        check_stack.pop()
                if len(check_stack) == 0:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    @decorate_time
    def merge_two_sorted_lists(self, in_list_a, in_list_b):
        """
        Merge two sorted linked lists and return it as a new list.
        The new list should be made by splicing together the nodes of the first two lists.
        :param in_list_a: list
        :param in_list_b: list
        :return:
        """
        p_node_a = in_list_a
        p_node_b = in_list_b
        new_root = ListNode(p_node_a.val if p_node_a.val <= p_node_b.val else p_node_b.val)
        new_p_node = new_root
        if p_node_a.val <= p_node_b.val:
            p_node_a = p_node_a.next
        else:
            p_node_b = p_node_b.next

        while p_node_a or p_node_b:
            if p_node_a is None:
                new_i_node = ListNode(p_node_b.val)
                p_node_b = p_node_b.next
                new_p_node.next = new_i_node
                new_p_node = new_i_node
            elif p_node_b is None:
                new_i_node = ListNode(p_node_a.val)
                p_node_a = p_node_a.next
                new_p_node.next = new_i_node
                new_p_node = new_i_node
            else:
                if p_node_a.val <= p_node_b.val:
                    new_i_node = ListNode(p_node_a.val)
                    p_node_a = p_node_a.next
                    new_p_node.next = new_i_node
                    new_p_node = new_i_node
                else:
                    new_i_node = ListNode(p_node_b.val)
                    p_node_b = p_node_b.next
                    new_p_node.next = new_i_node
                    new_p_node = new_i_node
        return new_root

    @decorate_time
    def remove_dup_from_sorted_array(self, nums):
        """
        Given a sorted array, remove the duplicates in-place such that
        each element appear only once and return the new length.
        Do not allocate extra space for another array,
        you must do this by modifying the input array in-place with O(1) extra memory.
        :param nums: list[int]
        :return: int
        """
        base_key = 0
        if not len(nums):
            return 0
        while True:
            if base_key + 1 == len(nums):
                return base_key + 1
            else:
                if nums[base_key] == nums[base_key+1]:
                    del nums[base_key+1]
                else:
                    base_key += 1

    @decorate_time
    def remove_element(self, nums, val):
        """
        Given an array and a value, remove all instances of that value in-place and return the new length.
        Do not allocate extra space for another array, you must do this by modifying the input array
        in-place with O(1) extra memory.
        The order of elements can be changed. It doesn't matter what you leave beyond the new length.
        :param nums: list[int]
        :param val: int
        :return: int
        """
        i_index = 0
        if len(nums) == 0:
            return 0
        while True:
            if nums[i_index] == val:
                del nums[i_index]
            else:
                i_index += 1
            if len(nums) == 0:
                return 0
            elif i_index + 1 == len(nums):
                if nums[i_index] == val:
                    del nums[i_index]
                    return i_index
                else:
                    return i_index + 1

    @decorate_time
    def c_str_str(self, haystack, needle):
        """
        Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
        :param haystack:
        :param needle:
        :return:
        """
        if haystack is None or needle is None:
            return -1
        len_s, len_t = len(haystack), len(needle)
        for i in range(len_s - len_t + 1):
            j = 0
            while j < len_t:
                if haystack[i + j] != needle[j]:
                    break
                j += 1
            if j == len_t:
                return i
        return -1

    @decorate_time
    def search_insert_position(self, nums, target):
        """
        Given a sorted array and a target value, return the index if the target is found.
        If not, return the index where it would be if it were inserted in order.
        You may assume no duplicates in the array.

        Example 1:
        Input: [1,3,5,6], 5
        Output: 2

        Example 2:
        Input: [1,3,5,6], 2
        Output: 1

        :param nums: list[int]
        :param target: int
        :return: int
        """
        list_len = len(nums)
        for i in range(list_len):
            if nums[i] == target:
                return i
            else:
                if i + 1 < list_len and nums[i + 1] == target:
                    return i + 1
                elif i + 1 < list_len and nums[i + 1] > target:
                    return i + 1
                else:
                    pass
        else:
            return list_len

    @decorate_time
    def add_two_numbers(self, l1, l2):
        """
        You are given two non-empty linked lists representing two non-negative integers.
        The digits are stored in reverse order and each of their nodes contain a single digit.
        Add the two numbers and return it as a linked list.
        You may assume the two numbers do not contain any leading zero, except the number 0 itself.

        Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
        Output: 7 -> 0 -> 8
        :param l1: linked nodes
        :param l2: linked nodes
        :return:
        """
        def get_linked_num(in_linked_):
            l_stack = [in_linked_.val]
            l_n = in_linked_.next
            while l_n:
                l_stack.append(l_n.val)
                l_n = l_n.next
            return int(''.join([str(i) for i in l_stack[::-1]]))
        num1 = get_linked_num(l1)
        num2 = get_linked_num(l2)
        res = num1 + num2
        return produce_node_list([int(i) for i in list(str(res))[::-1]])

    @decorate_time
    def merge_two_binary_trees(self, t1, t2):
        """
        Given two binary trees and imagine that when you put one of them to cover the other,
        some nodes of the two trees are overlapped while the others are not.
        You need to merge them into a new binary tree.
        The merge rule is that if two nodes overlap,
        then sum node values up as the new value of the merged node.
        Otherwise, the NOT null node will be used as the node of new tree.
        Input:
        Tree 1                     Tree 2
              1                         2
             / \                       / \
            3   2                     1   3
           /                           \   \
          5                             4   7
        Output:
        Merged tree:
                 3
                / \
               4   5
              / \   \
             5   4   7
        :return:
        """
        if not t1 and not t2:
            return None
        ans = TreeNode((t1.val if t1 else 0) + (t2.val if t2 else 0))
        ans.left = self.merge_two_binary_trees(t1 and t1.left, t2 and t2.left)
        ans.right = self.merge_two_binary_trees(t1 and t1.right, t2 and t2.right)
        return ans

    @decorate_time
    def array_partition_i(self, i_nums):
        """
        Given an array of 2n integers, your task is to group these integers into n pairs of integer,
        say (a1, b1), (a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i from 1 to n as large as possible.

        Example 1:
        Input: [1,4,3,2]

        Output: 4
        Explanation: n is 2, and the maximum sum of pairs is 4 = min(1, 2) + min(3, 4).
        Note:
        n is a positive integer, which is in the range of [1, 10000].
        All the integers in the array will be in the range of [-10000, 10000].

        :param i_nums:
        :return:
        """
        return sum(sorted(i_nums)[::2])

    @decorate_time
    def number_complement(self, in_num):
        """
        Given a positive integer, output its complement number.
        The complement strategy is to flip the bits of its binary representation.

        Note:
        The given integer is guaranteed to fit within the range of a 32-bit signed integer.
        You could assume no leading zero bit in the integer’s binary representation.
        Example 1:
        Input: 5
        Output: 2
        Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010.
        So you need to output 2.
        Example 2:
        Input: 1
        Output: 0
        Explanation: The binary representation of 1 is 1 (no leading zero bits), and its complement is 0.
        So you need to output 0.

        :param in_num:
        :return:
        """
        i = 1
        while i <= in_num:
            i <<= 1
        return (i - 1) ^ in_num

    @decorate_time
    def reverse_words_in_string(self, in_string):
        """
        Given a string, you need to reverse the order of characters in each word within a sentence
        while still preserving whitespace and initial word order.

        Example 1:
        Input: "Let's take LeetCode contest"
        Output: "s'teL ekat edoCteeL tsetnoc"
        Note: In the string, each word is separated by single space and there will not be any extra space in the string.
        :param in_string:
        :return:
        """
        word_list = in_string.split(' ')
        return ' '.join([i[::-1] for i in word_list])

    @decorate_time
    def fizz_buzz_fizzbuzz(self, in_num):
        """
        Write a program that outputs the string representation of numbers from 1 to n.

        But for multiples of three it should output “Fizz” instead of the number and for the multiples of five output “Buzz”.
        For numbers which are multiples of both three and five output “FizzBuzz”.
        Example:
        n = 15,
        Return:
        [
            "1",
            "2",
            "Fizz",
            "4",
            "Buzz",
            "Fizz",
            "7",
            "8",
            "Fizz",
            "Buzz",
            "11",
            "Fizz",
            "13",
            "14",
            "FizzBuzz"
        ]
        :param in_num:
        :return:
        """
        res_list = []
        for i in range(1, in_num + 1):
            if i % 3 == 0 and i % 5 == 0:
                res_list.append('FizzBuzz')
            elif i % 3 == 0:
                res_list.append('Fizz')
            elif i % 5 == 0:
                res_list.append('Buzz')
            else:
                res_list.append(str(i))
        return res_list

    @decorate_time
    def trim_binary_search_tree(self, in_root, in_L, in_R):
        """
        Given a binary search tree and the lowest and highest boundaries as L and R,
        trim the tree so that all its elements lies in [L, R] (R >= L).
        You might need to change the root of the tree,
        so the result should return the new root of the trimmed binary search tree.
        Example 1:
        Input:
            1
           / \
          0   2

          L = 1
          R = 2

        Output:
            1
              \
               2
        Example 2:
        Input:
            3
           / \
          0   4
           \
            2
           /
          1

          L = 1
          R = 3

        Output:
              3
             /
           2
          /
         1
        :param in_root:
        :param in_L:
        :param in_R:
        :return:
        """
        if not in_root:
            return None
        if in_L > in_root.val:
            return self.trim_binary_search_tree(in_root.right, in_L, in_R)
        elif in_R < in_root.val:
            return self.trim_binary_search_tree(in_root.left, in_L, in_R)
        in_root.left = self.trim_binary_search_tree(in_root.left, in_L, in_R)
        in_root.right = self.trim_binary_search_tree(in_root.right, in_L, in_R)
        return in_root

    @decorate_time
    def next_greater_element_1(self, nums1, nums2):
        """
        You are given two arrays (without duplicates) nums1 and nums2 where nums1’s elements are subset of nums2.
        Find all the next greater numbers for nums1's elements in the corresponding places of nums2.
        The Next Greater Number of a number x in nums1 is the first greater number to its right in nums2.
        If it does not exist, output -1 for this number.

        Example 1:
        Input: nums1 = [4,1,2], nums2 = [1,3,4,2].
        Output: [-1,3,-1]
        Explanation:
            For number 4 in the first array, you cannot find the next greater number for it in the second array, so output -1.
            For number 1 in the first array, the next greater number for it in the second array is 3.
            For number 2 in the first array, there is no next greater number for it in the second array, so output -1.
        Example 2:
        Input: nums1 = [2,4], nums2 = [1,2,3,4].
        Output: [3,-1]
        Explanation:
            For number 2 in the first array, the next greater number for it in the second array is 3.
            For number 4 in the first array, there is no next greater number for it in the second array, so output -1.
        Note:
        All elements in nums1 and nums2 are unique.
        The length of both nums1 and nums2 would not exceed 1000.
        :param nums1:
        :param nums2:
        :return:
        """
        res_list = []
        for i in nums1:
            for j in range(nums2.index(i), len(nums2)):
                if i < nums2[j]:
                    res_list.append(nums2[j])
                    break
            else:
                res_list.append(-1)
        return res_list

    @decorate_time
    def island_perimeter(self, in_grid):
        """
        You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water.
        Grid cells are connected horizontally/vertically (not diagonally).
        The grid is completely surrounded by water,
        and there is exactly one island (i.e., one or more connected land cells).
        The island doesn't have "lakes" (water inside that isn't connected to the water around the island).
        One cell is a square with side length 1.
        The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

        Example:

        [[0,1,0,0],
         [1,1,1,0],
         [0,1,0,0],
         [1,1,0,0]]

        Answer: 16
        Explanation: The perimeter is the 16 yellow stripes in the image below

        https://leetcode.com/problems/island-perimeter/description/
        :param in_grid: List[List[int]]
        :return:
        """
        import operator
        return sum(sum(map(operator.ne, [0] + row, row + [0])) for row in in_grid + [i for i in map(list, zip(*in_grid))])

    @decorate_time
    def longest_uncommon_subsequence_i(self, i_a: str, i_b: str) -> int:
        """
        Given a group of two strings, you need to find the longest uncommon subsequence of this group of two strings.
        The longest uncommon subsequence is defined as the longest subsequence of one of these strings and
        this subsequence should not be any subsequence of the other strings.

        A subsequence is a sequence that can be derived from one sequence by deleting some characters without changing
        the order of the remaining elements. Trivially, any string is a subsequence of itself and
        an empty string is a subsequence of any string.

        The input will be two strings, and the output needs to be the length of the longest uncommon subsequence.
        If the longest uncommon subsequence doesn't exist, return -1.

        Example 1:
        Input: "aba", "cdc"
        Output: 3
        Explanation: The longest uncommon subsequence is "aba" (or "cdc"),
        because "aba" is a subsequence of "aba",
        but not a subsequence of any other strings in the group of two strings.
        Note:

        Both strings' lengths will not exceed 100.
        Only letters from a ~ z will appear in input strings.
        :param i_a:
        :param i_b:
        :return: len int
        """
        if i_a == i_b:
            return -1
        else:
            return max(len(i_a), len(i_b))

    @decorate_time
    def average_of_levels_in_binary_tree(self, root):
        """
        Given a non-empty binary tree, return the average value of the nodes on each level in the form of an array.

        Example 1:
        Input:
            3
           / \
          9  20
            /  \
           15   7
        Output: [3, 14.5, 11]
        Explanation:
        The average value of nodes on level 0 is 3,  on level 1 is 14.5, and on level 2 is 11. Hence return [3, 14.5, 11].
        Note:
        The range of node's value is in the range of 32-bit signed integer.
        :param root:
        :return:
        """
        info = []

        def dfs(node, depth=0):
            if node:
                if len(info) <= depth:
                    info.append([0, 0])
                info[depth][0] += node.val
                info[depth][1] += 1
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)

        dfs(root)
        return [s / float(c) for s, c in info]

    @decorate_time
    def nim_game(self, in_n):
        """
        You are playing the following Nim Game with your friend: There is a heap of stones on the table,
        each time one of you take turns to remove 1 to 3 stones. The one who removes the last stone will be the winner.
        You will take the first turn to remove the stones.
        Both of you are very clever and have optimal strategies for the game.
        Write a function to determine whether you can win the game given the number of stones in the heap.
        For example, if there are 4 stones in the heap,
        then you will never win the game: no matter 1, 2, or 3 stones you remove,
        the last stone will always be removed by your friend.

        Credits:
        Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
        https://leetcode.com/problems/nim-game/discuss/
        :param in_n:
        :return:
        """
        return (in_n % 4) != 0

    @decorate_time
    def single_number(self, in_nums):
        """
        Given an array of integers, every element appears twice except for one. Find that single one.
        Note:
        Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
        :param in_nums:
        :return:
        """
        res = 0
        for n in in_nums:
            res ^= n
        return res

    @decorate_time
    def binary_number_with_alternating_bits(self, in_n):
        """
        Given a positive integer, check whether it has alternating bits: namely, if two adjacent bits will always have
        different values.

        Example 1:
        Input: 5
        Output: True
        Explanation:
        The binary representation of 5 is: 101
        Example 2:
        Input: 7
        Output: False
        Explanation:
        The binary representation of 7 is: 111.
        Example 3:
        Input: 11
        Output: False
        Explanation:
        The binary representation of 11 is: 1011.
        Example 4:
        Input: 10
        Output: True
        Explanation:
        The binary representation of 10 is: 1010.
        :param in_n:
        :return:
        """
        if '00' not in bin(in_n) and '11' not in bin(in_n):
            return True
        else:
            return False

    @decorate_time
    def max_consecutive_ones(self, in_bin_nums):
        """
        Given a binary array, find the maximum number of consecutive 1s in this array.

        Example 1:
        Input: [1,1,0,1,1,1]
        Output: 3
        Explanation: The first two digits or the last three digits are consecutive 1s.
            The maximum number of consecutive 1s is 3.
        Note:
        The input array will only contain 0 and 1.
        The length of input array is a positive integer and will not exceed 10,000
        :param in_bin_nums:
        :return:
        """
        return max([len(j) for j in ''.join([str(i) for i in in_bin_nums]).split('0')])

    @decorate_time
    def maximum_depth_of_binary_tree(self, in_root):
        """
        Given a binary tree, find its maximum depth.
        The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
        :param in_root:
        :return:
        """
        dep_rec = []

        def loop_binary_tree(in_node, in_depth=0):
            if in_node:
                if len(dep_rec) <= in_depth:
                    dep_rec.append(in_depth)
                loop_binary_tree(in_node.left, in_depth + 1)
                loop_binary_tree(in_node.right, in_depth + 1)

        loop_binary_tree(in_root)
        return max(dep_rec)

    @decorate_time
    def employee_importance(self, in_employees, in_id):
        """
        You are given a data structure of employee information, which includes the employee's unique id,
        his importance value and his direct subordinates' id.
        For example, employee 1 is the leader of employee 2,
        and employee 2 is the leader of employee 3. They have importance value 15, 10 and 5, respectively.
        Then employee 1 has a data structure like [1, 15, [2]], and employee 2 has [2, 10, [3]],
        and employee 3 has [3, 5, []]. Note that although employee 3 is also a subordinate of employee 1,
        the relationship is not direct.
        Now given the employee information of a company,
        and an employee id, you need to return the total importance value of this employee and all his subordinates.

        Example 1:
        Input: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
        Output: 11
        Explanation:
        Employee 1 has importance value 5, and he has two direct subordinates: employee 2 and employee 3.
        They both have importance value 3. So the total importance value of employee 1 is 5 + 3 + 3 = 11.
        Note:
        One employee has at most one direct leader and may have several subordinates.
        The maximum number of employees won't exceed 2000.
        :param in_employees:
        :param in_id:
        :return:
        """

        def find_id_employee(i_in_employees, i_in_id):
            for i in i_in_employees:
                if i[0] == i_in_id:
                    return i

        emp_leader = find_id_employee(in_employees, in_id)
        res_importance = emp_leader[1] + sum([i[1] for i in in_employees if i[0] in emp_leader[2]])

        return res_importance

    @decorate_time
    def add_binary(self, in_a, in_b):
        """
        Given two binary strings, return their sum (also a binary string).

        For example,
        a = "11"
        b = "1"
        Return "100".
        :param in_a:
        :param in_b:
        :return:
        """
        return bin(int(in_a, base=2) + int(in_b, base=2))[2:]

    @decorate_time
    def array_digits_plus_one(self, in_digits):
        """
        Given a non-negative integer represented as a non-empty array of digits, plus one to the integer.
        You may assume the integer do not contain any leading zero, except the number 0 itself.
        The digits are stored such that the most significant digit is at the head of the list.
        :param in_digits:
        :return:
        """
        return [int(i) for i in bin(int(''.join([str(i) for i in in_digits]), base=2) + 1)[2:]]

    @decorate_time
    def sqrt_imp(self, in_x):
        """
        Implement int sqrt(int x).
        Compute and return the square root of x.
        x is guaranteed to be a non-negative integer.

        Example 1:
        Input: 4
        Output: 2
        Example 2:
        Input: 8
        Output: 2
        Explanation: The square root of 8 is 2.82842..., and since we want to return an integer, the decimal part will be truncated.

        :param in_x:
        :return:
        """
        left = 0.0
        right = in_x
        eps = 1e-12

        if in_x < 1.0:
            right = 1.0

        while right - left > eps:
            mid = (right + left) / 2
            if mid * mid < in_x:
                left = mid
            else:
                right = mid

        return int(left)

    @decorate_time
    def one_bit_and_two_bit_characters(self, in_bits):
        """
        We have two special characters. The first character can be represented by one bit 0.
        The second character can be represented by two bits (10 or 11).
        Now given a string represented by several bits.
        Return whether the last character must be a one-bit character or not.
        The given string will always end with a zero.
        Example 1:
        Input:
        bits = [1, 0, 0]
        Output: True
        Explanation:
        The only way to decode it is two-bit character and one-bit character. So the last character is one-bit character.
        Example 2:
        Input:
        bits = [1, 1, 1, 0]
        Output: False
        Explanation:
        The only way to decode it is two-bit character and two-bit character. So the last character is NOT one-bit character.
        Note:
        1 <= len(bits) <= 1000.
        bits[i] is always 0 or 1.
        :param in_bits:
        :return:
        """
        cur = 0
        bits_len = len(in_bits)
        while cur < bits_len - 2:
            if in_bits[cur] == 0:
                cur += 1
            else:
                cur += 2
        if in_bits[cur] == 1:
            return False
        else:
            return True

    @decorate_time
    def max_area_of_island(self, in_grid):
        """
        answer from https://discuss.leetcode.com/topic/106370/easy-python/2
        Given a non-empty 2D array grid of 0's and 1's,
        an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.)
        You may assume all four edges of the grid are surrounded by water.
        Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)

        Example 1:
        [[0,0,1,0,0,0,0,1,0,0,0,0,0],
         [0,0,0,0,0,0,0,1,1,1,0,0,0],
         [0,1,1,0,1,0,0,0,0,0,0,0,0],
         [0,1,0,0,1,1,0,0,1,0,1,0,0],
         [0,1,0,0,1,1,0,0,1,1,1,0,0],
         [0,0,0,0,0,0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,0,1,1,1,0,0,0],
         [0,0,0,0,0,0,0,1,1,0,0,0,0]]
        Given the above grid, return 6. Note the answer is not 11, because the island must be connected 4-directionally.
        Example 2:
        [[0,0,0,0,0,0,0,0]]
        Given the above grid, return 0.
        Note: The length of each dimension in the given grid does not exceed 50.
        :param in_grid:
        :return:
        """
        m, n = len(in_grid), len(in_grid[0])

        def dfs(i, j):
            if 0 <= i < m and 0 <= j < n and in_grid[i][j]:
                in_grid[i][j] = 0
                return 1 + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i + 1, j) + dfs(i, j - 1)
            return 0

        areas = [dfs(i, j) for i in range(m) for j in range(n) if in_grid[i][j]]
        print(areas)
        return max(areas) if areas else 0

    @decorate_time
    def invert_binary_tree(self, in_root):
        """
        Invert a binary tree.

             4
           /   \
          2     7
         / \   / \
        1   3 6   9
        to
             4
           /   \
          7     2
         / \   / \
        9   6 3   1
        Trivia:
        This problem was inspired by this original tweet by Max Howell:
        Google: 90% of our engineers use the software you wrote (Homebrew),
        but you can’t invert a binary tree on a whiteboard so fuck off.
        :param in_root:
        :return:
        """
        if in_root:
            in_root.left, in_root.right = self.invert_binary_tree(in_root.right), self.invert_binary_tree(in_root.left)
        return in_root

    @decorate_time
    def detect_capital(self, in_word):
        """
        Given a word, you need to judge whether the usage of capitals in it is right or not.
        We define the usage of capitals in a word to be right when one of the following cases holds:
        All letters in this word are capitals, like "USA".
        All letters in this word are not capitals, like "leetcode".
        Only the first letter in this word is capital if it has more than one letter, like "Google".
        Otherwise, we define that this word doesn't use capitals in a right way.
        Example 1:
        Input: "USA"
        Output: True
        Example 2:
        Input: "FlaG"
        Output: False
        Note: The input will be a non-empty word consisting of uppercase and lowercase latin letters.
        :param in_word:
        :return:
        """
        if ord(in_word[1]) >= ord('a'):
            for i in in_word[2:]:
                if ord(i) < ord('a'):
                    return False
        else:
            for i in in_word[2:]:
                if ord(i) >= ord('a'):
                    return False
        return True

    @decorate_time
    def add_digits(self, in_num):
        """
        Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
        For example:
        Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.
        Follow up:
        Could you do it without any loop/recursion in O(1) runtime?
        Credits:
        Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
        this method depends on the truth:
        N=(a[0] * 1 + a[1] * 10 + ...a[n] * 10 ^n),and a[0]...a[n] are all between [0,9]
        we set M = a[0] + a[1] + ..a[n]
        and another truth is that:
        1 % 9 = 1
        10 % 9 = 1
        100 % 9 = 1
        so N % 9 = a[0] + a[1] + ..a[n]
        means N % 9 = M
        so N = M (% 9)
        as 9 % 9 = 0,so we can make (n - 1) % 9 + 1 to help us solve the problem when n is 9.as N is 9,
        ( 9 - 1) % 9 + 1 = 9
        :param in_num:
        :return:
        """
        if in_num == 0:
            return 0
        else:
            return (in_num - 1) % 9 + 1

    @decorate_time
    def count_binary_substrings(self, in_bin_str):
        """
        Give a string s, count the number of non-empty (contiguous) substrings that have the same number of 0's and 1's,
        and all the 0's and all the 1's in these substrings are grouped consecutively.
        Substrings that occur multiple times are counted the number of times they occur.

        Example 1:
        Input: "00110011"
        Output: 6
        Explanation: There are 6 substrings that have equal number of consecutive
        1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
        Notice that some of these substrings repeat and are counted the number of times they occur.
        Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
        Example 2:
        Input: "10101"
        Output: 4
        Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number of consecutive 1's and 0's.
        Note:
        s.length will be between 1 and 50,000.
        s will only consist of "0" or "1" characters.
        :param in_bin_str:
        :return:
        """
        s = [i for i in map(len, in_bin_str.replace('01', '0 1').replace('10', '1 0').split())]
        print(s)
        return sum(min(a, b) for a, b in zip(s, s[1:]))

    @decorate_time
    def find_all_numbers_disappeared_in_an_array(self, in_nums):
        """
        Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.
        Find all the elements of [1, n] inclusive that do not appear in this array.
        Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.

        Example:
        Input:
        [4,3,2,7,8,2,3,1]
        Output:
        [5,6]
        :param in_nums:
        :return:
        """
        lost_list = []
        l_max = max(in_nums)
        for i in range(1, l_max+1):
            if i not in in_nums:
                lost_list.append(i)
        return lost_list

    @decorate_time
    def sum_of_two_integers(self, in_a, in_b):
        """
        Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.
        Example:
        Given a = 1 and b = 2, return 3.
        Credits:
        Special thanks to @fujiaozhu for adding this problem and creating all test cases.
        :param in_a:
        :param in_b:
        :return:
        """
        # 32 bits integer max
        MAX = 0x7FFFFFFF
        # 32 bits interger min
        MIN = 0x80000000
        # mask to get last 32 bits
        mask = 0xFFFFFFFF
        while in_b != 0:
            # ^ get different bits and & gets double 1s, << moves carry
            in_a, in_b = (in_a ^ in_b) & mask, ((in_a & in_b) << 1) & mask
        # if a is negative, get a's 32 bits complement positive first
        # then get 32-bit positive's Python complement negative
        return in_a if in_a <= MAX else ~(in_a ^ mask)

    @decorate_time
    def find_the_difference(self, in_str1, in_str2):
        """
        Given two strings s and t which consist of only lowercase letters.
        String t is generated by random shuffling string s and then add one more letter at a random position.
        Find the letter that was added in t.

        Example:
        Input:
        s = "abcd"
        t = "abcde"
        Output:
        e

        Explanation:
        'e' is the letter that was added.
        :param in_str1:
        :param in_str2:
        :return:
        """
        import collections
        return ''.join(list(collections.Counter(in_str2) - collections.Counter(in_str1)))

    @decorate_time
    def move_zeros(self, in_nums):
        """
        Given an array nums,
        write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
        For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

        Note:
        You must do this in-place without making a copy of the array.
        Minimize the total number of operations.
        Credits:
        Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
        :param in_nums:
        :return:
        """
        i_zero = 0
        for i in range(len(in_nums)):
            if in_nums[i] != 0:
                in_nums[i], in_nums[i_zero] = in_nums[i_zero], in_nums[i]
                i_zero += 1
        return in_nums

    @decorate_time
    def two_sum_4(self, in_root, in_k):
        """
        Given a Binary Search Tree and a target number,
        return true if there exist two elements in the BST such that their sum is equal to the given target.

        Example 1:
        Input:
            5
           / \
          3   6
         / \   \
        2   4   7

        Target = 9

        Output: True
        Example 2:
        Input:
            5
           / \
          3   6
         / \   \
        2   4   7

        Target = 28

        Output: False
        :param in_root:
        :param in_k:
        :return:
        """
        if not in_root:
            return False
        dfs, num_set = [in_root], set()
        for i in dfs:
            if in_k - i.val in num_set:
                return True
            num_set.add(i.val)
            if i.left:
                dfs.append(i.left)
            if i.right:
                dfs.append(i.right)
        else:
            return False

    @decorate_time
    def construct_string_from_binary_tree(self, in_root):
        """
        You need to construct a string consists of parenthesis and integers from a binary tree with the preorder traversing way.
        The null node needs to be represented by empty parenthesis pair "()".
        And you need to omit all the empty parenthesis pairs that don't affect the one-to-one mapping relationship
        between the string and the original binary tree.
        Example 1:
        Input: Binary tree: [1,2,3,4]
               1
             /   \
            2     3
           /
          4
        Output: "1(2(4))(3)"
        Explanation: Originallay it needs to be "1(2(4)())(3()())",
        but you need to omit all the unnecessary empty parenthesis pairs.
        And it will be "1(2(4))(3)".
        Example 2:
        Input: Binary tree: [1,2,3,null,4]
               1
             /   \
            2     3
             \
              4
        Output: "1(2()(4))(3)"
        Explanation: Almost the same as the first example,
        except we can't omit the first parenthesis pair to break the one-to-one mapping relationship
         between the input and the output.
        :param in_root:
        :return:
        """
        if not in_root:
            return ''
        i_left = '({})'.format(self.construct_string_from_binary_tree(in_root.left) if (in_root.left or in_root.right) else '')
        i_right = '({})'.format(self.construct_string_from_binary_tree(in_root.right if in_root.right else ''))
        return '{}{}{}'.format(in_root.val, i_left, i_right)

    @decorate_time
    def convert_bst_to_greater_tree(self, in_root):
        """
        Given a Binary Search Tree (BST),
        convert it to a Greater Tree such that every key of the original
        BST is changed to the original key plus sum of all keys greater than the original key in BST.
        Example:
        Input: The root of a Binary Search Tree like this:
                      5
                    /   \
                   2     13
        Output: The root of a Greater Tree like this:
                     18
                    /   \
                  20     13
        :param in_root:
        :return:
        """
        sum_val = 0

        def visit_node(in_node):
            nonlocal sum_val
            if in_node:
                visit_node(in_node.right)
                in_node.val += sum_val
                sum_val = in_node.val
                visit_node(in_node.left)

        visit_node(in_root)
        return in_root

    @decorate_time
    def range_addition_second(self, in_m, in_n, in_ops):
        """
        Given an m * n matrix M initialized with all 0's and several update operations.
        Operations are represented by a 2D array,
        and each operation is represented by an array with two positive integers a and b,
        which means M[i][j] should be added by one for all 0 <= i < a and 0 <= j < b.
        You need to count and return the number of maximum integers in the matrix after performing all the operations.
        Example 1:
        Input:
        m = 3, n = 3
        operations = [[2,2],[3,3]]
        Output: 4
        Explanation:
        Initially, M =
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
        After performing [2,2], M =
        [[1, 1, 0],
         [1, 1, 0],
         [0, 0, 0]]
        After performing [3,3], M =
        [[2, 2, 1],
         [2, 2, 1],
         [1, 1, 1]]
        So the maximum integer in M is 2, and there are four of it in M. So return 4.
        Note:
        The range of m and n is [1,40000].
        The range of a is [1,m], and the range of b is [1,n].
        The range of operations size won't exceed 10,000.
        :param in_m:
        :param in_n:
        :param in_ops:
        :return:
        """
        if not in_ops:
            return in_m * in_n
        else:
           return min([i[0] for i in in_ops]) * min([i[1] for i in in_ops])

    @decorate_time
    def construct_the_rectangle(self, in_area):
        """
        For a web developer,
        it is very important to know how to design a web page's size.
        So, given a specific rectangular web page’s area,
        your job by now is to design a rectangular web page, whose length L and width W satisfy the following requirements:
        1. The area of the rectangular web page you designed must equal to the given target area.
        2. The width W should not be larger than the length L, which means L >= W.
        3. The difference between length L and width W should be as small as possible.
        You need to output the length L and the width W of the web page you designed in sequence.
        Example:
        Input: 4
        Output: [2, 2]
        Explanation: The target area is 4, and all the possible ways to construct it are [1,4], [2,2], [4,1].
        But according to requirement 2, [1,4] is illegal; according to requirement 3,
        [4,1] is not optimal compared to [2,2]. So the length L is 2, and the width W is 2.
        Note:
        The given area won't exceed 10,000,000 and is a positive integer
        The web page's width and length you designed must be positive integers.
        :param in_area:
        :return:
        """
        import math
        mid = math.floor(math.sqrt(in_area))
        while in_area % mid != 0:
            mid -= 1
        return [int(in_area/mid), int(mid)]

    @decorate_time
    def excel_sheet_column_number(self, in_str):
        """
        Related to question Excel Sheet Column Title
        Given a column title as appear in an Excel sheet, return its corresponding column number.
        For example:
        A -> 1
        B -> 2
        C -> 3
        ...
        Z -> 26
        AA -> 27
        AB -> 28
        Credits:
        Special thanks to @ts for adding this problem and creating all test cases.
        :param in_str:
        :return:
        """
        import functools
        return functools.reduce(lambda x, y: x * 26 + y, map(lambda x: ord(x) - ord('A') + 1, in_str))

    @decorate_time
    def minimum_moves_to_equal_array_elements(self, in_nums):
        """
        Given a non-empty integer array of size n,
        find the minimum number of moves required to make all array elements equal,
        where a move is incrementing n - 1 elements by 1.
        Example:
        Input:
        [1,2,3]
        Output:
        3
        Explanation:
        Only three moves are needed (remember each move increments two elements):
        [1,2,3]  =>  [2,3,3]  =>  [3,4,3]  =>  [4,4,4]
        [1,3,5][2,4,5][3,5,5][4,6,5][5,6,6][6,7,6][7,7,7]
        :param in_nums:
        :return:
        """
        move_steps = 0
        in_nums.sort()
        for i in range(len(in_nums)-1, -1, -1):
            if in_nums[i] == in_nums[0]:
                break
            move_steps += in_nums[i] - in_nums[0]
        return move_steps

    @decorate_time
    def best_time_to_buy_and_sell_stock(self, in_prices):
        """
        Say you have an array for which the ith element is the price of a given stock on day i.
        Design an algorithm to find the maximum profit. You may complete as many transactions as you like
        (ie, buy one and sell one share of the stock multiple times). However,
        you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
        :param in_prices:
        :return:
        """
        return sum(max(in_prices[i + 1] - in_prices[i], 0) for i in range(len(in_prices) - 1))

    @decorate_time
    def intersection_of_two_arrays(self, in_nums1, in_nums2):
        """
        Given two arrays, write a function to compute their intersection.
        Example:
        Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].
        Note:
        Each element in the result must be unique.
        The result can be in any order.
        :param in_nums1:
        :param in_nums2:
        :return:
        """
        in_sect = []
        for i in in_nums1:
            if i in in_nums2 and i not in in_sect:
                in_sect.append(i)
        return in_sect

    @decorate_time
    def ransom_note(self, in_ransom_note, in_magazine):
        """
        Given an arbitrary ransom note string and another string containing letters from all the magazines,
        write a function that will return true if the ransom note can be constructed from the magazines ;
        otherwise, it will return false.
        Each letter in the magazine string can only be used once in your ransom note.
        Note:
        You may assume that both strings contain only lowercase letters.

        canConstruct("a", "b") -> false
        canConstruct("aa", "ab") -> false
        canConstruct("aa", "aab") -> true
        :param in_ransom_note: str
        :param in_magazine: str
        :return: bool
        """
        import collections
        r_note = collections.Counter(in_ransom_note)
        mag = collections.Counter(in_magazine)
        for k, v in r_note.items():
            if mag[k] != v:
                return False
        else:
            return True

    @decorate_time
    def sum_of_left_leaves(self, in_root):
        """
        Find the sum of all left leaves in a given binary tree.
        Example:

            3
           / \
          9  20
            /  \
           15   7
        There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.
        :param in_root:
        :return:
        """
        def node_cal(in_node, in_is_left):
            if in_node:
                if in_is_left and not in_node.left and in_node.right:
                    return in_node.val
                return node_cal(in_node.left, True) + node_cal(in_node.right, False)
            return 0
        return node_cal(in_root)

    @decorate_time
    def degree_of_array(self, in_nums):
        """
        Given a non-empty array of non-negative integers nums,
        the degree of this array is defined as the maximum frequency of any one of its elements.
        Your task is to find the smallest possible length of a (contiguous) subarray of nums,
        that has the same degree as nums.
        Example 1:
        Input: [1, 2, 2, 3, 1]
        Output: 2
        Explanation:
        The input array has a degree of 2 because both elements 1 and 2 appear twice.
        Of the subarrays that have the same degree:
        [1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
        The shortest length is 2. So return 2.
        Example 2:
        Input: [1,2,2,3,1,4,2]
        Output: 6
        Note:
        nums.length will be between 1 and 50,000.
        nums[i] will be an integer between 0 and 49,999.
        :param in_nums:
        :return:
        """
        import collections
        return max([i for i in collections.Counter(in_nums).values()])

    @decorate_time
    def binary_tree_tilt(self, in_root):
        """
        Given a binary tree, return the tilt of the whole tree.
        The tilt of a tree node is defined as the absolute difference between
        the sum of all left subtree node values and the sum of all right subtree node values. Null node has tilt 0.
        The tilt of the whole tree is defined as the sum of all nodes' tilt.
        Example:
        Input:
                 1
               /   \
              2     3
        Output: 1
        Explanation:
        Tilt of node 2 : 0
        Tilt of node 3 : 0
        Tilt of node 1 : |2-3| = 1
        Tilt of binary tree : 0 + 0 + 1 = 1
        Note:
        The sum of node values in any subtree won't exceed the range of 32-bit integer.
        All the tilt values won't exceed the range of 32-bit integer.
        :param in_root:
        :return:
        """
        ans = 0

        def _sum(node):
            nonlocal ans
            if not node:
                return 0
            left, right = _sum(node.left), _sum(node.right)
            ans += abs(left - right)
            return node.val + left + right

        _sum(in_root)
        return ans

    @decorate_time
    def majority_element(self, in_nums):
        """
        Given an array of size n, find the majority element. The majority element is the element that appears more than
        ⌊ n/2 ⌋ times.
        You may assume that the array is non-empty and the majority element always exist in the array.
        Credits:
        Special thanks to @ts for adding this problem and creating all test cases.
        :param in_nums:
        :return:
        """
        import collections
        num_dict = collections.Counter(in_nums)
        num_len = len(in_nums)
        for k, v in num_dict.items():
            if v > num_len:
                return k
        else:
            return 0

    @decorate_time
    def assign_cookies(self, in_g, in_s):
        """
        Assume you are an awesome parent and want to give your children some cookies.
        But, you should give each child at most one cookie. Each child i has a greed factor gi,
        which is the minimum size of a cookie that the child will be content with;
        and each cookie j has a size sj. If sj >= gi, we can assign the cookie j to the child i,
        and the child i will be content.
        Your goal is to maximize the number of your content children and output the maximum number.
        Note:
        You may assume the greed factor is always positive.
        You cannot assign more than one cookie to one child.
        Example 1:
        Input: [1,2,3], [1,1]
        Output: 1
        Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3.
        And even though you have 2 cookies, since their size is both 1,
        you could only make the child whose greed factor is 1 content.
        You need to output 1.
        Example 2:
        Input: [1,2], [1,2,3]
        Output: 2
        Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2.
        You have 3 cookies and their sizes are big enough to gratify all of the children,
        You need to output 2.
        :param in_g:
        :param in_s:
        :return:
        """
        in_g.sort()
        in_s.sort()

        childi = 0
        cookiei = 0

        while cookiei < len(in_s) and childi < len(in_g):
            if in_s[cookiei] >= in_g[childi]:
                childi += 1
            cookiei += 1

        return childi

    @decorate_time
    def two_sum_second(self, in_nums, in_tar):
        """
        Given an array of integers that is already sorted in ascending order,
        find two numbers such that they add up to a specific target number.
        The function twoSum should return indices of the two numbers such that they add up to the target,
        where index1 must be less than index2. Please note that your returned answers (both index1 and index2)
        are not zero-based.
        You may assume that each input would have exactly one solution and you may not use the same element twice.
        Input: numbers={2, 7, 11, 15}, target=9
        Output: index1=1, index2=2
        :param in_nums:
        :param in_tar:
        :return:
        """
        for i in range(len(in_nums)):
            for j in range(i+1, len(in_nums)):
                if in_nums[i] + in_nums[j] == in_tar:
                    return i+1, j+1
        else:
            return 0, 0

    @decorate_time
    def first_unique_character_in_a_string(self, in_str):
        """
        Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
        Examples:
        s = "leetcode"
        return 0.
        s = "loveleetcode",
        return 2.
        Note: You may assume the string contain only lowercase letters.
        :param in_str:
        :return:
        """
        for i in range(len(in_str)):
            if in_str.count(in_str) == 1:
                return i
        else:
            return -1

    @decorate_time
    def minimum_absolute_difference_in_bst(self, in_root):
        """
        Given a binary search tree with non-negative values, find the minimum absolute difference between values of any two nodes.
        Example:
        Input:
           1
            \
             3
            /
           2
        Output:
        1
        Explanation:
        The minimum absolute difference is 1, which is the difference between 2 and 1 (or between 2 and 3).
        Note: There are at least two nodes in this BST.
        :param in_root:
        :return:
        """
        diff_list = []

        def dfs(in_node):
            if in_node.left:
                dfs(in_node.left)
                left_v = abs(in_node.val - in_node.left.val)
                diff_list.append(left_v)

            if in_node.right:
                dfs(in_node.right)
                right_v = abs(in_node.val - in_node.right.val)
                diff_list.append(right_v)
            return diff_list

        return min(dfs(in_root))

    @decorate_time
    def delete_node_in_a_linked_list(self, in_node):
        """
        Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.
        Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3,
        the linked list should become 1 -> 2 -> 4 after calling your function.
        :param in_node:
        :return:
        """
        in_node.val = in_node.next.val
        in_node.next = in_node.next.next

    @decorate_time
    def same_binary_trees(self, in_root1, in_root2):
        """
        Given two binary trees, write a function to check if they are the same or not.
        Two binary trees are considered the same if they are structurally identical and the nodes have the same value.
        preorder loop the binary tree
        Example 1:
        Input:     1         1
                  / \       / \
                 2   3     2   3
                [1,2,3],   [1,2,3]
        Output: true
        Example 2:
        Input:     1         1
                  /           \
                 2             2
                [1,2],     [1,null,2]
        Output: false
        Example 3:
        Input:     1         1
                  / \       / \
                 2   1     1   2
                [1,2,1],   [1,1,2]
        Output: false
        :param in_root1:
        :param in_root2:
        :return:
        """
        p_list = []

        def dfs(in_node):
            if in_node:
                p_list.append(in_node.val)
            if in_node.left:
                dfs(in_node.left)
            else:
                if in_node.right:
                    p_list.append(None)
            if in_node.right:
                dfs(in_node.right)
            return p_list

        p1_list = dfs(in_root1)
        p_list = []
        p2_list = dfs(in_root2)

        return True if p1_list == p2_list else False

    @decorate_time
    def valid_anagram(self, in_s, in_t):
        """
        Given two strings s and t, write a function to determine if t is an anagram of s.
        For example,
        s = "anagram", t = "nagaram", return true.
        s = "rat", t = "car", return false.
        Note:
        You may assume the string contains only lowercase alphabets.
        Follow up:
        What if the inputs contain unicode characters? How would you adapt your solution to such case?
        :param in_s:
        :param in_t:
        :return:
        """
        import collections
        return True if collections.Counter(in_s) == collections.Counter(in_t) else False

    @decorate_time
    def relative_ranks(self, in_nums):
        """
        Given scores of N athletes, find their relative ranks and the people with the top three highest scores,
        who will be awarded medals: "Gold Medal", "Silver Medal" and "Bronze Medal".
        Example 1:
        Input: [5, 4, 3, 2, 1]
        Output: ["Gold Medal", "Silver Medal", "Bronze Medal", "4", "5"]
        Explanation: The first three athletes got the top three highest scores, so they got "Gold Medal",
        "Silver Medal" and "Bronze Medal".
        For the left two athletes, you just need to output their relative ranks according to their scores.
        Note:
        N is a positive integer and won't exceed 10,000.
        All the scores of athletes are guaranteed to be unique.
        :param in_nums:
        :return:
        """
        for i in range(len(in_nums)):
            if i + 1 == 1:
                in_nums[i] = "Gold Medal"
            elif i + 1 == 2:
                in_nums[i] = "Silver Medal"
            elif i + 1 == 3:
                in_nums[i] = "Bronze Medal"
            else:
                in_nums[i] = str(i+1)

    @decorate_time
    def minimum_index_sum_of_two_lists(self, in_list1, in_list2):
        """
        Suppose Andy and Doris want to choose a restaurant for dinner, and they both have a list of favorite restaurants represented by strings.
        You need to help them find out their common interest with the least list index sum. If there is a choice tie between answers, output all of them with no order requirement. You could assume there always exists an answer.
        Example 1:
        Input:
        ["Shogun", "Tapioca Express", "Burger King", "KFC"]
        ["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"]
        Output: ["Shogun"]
        Explanation: The only restaurant they both like is "Shogun".
        Example 2:
        Input:
        ["Shogun", "Tapioca Express", "Burger King", "KFC"]
        ["KFC", "Shogun", "Burger King"]
        Output: ["Shogun"]
        Explanation: The restaurant they both like and have the least index sum is "Shogun" with index sum 1 (0+1).
        Note:
        The length of both lists will be in the range of [1, 1000].
        The length of strings in both lists will be in the range of [1, 30].
        The index is starting from 0 to the list length minus 1.
        No duplicates in both lists.

        :param in_list1:
        :param in_list2:
        :return:
        """
        for i in range(len(in_list1)):
            for j in range(len(in_list2)):
                if in_list1[i] == in_list2[j]:
                    return [in_list1[i]]

    @decorate_time
    def contain_duplicates(self, in_nums):
        """
        Given an array of integers, find if the array contains any duplicates.
        Your function should return true if any value appears at least twice in the array,
        and it should return false if every element is distinct.
        :param in_nums:
        :return:
        """
        for i in range(len(in_nums)):
            for j in range(i+1, len(in_nums)):
                if in_nums[i] == in_nums[j]:
                    return True
        else:
            return False

    @decorate_time
    def reverse_linked_list(self, in_head):
        """
        Reverse a singly linked list.
        :param in_head:
        :return:
        """
        prev = None
        while in_head:
            curr = in_head
            in_head = in_head.next
            curr.next = prev
            prev = curr
        return prev

    @decorate_time
    def image_smoother(self, in_img):
        """
        Given a 2D integer matrix M representing the gray scale of an image,
        you need to design a smoother to make the gray scale of each cell becomes the average gray scale (rounding down) of all
        the 8 surrounding cells and itself. If a cell has less than 8 surrounding cells, then use as many as you can.
        Example 1:
        Input:
        [[1,1,1],
         [1,0,1],
         [1,1,1]]
        Output:
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
        Explanation:
        For the point (0,0), (0,2), (2,0), (2,2): floor(3/4) = floor(0.75) = 0
        For the point (0,1), (1,0), (1,2), (2,1): floor(5/6) = floor(0.83333333) = 0
        For the point (1,1): floor(8/9) = floor(0.88888889) = 0
        Note:
        The value in the given matrix is in the range of [0, 255].
        The length and width of the given matrix are in the range of [1, 150].
        :param in_img:
        :return:
        """
        from copy import deepcopy
        x_len = len(in_img)
        y_len = len(in_img[0]) if x_len else 0
        res = deepcopy(in_img)
        for x in range(x_len):
            for y in range(y_len):
                neighbors = [
                    in_img[_x][_y]
                    for _x in (x - 1, x, x + 1)
                    for _y in (y - 1, y, y + 1)
                    if 0 <= _x < x_len and 0 <= _y < y_len
                ]
                res[x][y] = sum(neighbors) // len(neighbors)
        return res

    @decorate_time
    def number_of_boomerangs(self, in_points):
        """
        Given n points in the plane that are all pairwise distinct, a "boomerang" is a tuple of points (i, j, k)
        such that the distance between i and j equals the distance between i and k (the order of the tuple matters).
        Find the number of boomerangs. You may assume that n will be at most 500 and coordinates of points are all
        in the range [-10000, 10000] (inclusive).
        Example:
        Input:
        [[0,0],[1,0],[2,0]]
        Output:
        2
        Explanation:
        The two boomerangs are [[1,0],[0,0],[2,0]] and [[1,0],[2,0],[0,0]]
        :param in_points:
        :return:
        Put each point p at the center of boomrang
        cmap[k] are number of points at distance k
        p at the center leaves two places open in boomrang
        cmap[k] * (cmap[k] - 1) = cmap[k] P 2 or cmap[k]!/(cmap[k]-2)!
        """
        res = 0
        for p in in_points:
            cmap = {}
            for q in in_points:
                f = p[0]-q[0]
                s = p[1]-q[1]
                cmap[f*f + s*s] = 1 + cmap.get(f*f + s*s, 0)
            for k in cmap:
                res += cmap[k] * (cmap[k] - 1)
        return res

    @decorate_time
    def longest_palindrome_case_sensitive(self, in_str):
        """
        Given a string which consists of lowercase or uppercase letters,
        find the length of the longest palindromes that can be built with those letters.
        This is case sensitive, for example "Aa" is not considered a palindrome here.
        Note:
        Assume the length of given string will not exceed 1,010.
        Example:
        Input:
        "abccccdd"
        Output:
        7
        Explanation:
        One longest palindrome that can be built is "dccaccd", whose length is 7.
        :param in_str:
        :return:
        """
        import collections
        longest_len = 0
        have_odd = 0
        alp_dict = collections.Counter(in_str)
        for i in alp_dict.values():
            if i % 2 == 0:
                longest_len += i
            else:
                longest_len += (i - 1)
                have_odd = 1
        return longest_len if have_odd == 0 else longest_len + 1

    @decorate_time
    def maximum_product_of_three_numbers(self, in_nums):
        """
        Given an integer array, find three numbers whose product is maximum and output the maximum product.
        Example 1:
        Input: [1,2,3]
        Output: 6
        Example 2:
        Input: [1,2,3,4]
        Output: 24
        Note:
        The length of the given array will be in range [3,104] and all elements are in the range [-1000, 1000].
        Multiplication of any three numbers in the input won't exceed the range of 32-bit signed integer.
        :param in_nums:
        :return:
        """
        from copy import deepcopy
        from functools import reduce
        import operator
        if len(in_nums) < 3:
            return -1
        cop_in_nums = deepcopy(in_nums)
        cop_in_nums.sort()
        return reduce(operator.mul, cop_in_nums[-3:])

    @decorate_time
    def binary_watch(self, in_num):
        """
        A binary watch has 4 LEDs on the top which represent the hours (0-11),
        and the 6 LEDs on the bottom represent the minutes (0-59).
        Each LED represents a zero or one, with the least significant bit on the right.
        For example, the above binary watch reads "3:25".
        Given a non-negative integer n which represents the number of LEDs that are currently on,
        return all possible times the watch could represent.
        Example:
        Input: n = 1
        Return: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
        Note:
        The order of output does not matter.
        The hour must not contain a leading zero, for example "01:00" is not valid, it should be "1:00".
        The minute must be consist of two digits and may contain a leading zero, for example "10:2" is not valid,
        it should be "10:02".
        :param in_num:
        :return:
        """
        if in_num < 1 or in_num > 10:
            return []
        return ['%d:%02d' % (h, m) for h in range(12) for m in range(60) if (bin(h) + bin(m)).count('1') == in_num]

    @decorate_time
    def intersection_of_two_arrays_ii(self, in_nums1, in_nums2):
        """
        Given two arrays, write a function to compute their intersection.
        Example:
        Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].
        Note:
        Each element in the result should appear as many times as it shows in both arrays.
        The result can be in any order.
        Follow up:
        What if the given array is already sorted? How would you optimize your algorithm?
        What if nums1's size is small compared to nums2's size? Which algorithm is better?
        What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?
        :param in_nums1:
        :param in_nums2:
        :return:
        """
        from collections import Counter
        c1, c2 = Counter(in_nums1), Counter(in_nums2)
        return sum([[num] * min(c1[num], c2[num]) for num in c1 & c2], [])

    @decorate_time
    def diameter_of_binary_tree(self, in_root):
        """
        Given a binary tree, you need to compute the length of the of the tree.
        The diameter of a binary tree is the length of the longest path between any two nodes in a tree.
        This path may or may not pass through the root.
        Example:
        Given a binary tree
                  1
                 / \
                2   3
               / \
              4   5
        Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].
        Note: The length of path between two nodes is represented by the number of edges between them.
        :param in_root:
        :return:
        """
        tree_dep = 1

        def dfs(in_node):
            nonlocal tree_dep
            if not in_root:
                return 0
            else:
                left_dep = dfs(in_node.left)
                right_dep = dfs(in_node.right)
                tree_dep = max(tree_dep, left_dep + right_dep + 1)
                return 1 + max(left_dep, right_dep)
        dfs(in_root)
        return tree_dep - 1

    @decorate_time
    def missing_number(self, in_nums):
        """
        Given an array containing n distinct numbers taken from 0, 1, 2, ..., n,
        find the one that is missing from the array.
        For example,
        Given nums = [0, 1, 3] return 2.
        Note:
        Your algorithm should run in linear runtime complexity.
        Could you implement it using only constant extra space complexity?
        Credits:
        Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
        :param in_nums:
        :return:
        """
        return (in_nums[-1]*(in_nums[-1]+1))//2 - sum(in_nums)

    @decorate_time
    def student_attendance_record_i(self, in_str):
        """
        You are given a string representing an attendance record for a student.
        The record only contains the following three characters:
        'A' : Absent.
        'L' : Late.
        'P' : Present.
        A student could be rewarded if his attendance record doesn't contain more than
        one 'A' (absent) or more than two continuous 'L' (late).
        You need to return whether the student could be rewarded according to his attendance record.
        Example 1:
        Input: "PPALLP"
        Output: True
        Example 2:
        Input: "PPALLL"
        Output: False
        :param in_str:
        :return:
        """
        import collections
        p_dict = collections.Counter(in_str)
        return False if p_dict['A'] > 1 or p_dict['L'] > 2 else True

    @decorate_time
    def base_7(self, in_num):
        """
        Given an integer, return its base 7 string representation.
        Example 1:
        Input: 100
        Output: "202"
        Example 2:
        Input: -7
        Output: "-10"
        Note: The input will be in range of [-1e7, 1e7].
        :param in_num:
        :return:
        """
        if in_num < 0:
            return '-' + self.base_7(-in_num)
        if in_num < 7:
            return str(in_num)
        return self.base_7(in_num//7) + str(in_num%7)

    @decorate_time
    def reverse_string_ii(self, in_str, in_k):
        """
        Given a string and an integer k,
        you need to reverse the first k characters for every 2k characters counting from the start of the string.
        If there are less than k characters left, reverse all of them.
        If there are less than 2k but greater than or equal to k characters,
        then reverse the first k characters and left the other as original.
        Example:
        Input: s = "abcdefg", k = 2
        Output: "bacdfeg"
        Restrictions:
        The string consists of lower English letters only.
        Length of the given string and k will in the range [1, 10000]
        :param in_str:
        :param in_k:
        :return:
        """
        res_str = ''
        for i in range(0, len(in_str), in_k):
            if i % in_k == 0:
                res_str += in_str[i:i+in_k][::-1]
            else:
                res_str += in_str[i:i + in_k]
        return res_str

    @decorate_time
    def longest_continuous_increasing_subsequence(self, in_nums):
        """
        Given an unsorted array of integers, find the length of longest continuous increasing subsequence.
        Example 1:
        Input: [1,3,5,4,7]
        Output: 3
        Explanation: The longest continuous increasing subsequence is [1,3,5], its length is 3.
        Even though [1,3,5,7] is also an increasing subsequence, it's not a continuous one where 5 and 7 are separated by 4.
        Example 2:
        Input: [2,2,2,2,2]
        Output: 1
        Explanation: The longest continuous increasing subsequence is [2], its length is 1.
        Note: Length of the array will not exceed 10,000.
        :param in_nums:
        :return:
        """
        longest_len = 0
        for i in range(len(in_nums)):
            temp_len = 1
            for j in range(i, len(in_nums) - 1):
                if in_nums[j] < in_nums[j+1]:
                    temp_len += 1
            longest_len = temp_len if temp_len > longest_len else longest_len
        return longest_len

    @decorate_time
    def convert_sorted_array_to_binary_search_tree(self, in_nums):
        """
        Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
        :param in_nums:
        :return:
        """
        if not in_nums:
            return None
        mid = len(in_nums) // 2
        root = TreeNode(in_nums[mid])
        root.left = self.convert_sorted_array_to_binary_search_tree(in_nums[:mid])
        root.right = self.convert_sorted_array_to_binary_search_tree(in_nums[mid+1:])
        return root

    @decorate_time
    def find_pivot_index(self, in_nums):
        """
        Given an array of integers nums, write a method that returns the "pivot" index of this array.
        We define the pivot index as the index where the sum of the numbers to the left of the index
        is equal to the sum of the numbers to the right of the index.
        If no such index exists, we should return -1. If there are multiple pivot indexes,
        you should return the left-most pivot index.
        Example 1:
        Input:
        nums = [1, 7, 3, 6, 5, 6]
        Output: 3
        Explanation:
        The sum of the numbers to the left of index 3 (nums[3] = 6) is equal to the sum of numbers to the right of index 3.
        Also, 3 is the first index where this occurs.
        Example 2:
        Input:
        nums = [1, 2, 3]
        Output: -1
        Explanation:
        There is no index that satisfies the conditions in the problem statement.
        Note:

        The length of nums will be in the range [0, 10000].
        Each element nums[i] will be an integer in the range [-1000, 1000].

        :param in_nums:
        :return:
        """
        for i in range(1, len(in_nums)-1):
            if sum(in_nums[:i]) == sum(in_nums[i+1:]):
                return i
        else:
            return -1

    @decorate_time
    def second_minimum_node_in_a_binary_tree(self, in_root):
        """
        Given a non-empty special binary tree consisting of nodes with the non-negative value,
        where each node in this tree has exactly two or zero sub-node.
        If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes.
        Given such a binary tree,
        you need to output the second minimum value in the set made of all the nodes' value in the whole tree.
        If no such second minimum value exists, output -1 instead.
        Example 1:
        Input:
            2
           / \
          2   5
             / \
            5   7
        Output: 5
        Explanation: The smallest value is 2, the second smallest value is 5.
        Example 2:
        Input:
            2
           / \
          2   2
        Output: -1
        Explanation: The smallest value is 2, but there isn't any second smallest value.
        :param in_root:
        :return:
        """
        res = [float('inf')]

        def traverse(node):
            if not node:
                return
            if in_root.val < node.val < res[0]:
                res[0] = node.val
            traverse(node.left)
            traverse(node.right)
        traverse(in_root)
        return -1 if res[0] == float('inf') else res[0]

    @decorate_time
    def add_strings(self, in_str1, in_str2):
        """
        Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.
        Note:
        The length of both num1 and num2 is < 5100.
        Both num1 and num2 contains only digits 0-9.
        Both num1 and num2 does not contain any leading zero.
        You must not use any built-in BigInteger library or convert the inputs to integer directly.
        :param in_str1: str
        :param in_str2: str
        :return: str
        """
        import itertools
        res_str = ''
        left_i = 0
        for i, j in itertools.zip_longest(in_str1[::-1], in_str2[::-1], fillvalue='0'):
            temp = int(i) + int(j) + left_i
            if temp >= 10:
                temp -= 10
                left_i = 1
            else:
                left_i = 0
            res_str += str(temp)
        return res_str[::-1]

    @decorate_time
    def happy_number(self, in_num):
        """
        Write an algorithm to determine if a number is "happy".
        A happy number is a number defined by the following process:
        Starting with any positive integer,
        replace the number by the sum of the squares of its digits,
        and repeat the process until the number equals 1 (where it will stay),
        or it loops endlessly in a cycle which does not include 1.
        Those numbers for which this process ends in 1 are happy numbers.
        Example: 19 is a happy number
        1**2 + 9**2 = 82
        8**2 + 2**2 = 68
        6**2 + 8**2 = 100
        1**2 + 0**2 + 0**2 = 1
        Credits:
        Special thanks to @mithmatt and @ts for adding this problem and creating all test cases.
        :param in_num:
        :return:
        """

        def dig_squ(num):
            sum_i = 0
            while num > 0:
                i = num % 10
                sum_i += i**2
                num //= 10
            return sum_i

        slow, fast = in_num, in_num
        slow = dig_squ(slow)
        fast = dig_squ(fast)
        fast = dig_squ(fast)

        while slow != fast:
            slow = dig_squ(slow)
            fast = dig_squ(fast)
            fast = dig_squ(fast)
        if slow == 1:
            return True
        else:
            return False

    @decorate_time
    def binary_tree_level_order_traversal_ii(self, in_root):
        """
        Given a binary tree, return the bottom-up level order traversal of its nodes' values.
        (ie, from left to right, level by level from leaf to root).
        For example:
        Given binary tree [3,9,20,null,null,15,7],
            3
           / \
          9  20
            /  \
           15   7
        return its bottom-up level order traversal as:
        [
          [15,7],
          [9,20],
          [3]
        ]
        :param in_root:
        :return:
        """
        info = []

        def dfs(in_node, in_level=0):
            if in_node:
                if len(info) <= in_level:
                    info.append([])
                    info[in_level].append(in_node.val)
                dfs(in_node.left, in_level + 1)
                dfs(in_node.right, in_level + 1)

        dfs(in_root)
        return info[::-1]

    @decorate_time
    def longest_harmonious_subsequence(self, in_nums):
        """
        We define a harmonious array is an array where the difference between its maximum value and its minimum value is exactly 1.
        Now, given an integer array, you need to find the length of its longest harmonious subsequence among all its possible subsequences.
        Example 1:
        Input: [1,3,2,2,5,2,3,7]
        Output: 5
        Explanation: The longest harmonious subsequence is [3,2,2,2,3].
        Note: The length of the input array will not exceed 20,000.
        :param in_nums:
        :return:
        """
        import collections
        count = collections.Counter(in_nums)
        ans = 0
        for x in count:
            if x + 1 in count:
                ans = max(ans, count[x] + count[x + 1])
        return ans

    @decorate_time
    def convert_a_number_to_hexadecimal(self, in_num):
        """
        Given an integer, write an algorithm to convert it to hexadecimal. For negative integer,
        two’s complement method is used.
        Note:
        All letters in hexadecimal (a-f) must be in lowercase.
        The hexadecimal string must not contain extra leading 0s. If the number is zero,
        it is represented by a single zero character '0'; otherwise,
        the first character in the hexadecimal string will not be the zero character.
        The given number is guaranteed to fit within the range of a 32-bit signed integer.
        You must not use any method provided by the library which converts/formats the number to hex directly.
        Example 1:
        Input:
        26
        Output:
        "1a"
        Example 2:
        Input:
        -1
        Output:
        "ffffffff"
        :param in_num:
        :return:
        """
        return ''.join('0123456789abcdef'[(in_num >> 4 * i) & 15] for i in range(8))[::-1].lstrip('0') or '0'

    @decorate_time
    def subtree_of_another_tree_one(self, in_s, in_t):
        """
        Given two non-empty binary trees s and t,
        check whether tree t has exactly the same structure and node values with a subtree of s.
        A subtree of s is a tree consists of a node in s and all of this node's descendants.
        The tree s could also be considered as a subtree of itself.
        Example 1:
        Given tree s:
             3
            / \
           4   5
          / \
         1   2
        Given tree t:
           4
          / \
         1   2
        Return true, because t has the same structure and node values with a subtree of s.
        Example 2:
        Given tree s:
             3
            / \
           4   5
          / \
         1   2
            /
           0
        Given tree t:
           4
          / \
         1   2
        Return false.
        :param in_s:
        :param in_t:
        :return:
        """
        def is_match(s, t):
            if not (s and t):
                return s is t
            return s.val == t.val and is_match(s.left, t.left) and is_match(s.right, t.right)

        def is_sub_tree(s, t):
            if is_match(s, t):
                return True
            if not s:
                return False
            return is_sub_tree(s.left, t) or is_sub_tree(s.right, t)

        return is_sub_tree(in_s, in_t)

    @decorate_time
    def subtree_of_another_tree_two(self, in_s, in_t):
        """
        For each node in a tree, we can create node.merkle, a hash representing it's subtree.
        This hash is formed by hashing the concatenation of the merkle of the left child,
        the node's value,
        and the merkle of the right child.
        Then, two trees are identical if and only if the merkle hash of their roots are equal
        (except when there is a hash collision.) From there, finding the answer is straightforward:
        we simply check if any node in s has node.merkle == t.merkle
        :param in_s:
        :param in_t:
        :return:
        """
        from hashlib import sha256

        def hash_sha256(x):
            sha_ = sha256()
            sha_.update(x)
            return sha_.hexdigest()

        def merkle_node(node):
            if not node:
                return '#'
            m_left = merkle_node(node.left)
            m_right = merkle_node(node.right)
            node.merkle = hash_sha256(m_left + node.val + m_right)
            return node

        merkle_node(in_s)
        merkle_node(in_t)

        def dfs(node):
            if not node:
                return False
            else:
                return node.merkle == in_t.merkle or dfs(node.left) or dfs(node.right)

    @decorate_time
    def climbing_stairs(self, in_n):
        """
        You are climbing a stair case. It takes n steps to reach to the top.
        Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
        Note: Given n will be a positive integer.
        Example 1:
        Input: 2
        Output:  2
        Explanation:  There are two ways to climb to the top.
        1. 1 step + 1 step
        2. 2 steps
        Example 2:
        Input: 3
        Output:  3
        Explanation:  There are three ways to climb to the top.
        1. 1 step + 1 step + 1 step
        2. 1 step + 2 steps
        3. 2 steps + 1 step
        :param in_n:
        :return:
        """
        a = b = 1
        for _ in range(in_n):
            a, b = b, a+b
        return a

    @decorate_time
    def climbing_stairs_two(self, in_n):
        if in_n == 1:
            return 1
        if in_n == 2:
            return 2
        return self.climbing_stairs_two(in_n-1) + self.climbing_stairs_two(in_n-2)

    @decorate_time
    def power_of_three(self, in_n):
        """
        Given an integer, write a function to determine if it is a power of three.
        Follow up:
        Could you do it without using any loop / recursion?
        Credits:
        Special thanks to @dietpepsi for adding this problem and creating all test cases.
        :param in_n:
        :return:
        """
        if in_n <= 0 or in_n > 3**19:
            return False
        else:
            return in_n % 3

    @decorate_time
    def path_sum_three_one(self, in_root, in_sum):
        """
        You are given a binary tree in which each node contains an integer value.
        Find the number of paths that sum to a given value.
        The path does not need to start or end at the root or a leaf,
        but it must go downwards (traveling only from parent nodes to child nodes).
        The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.
        Example:
        root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
              10
             /  \
            5   -3
           / \    \
          3   2   11
         / \   \
        3  -2   1
        Return 3. The paths that sum to 8 are:
        1.  5 -> 3
        2.  5 -> 2 -> 1
        3. -3 -> 11
        :param in_root:
        :param in_sum:
        :return:
        """
        def find_path(node, i_sum):
            if node:
                return int(node.val == i_sum) + find_path(node.left, i_sum - node.val) +\
                       find_path(node.right, i_sum - node.val)
            return 0

        def path_sum(root, target):
            if root:
                return find_path(root, target) + path_sum(root.left, target) + path_sum(root.right, target)
            return 0
        return path_sum(in_root, in_sum)

    @decorate_time
    def path_sum_three_two(self, in_root, in_sum):
        """
        A more efficient implementation uses the Two Sum idea. It uses a hash table (extra memory of order N).
        With more space, it gives us an O(N) complexity.
        As we traverse down the tree, at an arbitrary node N, we store the sum until this node N
        (sum_so_far (prefix) + N.val). in hash-table. Note this sum is the sum from root to N.
        Now at a grand-child of N, say G,
        we can compute the sum from the root until G since we have the prefix_sum until this grandchild available.
        We pass in our recursive routine.
        How do we know if we have a path of target sum which ends at this grand-child G?
        Say there are multiple such paths that end at G and say they start at A, B, C where A,B,C are predecessors of G.
        Then sum(root->G) - sum(root->A) = target. Similarly sum(root->G)-sum(root>B) = target.
        Therefore we can compute the complement at G as sum_so_far+G.
        val-target and look up the hash-table for the number of paths which had this sum
        Now after we are done with a node and all its grandchildren, we remove it from the hash-table.
        This makes sure that the number of complement paths returned always correspond to paths that ended at
        a predecessor node.
        :param in_root:
        :param in_sum:
        :return:
        """

        def helper(root, target, so_far, cache):
            if root:
                complement = so_far + root.val - target
                if complement in cache:
                    self.result += cache[complement]
                cache.setdefault(so_far + root.val, 0)
                cache[so_far + root.val] += 1
                self.helper(root.left, target, so_far + root.val, cache)
                self.helper(root.right, target, so_far + root.val, cache)
                cache[so_far + root.val] -= 1
            else:
                return 0

        def path_sum(root, i_sum):
            result = 0
            helper(root, i_sum, result, {0: 1})
            return result

        return path_sum(in_root, in_sum)

    @decorate_time
    def set_mismatch(self, in_nums):
        """
        The set S originally contains numbers from 1 to n. But unfortunately,
        due to the data error, one of the numbers in the set got duplicated to another number in the set,
        which results in repetition of one number and loss of another number.
        Given an array nums representing the data status of this set after the error.
        Your task is to firstly find the number occurs twice and then find the number that is missing.
        Return them in the form of an array.
        Example 1:
        Input: nums = [1,2,2,4]
        Output: [2,3]
        Note:
        The given array size will in the range [2, 10000].
        The given array's numbers won't have any order.
        :param in_nums:
        :return:
        """
        N = len(in_nums)
        alpha = sum(in_nums) - N * (N + 1) / 2
        beta = (sum(x * x for x in in_nums) - N * (N + 1) * (2 * N + 1) / 6) / alpha
        return (alpha + beta) / 2, (beta - alpha) / 2

    @decorate_time
    def self_dividing_numbers(self, in_left, in_right):
        """
        A self-dividing number is a number that is divisible by every digit it contains.
        For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.
        Also, a self-dividing number is not allowed to contain the digit zero.
        Given a lower and upper number bound, output a list of every possible self dividing number,
        including the bounds if possible.
        Example 1:
        Input:
        left = 1, right = 22
        Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
        Note:
        The boundaries of each input argument are 1 <= left <= right <= 10000.
        :param in_left:
        :param in_right:
        :return:
        """
        res_list = []
        from copy import deepcopy

        def is_self_div(in_num):
            is_div = False
            i_in_num = deepcopy(in_num)
            while i_in_num > 0:
                if in_num % (i_in_num % 10) == 0:
                    is_div = True
                else:
                    return False
                i_in_num //= 10
            return is_div

        for i in range(in_left, in_right+1):
            if '0' not in str(i):
                if is_self_div(i):
                    res_list.append(i)
        return res_list

    @decorate_time
    def binary_number_with_alternating_bits(self, in_num):
        """
        Given a positive integer, check whether it has alternating bits: namely,
        if two adjacent bits will always have different values.
        Example 1:
        Input: 5
        Output: True
        Explanation:
        The binary representation of 5 is: 101
        Example 2:
        Input: 7
        Output: False
        Explanation:
        The binary representation of 7 is: 111.
        Example 3:
        Input: 11
        Output: False
        Explanation:
        The binary representation of 11 is: 1011.
        Example 4:
        Input: 10
        Output: True
        Explanation:
        The binary representation of 10 is: 1010.
        :param in_num:
        :return:
        """
        if in_num <= 1:
            return False
        xor_bit = 0 if (in_num & 1) else 1
        while in_num > 0:
            if (in_num & 1) ^ xor_bit:
                xor_bit = in_num & 1
            else:
                return False
            in_num >>= 1
        else:
            return True

    @decorate_time
    def longest_univalue_path(self, in_root):
        """
        Given a binary tree, find the length of the longest path where each node in the path has the same value.
        This path may or may not pass through the root.
        Note: The length of path between two nodes is represented by the number of edges between them.
        Example 1:
        Input:
                      5
                     / \
                    4   5
                   / \   \
                  1   1   5
        Output:
        2
        Example 2:
        Input:
                      1
                     / \
                    4   5
                   / \   \
                  4   4   5
        Output:
        2
        Note: The given binary tree has not more than 10000 nodes. The height of the tree is not more than 1000.
        :param in_root:
        :return:
        """
        longest_path = [0]

        def dfs(node):
            if not node:
                return 0
            left_len, right_len = dfs(node.left), dfs(node.right)
            left = (left_len + 1) if node.left and node.val == node.left.val else 0
            right = (right_len + 1) if node.right and node.val == node.right.val else 0
            longest_path[0] = max(longest_path[0], left + right)
            return max(left, right)
        dfs(in_root)
        return longest_path[0]

    @decorate_time
    def repeated_string_match(self, in_a, in_b):
        """
        Given two strings A and B,
        find the minimum number of times A has to be repeated such that B is a substring of it.
        If no such solution, return -1.
        For example, with A = "abcd" and B = "cdabcdab".
        Return 3, because by repeating A three times (“abcdabcdabcd”),
        B is a substring of it; and B is not a substring of A repeated two times ("abcdabcd").
        Note:
        The length of A and B will be between 1 and 10000.
        :param in_a:
        :param in_b:
        :return:
        """
        from collections import Counter
        a_d = Counter(in_a)
        b_d = Counter(in_b)
        if not all([i for i in map(lambda i: i in [j for j in a_d.keys()], b_d.keys())]):
            return -1
        rep_cnt = 1
        while rep_cnt < 10000:
            tmp_str = in_a * rep_cnt
            if in_b in tmp_str:
                return rep_cnt
            else:
                rep_cnt += 1
        else:
            return -1

    @decorate_time
    def valid_palindrome_two(self, in_str):
        """
        Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome.
        Example 1:
        Input: "aba"
        Output: True
        Example 2:
        Input: "abcdca"
        Output: True
        Explanation: You could delete the character 'c'.
        :param in_str:
        :return:
        """
        start_p = 0
        end_p = len(in_str) - 1
        jump_cnt = 0
        while start_p < len(in_str)//2:
            if in_str[start_p] == in_str[end_p]:
                start_p += 1
                end_p -= 1
            else:
                jump_cnt += 1
                if in_str[start_p] == in_str[end_p-1]:
                    end_p -= 1
                elif in_str[start_p+1] == in_str[end_p]:
                    start_p += 1
                else:
                    return False
        if jump_cnt <= 1:
            return True
        else:
            return False

    @decorate_time
    def non_decreasing_array(self, in_nums):
        """
        Given an array with n integers,
        your task is to check if it could become non-decreasing by modifying at most 1 element.
        We define an array is non-decreasing if array[i] <= array[i + 1] holds for every i (1 <= i < n).
        Example 1:
        Input: [4,2,3]
        Output: True
        Explanation: You could modify the first
        4
         to
        1
         to get a non-decreasing array.
        Example 2:
        Input: [4,2,1]
        Output: False
        Explanation: You can't get a non-decreasing array by modify at most one element.
        Note: The n belongs to [1, 10,000].
        :param in_nums:
        :return:
        """
        re_cnt = 0
        for i in range(1, len(in_nums)):
            if in_nums[i] < in_nums[i-1]:
                re_cnt += 1
        if re_cnt <= 1:
            return True
        else:
            return False


if __name__ == '__main__':
    sol = Solution1()
    # test_list = ['xushiyin4', 'xushiyin_yu', 'xushiyin4556', 'xushiyi0']
    # test_parent = '{{dd{ee}}[ff](123)}'
    # test_dup_int_list = [1, 6, 6, 6, 6, 6, 7, 7, 7]

    # print(sol.hamming_distance(1, 4))
    # print(sol.judge_route_circle('UDLR'))
    # print(sol.judge_route_circle('LLRRUUUUUDD'))
    # print(sol.two_sum_add([2, 7, 12, 15], 27))
    # print(sol.is_palindrome(121))
    # print(sol.longest_common_prefix(test_list))
    # print(sol.valid_parentheses(test_parent))
    #
    # test_list1 = produce_random_list(10, 1)
    # loop_print_linked_value(test_list1)
    # test_list2 = produce_random_list(10, 2)
    # loop_print_linked_value(test_list2)
    # merge_list = sol.merge_two_sorted_lists(test_list1, test_list2)
    # loop_print_linked_value(merge_list)
    #
    # print(sol.remove_dup_from_sorted_array(test_dup_int_list))
    # print(sol.remove_element(test_dup_int_list, 6))
    # print(test_dup_int_list)
    # test_h_str = 'hello'
    # test_n_str = 'll'
    # print(sol.c_str_str(test_h_str, test_n_str))
    # test_s_list = [1, 3, 4, 7, 8, 9]
    # print(sol.search_insert_position(test_s_list, 4))
    # l1 = produce_node_list([1, 5, 3])
    # l2 = produce_node_list([3, 2, 1])
    # loop_print_linked_value(sol.add_two_numbers(l1, l2))
    # print(sol.count_and_say(5))
    # root1 = yield_nodes_tree(i_rev=1)
    # root2 = yield_nodes_tree()
    # sum_root = sol.merge_two_binary_trees(root1, root2)
    # loop_binary_tree_nodes(sum_root)
    # print(sol.number_complement(10))
    # print(sol.reverse_words_in_string("Let's take LeetCode contest"))
    # print(sol.fizz_buzz_fizzbuzz(15))
    # nums1 = [4, 1, 2]
    # nums2 = [1, 3, 4, 2]
    # nums1 = [2, 4]
    # nums2 = [1, 2, 3, 4]
    # print(sol.next_greater_element_1(nums1, nums2))
    # test_grid = [[0, 1, 0, 0],
    #              [1, 1, 1, 0],
    #              [0, 1, 0, 0],
    #              [1, 1, 0, 1]]
    # print(sol.island_perimeter(test_grid))
    # import operator
    # print(sum(map(operator.ne, [0] + test_grid[0], test_grid[0] + [0])))
    # print(sol.binary_number_with_alternating_bits(7))
    # print(sol.max_consecutive_ones([1, 1, 0, 0, 0, 0, 1, 1]))
    # o_root = yield_nodes_tree()
    # print(sol.maximum_depth_of_binary_tree(o_root))
    # print(sol.add_binary('101', '101'))
    # print(sol.array_digits_plus_one([1, 0, 1, 1]))
    # print(sol.sqrt_imp(15))
    # print(sol.one_bit_and_two_bit_characters([1,0,1,1,1,0,0,0]))
    # test_arr = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
    #             [0,0,0,0,0,0,0,1,1,1,0,0,0],
    #             [0,1,1,0,1,0,0,0,0,0,0,0,0],
    #             [0,1,0,0,1,1,0,0,1,0,1,0,0],
    #             [0,1,0,0,1,1,0,0,1,1,1,0,0],
    #             [0,0,0,0,0,0,0,0,0,0,1,0,0],
    #             [0,0,0,0,0,0,0,1,1,1,1,0,0],
    #             [0,0,0,0,0,0,0,1,1,0,0,0,0]
    #             ]
    # print(sol.max_area_of_island(test_arr))
    # print(sol.detect_capital('USAa'))
    # print(sol.count_binary_substrings("00110011"))
    # test_l = [4,3,2,7,8,2,3,1]
    # print(sol.find_all_numbers_disappeared_in_an_array(test_l))
    # print(sol.find_the_difference('abcd', 'abcdef'))
    # print(sol.move_zeros([0, 1, 0, 3, 0, 1, 12]))
    # print(sol.construct_the_rectangle(24))
    # print(sol.range_addition_second(5, 4, [[3,2], [3,3]]))
    # print(sol.two_sum_second([2, 7, 11, 15], 9))
    # print(sol.number_of_boomerangs([[0,0],[1,0],[2,0]]))
    # print(sol.longest_palindrome_case_sensitive("abbccccdd"))
    # print(sol.maximum_product_of_three_numbers([1,2,3]))
    # print(sol.intersection_of_two_arrays_ii([1, 2, 2, 1], [2, 2]))
    # print(sol.reverse_string_ii('abcdefg', 2))
    # print(sol.longest_continuous_increasing_subsequence([2,2,2,2,2]))
    # print(sol.find_pivot_index([2,2,2,2,2]))
    # print(sol.add_strings('1291', '67'))
    # print(sol.happy_number(19))
    # print(sol.convert_a_number_to_hexadecimal(16))
    # print(sol.self_dividing_numbers(1, 22))
    # print(sol.binary_number_with_alternating_bits(5))
    # print(sol.repeated_string_match('acbdbca', 'cdabcdab'))
    # print(sol.valid_palindrome_two('acffdfdca'))
    root = yield_nodes_tree()
    for i in level_output_tree(TreeNode(1)):
        print(i, sep=' ', end=' ')





