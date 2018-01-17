#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time : 2017/11/21 上午10:27
@Author : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site : 
@File : leetcode2.py
@desc : 101---200
"""
from algorithm_files.leetcode1 import decorate_time


class Solution2(object):
    def __init__(self):
        self.logger_dict = dict()

    @decorate_time
    def maximum_average_subarray_one(self, in_nums, in_key):
        """
        Given an array consisting of n integers,
        find the contiguous subarray of given length k that has the maximum average value.
        And you need to output the maximum average value.
        Example 1:
        Input: [1,12,-5,-6,50,3], k = 4
        Output: 12.75
        Explanation: Maximum average is (12-5-6+50)/4 = 51/4 = 12.75
        Note:
        1 <= k <= n <= 30,000.
        Elements of the given array will be in the range [-10,000, 10,000].
        :param in_nums:
        :param in_key:
        :return:
        """
        max_ave = 0
        for i in range(len(in_nums) - in_key):
            max_ave = max(max_ave, sum(in_nums[i:i + in_key]) / in_key)
        return max_ave

    @decorate_time
    def flood_fill(self, in_image, in_sr, in_sc, in_new_color):
        """
        An image is represented by a 2-D array of integers, each integer representing the pixel value of
        the image (from 0 to 65535).
        Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill,
        and a pixel value newColor, "flood fill" the image.
        To perform a "flood fill", consider the starting pixel,
        plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel,
        plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel),
        and so on. Replace the color of all of the aforementioned pixels with the newColor.
        At the end, return the modified image.
        Example 1:
        Input:
        image = [[1,1,1],[1,1,0],[1,0,1]]
        sr = 1, sc = 1, newColor = 2
        Output: [[2,2,2],[2,2,0],[2,0,1]]
        Explanation:
        From the center of the image (with position (sr, sc) = (1, 1)), all pixels connected
        by a path of the same color as the starting pixel are colored with the new color.
        Note the bottom corner is not colored 2, because it is not 4-directionally connected
        to the starting pixel.
        Note:
        The length of image and image[0] will be in the range [1, 50].
        The given starting pixel will satisfy 0 <= sr < image.length and 0 <= sc < image[0].length.
        The value of each color in image[i][j] and newColor will be an integer in [0, 65535].
        :param in_image:
        :param in_sr:
        :param in_sc:
        :param in_new_color:
        :return:
        """
        rows, columns, orign_color = len(in_image), len(in_image[0]), in_image[in_sr][in_sc]

        def dfs(row, col):
            if (not (0 <= row < rows and 0 <= col < columns)) or in_image[row][col] != orign_color:
                return
            in_image[row][col] = in_new_color
            [dfs(row + x, col + y) for x, y in [(0, 1), (1, 0), (0, -1), (-1, 0)]]

        if orign_color != in_new_color:
            dfs(in_sr, in_sc)
        return in_image

    @decorate_time
    def longest_word_in_dictionary(self, in_str_s):
        """
        Given a list of strings words representing an English Dictionary,
        find the longest word in words that can be built one character at a time by other words in words.
        If there is more than one possible answer, return the longest word with the smallest lexicographical order.
        If there is no answer, return the empty string.
        Example 1:
        Input:
        words = ["w","wo","wor","worl", "world"]
        Output: "world"
        Explanation:
        The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".
        Example 2:
        Input:
        words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
        Output: "apple"
        Explanation:
        Both "apply" and "apple" can be built from other words in the dictionary.
        However, "apple" is lexicographically smaller than "apply".
        Note:
        All the strings in the input will only contain lowercase letters.
        The length of words will be in the range [1, 1000].
        The length of words[i] will be in the range [1, 30].
        :param in_str_s:
        :return:
        """
        sort_str_s, res_word, str_set = sorted(in_str_s), '', set()
        for i in sort_str_s:
            if len(i) == 1 or i[:-1] in str_set:
                str_set.add(i)
                res_word = i if res_word == '' else i if len(i) > len(res_word) else res_word
        return res_word

    @decorate_time
    def maximum_sub_array(self, in_nums):
        """
        Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
        For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
        the contiguous subarray [4,-1,2,1] has the largest sum = 6.
        :param in_nums:
        :return:
        """
        if not in_nums:
            return 0
        cur_sum, max_sum = in_nums[0], in_nums[0]
        for i in in_nums[1:]:
            cur_sum = max(i, cur_sum + i)
            max_sum = max(max_sum, cur_sum)
        return max_sum

    @decorate_time
    def number_of_1_bit(self, in_num):
        """
        Write a function that takes an unsigned integer and returns the number of ’1' bits it has
        (also known as the Hamming weight).
        For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011,
        so the function should return 3.
        Credits:
        Special thanks to @ts for adding this problem and creating all test cases.
        :param in_num:
        :return:
        """
        from ctypes import c_int
        res_cnt = 0
        while c_int(in_num).value:
            res_cnt += 1
            in_num &= in_num - 1
        return res_cnt

    @decorate_time
    def binary_tree_paths(self, in_root):
        """
        Given a binary tree, return all root-to-leaf paths.
        For example, given the following binary tree:
           1
         /   \
        2     3
         \
          5
        All root-to-leaf paths are:
        ["1->2->5", "1->3"]
        Credits:
        Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
        :param in_root:
        :return:
        """
        if not in_root:
            return []
        return [str(in_root.val) + '->' + path
                for kid in (in_root.left, in_root.right) if kid
                for path in self.binary_tree_paths(kid)] or [str(in_root.val)]

    @decorate_time
    def binary_tree_paths_bfs(self, in_root):
        """
        bfs and queue
        :param in_root:
        :return:
        """
        import collections
        if not in_root:
            return []
        res, n_que = [], collections.deque([(in_root, '')])
        while n_que:
            node, ls = n_que.popleft()
            if not node.left and not node.right:
                res.append(ls + str(node.val))
            if node.left:
                n_que.append((node.left, ls + str(node.val) + '->'))
            if node.right:
                n_que.append((node.right, ls + str(node.val) + '->'))
        return res

    @decorate_time
    def symmetric_tree_bfs(self, in_root):
        """
        Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
        For example, this binary tree [1,2,2,3,4,4,3] is symmetric:
            1
           / \
          2   2
         / \ / \
        3  4 4  3
        But the following [1,2,2,null,3,null,3] is not:
            1
           / \
          2   2
           \   \
           3    3
        :param in_root:
        :return:
        """
        import collections
        node_val_list = []

        def bfs(node, is_left):
            if node:
                n_que = collections.deque([node])
                while n_que:
                    node = n_que.popleft()
                    if node:
                        node_val_list.append(node.val)
                    else:
                        node_val_list.append(None)
                        continue
                    if is_left:
                        if node.left:
                            n_que.append(node.left)
                        else:
                            if node.right:
                                n_que.append(None)
                        if node.right:
                            n_que.append(node.right)
                        else:
                            if node.left:
                                n_que.append(None)
                    else:
                        if node.right:
                            n_que.append(node.right)
                        else:
                            if node.left:
                                n_que.append(None)
                        if node.left:
                            n_que.append(node.left)
                        else:
                            if node.right:
                                n_que.append(None)
            return node_val_list

        left_tree = bfs(in_root.left, 1)
        node_val_list = []
        right_tree = bfs(in_root.right, 0)
        print(left_tree)
        print(right_tree)
        if left_tree == right_tree:
            return True
        else:
            return False

    @decorate_time
    def symmetric_tree_dfs(self, in_root):
        """
        :param in_root:
        :return:
        """
        if not in_root:
            return False

        def is_mirror(left, right):
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False

            if left.val == right.val:
                one_pair = is_mirror(left.left, right.right)
                two_piar = is_mirror(left.right, right.left)
                return one_pair and two_piar
            else:
                return False

        return is_mirror(in_root.left, in_root.right)

    @decorate_time
    def ugly_number(self, in_num):
        """
        Write a program to check whether a given number is an ugly number.
        Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.
        For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.
        Note that 1 is typically treated as an ugly number.
        Credits:
        Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
        :param in_num:
        :return:
        """
        for i in [2, 3, 5]:
            while in_num % i == 0:
                in_num /= i
        return in_num == 1

    @decorate_time
    def house_robber(self, in_nums):
        """
        You are a professional robber planning to rob houses along a street.
        Each house has a certain amount of money stashed,
        the only constraint stopping you from robbing each of them is that adjacent houses have security system
        connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
        Given a list of non-negative integers representing the amount of money of each house,
        determine the maximum amount of money you can rob tonight without alerting the police.
        Credits:
        Special thanks to @ifanchu for adding this problem and creating all test cases.
        Also thanks to @ts for adding additional test cases.
        :param in_nums:
        :return:
        """
        last, now = 0, 0
        for i in in_nums:
            last, now = now, max(last + i, now)
        return now

    @decorate_time
    def pascal_triangle(self, in_num_rows):
        """
        Given numRows, generate the first numRows of Pascal's triangle.

        For example, given numRows = 5,
        Return
        [
             [1],
            [1,1],
           [1,2,1],
          [1,3,3,1],
         [1,4,6,4,1],
       [1,5,10,10,5,1]
        ]
        :param in_num_rows:
        :return:
        """
        res = [[1]]
        for i in range(1, in_num_rows):
            res += [[i for i in map(lambda x, y: x + y, res[-1] + [0], [0] + res[-1])]]
        return res[:in_num_rows]

    @decorate_time
    def lowest_common_ancestor_of_a_bst(self, in_root, in_p, in_q):
        """
        Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
        According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes
        v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a
         descendant of itself).”
            _______6______
           /              \
        ___2__          ___8__
       /      \        /      \
       0      _4       7       9
             /  \
             3   5
        For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. Another example is LCA of nodes 2 and 4 is 2,
        since a node can be a descendant of itself according to the LCA definition.
        :param in_root:
        :param in_p:
        :param in_q:
        :return:
        """
        while in_root:
            if in_root.val > in_p.val and in_root.val > in_q.val:
                in_root = in_root.left
            elif in_root.val < in_p.val and in_root.val < in_q.val:
                in_root = in_root.right
            else:
                return in_root

    @decorate_time
    def power_of_four(self, in_num):
        """
        Given an integer (signed 32 bits), write a function to check whether it is a power of 4.
        Example:
        Given num = 16, return true. Given num = 5, return false.
        Follow up: Could you solve it without loops/recursion?
        Credits:
        Special thanks to @yukuairoy for adding this problem and creating all test cases.
        1
        100
        10000
        1000000
        100000000
        10000000000
        1000000000000
        100000000000000
        10000000000000000
        1000000000000000000
        100000000000000000000
        10000000000000000000000
        1000000000000000000000000
        100000000000000000000000000
        10000000000000000000000000000
        1000000000000000000000000000000
        xor all of them, equal to 1431655765
        :param in_num:
        :return:
        """
        return in_num != 0 and in_num & (in_num - 1) == 0 and in_num & 1431655765 == in_num

    @decorate_time
    def reverse_vowels_of_a_string(self, in_str):
        """
        Write a function that takes a string as input and reverse only the vowels of a string.
        Example 1:
        Given s = "hello", return "holle".
        Example 2:
        Given s = "leetcode", return "leotcede".
        Note:
        The vowels does not include the letter "y".
        vowel a e i o u
        :param in_str:
        :return:
        """
        vowels = ['a', 'e', 'i', 'o', 'u']
        first = 0
        str_list = list(in_str)
        for i in range(len(str_list)):
            if str_list[i] in vowels:
                if first == 0:
                    first = i
                else:
                    second = i
                    str_list[first], str_list[second] = str_list[second], str_list[first]
                    first = second
        return ''.join(str_list)

    @decorate_time
    def valid_perfect_square(self, in_num):
        """
        Given a positive integer num, write a function which returns True if num is a perfect square else False.
        Note: Do not use any built-in library function such as sqrt.
        Example 1:
        Input: 16
        Returns: True
        Example 2:
        Input: 14
        Returns: False
        Credits:
        Special thanks to @elmirap for adding this problem and creating all test cases.
        :param in_num:
        :return:
        """
        r = in_num
        while r * r > in_num:
            r = (r + in_num // r) // 2
        return r * r == in_num

    @decorate_time
    def repeated_substring_pattern(self, in_str):
        """
        Given a non-empty string check if it can be constructed by taking a substring of it and appending multiple
        copies of the substring together. You may assume the given string consists of lowercase English letters only
        and its length will not exceed 10000.
        Example 1:
        Input: "abab"
        Output: True
        Explanation: It's the substring "ab" twice.
        Example 2:
        Input: "aba"
        Output: False
        Example 3:
        Input: "abcabcabcabc"
        Output: True
        Explanation: It's the substring "abc" four times. (And the substring "abcabc" twice.)
        :param in_str:
        :return:
        """
        if len(in_str) % 2 != 0 or not in_str:
            return False
        if in_str[:len(in_str) // 2] == in_str[len(in_str) // 2:]:
            return True
        else:
            return False

    @decorate_time
    def balanced_binary_tree(self, in_root):
        """
        Given a binary tree, determine if it is height-balanced.
        For this problem,
        a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node
        never differ by more than 1.
        :param in_root:
        :return:
        """

        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            return 1 + max(left, right)

        return dfs(in_root) != -1

    @decorate_time
    def find_mode_in_bst(self, in_root):
        """
        Given a binary search tree (BST) with duplicates,
        find all the mode(s) (the most frequently occurred element) in the given BST.
        Assume a BST is defined as follows:
        The left subtree of a node contains only nodes with keys less than or equal to the node's key.
        The right subtree of a node contains only nodes with keys greater than or equal to the node's key.
        Both the left and right subtrees must also be binary search trees.
        For example:
        Given BST [1,null,2,2],
           1
            \
             2
            /
           2
        return [2].
        Note: If a tree has more than one mode, you can return them in any order.
        Follow up: Could you do that without using any extra space? (Assume that the implicit stack space incurred
        due to recursion does not count).
        :param in_root:
        :return:
        """
        cnt_dict = dict()

        def dfs(node):
            if node:
                if node.val in cnt_dict:
                    cnt_dict[node.val] += 1
                else:
                    cnt_dict[node.val] = 1
            dfs(node.left)
            dfs(node.right)

        dfs(in_root)
        return [k for k, v in cnt_dict.items() if v > 1]

    @decorate_time
    def pascal_triangle_two(self, in_row_index):
        """Given an index k, return the kth row of the Pascal's triangle.
        For example, given k = 3,
        Return [1,3,3,1].
        Note:
        Could you optimize your algorithm to use only O(k) extra space?
        """
        if in_row_index < 1:
            return [1]
        start_l = [1]
        for i in range(in_row_index):
            start_l = [j for j in map(lambda x, y: x + y, start_l + [0], [0] + start_l)]
        return start_l

    @decorate_time
    def sentence_similarity(self, in_words1, in_words2, in_pair):
        """
        Given two sentences words1, words2 (each represented as an array of strings),
        and a list of similar word pairs pairs, determine if two sentences are similar.
        For example, "great acting skills" and "fine drama talent" are similar,
        if the similar word pairs are pairs = [["great", "fine"], ["acting","drama"], ["skills","talent"]].
        Note that the similarity relation is not transitive.
        For example, if "great" and "fine" are similar,
        and "fine" and "good" are similar, "great" and "good" are not necessarily similar.
        However, similarity is symmetric. For example, "great" and "fine" being similar is the same as
        "fine" and "great" being similar.
        Also, a word is always similar with itself. For example,
        the sentences words1 = ["great"], words2 = ["great"], pairs = [] are similar,
        even though there are no specified similar word pairs.
        Finally, sentences can only be similar if they have the same number of words.
        So a sentence like words1 = ["great"] can never be similar to words2 = ["doubleplus","good"].
        Note:
        The length of words1 and words2 will not exceed 1000.
        The length of pairs will not exceed 2000.
        The length of each pairs[i] will be 2.
        The length of each words[i] and pairs[i][j] will be in the range [1, 20].
        :param in_words1:
        :param in_words2:
        :param in_pair:
        :return:
        """
        if len(in_words1) != len(in_words2):
            return False
        for i, j in zip(in_words1, in_words2):
            if i != j:
                if [i, j] in in_pair or [j, i] in in_pair:
                    continue
                else:
                    return False
        else:
            return True

    @decorate_time
    def number_of_segments_in_a_string(self, in_str):
        """
        Count the number of segments in a string,
        where a segment is defined to be a contiguous sequence of non-space characters.
        Please note that the string does not contain any non-printable characters.
        Example:
        Input: "Hello, my name is John"
        Output: 5
        :param in_str:
        :return:
        """
        return len(in_str.split())

    @decorate_time
    def arranging_coins(self, in_num):
        """
        You have a total of n coins that you want to form in a staircase shape, where every k-th row must have exactly k coins.
        Given n, find the total number of full staircase rows that can be formed.
        n is a non-negative integer and fits within the range of a 32-bit signed integer.
        Example 1:
        n = 5
        The coins can form the following rows:
        ¤
        ¤ ¤
        ¤ ¤
        Because the 3rd row is incomplete, we return 2.
        Example 2:
        n = 8
        The coins can form the following rows:
        ¤
        ¤ ¤
        ¤ ¤ ¤
        ¤ ¤
        Because the 4th row is incomplete, we return 3.
        :param in_num:
        :return:
        """
        i = 1
        if in_num < 1:
            return 0
        elif in_num == 1:
            return 1
        else:
            while i * (i + 1) // 2 <= in_num:
                i += 1
            return i - 1

    @decorate_time
    def guess_number_higher_or_lower(self, in_num):
        """
        We are playing the Guess Game. The game is as follows:
        I pick a number from 1 to n. You have to guess which number I picked.
        Every time you guess wrong, I'll tell you whether the number is higher or lower.
        You call a pre-defined API guess(int num) which returns 3 possible results (-1, 1, or 0):
        -1 : My number is lower
         1 : My number is higher
         0 : Congrats! You got it!
        Example:
        n = 10, I pick 6.
        Return 6.
        :param in_num:
        :return:
        """

        def guess(n):
            pass

        low = 0
        high = in_num
        while True:
            mid = (low + high) // 2
            if guess(mid) > 0:
                high = mid
            elif guess(mid) < 0:
                low = mid
            else:
                return mid

    @decorate_time
    def linked_list_cycle(self, in_head):
        """
        Given a linked list, determine if it has a cycle in it.
        Follow up:
        Can you solve it without using extra space?
        :param in_head:
        :return:
        """
        try:
            slow = in_head
            fast = in_head.next
            while slow is not fast:
                slow = slow.next
                fast = fast.next.next
            return True
        except:
            return False

    @decorate_time
    def path_sum_binary_tree(self, in_root, in_sum):
        """
        Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values
        along the path equals the given sum.
        For example:
        Given the below binary tree and sum = 22,
                      5
                     / \
                    4   8
                   /   / \
                  11  13  4
                 /  \      \
                7    2      1
        return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
        :param in_root:
        :param in_sum:
        :return:
        """
        if not in_root:
            return False
        path_list = \
            [[in_root.val] + path for kid in (in_root.left, in_root.right) if kid for path in
             self.binary_tree_paths(kid)] or [in_root.val]
        for i in path_list:
            if sum(i) == in_sum:
                return True
        else:
            return False

    @decorate_time
    def isomorphic_strings(self, in_s, in_t):
        """
        Given two strings s and t, determine if they are isomorphic.
        Two strings are isomorphic if the characters in s can be replaced to get t.
        All occurrences of a character must be replaced with another character while preserving the order of characters.
        No two characters may map to the same character but a character may map to itself.
        For example,
        Given "egg", "add", return true.
        Given "foo", "bar", return false.
        Given "paper", "title", return true.
        Note:
        You may assume both s and t have the same length.
        :param in_s:
        :param in_t:
        :return:
        """
        # return len(set(zip(in_s, in_t))) == len(in_s) == len(in_t)
        return [in_s.find(i) for i in in_s] == [in_t.find(j) for j in in_t]

    @decorate_time
    def find_all_anagrams_in_a_string(self, in_s, in_p):
        """
        Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
        Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.
        The order of output does not matter.
        Example 1:
        Input:
        s: "cbaebabacd" p: "abc"
        Output:
        [0, 6]
        Explanation:
        The substring with start index = 0 is "cba", which is an anagram of "abc".
        The substring with start index = 6 is "bac", which is an anagram of "abc".
        Example 2:
        Input:
        s: "abab" p: "ab"
        Output:
        [0, 1, 2]
        Explanation:
        The substring with start index = 0 is "ab", which is an anagram of "ab".
        The substring with start index = 1 is "ba", which is an anagram of "ab".
        The substring with start index = 2 is "ab", which is an anagram of "ab".
        :param in_s:
        :param in_p:
        :return:
        """
        from collections import Counter
        pos_list = []
        dic_p = Counter(in_p)
        for i in range(len(in_s) - len(in_p) + 1):
            dic_s = Counter(in_s[i:i + len(in_p)])
            is_true = True
            for k, v in dic_p.items():
                if v != dic_s.get(k, 0):
                    is_true = False
                    break
            if is_true is True:
                pos_list.append(i)
        return pos_list

    @decorate_time
    def perfect_number(self, in_num):
        """
        We define the Perfect Number is a positive integer that is equal to the sum of all its positive divisors
        except itself.
        Now, given an integer n, write a function that returns true when it is a perfect number and false
        when it is not.
        Example:
        Input: 28
        Output: True
        Explanation: 28 = 1 + 2 + 4 + 7 + 14
        Note: The input number n will not exceed 100,000,000. (1e8)
        :param in_num:
        :return:
        """
        div_list = []
        i = 1
        while i ** 2 < in_num:
            if in_num % i == 0:
                div_list.append(i)
                if in_num // i != in_num:
                    div_list.append(in_num // i)
            i += 1
        print(div_list)
        if in_num == sum(div_list):
            return True
        else:
            return False

    @decorate_time
    def minimum_depth_of_binary_tree(self, in_root):
        """
        Given a binary tree, find its minimum depth.
        The minimum depth is the number of nodes along
        the shortest path from the root node down to the nearest leaf node.
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
        return min(dep_rec)

    @decorate_time
    def word_pattern(self, in_pat, in_str_s):
        """
        Given a pattern and a string str, find if str follows the same pattern.
        Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in str.
        Examples:
        pattern = "abba", str = "dog cat cat dog" should return true.
        pattern = "abba", str = "dog cat cat fish" should return false.
        pattern = "aaaa", str = "dog cat cat dog" should return false.
        pattern = "abba", str = "dog dog dog dog" should return false.
        Notes:
        You may assume pattern contains only lowercase letters, and str contains lowercase letters separated by a single space.
        Credits:
        Special thanks to @minglotus6 for adding this problem and creating all test cases.
        :param in_pat:
        :param in_str_s:
        :return:
        """
        word_list = in_str_s.split()
        return [in_pat.find(i) for i in in_pat] == [word_list.index(j) for j in word_list]

    @decorate_time
    def palindrome_linked_list(self, in_head):
        """
        Given a singly linked list, determine if it is a palindrome.
        Follow up:
        Could you do it in O(n) time and O(1) space?
        :param in_head:
        :return:
        """
        fast = slow = in_head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        node = None
        while slow:
            nxt = slow.next
            slow.next = node
            node = slow
            slow = nxt
        while node:
            if node.val != in_head.val:
                return False
            node = node.next
            in_head = in_head.next
        return True

    @decorate_time
    def remove_linked_list_elements(self, in_head, in_val):
        """
        Remove all elements from a linked list of integers that have value val.
        Example
        Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
        Return: 1 --> 2 --> 3 --> 4 --> 5
        Credits:
        Special thanks to @mithmatt for adding this problem and creating all test cases.
        :param in_head:
        :param in_val:
        :return:
        """
        node = in_head
        while node:
            if in_val == node.val == in_head.val:
                node = node.next
                in_head = in_head.next
            if node.next and in_val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        return in_head

    @decorate_time
    def contain_duplicate_two(self, in_nums, in_k):
        """
        Given an array of integers and an integer k,
        find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and
        the absolute difference between i and j is at most k.
        :param in_nums:
        :param in_k:
        :return:
        """
        for i in range(len(in_nums)):
            for j in range(i, len(in_nums)):
                if in_nums[i] == in_nums[j] and abs(j - i) <= in_k:
                    return True
        else:
            return False

    @decorate_time
    def sum_of_square_numbers(self, in_num):
        """
        Given a non-negative integer c, your task is to decide whether there're two integers a and
         b such that a**2 + b**2 = c.
        Example 1:
        Input: 5
        Output: True
        Explanation: 1 * 1 + 2 * 2 = 5
        Example 2:
        Input: 3
        Output: False
        :param in_num:
        :return:
        """
        r = in_num
        while r * r > in_num:
            r = (r + in_num // r) // 2
        if r ** 2 + 1 == in_num:
            return True
        else:
            return False

    @decorate_time
    def merge_sorted_array_in_place(self, nums1, nums2):
        """
        Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
        Note:
        You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements
        from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        nums1.extend(nums2)
        nums1.sort()
        return nums1

    @decorate_time
    def length_of_last_word(self, in_str):
        """
        Given a string s consists of upper/lower-case alphabets and empty space characters ' ',
        return the length of last word in the string.
        If the last word does not exist, return 0.
        Note: A word is defined as a character sequence consists of non-space characters only.
        Example:
        Input: "Hello World"
        Output: 5
        :param in_str:
        :return:
        """
        return len(in_str.split()[-1])

    @decorate_time
    def range_sum_query_immutable(self, in_i, in_j):
        """
        Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.
        Example:
        Given nums = [-2, 0, 3, -5, 2, -1]
        sumRange(0, 2) -> 1
        sumRange(2, 5) -> -1
        sumRange(0, 5) -> -3
        Note:
        You may assume that the array does not change.
        There are many calls to sumRange function.
        :param in_i:
        :param in_j:
        :return:
        """
        inner_num = [-2, 0, 3, -5, 2, -1]
        import functools
        import operator

        def sum_range(i, j):
            if i < 0 or j >= len(inner_num):
                return None
            return functools.reduce(operator.add, inner_num[i:j + 1])

        return sum_range(in_i, in_j)

    @decorate_time
    def intersection_of_two_linked_lists(self, in_head_a, in_head_b):
        """
        Write a program to find the node at which the intersection of two singly linked lists begins.
        For example, the following two linked lists:
        A:          a1 → a2
                           ↘
                             c1 → c2 → c3
                           ↗
        B:     b1 → b2 → b3
        begin to intersect at node c1.
        Notes:
        If the two linked lists have no intersection at all, return null.
        The linked lists must retain their original structure after the function returns.
        You may assume there are no cycles anywhere in the entire linked structure.
        Your code should preferably run in O(n) time and use only O(1) memory.
        Credits:
        Special thanks to @stellari for adding this problem and creating all test cases.
        :param in_head_a:
        :param in_head_b:
        :return:
        """
        if in_head_b is None or in_head_a is None:
            return None
        h_a, h_b = in_head_a, in_head_b
        while h_a is not h_b:
            h_a = in_head_a if h_a is None else h_a.next
            h_b = in_head_b if h_b is None else h_b.next
        return h_a

    @decorate_time
    def can_place_flowers(self, in_flowers, in_n):
        """
        Suppose you have a long flowerbed in which some of the plots are planted and some are not.
        However, flowers cannot be planted in adjacent plots - they would compete for water and both would die.
        Given a flowerbed (represented as an array containing 0 and 1, where 0 means empty and 1 means not empty),
        and a number n, return if n new flowers can be planted in it without violating the no-adjacent-flowers rule.
        Example 1:
        Input: flowerbed = [1,0,0,0,1], n = 1
        Output: True
        Example 2:
        Input: flowerbed = [1,0,0,0,1], n = 2
        Output: False
        Note:
        The input array won't violate no-adjacent-flowers rule.
        The input array size is in the range of [1, 20000].
        n is a non-negative integer which won't exceed the input array size.
        :param in_flowers:
        :param in_n:
        :return:
        """
        i = 1
        cnt = 0
        while i < len(in_flowers) - 1:
            if in_flowers[i - 1] == in_flowers[i] == in_flowers[i + 1] == 0:
                cnt += 1
                i += 2
            else:
                i += 1
        if cnt >= in_n:
            return True
        else:
            return False

    @decorate_time
    def nth_digit(self, in_num):
        """
        Find the nth digit of the infinite integer sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...
        Note:
        n is positive and will fit within the range of a 32-bit signed integer (n < 231).
        Example 1:
        Input:
        3
        Output:
        3
        Example 2:
        Input:
        11
        Output:
        0
        Explanation:
        The 11th digit of the sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... is a 0, which is part of the number 10.

        1 * 9 (size 1, 1... 9)
        2 * 90 (size 2, 10... 99)
        3 * 900 (size 3, 100... 999)
        So we can "fast-skip" those numbers until we find the size of the number that will hold our digit.
        At the end of the loop, we will have:
        start: first number of size size (will be power of 10)
        n: will be the number of digits that we need to count after start
        How do we get the number that will hold the digit? It will be start + (n - 1) // size
        (we use n - 1 because we need zero-based index). Once we have that number, we can get the n - 1 % size-th digit
        of that number, and that will be our result.
        :param in_num:
        :return:
        """
        start, size, step = 1, 1, 9
        while in_num > size * step:
            in_num, size, step, start = in_num - (size * step), size + 1, step * 10, start * 10
        return int(str(start + (in_num - 1) // size)[(in_num - 1) % size])

    @decorate_time
    def min_stack(self, in_x):
        """
        Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
        push(x) -- Push element x onto stack.
        pop() -- Removes the element on top of the stack.
        top() -- Get the top element.
        getMin() -- Retrieve the minimum element in the stack.
        Example:
        MinStack minStack = new MinStack();
        minStack.push(-2);
        minStack.push(0);
        minStack.push(-3);
        minStack.getMin();   --> Returns -3.
        minStack.pop();
        minStack.top();      --> Returns 0.
        minStack.getMin();   --> Returns -2.
        :param in_x:
        :return:
        """

        class MinStack(object):
            def __init__(self, in_list=None):
                """
                initialize your data structure here.
                """
                import copy
                self._stack = copy.deepcopy(in_list)

            def push(self, x):
                """
                :type x: int
                :rtype: void
                """
                if self._stack is None:
                    self._stack = []
                self._stack.append(x)

            def pop(self):
                """
                :rtype: void
                """
                if self._stack is None or len(self._stack) == 0:
                    return None
                else:
                    return self._stack.pop(0)

            def top(self):
                """
                :rtype: int
                """
                if self._stack is None or len(self._stack) == 0:
                    return None
                else:
                    return self._stack[0]

            def getMin(self):
                """
                :rtype: int
                """
                if self._stack is None or len(self._stack) == 0:
                    return None
                i_min = min(self._stack)
                return i_min

    @decorate_time
    def reverse_bits(self, in_num):
        """
        Reverse bits of a given 32 bits unsigned integer.
        For example, given input 43261596 (represented in binary as 00000010100101000001111010011100),
        return 964176192 (represented in binary as 00111001011110000010100101000000).
        Follow up:
        If this function is called many times, how would you optimize it?
        Related problem: Reverse Integer
        Credits:
        Special thanks to @ts for adding this problem and creating all test cases.
        :param in_num:
        :return:
        """
        # m = 0
        # for i in range(32):
        #     m <<= 1
        #     m |= in_num & 1
        #     in_num >>= 1
        # return m
        ori_bin = '{0:032b}'.format(in_num)
        reverse_bin = ori_bin[::-1]
        return int(reverse_bin, 2)

    @decorate_time
    def shortest_unsorted_continuous_subarray(self, in_nums):
        """
        Given an integer array, you need to find one continuous subarray that if you only sort this subarray
        in ascending order, then the whole array will be sorted in ascending order, too.
        You need to find the shortest such subarray and output its length.
        Example 1:
        Input: [2, 6, 4, 8, 10, 9, 15]
        Output: 5
        Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array
        sorted in ascending order.
        Note:
        Then length of the input array is in range [1, 10,000].
        The input array may contain duplicates, so ascending order here means <=.
        :param in_nums:
        :return:
        """
        left = 0
        right = 0
        for i in range(len(in_nums)):
            if in_nums[i] != min(in_nums[i:]):
                left = i
                break
        for j in range(len(in_nums) - 1, 0, -1):
            if in_nums[j] != max(in_nums[:j + 1]):
                right = j
                break
        print(left, right)
        if left >= right:
            return 0
        else:
            return right - left + 1

    @decorate_time
    def k_diff_pairs_in_an_array(self, in_nums, in_k):
        """
        Given an array of integers and an integer k, you need to find the number of unique k-diff pairs in the array.
        Here a k-diff pair is defined as an integer pair (i, j),
        where i and j are both numbers in the array and their absolute difference is k.
        Example 1:
        Input: [3, 1, 4, 1, 5], k = 2
        Output: 2
        Explanation: There are two 2-diff pairs in the array, (1, 3) and (3, 5).
        Although we have two 1s in the input, we should only return the number of unique pairs.
        Example 2:
        Input:[1, 2, 3, 4, 5], k = 1
        Output: 4
        Explanation: There are four 1-diff pairs in the array, (1, 2), (2, 3), (3, 4) and (4, 5).
        Example 3:
        Input: [1, 3, 1, 5, 4], k = 0
        Output: 1
        Explanation: There is one 0-diff pair in the array, (1, 1).
        Note:
        The pairs (i, j) and (j, i) count as the same pair.
        The length of the array won't exceed 10,000.
        All the integers in the given input belong to the range: [-1e7, 1e7].
        :param in_nums:
        :param in_k:
        :return:
        """
        res_set = set()
        for i in range(len(in_nums)):
            for j in range(i + 1, len(in_nums)):
                if abs(in_nums[i] - in_nums[j]) == in_k:
                    if (in_nums[i], in_nums[j]) not in res_set and (in_nums[j], in_nums[i]) not in res_set:
                        res_set.add((in_nums[i], in_nums[j]))
        return len(res_set)

    @decorate_time
    def largest_palindrome_product(self, in_num):
        """
        Find the largest palindrome made from the product of two n-digit numbers.
        Since the result could be very large, you should return the largest palindrome mod 1337.
        Example:
        Input: 2
        Output: 987
        Explanation: 99 x 91 = 9009, 9009 % 1337 = 987
        Note:
        The range of n is [1,8].
        :param in_num:
        :return:
        """

        def is_palindrome(x):
            if str(x) == str(x)[::-1]:
                return True
            else:
                return False

        start_s = '9' * in_num
        start_v = int(start_s)
        p_list = []
        for i in range(start_v, 0, -1):
            for j in range(i - 1, 0, -1):
                if is_palindrome(i * j):
                    p_list.append(i * j)
        return max(p_list) % 1337

    @decorate_time
    def rotate_array(self, in_nums, in_k):
        """
        Rotate an array of n elements to the right by k steps.
        For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
        Note:
        Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
        [show hint]
        Related problem: Reverse Words in a String II
        Credits:
        Special thanks to @Freezen for adding this problem and creating all test cases.
        :param in_nums:
        :param in_k:
        :return:
        """
        for i in range(in_k, len(in_nums), in_k):
            if i + 3 > len(in_nums) - 1:
                break
            else:
                in_nums = in_nums[i + 1:i + in_k + 1] + in_nums[:i + 1] + in_nums[i + in_k + 1:]
        return in_nums

    @decorate_time
    def count_primes(self, in_num):
        """
        Description:
        Count the number of prime numbers less than a non-negative number, n.
        Credits:
        Special thanks to @mithmatt for adding this problem and creating all test cases.
        :param in_num:
        :return:
        """
        if in_num < 3:
            return 0
        primes = [True] * in_num
        primes[0] = primes[1] = False
        for i in range(2, int(in_num ** 0.5) + 1):
            if primes[i]:
                primes[i * i:: i] = [False] * len(primes[i * i:: i])
        return sum(primes)

    @decorate_time
    def third_maximum_number(self, in_nums):
        """
        Given a non-empty array of integers, return the third maximum number in this array. If it does not exist,
        return the maximum number. The time complexity must be in O(n).
        Example 1:
        Input: [3, 2, 1]
        Output: 1
        Explanation: The third maximum is 1.
        Example 2:
        Input: [1, 2]
        Output: 2
        Explanation: The third maximum does not exist, so the maximum (2) is returned instead.
        Example 3:
        Input: [2, 2, 3, 1]
        Output: 1
        Explanation: Note that the third maximum here means the third maximum distinct number.
        Both numbers with value 2 are both considered as second maximum.
        :param in_nums:
        :return:
        """
        from heapq import nlargest
        u_list = list(set(in_nums))
        if 1 < len(u_list) < 3:
            return max(u_list)
        elif 1 > len(u_list):
            return None
        else:
            return min(nlargest(3, u_list))

    @decorate_time
    def nested_list_weight_sum(self, in_n_list):
        """
        Given a nested list of integers, return the sum of all integers in the list weighted by their depth.
        Each element is either an integer, or a list -- whose elements may also be integers or other lists.
        Example 1:
        Given the list [[1,1],2,[1,1]], return 10. (four 1's at depth 2, one 2 at depth 1)
        Example 2:
        Given the list [1,[4,[6]]], return 27. (one 1 at depth 1, one 4 at depth 2, and one 6 at depth 3; 1 + 4*2 + 6*3 = 27)
        :param in_n_list:
        :return:
        """

        def list_sum(list_1, dep):
            i_res_sum = 0
            for i in list_1:
                if isinstance(i, int):
                    i_res_sum += i * dep
                else:
                    i_res_sum += list_sum(i, dep + 1)
            return i_res_sum

        return list_sum(in_n_list, 1)

    @decorate_time
    def logger_rate_limiter(self, in_sec, in_info):
        """
        Design a logger system that receive stream of messages along with its timestamps,
        each message should be printed if and only if it is not printed in the last 10 seconds.
        Given a message and a timestamp (in seconds granularity),
        return true if the message should be printed in the given timestamp, otherwise returns false.
        It is possible that several messages arrive roughly at the same time.
        Example:
        Logger logger = new Logger();
        // logging string "foo" at timestamp 1
        logger.shouldPrintMessage(1, "foo"); returns true;
        // logging string "bar" at timestamp 2
        logger.shouldPrintMessage(2,"bar"); returns true;
        // logging string "foo" at timestamp 3
        logger.shouldPrintMessage(3,"foo"); returns false;
        // logging string "bar" at timestamp 8
        logger.shouldPrintMessage(8,"bar"); returns false;
        // logging string "foo" at timestamp 10
        logger.shouldPrintMessage(10,"foo"); returns false;
        // logging string "foo" at timestamp 11
        logger.shouldPrintMessage(11,"foo"); returns true;
        :param in_sec:
        :param in_info:
        :return:
        """
        if in_info not in self.logger_dict:
            self.logger_dict[in_info] = in_sec
            return True
        else:
            if in_sec > self.logger_dict[in_info] + 10:
                self.logger_dict[in_info] = in_sec
                return True
            else:
                return False

    @decorate_time
    def moving_average_from_data_stream(self, in_window_size):
        """
        Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.
        For example,
        MovingAverage m = new MovingAverage(3);
        m.next(1) = 1
        m.next(10) = (1 + 10) / 2
        m.next(3) = (1 + 10 + 3) / 3
        m.next(5) = (10 + 3 + 5) / 3
        :param in_window_size:
        :return:
        """
        inner_list = []
        inner_w_size = in_window_size

        def next_(n_num):
            nonlocal inner_list
            nonlocal inner_w_size
            if len(inner_list) < 3:
                inner_list.append(n_num)
                return sum(inner_list) / len(inner_list)
            else:
                inner_list.pop(0)
                inner_list.append(n_num)
                return sum(inner_list) / inner_w_size

        return next_

    @decorate_time
    def palindrome_permutation(self, in_str):
        """
        Given a string, determine if a permutation of the string could form a palindrome.
        For example,
        "code" -> False, "aab" -> True, "carerac" -> True.
        Hint:
        Consider the palindromes of odd vs even length. What difference do you notice?
        Count the frequency of each character.
        If each character occurs even number of times,
        then it must be a palindrome. How about character which occurs odd number of times?
        :param in_str:
        :return:
        """
        from collections import Counter
        ap_dict = Counter(in_str)
        odd_cnt = 0
        for v in ap_dict.values():
            if v % 2 == 1:
                odd_cnt += 1
        if odd_cnt <= 1:
            return True
        else:
            return False

    @decorate_time
    def flip_game(self, in_str):
        """
        You are playing the following Flip Game with your friend:
         Given a string that contains only these two characters:
          + and -, you and your friend take turns to flip two consecutive "++" into "--".
           The game ends when a person can no longer make a move and therefore the other person will be the winner.
        Write a function to compute all possible states of the string after one valid move.
        For example, given s = "++++", after one move, it may become one of the following states:

        [
          "--++",
          "+--+",
          "++--"
        ]
        :param in_str:
        :return:
        """
        return [in_str[:i - 1] + "--" + in_str[i + 1:] for i in range(1, len(in_str)) if in_str[i - 1:i + 1] == "++"]

    @decorate_time
    def shortest_word_distance(self, words, word1, word2):
        """
        Given a list of words and two words word1 and word2,
         return the shortest distance between these two words in the list.
        For example, Assume that words = ["practice", "makes", "perfect", "coding", "makes"].
        Given word1 = “coding”, word2 = “practice”, return 3. Given word1 = "makes", word2 = "coding", return 1.
        Note: You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.
        :return:
        """
        w1_list = [i for i in range(len(words)) if words[i] == word1]
        w2_list = [i for i in range(len(words)) if words[i] == word2]
        return min([abs(i - j) for i in w1_list for j in w2_list])

    @decorate_time
    def meeting_rooms(self, in_list):
        """
        Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei),
        determine if a person could attend all meetings.
        For example,
        Given [[0, 30],[5, 10],[15, 20]],
        return false
        :param in_list:
        :return:
        """
        from itertools import combinations
        for i in combinations(in_list, 2):
            if i[0][0] < i[1][0] and i[0][1] > i[1][1]:
                return False
            elif i[0][0] > i[1][0] and i[0][1] < i[1][1]:
                return False
        else:
            return True

    @decorate_time
    def paint_house(self, in_costs):
        """
        There are a row of n houses, each house can be painted with one of the three colors: red, blue or green.
        The cost of painting each house with a certain color is different.
        You have to paint all the houses such that no two adjacent houses have the same color.
        The cost of painting each house with a certain color is represented by a n x 3 cost matrix.
        For example, costs[0][0] is the cost of painting house 0 with color red;
        costs[1][2] is the cost of painting house 1 with color green,
        and so on... Find the minimum cost to paint all houses.
        Note: All costs are positive integers.
        :param in_costs: List[List[int]]
        :return:
        """
        if not in_costs:
            return None
        prev = in_costs[0][:]
        curr = [0] * 3
        for i in range(len(in_costs) - 1):
            curr[0] = min(prev[1], prev[2]) + in_costs[i + 1][0]
            curr[1] = min(prev[0], prev[2]) + in_costs[i + 1][1]
            curr[2] = min(prev[1], prev[0]) + in_costs[i + 1][2]
            prev[:] = curr[:]
        return min(prev)

    @decorate_time
    def closest_binary_search_tree_value(self, in_root, in_target):
        """
        Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.
        Note: Given target value is a floating point.
        You are guaranteed to have only one unique value in the BST that is closest to the target.
        :param in_root:
        :param in_target:
        :return:
        """
        res_value = in_root.val

        def dfs(node):
            nonlocal res_value
            if not node:
                return
            if abs(node.val - in_target) < abs(res_value - in_target):
                res_value = node.val
            dfs(node.left)
            dfs(node.right)
            return

        dfs(in_root)
        return res_value

    @decorate_time
    def strobogrammatic_number(self, in_num):
        """
        A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
        Write a function to determine if a number is strobogrammatic. The number is represented as a string.
        For example, the numbers “69”, “88”, and “818” are all strobogrammatic.
        0123456789
        :param in_num: str
        :return: bool
        """
        for i in in_num:
            if i in '23457':
                return False
        r_dict = {'6': '9', '9': '6', '1': '1', '0': '0', '8': '8'}
        rev_str = in_num[::-1]
        for i, j in zip(in_num, rev_str):
            if i != r_dict[j]:
                return False
        else:
            return True

    @decorate_time
    def valid_word_square(self, in_list_str):
        """
        Given a sequence of words, check whether it forms a valid word square.
        A sequence of words forms a valid word square if the kth row and column read the exact same string,
        where 0 ≤ k < max(numRows, numColumns).
        Note:
        The number of words given is at least 1 and does not exceed 500.
        Word length will be at least 1 and does not exceed 500.
        Each word contains only lowercase English alphabet a-z.
        Example 1:
        Input:
        [
          "abcd",
          "bnrt",
          "crmy",
          "dtye"
        ]
        Output:
        true
        Explanation:
        The first row and first column both read "abcd".
        The second row and second column both read "bnrt".
        The third row and third column both read "crmy".
        The fourth row and fourth column both read "dtye".
        Therefore, it is a valid word square.
        Example 2:
        Input:
        [
          "abcd",
          "bnrt",
          "crm",
          "dt"
        ]
        Output:
        true
        Explanation:
        The first row and first column both read "abcd".
        The second row and second column both read "bnrt".
        The third row and third column both read "crm".
        The fourth row and fourth column both read "dt".
        Therefore, it is a valid word square.
        Example 3:
        Input:
        [
          "ball",
          "area",
          "read",
          "lady"
        ]
        Output:
        false
        Explanation:
        The third row reads "read" while the third column reads "lead".
        Therefore, it is NOT a valid word square.
        :param in_list_str:
        :return: bool
        """
        max_len = len(in_list_str[0])
        if max_len != max([len(i) for i in in_list_str]):
            return False
        for i in range(len(in_list_str)):
            if len(in_list_str[i]) < max_len:
                in_list_str[i] = '{:*<{i_len}}'.format(in_list_str[i], i_len=max_len)
        for i in range(len(in_list_str)):
            if in_list_str[i] != ''.join([j[i] for j in in_list_str]):
                return False
        else:
            return True

    @decorate_time
    def paint_fence(self, in_n, in_k):
        """
        There is a fence with n posts, each post can be painted with one of the k colors.
        You have to paint all the posts such that no more than two adjacent fence posts have the same color.
        Return the total number of ways you can paint the fence.
        Note:
        n and k are non-negative integers.
        :param in_n:
        :param in_k:
        :return:
        """
        if in_n == 0:
            return 0
        if in_n == 1:
            return in_k
        same_color = 0
        diff_color = in_k
        total = same_color + diff_color
        for i in range(1, in_n):
            same_color = diff_color
            diff_color = (in_k - 1)*total
            total = same_color + diff_color
        return total

    @decorate_time
    def valid_word_abbreviation(self, in_str, in_abbr):
        """
        Given a non-empty string s and an abbreviation abbr, return whether the string matches with the given abbreviation.
        A string such as "word" contains only the following valid abbreviations:
        ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]
        Notice that only the above abbreviations are valid abbreviations of the string "word".
         Any other string is not a valid abbreviation of "word".
        Note:
        Assume s contains only lowercase letters and abbr contains only lowercase letters and digits.
        Example 1:
        Given s = "internationalization", abbr = "i12iz4n":
        Return true.
        Example 2:
        Given s = "apple", abbr = "a2e":
        Return false.
        :param in_str:
        :param in_abbr:
        :return:
        """
        i, j = 0, 0
        t_num = 0
        while i < len(in_str) and j < len(in_abbr):
            if in_abbr[j] in '0123456789':
                if t_num == 0 and in_abbr[j] == '0':
                    return False
                t_num = t_num * 10 + int(in_abbr[j])
                j += 1
            elif t_num > 0:
                i += t_num
                t_num = 0
            elif in_str[i] != in_abbr[j]:
                return False
            else:
                i += 1
                j += 1
        return i + t_num == len(in_str) and j == len(in_abbr)

    @decorate_time
    def maximum_distance_in_array(self, in_list_int):
        """
         Given m arrays, and each array is sorted in ascending order.
        Now you can pick up two integers from two different arrays (each array picks one) and calculate the distance.
        We define the distance between two integers a and b to be their absolute difference |a-b|.
        Your task is to find the maximum distance.
        Example 1:
        Input:
        [[1,2,3],
         [4,5],
         [1,2,3]]
        Output: 4
        Explanation:
        One way to reach the maximum distance 4 is to pick 1 in the first or third array and pick 5 in the second array.
        Note:

            Each given array will have at least 1 number. There will be at least two non-empty arrays.
            The total number of the integers in all the m arrays will be in the range of [2, 10000].
            The integers in the m arrays will be in the range of [-10000, 10000].
        :param in_list_int: list[list[int]]
        :return:
        """
        max_int = 0
        min_int = 0
        for i in in_list_int:
            if len(i) > 0:
                if max_int < i[-1]:
                    max_int = i[-1]
                if min_int > i[0]:
                    min_int = i[0]

        return max_int - min_int


class CompressedIterator(object):
    def __init__(self, in_str):
        self._str = self.recover(in_str)
        self.travel = 0

    def next(self):
        try:
            r_ch = self._str[self.travel]
        except IndexError:
            return ''
        except Exception:
            raise Exception
        self.travel += 1
        return r_ch

    def has_next(self):
        if self.travel < len(self._str):
            return True
        else:
            return False

    @staticmethod
    def recover(in_str):
        tmp_str = ''
        i = 0
        num = 0
        while i < len(in_str):
            if in_str[i].isdigit():
                num = num * 10 + int(in_str[i])
                i += 1
            elif num > 0:
                tmp_str += tmp_str[-1] * (num - 1)
                num = 0
            elif in_str[i].isalpha():
                tmp_str += in_str[i]
                i += 1
        return tmp_str


if __name__ == '__main__':
    sol = Solution2()
    # print(sol.maximum_average_subarray_one([1,12,-5,-6,50,3], 4))
    # print(sol.flood_fill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2))
    # print(sol.longest_word_in_dictionary(["a", "w","wo","wor","worl", "world", "app", "appl", "ap", "apply", "apple"]))
    # print(sol.maximum_sub_array([-2,1,-3,4,-1,2,1,-5,4]))
    # print(sol.number_of_1_bit(20))
    # print(sol.binary_tree_paths_bfs(yield_nodes_tree(1)))
    # print(sol.symmetric_tree(yield_nodes_tree(1)))
    # print(sol.house_robber([1,12,5,6,50,3,8]))
    # print(sol.pascal_triangle(5))
    # print(sol.reverse_vowels_of_a_string('leetcode'))
    # print(sol.valid_perfect_square(14))
    # print(sol.repeated_substring_pattern('abdabcabdabc'))
    # print(sol.pascal_triangle_two(3))
    # print(sol.number_of_segments_in_a_string('Hello, my name is John'))
    # print(sol.arranging_coins(5))
    # print(sol.find_all_anagrams_in_a_string('cbaebabacd', 'abc'))
    # print(sol.perfect_number(10))
    # print(sol.word_pattern('abba', 'dog cat fish dog'))
    # re_head = sol.remove_linked_list_elements(produce_node_list([6,2,6,4,5,6]), 6)
    # loop_print_linked_value(re_head)
    # print(sol.sum_of_square_numbers(16))
    # print(sol.merge_sorted_array_in_place([1,3,5,7,9],[2,4,6,8]))
    # print(sol.range_sum_query_immutable(2,5))
    # print(sol.can_place_flowers([1,0,0,0,1], 2))
    # print(sol.nth_digit(12))
    # print(sol.reverse_bits(43261596))
    # print(sol.shortest_unsorted_continuous_subarray([2, 6, 4, 8, 10,  12, 9, 15]))
    # print(sol.k_diff_pairs_in_an_array([1, 3, 1, 5, 4], 0))
    # print(sol.largest_palindrome_product(3))
    # print(sol.rotate_array([1,2,3,4,5,6,7,8,9,10,11], 3))
    # print(sol.third_maximum_number([2, 2, 3]))
    # print(sol.nested_list_weight_sum([1,[4,[6]]]))
    # print(sol.logger_rate_limiter(1, 'foo'))
    # print(sol.logger_rate_limiter(5, 'foo'))
    # print(sol.logger_rate_limiter(12, 'foo'))
    # n_func = sol.moving_average_from_data_stream(3)
    # print(n_func(1))
    # print(n_func(2))
    # print(n_func(3))
    # print(n_func(5))
    # print(sol.palindrome_permutation('code'))
    # print(sol.flip_game('+++++'))
    # print(sol.shortest_word_distance(["practice", "practice", "makes", "perfect", "coding", "makes"], 'coding', 'practice'))
    # print(sol.meeting_rooms([[5, 10],[15, 20]]))
    # print(sol.strobogrammatic_number('898'))
    # print(sol.valid_word_square(["abc", "bnrt", "crm", "dt"]))
    # print(sol.valid_word_abbreviation('apple', '5'))
    ci = CompressedIterator('L1e10t1c1o1d1e1')
    print(ci.next())
    print(ci.has_next())
    print(ci.next())
    print(ci.has_next())
    print(ci.next())
    print(ci.next())
    print(ci.next())
