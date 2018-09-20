#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time    : 2017/12/2 16:45
@Author  : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site    : 
@File    : leetcode_medium_1.py
@desc    :
"""
from algorithm_files.leetcode1 import decorate_time, TreeNode, level_output_tree, yield_nodes_tree
from urllib.parse import urlparse


class MapSumPairs(object):
    """
    Implement a MapSum class with insert, and sum methods.
    For the method insert, you'll be given a pair of (string, integer).
    The string represents the key and the integer represents the value. If the key already existed,
    then the original key-value pair will be overridden to the new one.
    For the method sum, you'll be given a string representing the prefix,
    and you need to return the sum of all the pairs' value whose key starts with the prefix.
    Example 1:
    Input: insert("apple", 3), Output: Null
    Input: sum("ap"), Output: 3
    Input: insert("app", 2), Output: Null
    Input: sum("ap"), Output: 5
    """
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._dict = {}

    def insert(self, key, val):
        """
        :type key: str
        :type val: int
        :rtype: void
        """
        if key in self._dict:
            self._dict[key] = val
        else:
            self._dict[key] = val

    def sum(self, prefix):
        """
        :type prefix: str
        :rtype: int
        """
        from functools import reduce
        import operator
        return reduce(operator.add, map(lambda i: i[1] if i[0].startswith(prefix) else 0, self._dict.items()))


class MagicDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._word_cand = {}
        self._word_set = None

    def _candidate(self, word):
        """
        yield candidate word
        :param word:
        :return:
        """
        for i in range(len(word)):
            yield word[:i] + '*' + word[i+1:]

    def build_dict(self, in_dict):
        """
        Build a dictionary through a list of words
        :type in_dict: List[str]
        :rtype: void
        """
        from collections import Counter
        self._word_set = set(in_dict)
        self._word_cand = Counter(c for w in self._word_set for c in self._candidate(w))

    def search(self, word):
        """
        Returns if there is any word in the trie that equals to the given word after modifying exactly one character
        :type word: str
        :rtype: bool
        """
        return any(self._word_cand[c] >= 1 and word not in self._word_set for c in self._candidate(word))


class Solution3(object):
    def __init__(self):
        self.full_tiny = {}
        self.tiny_full = {}
        self.global_count = 0
        self.beat_cache = {}

    @decorate_time
    def sentence_similarity_two(self, in_words1, in_words2, in_pairs):
        """
        Given two sentences words1, words2 (each represented as an array of strings),
        and a list of similar word pairs pairs, determine if two sentences are similar.
        For example, words1 = ["great", "acting", "skills"] and words2 = ["fine", "drama", "talent"] are similar,
        if the similar word pairs are pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]].
        Note that the similarity relation is transitive.
        For example, if "great" and "good" are similar, and "fine" and "good" are similar, then "great" and "fine" are similar.
        Similarity is also symmetric. For example, "great" and "fine" being similar is the same as "fine" and "great" being similar.
        Also, a word is always similar with itself. For example, the sentences words1 = ["great"], words2 = ["great"],
        pairs = [] are similar, even though there are no specified similar word pairs.
        Finally, sentences can only be similar if they have the same number of words.
        So a sentence like words1 = ["great"] can never be similar to words2 = ["doubleplus","good"].
        Note:
        The length of words1 and words2 will not exceed 1000.
        The length of pairs will not exceed 2000.
        The length of each pairs[i] will be 2.
        The length of each words[i] and pairs[i][j] will be in the range [1, 20].
        :type in_words1: List[str]
        :type in_words2: List[str]
        :type in_pairs: List[List[str]]
        :rtype: bool
        """
        def is_in_pairs(ii, jj):
            i_p_l = []
            j_p_l = []
            for k in in_pairs:
                if ii in k:
                    i_p_l.extend(k)
                if jj in k:
                    j_p_l.extend(k)
            if set(i_p_l) & set(j_p_l):
                return True
            else:
                return False

        if len(in_words1) != len(in_words2):
            return False
        for i, j in zip(in_words1, in_words2):
            if i == j:
                continue
            elif is_in_pairs(i, j):
                continue
            else:
                return False
        else:
            return True

    @decorate_time
    def encode_and_decode_tiny_url(self, is_encode, in_url):
        """
        TinyURL is a URL shortening service where you enter a URL such as
        https://leetcode.com/problems/design-tinyurl and it returns a short URL such as http://tinyurl.com/4e9iAk.
        Design the encode and decode methods for the TinyURL service.
        There is no restriction on how your encode/decode algorithm should work.
        You just need to ensure that a URL can be encoded to a tiny URL and the tiny URL can be decoded to the original URL.
        :param is_encode:
        :param in_url:
        :return:

        use base62 encode
        """
        import string
        letters = string.ascii_letters + string.digits

        def decto62(dec):
            ans = ''
            while True:
                ans = letters[dec % 62] + ans
                dec //= 62
                if not dec:
                    break
            return ans

        def decode(url):
            idx = urlparse(url, scheme='http').path[1:]
            if idx in self.tiny_full:
                return self.tiny_full[idx]
            else:
                return None

        def encode(url):
            suffix = decto62(self.global_count)
            if url not in self.full_tiny:
                self.full_tiny[url] = suffix
                self.tiny_full[suffix] = url
                self.global_count += 1
            return "http://tinyurl.com/" + suffix

        if is_encode is True:
            return encode(in_url)
        else:
            return decode(in_url)

    @decorate_time
    def maximum_binary_tree(self, in_nums):
        """
        Given an integer array with no duplicates. A maximum tree building on this array is defined as follow:
        The root is the maximum number in the array.
        The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
        The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
        Construct the maximum tree by the given array and output the root node of this tree.
        Example 1:
        Input: [3,2,1,6,0,5]
        Output: return the tree root node representing the following tree:
              6
            /   \
           3     5
            \    /
             2  0
               \
                1
        Note:
        The size of the given array will be in the range [1,1000].
        :param in_nums:
        :return:
        """
        def find_max_index(in_list):
            max_index = 0
            for i in range(len(in_list)):
                if in_list[max_index] < in_list[i]:
                    max_index = i
            return max_index

        def dfs(in_list):
            if not in_list:
                return None
            m_index = find_max_index(in_list)
            i_root = TreeNode(in_list[m_index])
            i_root.left = dfs(in_list[:m_index])
            i_root.right = dfs(in_list[m_index+1:])
            return i_root

        return dfs(in_nums)

    @decorate_time
    def complex_number_multiplication(self, in_co_num1, in_co_num2):
        """
        Given two strings representing two complex numbers.
        You need to return a string representing their multiplication. Note i**2 = -1 according to the definition.
        Example 1:
        Input: "1+1i", "1+1i"
        Output: "0+2i"
        Explanation: (1 + i) * (1 + i) = 1 + i**2 + 2 * i = 2i, and you need convert it to the form of 0+2i.
        Example 2:
        Input: "1+-1i", "1+-1i"
        Output: "0+-2i"
        Explanation: (1 - i) * (1 - i) = 1 + i**2 - 2 * i = -2i, and you need convert it to the form of 0+-2i.
        Note:
        The input strings will not have extra blank.
        The input strings will be given in the form of a+bi,
        where the integer a and b will both belong to the range of [-100, 100].
        And the output should be also in this form.
        :param in_co_num1:
        :param in_co_num2:
        :return:
        """
        a1, b1 = map(int, in_co_num1[:-1].split('+'))
        a2, b2 = map(int, in_co_num2[:-1].split('+'))
        return '{}+{}i'.format(a1 * b1 - a2 * b2, a1 * b2 + a2 * b1)

    @decorate_time
    def count_bits(self, in_num):
        """
        Given a non negative integer number num.
        For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and
        return them as an array.
        Example:
        For num = 5 you should return [0,1,1,2,1,2].
        Follow up:
        It is very easy to come up with a solution with run time O(n*sizeof(integer)).
        But can you do it in linear time O(n) /possibly in a single pass?
        Space complexity should be O(n).
        Can you do it like a boss? Do it without using any builtin function
        like __builtin_popcount in c++ or in any other language.
        Credits:
        Special thanks to @ syedee for adding this problem and creating all test cases.
        :param in_num:
        :return:
        """
        res_list = []

        def count_bit(num):
            cnt = 0
            while num:
                cnt += 1
                num &= (num-1)
            return cnt

        for i in range(in_num+1):
            res_list.append(count_bit(i))
        return res_list

    @decorate_time
    def battle_ships_in_a_board(self, board):
        """
        Given an 2D board, count how many battleships are in it. The battleships are represented with 'X's,
        empty slots are represented with '.'s. You may assume the following rules:
        You receive a valid board, made of only battleships or empty slots.
        Battleships can only be placed horizontally or vertically. In other words,
        they can only be made of the shape 1xN (1 row, N columns) or Nx1 (N rows, 1 column), where N can be of any size.
        At least one horizontal or vertical cell separates between two battleships - there are no adjacent battleships.
        Example:
        X..X
        ...X
        ...X
        In the above board there are 2 battleships.
        Invalid Example:
        ...X
        XXXX
        ...X
        This is an invalid board that you will not receive - as
        battleships will always have a cell separating between them.
        Follow up:
        Could you do it in one-pass, using only O(1) extra memory and without modifying the value of the board?
        :type board: List[List[str]]
        :rtype: int
        """
        total = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'X':
                    flag = 1
                    if j > 0 and board[i][j - 1] == 'X':
                        flag = 0
                    if i > 0 and board[i - 1][j] == 'X':
                        flag = 0
                    total += flag
        return total

    @decorate_time
    def find_all_duplicates_in_an_array(self, in_nums):
        """
        Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.
        Find all the elements that appear twice in this array.
        Could you do it without extra space and in O(n) runtime?
        Example:
        Input:
        [4,3,2,7,8,2,3,1]
        Output:
        [2,3]
        :param in_nums:
        :return:
        """
        res = []
        for x in in_nums:
            if in_nums[abs(x)-1] < 0:
                res.append(abs(x))
            else:
                in_nums[abs(x)-1] *= -1
        return res

    @decorate_time
    def queue_reconstruction_by_height(self, in_people):
        """
        Suppose you have a random list of people standing in a queue.
        Each person is described by a pair of integers (h, k),
        where h is the height of the person and k is the number of people in front of this person
        who have a height greater than or equal to h. Write an algorithm to reconstruct the queue.
        Note:
        The number of people is less than 1,100.
        Example
        Input:
        [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
        Output:
        [[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
        :type in_people: List[List[int]]
        :rtype: List[List[int]]
        """
        in_people.sort(key=lambda x: (-x[0], x[1]))
        res_list = []
        for i in in_people:
            res_list.insert(i[1], i)
        return res_list

    @decorate_time
    def find_bottom_left_tree_value(self, in_root):
        """
        Given a binary tree, find the leftmost value in the last row of the tree.
        Example 1:
        Input:
            2
           / \
          1   3
        Output:
        1
        Example 2:
        Input:
                1
               / \
              2   3
             /   / \
            4   5   6
               /
              7
        Output:
        7
        Note: You may assume the tree (i.e., the given root node) is not NULL.
        :type in_root: TreeNode
        :rtype: int
        """
        level_list = []

        def dfs(node, height):
            if not node:
                return None
            if len(level_list) <= height:
                level_list.append([node.val])
            else:
                level_list[height].append(node.val)
            if node.left:
                dfs(node.left, height + 1)
            if node.right:
                dfs(node.right, height + 1)

        dfs(in_root, 0)
        return level_list[-1][0]

    @decorate_time
    def single_element_in_a_sorted_array(self, in_nums):
        """
        Given a sorted array consisting of only integers where every element appears twice except for one element
        which appears once. Find this single element that appears only once.
        Example 1:
        Input: [1,1,2,2,3,3,4,8,8]
        Output: 2
        Example 2:
        Input: [3,3,7,7,10,11,11]
        Output: 10
        Note: Your solution should run in O(log n) time and O(1) space.
        :type in_nums: List[int]
        :rtype: int
        """
        lo, hi = 0, len(in_nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if in_nums[mid] == in_nums[mid ^ 1]:
                lo = mid + 1
            else:
                hi = mid
        return in_nums[lo]

    @decorate_time
    def palindromic_substrings(self, in_str):
        """
        Given a string, your task is to count how many palindromic substrings in this string.
        The substrings with different start indexes or end indexes are counted as
        different substrings even they consist of same characters.
        Example 1:
        Input: "abc"
        Output: 3
        Explanation: Three palindromic strings: "a", "b", "c".
        Example 2:
        Input: "aaa"
        Output: 6
        Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
        Note:
        The input string length won't exceed 1000.
        :param in_str: str
        :return: int
        """
        s_str = '#' + '#'.join(in_str) + '#'
        r_l = [0] * len(s_str)
        max_right = 0
        pos = 0

        for i in range(len(s_str)):
            if i < max_right:
                r_l[i] = min(r_l[2 * pos - i], max_right - i)
            else:
                r_l[i] = 1
            # 尝试扩展，注意处理边界
            while i - r_l[i] >= 0 and i + r_l[i] < len(s_str) and s_str[i - r_l[i]] == s_str[i + r_l[i]]:
                r_l[i] += 1
            # 更新MaxRight,pos
            if r_l[i] + i - 1 > max_right:
                max_right = r_l[i] + i - 1
                pos = i
        return len([i - 1 for i in r_l if i - 1 > 0])

    @decorate_time
    def find_largest_value_in_each_tree_row(self, in_root):
        """
        You need to find the largest value in each row of a binary tree.
        Example:
        Input:
                  1
                 / \
                3   2
               / \   \
              5   3   9
        Output: [1, 3, 9]
        :param in_root:
        :return:
        """
        max_list = []

        def dfs(node, height):
            if not node:
                return None
            if len(max_list) <= height:
                max_list.append(node.val)
            else:
                if max_list[height] < node.val:
                    max_list[height] = node.val
            if node.left:
                dfs(node.left, height + 1)
            if node.right:
                dfs(node.right, height + 1)

        dfs(in_root, 0)
        return max_list

    @decorate_time
    def most_frequent_subtree_sum(self, in_root):
        """
        Given the root of a tree, you are asked to find the most frequent subtree sum.
        The subtree sum of a node is defined as the sum of all the node values formed by the subtree rooted at that node
        (including the node itself). So what is the most frequent subtree sum value? If there is a tie,
        return all the values with the highest frequency in any order.
        Examples 1
        Input:
          5
         /  \
        2   -3
        return [2, -3, 4], since all the values happen only once, return all of them in any order.
        Examples 2
        Input:
          5
         /  \
        2   -5
        return [2], since 2 happens twice, however -5 only occur once.
        Note: You may assume the sum of values in any subtree is in the range of 32-bit signed integer.
        :type in_root: TreeNode
        :rtype: List[int]
        """
        from collections import Counter
        node_sum_list = []

        def sub_tree_sum(node):
            if not node:
                return 0
            r_sum = node.val + sub_tree_sum(node.left) + sub_tree_sum(node.right)
            return r_sum

        def dfs_tree(node):
            if not node:
                return None
            node_sum_list.append(sub_tree_sum(node))
            dfs_tree(node.left)
            dfs_tree(node.right)

        dfs_tree(in_root)
        sum_dict = Counter(node_sum_list)

        return [k for k, v in sum_dict.items() if v == max(sum_dict.values())]

    @decorate_time
    def find_duplicate_file_in_system(self, in_paths):
        """
        Given a list of directory info including directory path, and all the files with contents in this directory,
        you need to find out all the groups of duplicate files in the file system in terms of their paths.
        A group of duplicate files consists of at least two files that have exactly the same content.
        A single directory info string in the input list has the following format:
        "root/d1/d2/.../dm f1.txt(f1_content) f2.txt(f2_content) ... fn.txt(fn_content)"
        It means there are n files (f1.txt, f2.txt ... fn.txt with content f1_content, f2_content ... fn_content,
        respectively) in directory root/d1/d2/.../dm. Note that n >= 1 and m >= 0. If m = 0, it means the directory
        is just the root directory.
        The output is a list of group of duplicate file paths.
        For each group, it contains all the file paths of the files that have the same content.
        A file path is a string that has the following format:
        "directory_path/file_name.txt"
        Example 1:
        Input:
        ["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]
        Output:
        [["root/a/2.txt","root/c/d/4.txt","root/4.txt"],["root/a/1.txt","root/c/3.txt"]]
        Note:
        No order is required for the final output.
        You may assume the directory name, file name and file content only has letters and digits,
        and the length of file content is in the range of [1,50].
        The number of files given is in the range of [1,20000].
        You may assume no files or directories share the same name in the same directory.
        You may assume each given directory info represents a unique directory.
        Directory path and file info are separated by a single blank space.
        Follow-up beyond contest:
        Imagine you are given a real file system, how will you search files? DFS or BFS?
        If the file content is very large (GB level), how will you modify your solution?
        If you can only read the file by 1kb each time, how will you modify your solution?
        What is the time complexity of your modified solution? What is the most time-consuming part and memory
        consuming part of it? How to optimize?
        How to make sure the duplicated files you find are not false positive?
        :type in_paths: List[str]
        :rtype: List[List[str]]
        """
        con_dict = {}

        def get_content(i_str):
            for i in range(len(i_str)):
                if i_str[i] == '(':
                    return i, i_str[i:]
            else:
                return 0, ''

        def dir_deal(i_str):
            s_list = i_str.split()
            for i in s_list[1:]:
                p, c = get_content(i)
                if c not in con_dict:
                    con_dict[c] = ['/'.join([s_list[0], i[:p]])]
                else:
                    con_dict[c].append('/'.join([s_list[0], i[:p]]))

        for j in in_paths:
            dir_deal(j)

        return [i for i in con_dict.values()]

    @decorate_time
    def single_number_three(self, in_nums):
        """
        Given an array of numbers nums, in which exactly two elements appear only once and all the other elements
        appear exactly twice. Find the two elements that appear only once.
        For example:
        Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].
        Note:
        The order of the result is not important. So in the above example, [5, 3] is also correct.
        Your algorithm should run in linear runtime complexity.
        Could you implement it using only constant space complexity?
        Credits:
        Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
        :type in_nums: List[int]
        :rtype: List[int]
        """
        diff = in_nums[0]
        for i in in_nums[1:]:
            diff ^= i
        diff &= ~(diff - 1)

        num1 = 0
        num2 = 0
        for i in in_nums:
            if i & diff > 0:
                num1 ^= i
            else:
                num2 ^= i
        return [num1, num2]

    @decorate_time
    def beautiful_arrangement(self, in_n):
        """
        Suppose you have N integers from 1 to N. We define a beautiful arrangement as an array that is constructed by
        these N numbers successfully if one of the following is true for the ith position (1 <= i <= N) in this array:
        The number at the ith position is divisible by i.
        i is divisible by the number at the ith position.
        Now given N, how many beautiful arrangements can you construct?
        Example 1:
        Input: 2
        Output: 2
        Explanation:
        The first beautiful arrangement is [1, 2]:
        Number at the 1st position (i=1) is 1, and 1 is divisible by i (i=1).
        Number at the 2nd position (i=2) is 2, and 2 is divisible by i (i=2).
        The second beautiful arrangement is [2, 1]:
        Number at the 1st position (i=1) is 2, and 2 is divisible by i (i=1).
        Number at the 2nd position (i=2) is 1, and i (i=2) is divisible by 1.
        Note:
        N is a positive integer and will not exceed 15.
        :type in_n: int
        :rtype: int
        """
        def helper(i, X):
            if i == 1:
                return 1
            key = (i, X)
            if key in self.beat_cache:
                return self.beat_cache[key]
            total = 0
            for j in range(len(X)):
                if X[j] % i == 0 or i % X[j] == 0:
                    total += helper(i - 1, X[:j] + X[j + 1:])
            self.beat_cache[key] = total
            return total
        return helper(in_n, tuple(range(1, in_n + 1)))

    @decorate_time
    def arithmetic_slices(self, in_nums):
        """
        A sequence of number is called arithmetic if it consists of at least three elements and
        if the difference between any two consecutive elements is the same.
        For example, these are arithmetic sequence:
        1, 3, 5, 7, 9
        7, 7, 7, 7
        3, -1, -5, -9
        The following sequence is not arithmetic.
        1, 1, 2, 5, 7
        A zero-indexed array A consisting of N numbers is given. A slice of that array is any pair of integers
        (P, Q) such that 0 <= P < Q < N.
        A slice (P, Q) of array A is called arithmetic if the sequence:
        A[P], A[p + 1], ..., A[Q - 1], A[Q] is arithmetic. In particular, this means that P + 1 < Q.
        The function should return the number of arithmetic slices in the array A.
        Example:
        A = [1, 2, 3, 4]
        return: 3, for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and [1, 2, 3, 4] itself.
        :param in_nums: List[int]
        :return: int
        """
        if sum(in_nums) != len(in_nums) * (in_nums[0] + in_nums[-1])//2:
            return 0
        total = 1

        for i in range(3, len(in_nums)):
            for j in range(len(in_nums) - i + 1):
                total += 1
        return total

    @decorate_time
    def optimal_division(self, in_nums):
        """
        Given a list of positive integers, the adjacent integers will perform the float division. For example,
         [2,3,4] -> 2 / 3 / 4.
        However, you can add any number of parenthesis at any position to change the priority of operations.
        You should find out how to add parenthesis to get the maximum result, and return the corresponding expression
        in string format. Your expression should NOT contain redundant parenthesis.
        Example:
        Input: [1000,100,10,2]
        Output: "1000/(100/10/2)"
        Explanation:
        1000/(100/10/2) = 1000/((100/10)/2) = 200
        However, the bold parenthesis in "1000/((100/10)/2)" are redundant,
        since they don't influence the operation priority. So you should return "1000/(100/10/2)".
        Other cases:
        1000/(100/10)/2 = 50
        1000/(100/(10/2)) = 50
        1000/100/10/2 = 0.5
        1000/100/(10/2) = 2
        Note:
        The length of the input array is [1, 10].
        Elements in the given array will be in range [2, 1000].
        There is only one optimal division for each test case.
        :param in_nums: List[int]
        :return: str
        """
        s_list = [i for i in map(str, in_nums)]
        if len(s_list) <= 2:
            return '/'.join(s_list)
        return '{}/({})'.format(s_list[0], '/'.join(s_list[1:]))

    @decorate_time
    def minimum_moves_to_equal_array_elements_two(self, in_nums):
        """
        Given a non-empty integer array, find the minimum number of moves required to make all array elements equal,
        where a move is incrementing a selected element by 1 or decrementing a selected element by 1.
        You may assume the array's length is at most 10,000.
        Example:
        Input:
        [1,2,3]
        Output:
        2
        Explanation:
        Only two moves are needed (remember each move increments or decrements one element):
        [1,2,3]  =>  [2,2,3]  =>  [2,2,2]
        :param in_nums:
        :return:
        """
        median = sorted(in_nums)[len(in_nums) // 2]
        return sum(abs(num - median) for num in in_nums)

    @decorate_time
    def beautiful_arrangement_two(self, n, k):
        """
        Given two integers n and k,
        you need to construct a list which contains n different positive integers ranging from 1 to n and
        obeys the following requirement:
        Suppose this list is [a1, a2, a3, ... , an], then the list [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - an|]
        has exactly k distinct integers.
        If there are multiple answers, print any of them.
        Example 1:
        Input: n = 3, k = 1
        Output: [1, 2, 3]
        Explanation: The [1, 2, 3] has three different positive integers ranging from 1 to 3,
        and the [1, 1] has exactly 1 distinct integer: 1.
        Example 2:
        Input: n = 3, k = 2
        Output: [1, 3, 2]
        Explanation: The [1, 3, 2] has three different positive integers ranging from 1 to 3,
        and the [2, 1] has exactly 2 distinct integers: 1 and 2.
        Note:
        The n and k are in the range 1 <= k < n <= 104.
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        ans = range(1, n - k)
        for d in range(k + 1):
            if d % 2 == 0:
                ans.append(n - k + d // 2)
            else:
                ans.append(n - d // 2)
        return ans

    @decorate_time
    def minimum_ascii_delete_sum_for_two_strings(self, s1, s2):
        """
        Given two strings s1, s2, find the lowest ASCII sum of deleted characters to make two strings equal.
        Example 1:
        Input: s1 = "sea", s2 = "eat"
        Output: 231
        Explanation: Deleting "s" from "sea" adds the ASCII value of "s" (115) to the sum.
        Deleting "t" from "eat" adds 116 to the sum.
        At the end, both strings are equal, and 115 + 116 = 231 is the minimum sum possible to achieve this.
        Example 2:
        Input: s1 = "delete", s2 = "leet"
        Output: 403
        Explanation: Deleting "dee" from "delete" to turn the string into "let",
        adds 100[d]+101[e]+101[e] to the sum.  Deleting "e" from "leet" adds 101[e] to the sum.
        At the end, both strings are equal to "let", and the answer is 100+101+101+101 = 403.
        If instead we turned both strings into "lee" or "eet", we would get answers of 433 or 417, which are higher.
        Note:
        0 < s1.length, s2.length <= 1000.
        All elements of each string will have an ASCII value in [97, 122].
        :type s1: str
        :type s2: str
        :rtype: int
        """
        l1 = len(s1)
        l2 = len(s2)
        dp = [[0] * (l2 + 1) for i in range(l1 + 1)]
        for i in range(l1):
            for j in range(l2):
                if s1[i] == s2[j]:
                    dp[i+1][j+1] = dp[i][j] + ord(s1[i]) * 2
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        n1 = sum(ord(c) for c in s1)
        n2 = sum(ord(c) for c in s2)
        return n1 + n2 - dp[l1][l2]

    @decorate_time
    def sort_by_frequency(self, s):
        """
        Given a string, sort it in decreasing order based on the frequency of characters.
        Example 1:
        Input:
        "tree"
        Output:
        "eert"
        Explanation:
        'e' appears twice while 'r' and 't' both appear once.
        So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.
        Example 2:
        Input:
        "cccaaa"
        Output:
        "cccaaa"
        Explanation:
        Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
        Note that "cacaca" is incorrect, as the same characters must be together.
        Example 3:
        Input:
        "Aabb"
        Output:
        "bbAa"
        Explanation:
        "bbaA" is also a valid answer, but "Aabb" is incorrect.
        Note that 'A' and 'a' are treated as two different characters.
        :type s: str
        :rtype: str
        """
        from collections import Counter
        res_list = []
        f_dict = Counter(s)
        f_list = sorted(set(f_dict.values()), reverse=True)
        for i in f_list:
            temp = []
            for k, v in f_dict.items():
                if v == i:
                    temp.append(k)
            temp.sort()
            for j in temp:
                res_list.extend([j]*i)
        return ''.join(res_list)

    @decorate_time
    def product_of_array_except_self(self, nums):
        """
        Given an array of n integers where n > 1, nums,
        return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].
        Solve it without division and in O(n).
        For example, given [1,2,3,4], return [24,12,8,6].
        Follow up:
        Could you solve it with constant space complexity?
        (Note: The output array does not count as extra space for the purpose of space complexity analysis.)
        :type nums: List[int]
        :rtype: List[int]
        """
        from functools import reduce
        import operator
        return [reduce(operator.mul, nums[:i]+nums[i+1:]) for i in range(len(nums))]

    @decorate_time
    def print_binary_tree(self, in_root):
        """
        Print a binary tree in an m*n 2D string array following these rules:
        The row number m should be equal to the height of the given binary tree.
        The column number n should always be an odd number.
        The root node's value (in string format) should be put in the exactly middle of the first row it can be put.
        The column and the row where the root node belongs will separate the rest space into two parts
        (left-bottom part and right-bottom part). You should print the left subtree in the left-bottom part and
        print the right subtree in the right-bottom part. The left-bottom part and the right-bottom part should
        have the same size. Even if one subtree is none while the other is not, you don't need to print anything
        for the none subtree but still need to leave the space as large as that for the other subtree. However,
        if two subtrees are none, then you don't need to leave space for both of them.
        Each unused space should contain an empty string "".
        Print the subtrees following the same rules.
        Example 1:
        Input:
             1
            /
           2
        Output:
        [["", "1", ""],
         ["2", "", ""]]
        Example 2:
        Input:
             1
            / \
           2   3
            \
             4
        Output:
        [["", "", "", "1", "", "", ""],
         ["", "2", "", "", "", "3", ""],
         ["", "", "4", "", "", "", ""]]
        Example 3:
        Input:
              1
             / \
            2   5
           /
          3
         /
        4
        Output:
        [["",  "",  "", "",  "", "", "", "1", "",  "",  "",  "",  "", "", ""]
         ["",  "",  "", "2", "", "", "", "",  "",  "",  "",  "5", "", "", ""]
         ["",  "3", "", "",  "", "", "", "",  "",  "",  "",  "",  "", "", ""]
         ["4", "",  "", "",  "", "", "", "",  "",  "",  "",  "",  "", "", ""]]
        Note: The height of binary tree is in the range of [1, 10].
        :param in_root:
        :return:
        """
        def get_height(node):
            if not node:
                return 0
            return 1 + max(get_height(node.ledt), get_height(node.right))
        rows = get_height(in_root)
        cols = 2 ** rows - 1
        res = [['' for _ in range(cols)] for _ in range(rows)]

        def traverse(node, level, pos):
            if not node:
                return
            left_padding, spacing = 2 ** (rows - level - 1) - 1, 2 ** (rows - level) - 1
            index = left_padding + pos * (spacing + 1)
            print(level, index, node.val)
            res[level][index] = str(node.val)
            traverse(node.left, level + 1, pos << 1)
            traverse(node.right, level + 1, (pos << 1) + 1)
        traverse(in_root, 0, 0)
        return res

    @decorate_time
    def split_linked_list_in_parts(self, root, k):
        """
        Given a (singly) linked list with head node root,
        write a function to split the linked list into k consecutive linked list "parts".
        The length of each part should be as equal as possible:
        no two parts should have a size differing by more than 1. This may lead to some parts being null.
        The parts should be in order of occurrence in the input list,
        and parts occurring earlier should always have a size greater than or equal parts occurring later.
        Return a List of ListNode's representing the linked list parts that are formed.
        Examples 1->2->3->4, k = 5 // 5 equal parts [ [1], [2], [3], [4], null ]
        Example 1:
        Input:
        root = [1, 2, 3], k = 5
        Output: [[1],[2],[3],[],[]]
        Explanation:
        The input and each element of the output are ListNodes, not arrays.
        For example, the input root has root.val = 1, root.next.val = 2, \root.next.next.val = 3,
         and root.next.next.next = null.
        The first element output[0] has output[0].val = 1, output[0].next = null.
        The last element output[4] is null, but it's string representation as a ListNode is [].
        Example 2:
        Input:
        root = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], k = 3
        Output: [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
        Explanation:
        The input has been split into consecutive parts with size difference at most 1,
        and earlier parts are a larger size than the later parts.
        Note:
        The length of root will be in the range [0, 1000].
        Each value of a node in the input will be an integer in the range [0, 999].
        k will be an integer in the range [1, 50].
        :type root: ListNode
        :type k: int
        :rtype: List[ListNode]
        """
        curr, length = root, 0
        while curr:
            curr, length = curr.next, length + 1
        # Determine the length of each chunk
        chunk_size, longer_chunks = length // k, length % k
        res = [chunk_size + 1] * longer_chunks + [chunk_size] * (k - longer_chunks)
        # Split up the list
        prev, curr = None, root
        for index, num in enumerate(res):
            if prev:
                prev.next = None
            res[index] = curr
            for i in range(num):
                prev, curr = curr, curr.next
        return res

    @decorate_time
    def friend_circles(self, M):
        """
        There are N students in a class. Some of them are friends, while some are not.
        Their friendship is transitive in nature. For example,
        if A is a direct friend of B, and B is a direct friend of C,
        then A is an indirect friend of C.
        And we defined a friend circle is a group of students who are direct or indirect friends.
        Given a N*N matrix M representing the friend relationship between students in the class. If M[i][j] = 1,
        then the ith and jth students are direct friends with each other,
        otherwise not. And you have to output the total number of friend circles among all the students.
        Example 1:
        Input:
        [[1,1,0],
         [1,1,0],
         [0,0,1]]
        Output: 2
        Explanation:The 0th and 1st students are direct friends, so they are in a friend circle.
        The 2nd student himself is in a friend circle. So return 2.
        Example 2:
        Input:
        [[1,1,0],
         [1,1,1],
         [0,1,1]]
        Output: 1
        Explanation:The 0th and 1st students are direct friends, the 1st and 2nd students are direct friends,
        so the 0th and 2nd students are indirect friends. All of them are in the same friend circle, so return 1.
        Note:
        N is in range [1,200].
        M[i][i] = 1 for all students.
        If M[i][j] = 1, then M[j][i] = 1.
        :type M: List[List[int]]
        :rtype: int
        """
        N = len(M)
        seen = set()

        def dfs(node):
            for nei, adj in enumerate(M[node]):
                if adj and nei not in seen:
                    seen.add(nei)
                    dfs(nei)

        ans = 0
        for i in range(N):
            if i not in seen:
                dfs(i)
                ans += 1
        return ans

    @decorate_time
    def array_nesting(self, nums):
        """
        A zero-indexed array A of length N contains all integers from 0 to N-1.
        Find and return the longest length of set S, where S[i] = {A[i], A[A[i]], A[A[A[i]]], ... }
        subjected to the rule below.
        Suppose the first element in S starts with the selection of element A[i] of index = i,
        the next element in S should be A[A[i]], and then A[A[A[i]]]… By that analogy,
        we stop adding right before a duplicate element occurs in S.
        Example 1:
        Input: A = [5,4,0,3,1,6,2]
        Output: 6
        Explanation:
        A[0] = 5, A[1] = 4, A[2] = 0, A[3] = 3, A[4] = 1, A[5] = 6, A[6] = 2.
        One of the longest S[K]:
        S[0] = {A[0], A[5], A[6], A[2]} = {5, 6, 2, 0}
        Note:
        N is an integer within the range [1, 20,000].
        The elements of A are all distinct.
        Each element of A is an integer within the range [0, N-1].
        :type nums: List[int]
        :rtype: int
        """
        set_list = []
        for i in range(len(nums)):
            temp_set = set()
            j = i
            while True:
                if nums[j] in temp_set:
                    set_list.append(temp_set)
                    break
                else:
                    temp_set.add(nums[j])
                    j = nums[j]
        for i in set_list:
            print(i)
        return max([len(i) for i in set_list])

    @decorate_time
    def top_k_frequent_elements(self, nums, k):
        """
        Given a non-empty array of integers, return the k most frequent elements.
        For example,
        Given [1,1,1,2,2,3] and k = 2, return [1,2].
        Note:
        You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
        Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        from collections import Counter
        n_dict = Counter(nums)
        res_list = [i for i, v in n_dict.items() if v >= k]
        return res_list

    @decorate_time
    def binary_tree_inorder_traversal(self, in_root):
        """
        Given a binary tree, return the inorder traversal of its nodes' values.
        For example:
        Given binary tree [1,null,2,3],
           1
            \
             2
            /
           3
        return [1,3,2].
        Note: Recursive solution is trivial, could you do it iteratively?
        :param in_root:
        :return:
        """
        # Recursive solution
        res_list = []

        # def dfs(node):
        #     if not node:
        #         return None
        #     dfs(node.left)
        #     res_list.append(node.val)
        #     dfs(node.right)
        # dfs(in_root)
        # return res_list

        def bfs(node):
            stack = []
            while stack or node:
                if node:
                    stack.append(node)
                    node = node.left
                else:
                    node = stack.pop(-1)
                    res_list.append(node.val)
                    node = node.right
        bfs(in_root)
        return res_list

    @decorate_time
    def total_hamming_distance(self, nums):
        """
        The Hamming distance between two integers is the number of positions at which the corresponding bits are different.
        Now your job is to find the total Hamming distance between all pairs of the given numbers.
        Example:
        Input: 4, 14, 2
        Output: 6
        Explanation: In binary representation, the 4 is 0100, 14 is 1110, and 2 is 0010 (just
        showing the four bits relevant in this case). So the answer will be:
        HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.
        Note:
        Elements of the given array are in the range of 0 to 10^9
        Length of the array will not exceed 10^4.
        :type nums: List[int]
        :rtype: int
        """
        from itertools import combinations
        total_hamming = 0
        for i in combinations(nums, 2):
            temp = i[0] ^ i[1]
            res = 0
            while temp:
                res += 1
                temp &= (temp - 1)
            total_hamming += res
        return total_hamming

    @decorate_time
    def maximum_length_of_pair_chain(self, pairs):
        """
        You are given n pairs of numbers. In every pair, the first number is always smaller than the second number.
        Now, we define a pair (c, d) can follow another pair (a, b) if and only if b < c.
        Chain of pairs can be formed in this fashion.
        Given a set of pairs, find the length longest chain which can be formed.
        You needn't use up all the given pairs. You can select pairs in any order.
        Example 1:
        Input: [[1,2], [2,3], [3,4]]
        Output: 2
        Explanation: The longest chain is [1,2] -> [3,4]
        Note:
        The number of given pairs will be in the range [1, 1000].
        :type pairs: List[List[int]]
        :rtype: int
        """
        cur, res = float('-inf'), 0
        for p in sorted(pairs, key=lambda x: x[1]):
            if cur < p[0]:
                cur, res = p[1], res + 1
        return res

    @decorate_time
    def next_greater_element_two(self, nums):
        """
        Given a circular array (the next element of the last element is the first element of the array),
        print the Next Greater Number for every element.
         The Next Greater Number of a number x is the first greater
        number to its traversing-order next in the array,
         which means you could search circularly to find its next greater number.
          If it doesn't exist, output -1 for this number.
        Example 1:
        Input: [1,2,1]
        Output: [2,-1,2]
        Explanation: The first 1's next greater number is 2;
        The number 2 can't find next greater number;
        The second 1's next greater number needs to search circularly, which is also 2.
        Note: The length of given array won't exceed 10000.
        :type nums: List[int]
        :rtype: List[int]
        """
        stack, res = [], [-1] * len(nums)
        nums_list = [i for i in range(len(nums))] * 2
        for i in nums_list:
            while stack and (nums[stack[-1]] < nums[i]):
                res[stack.pop()] = nums[i]
            stack.append(i)
        return res

    @decorate_time
    def task_scheduler(self, tasks, n):
        """
        Given a char array representing tasks CPU need to do.
        It contains capital letters A to Z where different letters represent different tasks.
        Tasks could be done without original order. Each task could be done in one interval.
        For each interval, CPU could finish one task or just be idle.
        However, there is a non-negative cooling interval n that means between two same tasks,
        there must be at least n intervals that CPU are doing different tasks or just be idle.
        You need to return the least number of intervals the CPU will take to finish all the given tasks.
        Example 1:
        Input: tasks = ["A","A","A","B","B","B"], n = 2
        Output: 8
        Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
        Note:
        The number of tasks is in the range [1, 10000].
        The integer n is in the range [0, 100].
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        from collections import Counter, OrderedDict
        # error solution
        # t_dict = Counter(tasks)
        # ord_dict = OrderedDict(sorted(t_dict.items(), key=lambda t: t[1], reverse=True))
        # sent = max(ord_dict.values())
        # task_list = []
        # while sent > 0:
        #     t_str = ''
        #     for k, v in ord_dict.items():
        #         if v != 0:
        #             ord_dict[k] -= 1
        #             t_str += k
        #     if len(task_list) > 0:
        #         if task_list[-1] == t_str:
        #             task_list.append('$')
        #             task_list.append(t_str)
        #         else:
        #             task_list.append(t_str)
        #     else:
        #         task_list.append(t_str)
        #     sent -= 1
        # return sum([len(i) for i in task_list])
        task_counts = [i for i in Counter(tasks).values()]
        m = max(task_counts)
        mct = task_counts.count(m)
        return max(len(tasks), (m - 1) * (n + 1) + mct)

    @decorate_time
    def add_one_row_to_tree(self, root, v, d):
        """
        Given the root of a binary tree, then value v and depth d,
        you need to add a row of nodes with value v at the given depth d. The root node is at depth 1.
        The adding rule is: given a positive integer depth d, for each NOT null tree nodes N in depth d-1,
        create two tree nodes with value v as N's left subtree root and right subtree root.
        And N's original left subtree should be the left subtree of the new left subtree root,
        its original right subtree should be the right subtree of the new right subtree root.
        If depth d is 1 that means there is no depth d-1 at all,
        then create a tree node with value v as the new root of the whole original tree,
        and the original tree is the new root's left subtree.
        Example 1:
        Input:
        A binary tree as following:
               4
             /   \
            2     6
           / \   /
          3   1 5
        v = 1
        d = 2
        Output:
               4
              / \
             1   1
            /     \
           2       6
          / \     /
         3   1   5

        Example 2:
        Input:
        A binary tree as following:
              4
             /
            2
           / \
          3   1
        v = 1
        d = 3
        Output:
              4
             /
            2
           / \
          1   1
         /     \
        3       1
        Note:
        The given d is in range [1, maximum depth of the given tree + 1].
        The given binary tree has at least one tree node.
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        dummy, dummy.left = TreeNode(None), root
        row = [dummy]
        for _ in range(d - 1):
            row = [kid for node in row for kid in (node.left, node.right) if kid]
        for node in row:
            node.left, node.left.left = TreeNode(v), node.left
            node.right, node.right.right = TreeNode(v), node.right
        return dummy.left

    @decorate_time
    def linked_list_cycle_second(self, in_head):
        """
        Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
        Note: Do not modify the linked list.
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
        except Exception:
            return None
        slow = slow.next
        while in_head is not slow:
            in_head = in_head.next
            slow = slow.next
        return in_head

    @decorate_time
    def subarray_sum_equal_k(self, nums, k):
        """
        Given an array of integers and an integer k,
        you need to find the total number of continuous subarrays whose sum equals to k.
        Example 1:
        Input:nums = [1,1,1], k = 2
        Output: 2
        Note:
        The length of the array is in range [1, 20,000].
        The range of numbers in the array is [-1000, 1000] and the range of the integer k is [-1e7, 1e7].
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        count, cur, res = {0: 1}, 0, 0
        for v in nums:
            cur += v
            res += count.get(cur - k, 0)
            count[cur] = count.get(cur, 0) + 1
        return res

    @decorate_time
    def target_sum(self, nums, in_sum):
        """
        You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -.
        For each integer, you should choose one from + and - as its new symbol.
        Find out how many ways to assign symbols to make sum of integers equal to target S.
        Example 1:
        Input: nums is [1, 1, 1, 1, 1], S is 3.
        Output: 5
        Explanation:
        -1+1+1+1+1 = 3
        +1-1+1+1+1 = 3
        +1+1-1+1+1 = 3
        +1+1+1-1+1 = 3
        +1+1+1+1-1 = 3
        There are 5 ways to assign symbols to make the sum of nums be target 3.
        Note:
        The length of the given array is positive and will not exceed 20.
        The sum of elements in the given array will not exceed 1000.
        Your output answer is guaranteed to be fitted in a 32-bit integer.
        :param nums:
        :param in_sum:
        :return: int
        """
        from collections import defaultdict
        memo = {0: 1}
        for x in nums:
            m = defaultdict(int)
            for s, n in memo.items():
                m[s + x] += n
                m[s - x] += n
            memo = m
        return memo[in_sum]


def get_quasi_constant(in_list):
    """
    :param in_list: [6,6,7,8,9,10]
    :return: int: 3
    """
    from collections import Counter
    cnt_dict = Counter(in_list)
    max_couple = [0, 0]
    for k, v in cnt_dict.items():
        if v > 1 and v > max_couple[1]:
            max_couple[0] = k
            max_couple[1] = v
    if max_couple[1] == 0:
        return 0
    else:
        return max_couple[1] + max(cnt_dict.get(max_couple[0] - 1, 0), cnt_dict.get(max_couple[0] + 1, 0))


def get_file_type_size_statistics(in_str):
    """
    :param in_str: str
    :return: str
    """
    type_dict = {
        'mp3': 'music',
        'aac': 'music',
        'flac': 'music',
        'jpg': 'image',
        'bmp': 'image',
        'gif': 'image',
        'mp4': 'movie',
        'avi': 'movie',
        'mkv': 'movie',
        '7z': 'other',
        'text': 'other',
        'zip': 'other',
    }
    # 统计字典
    stat_dict = {
        'music': 0,
        'image': 0,
        'movie': 0,
        'other': 0
    }
    # 分成单行数组
    sep_lines = [i for i in in_str.strip().split('\n')]
    print(sep_lines, len(sep_lines))
    for i in sep_lines:
        if i:
            f_name, f_size = i.split(' ')
            f_ext = f_name.split('.')[-1]
            f_size = int(f_size[:-1])
            ext_type = type_dict.get(f_ext, 'other')
            stat_dict[ext_type] += f_size
    # 字典生成字符串
    ret_str = ''
    for k, v in stat_dict.items():
        if k in ('image', 'movie'):
            ret_str += k +'s ' + str(v) + 'b\n'
        else:
            ret_str += k +' ' + str(v) + 'b\n'
    return ret_str

def trans_time_to_int(in_d, in_t):
    day_dict = {
        'Mon': 0,
        'Tue': 1,
        'Wed': 2,
        'Thu': 3,
        'Fri': 4,
        'Sat': 5,
        'Sun': 6
    }
    t_hour, t_min = in_t.split(':')
    return day_dict[in_d] * 24 * 60 + int(t_hour) * 60 + int(t_min)

def get_longest_sleeping_time(in_str):
    """
    将所有的时间映射成数字轴上的数字，找到最大区间即可
    :param in_str:
    :return: int
    """
    def trans_time_to_int(in_d, in_t):
        cnt_dict = {
            'Mon': 0,
            'Tue': 1,
            'Wed': 2,
            'Thu': 3,
            'Fri': 4,
            'Sat': 5,
            'Sun': 6
        }
        t_hour, t_min = in_t.split(':')
        return cnt_dict[in_d] * 24 * 60 + int(t_hour) * 60 + int(t_min)

    sep_lines = [i for i in in_str.strip().split('\n')]
    day_dict = {
        'Mon': [],
        'Tue': [],
        'Wed': [],
        'Thu': [],
        'Fri': [],
        'Sat': [],
        'Sun': []
    }
    for i in sep_lines:
        f_day, t_dure = i.split(' ')
        t1, t2 = t_dure.split('-')
        day_dict[f_day].append([trans_time_to_int(f_day, t1), trans_time_to_int(f_day, t2)])
    res_list = []
    for k, v in day_dict.items():
        for i in v:
            res_list.append(i[0])
            res_list.append(i[1])
    print(day_dict)

    ret_max = 0
    print(res_list)
    res_list.insert(0, 0)
    res_list.append(24 * 7 * 60)
    print(res_list)
    for i in range(len(res_list)//2):

        if res_list[2*i + 1] - res_list[2*i] > ret_max:
            ret_max = res_list[2*i + 1] - res_list[2*i]
    print(ret_max)


if __name__ == '__main__':
    input_str = """Mon 01:00-23:00
Tue 01:00-23:00
Wed 01:00-23:00
Thu 01:00-23:00
Fri 01:00-23:00
Sat 01:00-23:00
Sun 01:00-21:00
"""
    get_longest_sleeping_time(input_str)



# if __name__ == '__main__':
#     sol = Solution3()
#     # print(sol.sentence_similarity_two(["great", "acting", "skills"], ["fine", "drama", "talent"],
#     #                                   [["great", "good"], ["fine", "good"], ["acting", "drama"],
#     #                                    ["skills", "talent"]]))
#     # print(sol.encode_and_decode_tiny_url(True, 'https://leetcode.com/problems/design-tinyurl'))
#     # print(sol.encode_and_decode_tiny_url(False, 'http://tinyurl.com/a'))
#     # root = sol.maximum_binary_tree([3, 2, 1, 6, 0, 5])
#     # for i in level_output_tree(root):
#     #     print(i)
#     # print(sol.complex_number_multiplication("1+-1i", "1+-1i"))
#     # print(sol.count_bits(5))
#     # print(sol.find_all_duplicates_in_an_array([4,3,2,7,8,2,3,1]))
#     # print([[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]])
#     # print(sol.queue_reconstruction_by_height([[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]))
#     # root = yield_nodes_tree()
#     # print(sol.find_bottom_left_tree_value(root))
#     # print(sol.single_element_in_a_sorted_array([1,1,2,3,3,4,4,8,8]))
#     # print(sol.find_largest_value_in_each_tree_row(root))
#     # print(sol.most_frequent_subtree_sum(root))
#     # msp = MapSumPairs()
#     # msp.insert('apple', 6)
#     # msp.insert('apple', 4)
#     # msp.insert('apply', 5)
#     # print(msp.sum('app'))
#     # print(sol.find_duplicate_file_in_system(["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)",
#     #                                          "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]))
#     # print(sol.single_number_three([1, 2, 1, 3, 2, 5]))
#     # print(sol.beautiful_arrangement(3))
#     # for k, v in sol.beat_cache.items():
#     #     print(k, v)
#     # print(sol.arithmetic_slices([1, 3, 5, 7, 9]))
#     # print(sol.optimal_division([1000, 10, 10, 2]))
#     # print(sol.minimum_ascii_delete_sum_for_two_strings('delete', 'leet'))
#     # print(sol.sort_by_frequency('Aabb'))
#     # print(sol.product_of_array_except_self([1,2,3,4]))
#     # print(sol.array_nesting([5,4,0,3,1,6,2]))
#     # print(sol.binary_tree_inorder_traversal(root))
#     # print(sol.total_hamming_distance([4, 14, 2]))
#     # print(sol.task_scheduler(["A","A","A","B","B","B"], 2))
#     # print(sol.subarray_sum_equal_k((1, 1, 1), 2))
#     # print(sol.target_sum((1, 1, 1, 1, 1), 3))
#     # print(get_quasi_constant([6, 6, 10, 9, 7, 7]))
#     input_str = """my.song.mp3 11b
# greateSong.flac 1000b
# not3.txt 5b
# video.mp4 200b
# game.exe 100b
# mov!e.mkv 10000b"""
#     print(get_file_type_size_statistics(input_str))








