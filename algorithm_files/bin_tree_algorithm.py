#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : algo_exe
@Time    : 2017/11/6 22:02
@Author  : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site    : 
@File    : bin_tree_algorithm.py
@desc    :
"""
import queue


class TimesNode(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self):
        return self.left.eval() * self.right.eval()

    def in_order(self):
        return '(' + self.left.in_order() + '*' + self.right.in_order() + ')'


class PlusNode(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self):
        return self.left.eval() + self.right.eval()

    def in_order(self):
        return '(' + self.left.in_order() + '+' + self.right.in_order() + ')'


class NumNode(object):
    def __init__(self, num):
        self.num = num

    def eval(self):
        return self.num

    def in_order(self):
        return str(self.num)


def in_order_main():
    x = NumNode(4)
    y = NumNode(5)
    plus_two = PlusNode(x, y)
    z = NumNode(6)
    times_two = TimesNode(plus_two, z)
    a = NumNode(3)
    res = PlusNode(times_two, a)
    print(res.in_order())


def E(q):
    if q.empty():
        raise ValueError('Invalid Prefix Expression')
    token = q.get()

    if token == '+':
        return PlusNode(E(q), E(q))
    if token == '*':
        return TimesNode(E(q), E(q))
    return NumNode(float(token))


def E_main():
    x = input('Please enter a prefix expression:')
    lst = x.split()
    q = queue.Queue()
    for token in lst:
        q.put(token)

    root = E(q)
    print(root.eval())
    print(root.in_order())


###################################################################
# binary search tree algorithm
# all code from http://www.laurentluce.com/posts/binary-search-tree-library-in-python/

class TreeNode(object):
    def __init__(self, in_data):
        self.data = in_data
        self.left = None
        self.right = None

    def insert(self, in_data):
        if self.data:
            if in_data <= self.data:
                if self.left is None:
                    self.left = TreeNode(in_data)
                else:
                    self.left.insert(in_data)
            elif in_data > self.data:
                if self.right is None:
                    self.right = TreeNode(in_data)
                else:
                    self.right.insert(in_data)
        else:
            self.data = in_data

    def lookup(self, in_data, in_parent=None):
        if in_data < self.data:
            if self.left is None:
                return None, None
            else:
                return self.left.lookup(in_data, self)
        elif in_data > self.data:
            if self.right is None:
                return None, None
            else:
                return self.right.lookup(in_data, self)
        else:
            return self, in_parent

    def delete(self, in_data):
        """
        1- The node to remove has no child.
        2- The node to remove has 1 child.
        3- The node to remove has 2 children.
        :param in_data:
        :return:
        """
        node, parent = self.lookup(in_data)
        if node is not None:
            children_count = node.children_ount()
            if children_count == 0:
                if parent:
                    if parent.left is node:
                        parent.left = None
                    else:
                        parent.right = None
                    del node
                else:
                    self.data = None
            elif children_count == 1:
                if node.left:
                    n = node.left
                else:
                    n = node.right
                if parent:
                    if parent.left is node:
                        parent.left = n
                    else:
                        parent.right = n
                    del node
                else:
                    self.left = n.left
                    self.right = n.right
                    self.data = n.data
            else:
                parent = node
                successor = node.right
                while successor.left:
                    parent = successor
                    successor = successor.left
                node.data = successor.data
                if parent.left == successor:
                    parent.left = successor.right
                else:
                    parent.right = successor.right

    def children_count(self):
        cnt = 0
        if self.left is not None:
            cnt += 1
        if self.right is not None:
            cnt += 1
        return cnt

    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print(self.data)
        if self.right:
            self.right.print_tree()

    def compare_trees(self, node):
        if node is None:
            return False
        if self.data != node.data:
            return False
        res = True
        if self.left is None:
            if node.left is not None:
                return False
        else:
            res = self.left.compare_trees(node.left)
        if res is False:
            return False
        if self.right is None:
            if node.right is not None:
                return False
        else:
            res = self.right.compare_trees(node.right)
        return res

    def tree_data(self):
        stack = []
        node = self
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                yield node.data
                node = node.right


if __name__ == '__main__':
    # in_order_main()
    E_main()
