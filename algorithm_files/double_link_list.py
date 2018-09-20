#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@license : (C) Copyright 2013-2017, Easy doesnt enter into grown-up life.
@Software: PyCharm
@Project : leet_code_some
@Time    : 2018/7/29 11:01
@Author  : achilles_xushy
@contact : yuqingxushiyin@gmail.com
@Site    : 
@File    : double_link_list.py
@desc    :
"""
from itertools import zip_longest
_absent = object()  # Marker that means no parameter was provided.


class ItemNode(object):

    """
    Dictionary key:value items wrapped in a node to be members of itemlist, the
    doubly linked list defined below.
    """

    def __init__(self, prev=None, next=None, key=_absent, value=_absent):
        self.prev = prev
        self.next = next
        self.key = key
        self.value = value


class ItemList(object):

    """
    Doubly linked list of ItemNodes.
    """

    def __init__(self, items=None):
        self.root = ItemNode()
        self.root.next = self.root.prev = self.root
        self.size = 0

        for key, value in items:
            self.append(key, value)

    def append(self, key, value):
        tail = self.root.prev if self.root.prev is not self.root else self.root
        node = ItemNode(tail, self.root, key=key, value=value)
        tail.next = node
        self.root.prev = node
        self.size += 1
        return node

    def remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
        return self

    def clear(self):
        for node, key, value in self:
            self.remove_node(node)
        return self

    def items(self):
        return list(self.iteritems())

    def keys(self):
        return list(self.iterkeys())

    def values(self):
        return list(self.itervalues())

    def iteritems(self):
        for node, key, value in self:
            yield key, value

    def iterkeys(self):
        for node, key, value in self:
            yield key

    def itervalues(self):
        for node, key, value in self:
            yield value

    def reverse(self):
        for node, key, value in self:
            node.prev, node.next = node.next, node.prev
        self.root.prev, self.root.next = self.root.next, self.root.prev
        return self

    def __len__(self):
        return self.size

    def __iter__(self):
        current = self.root.next
        while current and current is not self.root:
            # Record current.next here in case current.next changes after the
            # yield and before we return for the next iteration. For example,
            # methods like reverse() will change current.next() before yield
            # gets executed again.
            next_node = current.next
            yield current, current.key, current.value
            current = next_node

    def __contains__(self, item):
        """
        Params:
          item: Can either be a (key,value) tuple or an ItemNode reference.
        """
        node = key = value = _absent
        if hasattr(item, '__len__') and callable(item.__len__):
            if len(item) == 2:
                key, value = item
            elif len(item) == 3:
                node, key, value = item
        else:
            node = item

        if node is not _absent or _absent not in [key, value]:
            for self_node, self_key, self_value in self:
                if ((node is _absent and key == self_key and value == self_value)
                   or (node is not _absent and node == self_node)):
                    return True
        return False

    def __getitem__(self, index):
        # Only support direct access to the first or last element, as this is
        if index == 0 and self.root.next is not self.root:
            return self.root.next
        elif index == -1 and self.root.prev is not self.root:
            return self.root.prev
        raise IndexError(index)

    def __delitem__(self, index):
        self.remove_node(self[index])

    def __eq__(self, other):
        for (n1, key1, value1), (n2, key2, value2) in zip_longest(self, other):
            if key1 != key2 or value1 != value2:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self.size > 0

    def __str__(self):
        return '[%s]' % self.items()
