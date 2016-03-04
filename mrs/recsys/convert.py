#!/usr/bin/env python
# -*- coding: utf-8 -*-

def f_inverse_cap(l):
    no, val = 1, l[0]
    for i, x in enumerate(l):
        if x > val:
            no = i + 1
            val = x
            return no
    return no

def f_inverse(l):
    return l.index(1) + 1
