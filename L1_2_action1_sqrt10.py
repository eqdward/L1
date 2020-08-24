# -*- coding: utf-8 -*-
"""
"""

def sqrt10():
    E = 1e-10
    low, high = 3, 4
    mid = low + (high - low) / 2
    while high - low > E:
        if mid * mid > 10:
            high = mid
        else:
            low = mid
        mid = low + (high - low) / 2
    return mid

print(sqrt10())
