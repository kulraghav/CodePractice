"""
    This file contains python implementation of some of the basic algorithms

    References:
        - https://cp-algorithms.com/
"""

"""
    Binary Exponentiation (repeated squaring)
"""

def expo(a, b):
    if b == 0:
        return 1

    half_expo = expo(a, b//2)

    if b % 2 == 0:
        return half_expo*half_expo
    else:
        return half_expo*half_expo*a


def expo_iterative(a, b):
    answer = 1
    while b > 0:
        if b % 2 == 1:
            answer = answer * a
        a = a * a
        b = b//2
    return answer
