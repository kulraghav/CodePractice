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
            answer = answer*a
        a = a*a
        b = b//2
    return answer


"""
    Extended Euclidean Algorithm (gcd)
"""

def gcd(a, b):
    a, b = min(a, b), max(a, b)
    if a == 0:
        return b

    r = b % a
    return gcd(r, a)


def gcd_coef(a, b):
    if a > b:
        g, x, y = gcd_cof(b, a)
        return g, y, x

    if a == 0:
        return b, 0, 1

    r = b % a
    d = b // a

    g, x, y = gcd_coef(r, a)

    """
        x*r + y*a = g
        x*(b - d*a) + y*a = g
        x*b + (y-d*x)a = g
        => return g, (y-d*x), x
    """
    return g, (y-d*x), x
    

def gcd_iterative(a, b):
    a, b = min(a, b), max(a, b)
    while a > 0:
        r = b % a
        b = a
        a = r
    return b



