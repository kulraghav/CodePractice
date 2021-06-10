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

def gcd_coef_iterative(a, b):
    x_a, y_a = 1, 0
    x_b, y_b = 0, 1

    """
        (a, b) -> (b % a, a)

        a = x_a*a_init + y_a*a_init
        b = x_b*b_init + y_b*b_init

        r = b % a
        d = b // a

        r = b - d*a
          = (x_b*a_init + y_b*b_init) - d*(x_a*a_init + y_a*b_init)

        a = x_a*a_init + y_a*a_init

        =>
        x_a, y_a = x_b - d*x_a, y_b - d*y_a
        x_b, y_b = x_a, y_a
        
    """

    while a > 0:
        r = b % a
        d = b // a
        b = a
        a = r
        
        x_a_temp, y_a_temp = x_a, y_a
        x_b_temp, y_b_temp = x_b, y_b

        x_a, y_a = x_b_temp - d*x_a_temp, y_b_temp - d*y_a_temp
        x_b, y_b = x_a_temp, y_a_temp

    g, x, y = b, x_b, y_b
    return g, x, y

if __name__ == '__main__':
    from line_profiler import LineProfiler
    lp = LineProfiler()
    
    lp(expo)(2,30)
    lp.print_stats()

    lp(expo_iterative)(2,30)
    lp.print_stats()
