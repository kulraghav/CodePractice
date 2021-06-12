"""
    This file contains tests for the functions in algo.py
"""

def test_always_fails():
    assert False

def test_always_passes():
    assert True

def test_expo():
    from algo import expo

    a = 0
    b = 0
    assert expo(a, b) == 1

    a = 1
    b = 1
    assert expo(a, b) == 1

    a = 2
    b = 0
    assert expo(a, b) == 1

    a = 5
    b = 0
    assert expo(a, b) == 1

    a = 4
    b = 1
    assert expo(a, b) == 4

    a = 7
    b = 1
    assert expo(a, b) == 7

    a = 2
    b = 3
    assert expo(a, b) == 8

    a = 3
    b = 2 
    assert expo(a, b) == 9

    return

def test_expo_iterative():
    from algo import expo_iterative

    a = 0
    b = 0
    assert expo_iterative(a, b) == 1

    a = 1
    b = 1
    assert expo_iterative(a, b) == 1

    a = 2
    b = 0
    assert expo_iterative(a, b) == 1

    a = 5
    b = 0
    assert expo_iterative(a, b) == 1

    a = 4
    b = 1
    assert expo_iterative(a, b) == 4

    a = 7
    b = 1
    assert expo_iterative(a, b) == 7

    a = 2
    b = 3
    assert expo_iterative(a, b) == 8

    return

def test_gcd():
    from algo import gcd

    a = 0
    b = 2
    assert gcd(a, b) == 2

    a = 1
    b = 4
    assert gcd(a, b) == 1

    a = 6
    b = 8
    assert gcd(a, b) == 2

    a = 7
    b = 7
    assert gcd(a, b) == 7

    return

def test_gcd_iterative():
    from algo import gcd_iterative

    a = 0
    b = 2
    assert gcd_iterative(a, b) == 2

    a = 1
    b = 4
    assert gcd_iterative(a, b) == 1

    a = 6
    b = 8
    assert gcd_iterative(a, b) == 2

    a = 7
    b = 7
    assert gcd_iterative(a, b) == 7

    return

def test_gcd_coef():
    from algo import gcd_coef

    a = 0
    b = 2

    g, x, y = gcd_coef(a, b)
    assert g == 2
    assert x*a + y*b == g

    a = 1
    b = 4
    g, x, y = gcd_coef(a, b)
    assert x*a + y*b == g
    assert g == 1
    

    a = 6
    b = 8
    g, x, y = gcd_coef(a, b)
    assert x*a + y*b == g
    assert g == 2

    a = 7
    b = 7
    g, x, y = gcd_coef(a, b)
    assert x*a + y*b == g
    assert g == 7

    return

def test_gcd_coef_iterative():
    from algo import gcd_coef_iterative

    a = 0
    b = 2

    g, x, y = gcd_coef_iterative(a, b)
    assert g == 2
    assert x*a + y*b == g

    a = 1
    b = 4
    g, x, y = gcd_coef_iterative(a, b)
    assert x*a + y*b == g
    assert g == 1
    

    a = 6
    b = 8
    g, x, y = gcd_coef_iterative(a, b)
    assert x*a + y*b == g
    assert g == 2

    a = 7
    b = 7
    g, x, y = gcd_coef_iterative(a, b)
    assert x*a + y*b == g
    assert g == 7

    return

def test_fibo():
    from algo import fibo

    n = 0
    assert fibo(n) == 0

    n = 1
    assert fibo(n) == 1

    n = 2
    assert fibo(n) == 1

    n = 3
    assert fibo(n) == 2

    n = 4
    assert fibo(n) == 3

    n = 5
    assert fibo(n) == 5
    
    return

def test_min_stack():
    from algo import MinStack
    import math

    ms = MinStack()
    assert ms.get_min() == math.inf

    ms.push(3)
    assert ms.get_min() == 3

    ms.push(2)
    assert ms.get_min() == 2

    ms.push(5)
    assert ms.get_min() == 2

    ms.pop()
    assert ms.get_min() == 2

    ms.pop()
    assert ms.get_min() == 3

    return

def test_min_queue():
    from algo import MinQueue
    import math

    ms = MinQueue()
    assert ms.get_min() == math.inf

    ms.push(3)
    assert ms.get_min() == 3

    ms.push(2)
    assert ms.get_min() == 2

    ms.push(5)
    assert ms.get_min() == 2

    ms.pop()
    assert ms.get_min() == 2

    ms.pop()
    assert ms.get_min() == 5

    return

def test_rolling_min():
    from algo import rolling_min

    A = []
    k = 1
    assert rolling_min(A, k) == []

    A = [3,2,1]
    k = 1
    assert rolling_min(A, k) == [3,2,1]

    A = [5,3,1,2,4]
    k = 2
    assert rolling_min(A, k) == [3, 1, 1, 2] 

    A = [1,2,3]
    k = 3
    assert rolling_min(A, k) == [1]

def test_rolling_min_monotone_deque():
    from algo import rolling_min_monotone_deque

    A = []
    k = 1
    assert rolling_min_monotone_deque(A, k) == []

    A = [3,2,1]
    k = 1
    assert rolling_min_monotone_deque(A, k) == [3,2,1]

    A = [5,3,1,2,4]
    k = 2
    assert rolling_min_monotone_deque(A, k) == [3, 1, 1, 2] 

    A = [1,2,3]
    k = 3
    assert rolling_min_monotone_deque(A, k) == [1]

from algo import SparseRangeMin
class TestSparseRangeMin:

    def test_range_min(self):
        numbers = [0,1,2,3,4,5]
        srm = SparseRangeMin(numbers)

        assert srm.range_min(0, 5) == 0
        assert srm.range_min(1, 4) == 1
        assert srm.range_min(2, 3) == 2
        assert srm.range_min(4, 4) == 4
        
        return


if __name__ == '__main__':
    test = TestSparseRangeMin()
    test.test_range_min()
