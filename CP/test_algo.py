"""
    This file contains tests for the functions in algo.py
"""
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
