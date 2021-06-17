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


def transpose(A):
    A_t = []
    m = len(A)
    n = len(A[0])

    for i in range(m):
        A_t.append([])
        for j in range(n):
            A_t[i].append(A[j][i])
    return A_t

def dot(u, v):
    answer = 0
    for i in range(len(u)):
        answer = answer + u[i]*v[i]
    return answer

def matrix_mult(A, B):
    B_t = transpose(B)

    m = len(A)
    n = len(A[0])

    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(dot(A[i], B_t[j]))
    return C

def matrix_expo(M, n):
    I = [[1,0],[0,1]]
    answer = I
   
    while n > 0:
        if n % 2 == 1:
            answer = matrix_mult(answer, M)
        M = matrix_mult(M, M)
        n = n // 2
    return answer

def fibo(n):
    """
        (f_n, f_n+1) = (f_0, f_1) x ((0, 1),(1, 0))
    """

    M = [[0, 1],[1, 1]]
    P = matrix_expo(M, n)

    f_0, f_1 = 0, 1
    [[f, f_next]] = matrix_mult([[f_0, f_1]], P)
    
    return f

"""
    min-stack, min-queue, and rolling min
"""
import math
class MinStack:
    _infinity = math.inf

    def __init__(self):
        self.elems = []
        self.mins = []

    def push(self, elem):
        min_below = self.get_min()

        self.elems.append(elem)
        self.mins.append(min(elem, min_below))

    def pop(self):
        answer = self.elems.pop()
        self.mins.pop()
        return answer

    def get_min(self):
        if not self.mins:
            return self._infinity

        answer = self.mins[-1]
        return answer

    def is_empty(self):
        if not self.elems:
            return True
        return False

class MinQueue:
    def __init__(self):
        self.push_stack = MinStack()
        self.pop_stack = MinStack()

    def push(self, elem):
        self.push_stack.push(elem)

    def pop(self):
        if self.pop_stack.is_empty():
            while not self.push_stack.is_empty():
                self.pop_stack.push(self.push_stack.pop())

        return self.pop_stack.pop()

    def get_min(self):
        return min(self.push_stack.get_min(), self.pop_stack.get_min())

    def is_empty(self):
        return self.push_stack.is_empty() and self.pop_stack.is_empty()

def rolling_min(A, k):
    if not A:
        return []

    mq = MinQueue()
    for i in range(min(len(A), k)):
        mq.push(A[i])
    
    output = []
    output.append(mq.get_min())
    for i in range(k, len(A)):
        mq.pop()
        mq.push(A[i])
        output.append(mq.get_min())
    return output


from collections import deque
class MonotoneDeque:
    def __init__(self):
        self.deque = deque()

    def is_empty(self):
        if not self.deque:
            return True
        return False

    def peek_right(self):
        if self.is_empty():
            raise Exception('deque is empty')
        return self.deque[-1]

    def peek_left(self):
        if self.is_empty():
            raise Exception('deque is empty')
        return self.deque[0]

    def pop_right(self, x):
        if self.peek_right() == x:
            self.deque.pop()
        return

    def pop_left(self, x):
        if self.peek_left() == x:
            self.deque.popleft()
        return

    def push_right(self, x):
        while not self.is_empty() and self.peek_right() > x:
            self.pop_right(self.peek_right())
        self.deque.append(x)

    def min(self):
        if self.is_empty():
            raise Exception('stack is empty')
        return self.deque[0]

def rolling_min_monotone_deque(A, k):
    if not A:
        return []

    monotone_deque = MonotoneDeque()
    for i in range(min(len(A), k)):
        monotone_deque.push_right(A[i])

    output = []
    output.append(monotone_deque.min())
    for i in range(k, len(A)):
        monotone_deque.pop_left(A[i-k])
        monotone_deque.push_right(A[i])
        output.append(monotone_deque.min())
    return output

import math
class SparseRangeMin:
    def __init__(self, numbers):
        self.numbers = numbers

        """
            segment_mins[(i,j)] = min(A[i:i+2**j])

            min(A[i:i+2**j]) = min(min(A[i:i+2**(j-1)]), 
                                   min(A[i+2**(j-1):i+2**j]))

            segment_mins[(i,j)] = min(segment_mins[(i,j-1)],
                                      segment_mins[(i+2**(j-1),j-1)]
        """
        self.n = len(self.numbers)
        self.k = math.ceil(math.log2(self.n))
        
        self.segment_mins = {}
        for i in range(self.n):
            self.segment_mins[(i, 0)] = self.numbers[i]

        for j in range(1, self.k):
            for i in range(self.n):
                self.segment_mins[(i, j)] = self.segment_mins[(i, j-1)]

                i_next =  i + 2**(j-1) 
                if i_next < self.n:
                    self.segment_mins[(i, j)] = min(self.segment_mins[(i, j-1)],
                                                    self.segment_mins[(i_next, j-1)])

    def range_min(self, l, r):
        answer = math.inf
        for j in range(self.k, -1, -1):
            if 2**j <= r - l + 1:
                answer = min(answer, self.segment_mins[(l, j)])
                l = l + 2**j
        return answer


def fact_pow_prime(n, p):
    power = 0
    while p <= n:
        power = power + n // p
        n = n // p
    return power
    
def choose(n, k):
    answer = 1
    for i in range(n, n-k, -1):
        answer = answer*i

    for i in range(1, k+1):
        answer = answer/i
    return answer

def choose_stable(n, k):
    _epsilon = 0.01

    answer = 1
    for i in range(k):
        answer = answer*((n-i)/(k-i))
    return math.floor(answer + _epsilon)

def binomial_pascal(n, k):
    """
        C[(n, k)] := n choose k
    """
    C = {}
    C[(0, 0)] = 1
    for i in range(1, n+1):
        C[(i,0)] = 1
    for j in range(1, k+1):
        C[(0, j)] = 0

    for j in range(1, k+1):
        for i in range(1, n+1):
            C[(i, j)] = C[(i-1, j-1)] + C[(i-1, j)]
    return C[(n, k)]

def factorial_mod_p(n, p):
    answer = 1
    for i in range(1, n+1):
        answer = (answer * i) % p
    return answer

def inverse_mod_p(x, p):
    x = x % p
    if x == 0:
        raise Exception('{} is not invertible modulo {}'.format(x, p))

    for y in range(1, p):
        if (x*y) % p == 1:
            return y

    raise Exception('{} must be a prime number'.format(p))

def binomial_mod_large_p(n, k, p):
    assert p > n, "{} must be larger than {}".format(p, n)

    n_fact = factorial_mod_p(n, p)
    k_fact = factorial_mod_p(k, p)
    n_minus_k_fact = factorial_mod_p(n-k, p)

    return (n_fact * inverse_mod_p(k_fact * n_minus_k_fact, p)) % p

   
def catalan(n):
    C = {}
    C[0] = 1

    for k in range(1, n+1):
        C[k] = 0
        for i in range(0, k):
            C[k] = C[k] + C[i]*C[k-i-1]
    return C[n]

def compute_hash(s, p=31, m=10**9+1):
    total_mod_m = 0
    p_power = 1
    for i in range(len(s)):
        total_mod_m = (total_mod_m + ord(s[i])*p_power) % m
        p_power = p_power*p
    return total_mod_m

def count_distinct(strings):
    seen = set()
    count = 0
    for s in strings:
        h = compute_hash(s)
        if not h in seen:
            count = count + 1
        seen.add(h)
    return count



if __name__ == '__main__':
    #from line_profiler import LineProfiler
    #lp = LineProfiler()
    
    #lp(expo)(2,30)
    #lp.print_stats()

    #lp(expo_iterative)(2,30)
    #lp.print_stats()

    A = [5, 3, 1, 2, 4]
    k = 2
    rolling_min_monotone_deque(A, k)

    
