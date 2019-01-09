
"""
    Module: practice.py
    Purpose: practice algorithm coding problems
"""


"""
    22 December 2018
    Generate all permutations of a string
    Input: s (string)
    Output: perms (list of strings)
"""

"""
    Simpler case when all characters are distinct
"""
def generate_distinct(s):
    if len(s) == 0:
        return [""]

    if len(s) == 1:
        return [s]
    
    perms = []
    
    for i in range(len(s)):
        s_i = s[:i] + s[i+1:]
        perms_i = generate_distinct(s_i)
        print(perms_i)
        for perm in perms_i:
            perms.append(s[i] + perm)
    return perms    

"""
    When there are repeated characters
"""
def get_counts(s):
    counts = {}
    for i in range(len(s)):
        if not s[i] in counts:
            counts[s[i]] = 1
        else:
            counts[s[i]] = counts[s[i]] + 1
    return counts

def generate_perms(s):
    if len(s) == 0:
        return [""]
    if len(s) == 1:
        return [s]
    
    counts = get_counts(s)            
    perms = []
                
    for letter in counts:
        i = s.index(letter)
        s_new = s[:i] + s[i+1:]
        perms_new = generate_perms(s_new)
        for perm in perms_new:
            perms.append(letter + perm)
                
    return perms


"""
    Only print all the permutations
    Save space storing the output
"""

def increment(counts, letter):    
    if not letter in counts:
        counts[letter] = 1
    else:
        counts[letter] = counts[letter] + 1
    return    

def decrement(counts, letter):
    if not letter in counts:
        raise Exception("Invalid Input")
    elif counts[letter] == 1:
        del counts[letter]
    else:
        counts[letter] = counts[letter] - 1
    return     
        

def get_next_chars(s, prefix):
    next_chars = {}
    for i in range(len(s)):
        increment(next_chars, s[i])
        
    for i in range(len(prefix)):
        decrement(next_chars, prefix[i])
        
    return next_chars


def print_extensions(prefix, remaining_counts):

    if not remaining_counts:
        print("".join(prefix))
        
    next_chars = [char for char in remaining_counts]

    """
        Learning: Earlier I had written the for loop as follows
                 >> for char in remaining_counts: ...
                 This gives wrong results because remaining_counts changes in every iteration.
    """
    for char in next_chars: 
        prefix.append(char)
        decrement(remaining_counts, char)
        
        print_extensions(prefix, remaining_counts)

        """
            This is the backtracking step.
        """
        increment(remaining_counts, char)
        prefix.pop(-1)

    return

def print_perms(s):
    counts = get_counts(s)
    print_extensions([], counts)
    
"""
    Print only palindromic permutations
"""

def is_palindrome(A):
    begin = 0
    end = len(A)-1
    while begin < end:
        if not A[begin] == A[end]:
            return False
        begin = begin + 1
        end = end - 1

    return True

def print_palindrome_extensions(prefix, remaining_counts):

    if not remaining_counts:
        if is_palindrome(prefix):
            print("".join(prefix))
            return
        else:
            return
        
    next_chars = [char for char in remaining_counts]

    for char in next_chars: 
        prefix.append(char)
        decrement(remaining_counts, char)
        
        print_palindrome_extensions(prefix, remaining_counts)

        increment(remaining_counts, char)
        prefix.pop(-1)

    return

def print_palindrome_perms(s):
    counts = get_counts(s)
    print_palindrome_extensions([], counts)
    


"""
    [22 December 2018]
    [Facebook] Express an integer n as sum of squares using minimum numbers
    Input: n (integer)
    Output: squares (list of integers)
"""

import math
def sum_of_squares(n):
    if n == 0:
        return []
    if n == 1:
        return [1]
    squares = n*[1]
    min_squares = n
    for i in range(1, math.floor(n**0.5)+1):
        new_squares = sum_of_squares(n - (i*i))
        if 1 + len(new_squares) < min_squares:
            new_squares.append(i*i)
            squares = new_squares
            min_squares = len(new_squares)

    return squares        
    

"""
    [22 Dec 2018]
    [Square] Shortest unique prefixes
    Input: words (list of strings)
    Output: prefixes (list of strings)
"""
_end = "_end"

def make_trie(words):
    trie = {}
    for word in words:
        current = trie
        for letter in word:
            if not letter in current:
                current[letter] = {}
            current = current[letter]
        current[_end] = _end
    return trie

def shortest_unique_prefixes(words):
    trie = make_trie(words)
    prefixes = {}
    for word in words:
        current = trie
        prefix = ""

        """
            Learning:
            The 'break' inside the if statement breaks the nearest loop
            https://stackoverflow.com/questions/7022631/python-breaking-out-of-if-statement-inside-a-for-loop
            
        """
        for letter in word:
            current = current[letter]
            prefix = prefix + letter
            if len(current) == 1:
                prefixes[word] = prefix
                break

    return prefixes        

"""
    [22 Dec 2018]
    [Triplebyte] Generate a random sample from multinomial distribution
    Input: P (list of floats)  P = [p_1, p_2, ..., p_n] such that p_i >= 0 and sum_i p_i = 1
    Output: r (integer)
"""

def get_intervals(P):
    intervals = [0]
    for i in range(len(P)):
        intervals.append(intervals[-1]+P[i])
    return intervals

def binary_search(A, x):
    begin = 0
    end = len(A)-1

    """
        Invariant: A[begin] <= x < A[end] 
    """

    while begin < end:
        mid = (begin + end) // 2

        """
            Learning: Earlier the first if was A[mid] == x. This went into infinite loop on A of length 2.
        """
        if  A[mid] <= x < A[mid+1]:
            return mid

        if A[mid] > x:
            end = mid

        if A[mid] < x:
            begin = mid
    

    return begin        
            
import random
def generate_multinomial_sample(P):
    intervals = get_intervals(P)
    u = random.uniform(0, 1)

    r = binary_search(intervals, u)

    return r


""" 
    [22 Dec 2018]
    Running Top-k largest
    Input: A (list of integers)
           k (integer)
    Output: T (list of lists)
"""

"""
    heapq is min-heap
    to use it as a max-heap one has to negate the values
"""
import heapq
def top_k(A, k):
    T = []
    top_k = A[:k]
    heapq.heapify(top_k)

    T.append(list(top_k))
    for i in range(k, len(A)):
        heapq.heappush(top_k, A[i])
        heapq.heappop(top_k)
        T.append(list(top_k))
    return T    
             
    
    
"""
    [LinkedIn] k-nearest neighbors
"""
                

def get_distance(point, center):
    distance = (point[0]-center[0])**2 + (point[1]-center[1])**2
    return distance


def knn(points, center, k):
    heap = []

    """
        priorities are negative of the squared distance
        first co-ordinate of the tuple is the priority
    """
    for point in points[:k]:
        heapq.heappush(heap, (-get_distance(point, center), point))

    for point in points[k:]:
        heapq.heappush(heap, (-get_distance(point, center), point))
        heapq.heappop(heap)

    return list(heap)

"""
    [22 Dec 2018]
    [Apple] Generate a possible gray code of n bit strings
"""

def get_xor(A, B):
    C = []
    for i in range(len(A)):
        C.append((A[i] + B[i]) % 2)
    return C

def gray_code(n):
    if n == 0:
        return [[]]

    if n == 1:
        return [[0], [1]]

    codes = []
    
    sub_codes = gray_code(n-1)

    for sub_code in sub_codes:
        codes.append(sub_code + [0])

    offset = sub_codes[-1]

    for sub_code in sub_codes:
        codes.append(get_xor(offset, sub_code) + [1])

    return codes    

"""
    [22 Dec 2018]
    Dijkstra : single source shortest path
"""

def dij(G, W, s):
    heap = []

    heapq.heappush(heap, (0, s))
    distances = {s: 0}

    while heap:
        (distance, u) = heapq.heappop(heap)
        distances[u] = distance

        for v in G[u]:
            if v not in distances:
                distances[v] = distances[u] + W[(u,v)]
            if distances[v] > distances[u] + W[(u,v)]:
                distances[v] = distances[u] + W[(u,v)]
            heapq.heappush(distances[v], v)

    return distances        

"""
    [22 Dec 2018]
    DFS numbering
"""

_undefined = -1

def get_next_unvisited(G, u, visits):
    for v in G[u]:
        if v not in visits:
            return v
    return _undefined    


def dfs(G, s):
    t = 0
    stack = [s]
    visits = {s: [t, _undefined]}

    while stack:
        t = t + 1
        u = stack[-1]
        
        v = get_next_unvisited(G, u, visits)
     
        if v == _undefined:
            stack.pop(-1)
            visits[u][1] = t

        else:
            stack.append(v)
            visits[v] = [t, _undefined]

    return visits         
              
"""
    [22 Dec 2018]
    Add two binary integers given as list
"""

def add_one_to(X):
    for i in range(len(X)-1, -1, -1):
        if X[i] == 0:
            X[i] = 1
            return X
        else:
            X[i] = 0
    X = [1] + X
    return X

def add_binary_recursive(A, B):
    if len(A) == 0:
        return B
    if len(B) == 0:
        return A

    C = add_binary_recursive(A[:-1], B[:-1])

    last_bit = (A[-1] + B[-1]) % 2

    if A[-1] == 1 and B[-1] == 1:
        carry_bit = 1
    else:
        carry_bit = 0

    if carry_bit == 0:
        C.append(last_bit)
        return C
    else:
        D = add_one_to(C)
        D.append(last_bit)
        return D

def add_binary(A, B):
    carry = 0

    A.reverse()
    B.reverse()

    if len(A) < len(B):
        A = A + (len(B)-len(A))*[0]
    if len(B) < len(A):
        B = B + (len(A)-len(B))*[0]
 
    C = []

    for i in range(len(A)):
        C.append((A[i] + B[i] + carry) % 2)
        if A[i] + B[i] + carry > 1:
            carry = 1
        else:
            carry = 0

    if carry == 0:
        C.reverse()
        return C
    else:
        C.append(1)
        C.reverse()           
        return C

  
"""
    [22 Dec 2018]
    Max valid parenthesis
"""

def is_annihilating(a, stack):
    if not stack:
        return False
    else:
        if a == ")" and stack[-1] == "(":
            return True
        else:
            return False
        
def max_valid(s):
    stack = []
    for i in range(len(s)):
        if s[i] == "(":
            stack.append(s[i])

        if s[i] == ")":
            if is_annihilating(")", stack):
                stack.pop(-1)
            else:
                stack.append(s[i])

    return len(s) - len(stack)             
                   
                   
"""
     [23 Dec 2018]
     [Google] Nearest larger number on the right
"""

def nearest_larger(A):
    L = {}

    stack = []

    for i in range(len(A)):
        while stack and A[i] > stack[-1][1]:
            L[stack[-1][0]] = i
            stack.pop(-1)
        stack.append((i, A[i]))

    return L    
                
                
"""
    [23 Dec 2018]
    Dijkstra : single source shortest path 
    Using defaultdict
    G = defaultdict(list) or G = defaultdict(lambda: [])
"""
_infinity = 99999999
from collections import defaultdict
def dijk(G, W, s):
    heap = []

    heapq.heappush(heap, (0, s))
    distances = defaultdict(lambda: _infinity)
    distances = {s: 0}

    while heap:
        (distance, u) = heapq.heappop(heap)
        distances[u] = distance

        for v in G[u]:
            if distances[v] > distances[u] + W[(u,v)]:
                distances[v] = distances[u] + W[(u,v)]
            heapq.heappush(distances[v], v)

    return distances

"""
    [25 Dec 2018]
    [AirBnB] Alien Dictionary
    Input: words (list of string) - alphabetically sorted words
    Output: alphabetical_order (list of chars) - a possible order of alphabets
"""

from collections import OrderedDict
def make_ordered_trie(words):
    trie = OrderedDict()

    for word in words:
        current = trie
        for letter in word:
            if not letter in current:
                current[letter] = OrderedDict()
            current = current[letter]

        current[_end] = _end

    return trie

def build_graph(G, trie):
    if _end in trie:
        return G
    
    for letter in trie:
        if not letter in G:
            G[letter] = []

    """
        OrderedDict keys do not support indexing: TypeError        
    """
    
    keys = list(trie.keys())

    for i in range(1, len(keys)):
        G[keys[i-1]].append(keys[i])

    for letter in trie:
        build_graph(G, trie[letter])

    return G    
        

    
"""
    O(n^2) implementation
"""
_undefined = -1    
def get_sink(G):
    for u in G:
        if len(G[u]) == 0:
            return u
    return _undefined

def topological_sort(G):

    output = []

    while G:
        v = get_sink(G)
        output.append(v)

        for u in G:
            if v in G[u]:
                G[u].remove(v)
        del G[v]        

    output.reverse()
    
    return output


"""
   O(m + n) implementation  
"""

def get_vertex_to_indegree(G):
    # vertex_to_indegree = defaultdict(lambda: 0) # this did not work
    
    vertex_to_indegree = {}
    for u in G:
        vertex_to_indegree[u] = 0
        
    for u in G:
        for v in G[u]:
            vertex_to_indegree[v] = vertex_to_indegree[v] + 1

    return vertex_to_indegree

def get_indegree_to_vertices(vertex_to_indegree):
    indegree_to_vertices = defaultdict(lambda: [])

    for v in vertex_to_indegree:
        indegree_to_vertices[vertex_to_indegree[v]].append(v)

    return indegree_to_vertices

def get_source(indegree_to_vertices):
    return indegree_to_vertices[0][0]

def remove_source(source, G, vertex_to_indegree, indegree_to_vertices):
    indegree_to_vertices[0].remove(source)
    
    for v in G[source]:
        indegree = vertex_to_indegree[v]
        vertex_to_indegree[v] = indegree - 1
        indegree_to_vertices[indegree].remove(v)
        indegree_to_vertices[indegree-1].append(v)

    del G[source]

    return G
        
def top_sort(G):
    output = []

    vertex_to_indegree = get_vertex_to_indegree(G)
    indegree_to_vertices = get_indegree_to_vertices(vertex_to_indegree)

    while G:
        source = get_source(indegree_to_vertices)
        output.append(source)
        remove_source(source, G, vertex_to_indegree, indegree_to_vertices)

    return output    

        

    

    
    


"""
    The appraoch below was wrong:
        - trie is already built
        - trie keys are not processed in the order they were inserted

    Possible fixes:
        - use ordered dict for trie
    
"""
def make_graph_wrong(words):
    
    G = defaultdict(list)
    
    trie = make_trie(words)

    for word in words:
        current = trie
        
        for letter in word:
            for char in current:
                if  char != letter :
                    if letter not in G[char]:
                        G[char].append(letter)
        

            current = current[letter]

    return G
    
def get_alphabetical_order(words):

    trie = make_ordered_trie(words)
    
    G = {}
    
    G = build_graph(G, trie)

    alphabetical_order = top_sort(G)

    return alphabetical_order

    
"""
    
    [26 Dec 2018] [Shift] Check if the parentheses are balanced

    I mistakenly thought about the palindrome problem and gave the solution
    with two pointers.

"""        

def is_balanced(s):
    count_left = 0
    count_right = 0
    for i in range(len(s)):
        if count_left < count_right:
            return False
        
        if s[i] == "(":
            count_left = count_left + 1

        if s[i] -- ")":
            count_right = count_right + 1
    
    if count_left == count_right:
        return True
    else:
        return False
    
    
"""
    [26 Dec 2018] [Fabecook] Boggie solver for 4 x 4 board
    Args: board: 
              type: list of list of chars 
              description: input is a 4 x 4 matrix of characters
          words:
              type: list of string
              description: list of words forming a dictionary
    Returns:
          boggies:
              type: list of list of chars
              description: list of words in dictionary formed by 
                  traveling along adjacent places in the 4 x 4 grid 
"""


"""
    Learning: get_unvisited_neighbors function had bugs that took some time to discover.
"""
def is_outside(neighbor):
    if neighbor[0] > 3 or neighbor[1] > 3 or neighbor[0] < 0 or neighbor[1] < 0:
        return True
    else:
        return False
    
def get_unvisited_neighbors(i,j, visited):
    neighbors = [(i, j+1), (i, j-1), (i+1, j), (i-1, j)]

    output = []
    """
        Tried with .remove and it did not work.
    """
    for neighbor in neighbors:
        if not neighbor in visited and not is_outside(neighbor):
            output.append(neighbor)

    return output


def get_boggies(i,j, board, words, prefix, visited):
    boggies = []

    current_word = "".join(list(prefix))
    if current_word in words:
        boggies.append(current_word)
        
    neighbors = get_unvisited_neighbors(i,j, visited)


    if not neighbors:
        return boggies

    for neighbor in neighbors:
        prefix.append(board[neighbor[0]][neighbor[1]])
        visited.append(neighbor)
        boggies = boggies + get_boggies(neighbor[0], neighbor[1], board, words, prefix, visited)
        prefix.pop(-1)
        visited.pop(-1)

    return boggies    

        
    
def boggie(board, words):
    boggies = []
    for i in range(len(board)):
        for j in range(len(board)):
            prefix = [board[i][j]]
            visited = [(i,j)]
            boggies = boggies + get_boggies(i,j, board, words, prefix, visited)

    return boggies         

"""
    [27 Dec 2018] [Twitter] Form largest number by concatenating a list of numbers
"""

def get_digits(number):

    digits = []

    while number > 0:
        last_digit = int(number % 10)
        digits.append(last_digit)
        number = (number - last_digit) / 10

    digits.reverse()

    return digits


_end = -1
def make_digit_trie(numbers):
    trie = {}

    for number in numbers:
        current = trie
        for digit in get_digits(number):
            if not digit in current:
                current[digit] = {}

            current = current[digit]
        current[_end] = number

    return trie



def collect(trie):

    output = []

    if not trie:
        return []

    if -1 in trie:
        return [trie[-1]]

    sorted_first_digits = sorted(trie.keys(), reverse=True)

    for first_digit in sorted_first_digits:
        output = output + collect(trie[first_digit])

    return output

def get_largest_combination(numbers):

    trie = make_digit_trie(numbers)

    return collect(trie)

"""
    [27 Dec 2018] Encoding binary string as a graph and decoding string graph 
        It is known that string graphs can be characterize by excluding 
        P_4 (path of length 4), C_4 (cycle of length 4) and M_4 (perfect matching on 4 vertices) as induced subgraphs

    Lesson learned: It took me a lot of time to debug first version which went for optimized code.
                    When I modularized the code the structure became simple and clear and debugging was easy.
"""        

_A = '0'
_B = '1'
def encode_string(s):
    G = defaultdict(list)

    G[0]

    for i in range(1, len(s)+1):
        G[i] = []
    
        if s[i-1] == _B:
            for j in range(i):
                G[j].append(i)
                G[i].append(j)

    return G


def get_degree_to_vertices(G):
    degree_to_vertices = defaultdict(list)

    for u in G:
        degree_to_vertices[len(G[u])].append(u)    

    return degree_to_vertices        


_undefined = -1        
def get_star(G):
    n = len(G.keys())
    for v in G:
        if len(G[v]) == n-1:
            return v
    return _undefined

def get_iso(G):
    for v in G:
        if len(G[v]) == 0:
            return v
    return _undefined

def remove(u, G):
    for v in G[u]:
        G[v].remove(u)

    del G[u]    
    


def decode_string_graph(G):
    s = ""

    #degree_to_vertices = get_degree_to_vertices(G)
   
    n = len(G.keys())

    for i in range(n-1):
        star = get_star(G)
        iso = get_iso(G)

        if not star == _undefined:
            s = _B + s
            remove(star, G)

        elif not iso == _undefined:
            s = _A + s
            remove(iso, G)
        else:
            raise Exception("G is not a string graph!")
        
    return s    
            

            
_infinity = 99999999            
def jumpingOnClouds(c):
    if not c:
        return _infinity
    if c[0] == 1:
        return _infinity
    
    if len(c) == 1: # c[0] == 0 holds
        return 0
    
    if c[1] == 0:
        prefix_min = [0, 1]
    else:
        prefix_min = [0, _infinity]

    for i in range(2, len(c)):
        if c[i] == 0:
            prefix_min.append(1 + min(prefix_min[i-1], prefix_min[i-2]))
        else:
            prefix_min.append(_infinity)

    print(prefix_min)
    return prefix_min[-1]

"""
    Rotate an array
"""

def get_next(visited):
    for i in range(len(visited)):
        if not visited[i]:
            return i

    return len(visited)        
# Complete the rotLeft function below.
def rotLeft(a, d):

    visited = [False for i in range(d)]
    
    while not all(visited):
        start = get_next(visited)

        print(visited)
        print(start)
        print(a)
        print("\n")

        current_index = start
        next_index = (current_index + d) % len(a)

        visited[current_index] = True
        
        while not next_index == start:
            if next_index < d:
                visited[next_index] = True

            
            temp = a[current_index]
        
            a[current_index] = a[next_index]
            a[next_index] = temp

            current_index = next_index
            next_index = (current_index + d) % len(a)
            

    return a


"""
    [29 Dec 2018]
    Problems to practice:
    1. Generate all permutations of a string with repeated characters
    2. Johnson-Trotter: iteratively generate all permutations
    3. Generate all valid parenthesis obtained by removing minimum number of parenthesis from a string
    4. Lexicographically next number
    5. Solving cryptarithmetic puzzle
"""
    

# Complete the checkMagazine function below.
def checkMagazine(magazine, note):
    note_words = {}

    for word in note:
        if not word in note_words:
            note_words[word] = 1
        else:
            note_words[word] = note_words[word] + 1
   
    magazine_words = {}
    for word in magazine:
        if not word in magazine_words:
            magazine_words[word]  = 1
        else:
            magazine_words[word] = magazine_words[word] + 1

    answer = 'Yes'
    for word in note_words:

        if not (word in magazine_words) or note_words[word] > magazine_words[word]:
            answer = 'No'
            break

    return answer

def sherlockAndAnagrams(s):
    anagrams = {}

    for i in range(len(s)):
        for j in range(i, len(s)):
            sub = list(s[i:j+1])
            sub.sort()
            sub_hash = "".join(sub)
            if not sub_hash in anagrams:
                anagrams[sub_hash] = 1
            else:
                anagrams[sub_hash] = anagrams[sub_hash] + 1

    count = 0
    for key in anagrams:
        if anagrams[key] > 1:
            count = count + int(anagrams[key]*(anagrams[key]-1)/2)
    print(count)
    return count        


from collections import defaultdict
# Complete the freqQuery function below.
def freqQuery(queries):

    frequencies = defaultdict(lambda:0)
    answers = []

    freq_to_elems = defaultdict(list)

    for i in range(len(queries)):
        query_type, a = queries[i]
        if query_type == 1:
            if frequencies[a] > 0:
                freq_to_elems[frequencies[a]].remove(a)

            frequencies[a] = frequencies[a] + 1
            freq_to_elems[frequencies[a]].append(a)

        elif query_type == 2:
            if frequencies[a] > 0:
                freq_to_elems[frequencies[a]].remove(a)
                frequencies[a] = frequencies[a] - 1
                freq_to_elems[frequencies[a]].append(a)

            #if frequencies[a] < 0:
                #raise Exception("Keyerror for key {}".format(a))
        else:
            if not freq_to_elems[a]:
                answer = 0
            else:
                answer = 1

            answers.append(answer)

    return answer        
    
    
from collections import defaultdict

def construct_graph(arr, r):
    G = defaultdict(list)

    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[j] == r * arr[i]:
                G[i].append(j)

    return G


# Complete the countTriplets function below.
def countTriplets(arr, r):

    G = construct_graph(arr, r)
    
    count = 0
    for i in range(len(arr)):
        for j in G[i]:
            count = count + len(G[j])        

    return count                  
    
from collections import defaultdict
# Complete the countTriplets function below.
def countTriplets(arr, r):

    edge_counts = {}
    triplet_counts = {}
    
    count = 0
    value_to_indices = defaultdict(list)
    for i in range(len(arr)-1, -1, -1):
        edge_counts[i] = len(value_to_indices[r*arr[i]]) 
        triplet_counts[i] = sum([edge_counts[i] for i in value_to_indices[r*arr[i]]])
        value_to_indices[arr[i]].append(i)
        count = count + triplet_counts[i]
        
    return count      

from collections import defaultdict
# Complete the countTriplets function below.
def countTriplets(arr, r):
    indegrees = {}
    outdegrees = {}
    triplet_counts = {}
    
    count = 0
    value_to_indices = defaultdict(list)
    for i in range(len(arr)):
        if arr[i] % r == 0:
            indegrees[i] = len(value_to_indices[int(arr[i]/r)])
        else:
            indegrees[i] = 0
        value_to_indices[arr[i]].append(i)    


    value_to_indices = defaultdict(list)
    for i in range(len(arr)-1, -1, -1):
        outdegrees[i] = len(value_to_indices[r*arr[i]]) 
        value_to_indices[arr[i]].append(i)

    for i in range(len(arr)):
        triplet_counts[i] = indegrees[i] * outdegrees[i]
        count = count + triplet_counts[i]

    return count  


def maxSubsetSum(arr):
    max_sub_sums = {-2: 0, -1: 0}

    for i in range(len(arr)):
        max_sub_sums[i] = max(arr[i] + max_sub_sums[i-2], max_sub_sums[i-1])

    return max_sub_sums[len(arr)-1]  
    

def abbreviation(a, b):
    matches = {(0,0): True}

    for i in range(len(a)+1):
        for j in range(i+1, len(b)+1):
            matches[(i,j)] = False

    for i in range(1, len(a)+1):
        if a[:i].islower():
            matches[(i, 0)] = True
        else:
            matches[(i, 0)] = False
   
    for i in range(1, len(a)+1):
        for j in range(1, min(i+1, len(b)+1)):
            print(i,j)
            if a[i-1].upper() == b[j-1]:
                matches[(i,j)] = matches[(i-1, j-1)]
            else:
                if a[i-1].isupper():
                    matches[(i,j)] = False
                else:
                    matches[(i,j)] = matches[(i-1, j)]        

    return matches[(len(a), len(b))]

def update(i,j, matches):
    if  a[i-1] == b[j-1]:
        matches[(i,j)] = matches[(i-1, j-1)]
    
    if a[i-1].isupper() and a[i-1] != b[j-1]:
        matches[(i,j)] = False
    if a[i-1].islower() and a[i-1].upper() != b[j-1]:
        matches[(i,j)] = matches[(i-1, j)]
    if a[i-1].islower() and a[i-1].upper() == b[j-1]:
        matches[(i,j)] = matches[(i-1, j-1)] or matches[(i-1, j)]      
    return matches

# Complete the abbreviation function below.
def abbreviation(a, b):
    matches = {(0,0): True}

    for i in range(len(a)+1):
        for j in range(i+1, len(b)+1):
            matches[(i,j)] = False

    for i in range(1, len(a)+1):
        if a[:i].islower():
            matches[(i, 0)] = True
        else:
            matches[(i, 0)] = False
   
    for i in range(1, len(a)+1):
        for j in range(1, min(i+1, len(b)+1)):
            update(i,j, matches)  


    if matches[(len(a), len(b))]:
        return 'YES'
    else:
        return 'NO'


def get_slides(arr):
    up_slides = []
    down_slides = []
    """
        up_slides[i] = longest up-slide ending in i
        down_slides[i] = longest down-slide starting at i
    """
    for i in range(len(arr)):
        up_slides.append(1)
        down_slides.append(1)

    for i in range(1, len(arr)):
        if arr[i] > arr[i-1]:
            up_slides[i] = up_slides[i-1] + 1
        else:
            up_slides[i] = 1    

    for i in range(len(arr)-2, -1, -1):
        if arr[i] > arr[i+1]:
            down_slides[i] = down_slides[i+1] + 1
        else:
            down_slides[i] = 1


    return up_slides, down_slides   

# Complete the candies function below.
def candies(arr):
    if not arr:
        return 0

    up_slides, down_slides = get_slides(arr)

    count = 0
    for i in range(len(arr)):
        count = count + max(up_slides[i], down_slides[i])
        
    return count           


def alternatingCharacters(s):
    count = 0
    for i in range(len(s)-1):
        if s[i] == s[i+1]:
            count = count + 1
    return count        

partners = {"(": ")", "{": "}", "[": "]"}

def is_open(c):
    if c == "(" or c == "{" or c == "[":
        return True
    else:
        return False

def is_annihilating(char, stack):
    if not stack:
        return False
    
    if is_open(char):
        return False

    top = stack[-1]

    if not is_open(top):
        return False

    if partners[top] == char:
        return True
    else:
        return False

        
# Complete the isBalanced function below.
def isBalanced(s):
    stack = []

    for i in range(len(s)):
        if is_annihilating(s[i], stack):
            stack.pop(-1)
        else:
            stack.append(s[i])            

    if not stack:
        return 'YES'
    else:
        return 'NO'  


def minimum_bribes(q):
    count = 0
    for i in range(len(q)):
        if q[i] - (i+1) > 2:
            return 'Too chaotic'
        if q[i] - (i+1) > 0:
            count = count + q[i] - (i+1)
    return count

def minimumBribes(q):
    for i in range(len(q)):
        if q[i] - (i+1) > 2:
            return "Too chaotic"
            
    count = 0
    for i in range(len(q)):
       for j in range(i-1, -1, -1):
            if q[j+1] < q[j]:
               temp = q[j+1]
               q[j+1] = q[j]
               q[j] = temp
               count = count + 1
            else:  
                break      
    return count


####
def value(a):
    return a-1

_undefined = -1
def get_next_unvisited(root, visited):
    for i in range(root+1, len(visited)):
        if visited[i] == False:
            return i
    return _undefined        

# Complete the minimumSwaps function below.
def minimumSwaps(arr):
    if not arr:
        return 0

    visited = len(arr)*[False]
    root = 0
    count = 0

    while root != _undefined:
        count = count + 1
        print(root, visited)
        visited[root] = True
        current = root

        while value(arr[current]) != root:
            visited[current] = True
            current = value(arr[current])

        root = get_next_unvisited(root, visited)

    return len(arr) - count

def countSwaps(a):
    num_swaps = 0

    for i in range(len(a)):
        for j in range(len(a)-i-1):
            if a[j] > a[j+1]:
                temp = a[j]
                a[j] = a[j+1]
                a[j+1] = temp
                num_swaps = num_swaps + 1    


    first = a[0]
    last = a[-1]
    return num_swaps, first, last

def maximumToys(prices, k):
    prices.sort()

    num_toys = 0
    remaining_budget = k

    for i in range(len(prices)):
        if prices[i] <= remaining_budget:
            num_toys = num_toys + 1
            remaining_budget = remaining_budget - prices[i]
        else:
            break    

    return num_toys

from collections import defaultdict

class statsArray(object):
    def __init__(self, X):
        self.X = X

    def mean(self):
        mean = 0

        """
            numerically stable formula
            Learning: divisionbyzero error for (i/i+1) or (1/i+1)
        """
        for i in range(len(self.X)):
            mean = (i/(i+1))*mean + (1/(i+1))*self.X[i]

        return mean  

    def median(self):
        _undefined = None  
        A = sorted(self.X)
        if not A:
            return _undefined

        if len(A) == 0 % 2:
            median = (A[len(A)//2] + A[(len(A)//2)-1])/2
        else:
            median = A[len(A)//2]
        return median     

    def mode(self):
        """
            if you are using defaultdict then do not forget to initialize
            the default value, otherwise it might give keyerror
        """
        if not self.X:
            return None
        
        counts = defaultdict(lambda:0)

        A = self.X
        for i in range(len(A)):
            counts[A[i]] = counts[A[i]] + 1

        mode = sorted(counts, key=counts.get, reverse=True)[0]
        return mode

    def weighted_mean(self, W):
        X = self.X
        weighted_mean = 0

        w_sum = 0
        for i in range(len(X)):
            if not W[i] == 0:
                weighted_mean = (w_sum / (w_sum + W[i]))*weighted_mean + (W[i]/(w_sum + W[i]))*X[i]
                w_sum = w_sum + W[i]    
        return weighted_mean



def luckBalance(k, contests):
    contests.sort(key=lambda contest: (contest[0], -contest[1]), reverse=True)
    print(contests)

    luck_losses = [contest[1] for contest in contests if contests[0] == 1]

    return(sum(luck_losses[:-k]))

def luckBalance(k, contests):
    if not contests:
        return 0
    zero_luck_sum = 0
    for contest in contests:
        if contest[1] == 0:
            zero_luck_sum = zero_luck_sum + contest[0]

    

    one_lucks = []
    for contest in contests:
        if contest[1] == 1:
            one_lucks.append(contest[0])



    one_lucks.sort()

    

    one_win_sum = 0
    num_wins = max(len(one_lucks) - k, 0)

    for i in range(num_wins):
        one_win_sum = one_win_sum + one_lucks[i]
  
    one_loss_sum = 0
    for i in range(num_wins, len(one_lucks)):
        one_loss_sum = one_loss_sum + one_lucks[i] 

    return one_loss_sum + zero_luck_sum - one_win_sum      

"""    

from functools import cmp_to_key
class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score

        
    def __repr__(self):
        return("Name: {} Score: {}".format(name, score))
        
    def comparator(a, b):
        if a.score > b.score:
            return -1
        elif a.score < b.score:
            return 1
        elif a.score == b.score and a.name < b.name:
            return -1
        elif a.score == b.score and a.name > b.name:
            return 1
        else: 
            return 0                



n = int(input())
data = []
for i in range(n):
    name, score = input().split()
    score = int(score)
    player = Player(name, score)
    data.append(player)
    
data = sorted(data, key=cmp_to_key(Player.comparator))
for i in data:
    print(i.name, i.score)

"""

"""
    moving medians: simple
"""

def get_median(trail):
    if not trail:
        return -1

    trail.sort()

    if len(trail) == 0 % 2:
        median = (trail[(len(trail)//2)-1] + trail[len(trail)//2])/2
    else:
        median = trail[len(trail)//2]

    return median    



# Complete the activityNotifications function below.
def activityNotifications(expenditure, d):

    trail = expenditure[:d]
    median = get_median(trail)
    medians = [median]
    count = 0
    for i in range(d, len(expenditure)):
        trail.append(expenditure[i])
        trail.pop(0)
        if expenditure[i] >= 2*median:
            count = count + 1
        median = get_median(trail)
        medians.append(median)

    return(count)


undefined = -1
def height(root):
    if not root:
        return _undefined
    
    return 1 + max(height(root.left), height(root.right))


def is_left(v, root):
    if v < root.info:
        return True
    else:
        return False


def lca(root, v1, v2):
  #Enter your code here

  if root.info == v1 or root.info == v2:
      return root

  s1 = is_left(v1, root)
  s2 = is_left(v2, root) 

  if s1 != s2:
      return root
  else:
        if s1 == True:
            return lca(root.left, v1, v2)
        else:
            return lca(root.right, v1, v2) 


_infinity = 99999999
def get_max(root):
    if not root:
        return -_infinity
    
    current = root
    current_max = current.data
    while current.right:
        current = current.right
        current_max = max(current_max, current.data)
    return current_max
        
def get_min(root):
    if not root:
        return _infinity
    
    current = root
    current_min = current.data
    while current.left:
        current = current.left
        current_min = min(current_min, current.data)
    return current_min    
    
def checkBST(root):
    if not root:
        return True
    
    if not (checkBST(root.left) and checkBST(root.right)):
        return False
    
    max_left = get_max(root.left)
    min_right = get_min(root.right)
    
    if max_left > root.data or min_right < root.data:
        return False
    
    if max_left < root.data and min_right > root.data:
        return True        


def stepPerms(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:
        return 4

    # N = 10**9 + 7

    output = [1, 2, 4] 

    for i in range(3, n):
        output.append((output[i-1] + output[i-2] + output[i-3]))  

    return output[-1]                  

"""
    Fast Fibonacci
"""
def matrix_mult(A, B):
    a = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    b = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    c = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    d = A[1][0]*B[0][1] + A[1][1]*B[1][1]

    return [[a, b], [c, d]]
    


def matrix_pow(M, n):
    if n == 0:
        return [[1, 0], [0, 1]]
    if n % 2 == 1:
        return matrix_mult(M, matrix_pow(M, n-1))
    if n % 2 == 0:
        R = matrix_pow(M, n//2)
        return matrix_mult(R, R)

def right_mult(v, M):
    a = v[0]*M[0][0] + v[1]*M[1][0]
    b = v[0]*M[1][0] + v[1]*M[1][1]

    return (a, b)

    
    
    
def fibo(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    F = [[0, 1], [1, 1]]

    F_n = matrix_pow(F, n)

    (a, b) = right_mult((0, 1), F_n)

    return a

def intersect(A, B):
    if not A or not B:
        return []
    C = []
    i = 0
    j = 0
    
    while i < len(A) and j < len(B):
        if A[i] == B[j]:
            C.append(A[i])
            i = i + 1
            j = j + 1
        elif A[i] < B[j]:
            i = i + 1
        else:
            j = j + 1

    return C

def factors(m):
    divisors = []
    for i in range(1, m+1):
        if m % i == 0:
            divisors.append(i)
    return divisors

def gcd(m, n):
    m_factors = factors(m)
    n_factors = factors(n)

    common_factors = intersect(m_factors, n_factors)
    
    return common_factors[-1] 
       
    
from heapq import heapify, heappush, heappop   
def moving_medians(A, k):
    """
        Assume len(A) >= 2k  and A has no duplicate elements
    """
    left = [-a for a in A[:k]]
    right = A[k:2*k]

    heapify(left)
    heapify(right)

    for i in range(2*k, len(A)):
        a = A[i-2*k]
        b = A[i]

        #print(left, right)

        if -a in left and b < right[0]:
            i_a = left.index(-a)
            left[i_a] = -b
            heapify(left)

        elif -a in left and b > right[0]:
            heappush(right, b)
            left.remove(-a)
            c = heappop(right)
            heappush(left, -c)

        elif a in right and b > -left[0]:
            i_a = right.index(a)
            right[i_a] = b
            heapify(right)

        elif a in right and b < -left[0]:
            heappush(left, -b)
            right.remove(a)
            c = heappop(left)
            heappush(right, -c)

        else:
            pass


        print(-left[0], right[0])     

            
def square_root(n):
    if n == 0:
        return 0

    if n == 1:
        return 1

    begin = 2
    end = n

    """
        begin <= sqrt(n) <= end
    """
    while begin < end:
        mid = (begin + end)//2

        if mid*mid == n:
            return mid
        if mid*mid < n:
            begin = mid + 1
        if mid*mid > n:
            end = mid - 1

    if begin == end and begin*begin == n:
        return begin
    else:
        raise Exception("square root is not an integer")
    return None

def is_approx_equal(x, y, eps):
    if abs(x-y) < eps:
        return True
    else:
        return False

def approx_square_root(number, eps=0.01):
    if number == 0:
        return 0
    
    begin = 0
    end = number

    """
        begin <= sqrt(number) <= end
    """
    while begin < end + eps:
        mid = (begin + end)/2
        if is_approx_equal(mid*mid, number, eps):
            return mid

        if mid*mid < number:
            begin = mid

        if mid*mid > number:
            end = mid

    return begin        

def is_right(p, line):
    x = float(p[0])
    y = float(p[1])

    x1 = float(line[0][0])
    y1 = float(line[0][1])

    x2 = float(line[1][0])
    y2 = float(line[1][1])

    d = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)

    if d <= 0:
        return True
    else:
        return False
    

def is_inside(p, polygon):
    """
        Args: p (tuple of ints)
              polygon (list of tuples of ints)

        Return: True/False   
    """
    if len(polygon) < 3:
        return False

    for i in range(len(polygon)):
        line = polygon[i], polygon[(i+1) % len(polygon)]
        if not is_right(p, line):
            print(p, line)
            return False

    return True    

        

"""
    Number of swipe patterns of length k
"""

def get_neighbors(number):
    if number ==  1:
        return [2, 4]
    if number == 2:
        return [1, 3]
    if number == 3:
        return [2, 6]
    if number == 4:
        return [1, 5, 7]
    if number == 5:
        return [2, 4, 6, 8]
    if number == 6:
        return [3, 5, 9]
    if number == 7:
        return [4, 8]
    if number == 8:
        return [5, 7, 9]
    if number == 9:
        return [6, 8]

def get_extensions(prefix):
    if not prefix:
        raise Exception("Prefix is empty")

    last_number = prefix[-1]

    neighbors = get_neighbors(last_number)

    extensions = []
    for neighbor in neighbors:
        if not neighbor in prefix:
            extensions.append(neighbor)

    return extensions

def is_extendible(prefix):
    if not get_extensions(prefix):
        return False
    else:
        return True

def count_extensions(prefix, k):
    if len(prefix) == k:
        return 1

    if not is_extendible(prefix):
        return 0

    extensions = get_extensions(prefix)

    count = 0
    for extension in extensions:
        # new_prefix = [number for number in prefix]
        #new_prefix.append(extension)
        count = count + count_extensions(prefix + [extension], k)
        #new_prefix.pop(-1)

    return count
        
def count_swipes(k):
    count = 0
    for i in range(1, 10):
        prefix = [i]
        count = count + count_extensions(prefix, k)
    return count

"""
   maximum value path in a grid
"""
def update(i, j, table, matrix):
    table[(i,j)] = matrix[i][j] + max(table[(i+1, j)], table[(i,j+1)])
    return table

_zero = 0
def get_max_path(matrix):
    if not matrix:
        return _zero
    max_values = {}
    m = len(matrix)
    n = len(matrix[0])

    max_values[(m-1, n-1)] = matrix[m-1][n-1]

    for i in range(m+1):
        max_values[(i, n)] = _zero

    for j in range(n+1):
        max_values[(m, j)] = _zero


    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            update(i, j, max_values, matrix)

    return max_values[(0,0)]

def h_index(A):
    counts = defaultdict(lambda: 0)
    for i in range(len(A)):
        counts[A[i]] = counts[A[i]] + 1

    return counts     

        
