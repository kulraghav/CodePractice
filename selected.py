"""
    Selected algo-coding problems
"""


"""
    Print all the permutations of a string
    The string might contain duplicate characters.
    Follow up: print only palindrome permutations
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

    for char in next_chars:

        """
            This is the 'extension' step.
        """
        prefix.append(char)
        decrement(remaining_counts, char)
        
        print_extensions(prefix, remaining_counts)

        """
            This is the 'backtracking' step.
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
    Alien Dictionary
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
        Therefore we have to explicitly cast the keys to list type
        in order to iterate through them with an index        
    """
    
    keys = list(trie.keys())

    for i in range(1, len(keys)):
        G[keys[i-1]].append(keys[i])

    for letter in trie:
        build_graph(G, trie[letter])

    return G    
        

    
"""
    O(n^2) implementation of topological sort
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
   O(m + n) implementation of topological sort
"""

def get_vertex_to_indegree(G):
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
    Combining the subroutines
"""
def get_alphabetical_order(words):
    trie = make_ordered_trie(words)
    
    G = {}
    G = build_graph(G, trie)

    alphabetical_order = top_sort(G)
    return alphabetical_order    

"""
    leetcode: https://leetcode.com/explore/interview/card/top-interview-questions-medium/ 
"""

"""
    generate all distinct three sums
    learning: in python one can check if a list is a member of another list using "in"
    for instance: [1,3] in [[0,1], [1,3], [2,4]]

    To do: 
    1. find a reference to the membership for list of lists
    2. how does python internally store list of list
"""
def two_sum(A, target):
    seen = {}
    for i in range(len(A)):
        if target-A[i] in seen:
            return True
        seen[A[i]] = i
    return False

def three_sum(A):
    for i in range(len(A)):
        if two_sum(A[i+1:], -A[i]):
            return True
    return False

def generate_all_two_sums(A, target):
    seen = {}
    two_sums = []
    for i in range(len(A)):
        if target - A[i] in seen and sorted([A[i], target-A[i]]) not in two_sums:
            two_sums = two_sums + [sorted([A[i], target-A[i]])]
        seen[A[i]] = i
    return two_sums

def generate_all_three_sums(A):
    three_sums = []
    for i in range(len(A)):
        two_sums = generate_all_two_sums(A[i+1:], -A[i])
        three_sums = three_sums + [sorted([A[i]] + x) for x in two_sums if sorted([A[i]] + x) not in three_sums]
    return three_sums    

def is_equal(X, Y):
    for x in X:
        if not x in Y:
            return False
    return True

def test_three_sum():
    A = [-1, 0, 1, 2, -1, -4]
    B = [1,2,3,4]
    assert three_sum(A) == True
    assert three_sum(B) == False
    print("Successfuly completed: three sum decision test")
    assert is_equal(generate_all_three_sums(A), [[-1, 0, 1], [-1, -1, 2], [-1, 0, 1]])
    print("Successfuly completed: three sum generate all test")
    print(".")

"""
    set matrix zero: erase rows and columns
"""

_erased = -1
def erase_row(M, i):
    for j in range(len(M[i])):
        if not M[i][j] == 0:
            M[i][j] = _erased
    return M

def erase_column(M, j):
    for i in range(len(M)):
        if not M[i][j] == 0:
            M[i][j] = _erased
        
def erase(M):
    if not M:
        return M
    
    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] == 0:
                erase_row(M, i)
                erase_column(M, j)

    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] == _erased:
                M[i][j] = 0
    return M             


def test_erase():
    M = [[1,1,1],
         [1,0,1],
         [1,1,1]]

    M_e = [[1,0,1],
           [0,0,0],
           [1,0,1]]

    N =  [[0,1,2,0],
          [3,4,5,2],
          [1,3,1,5]]
    N_e = [[0,0,0,0],
           [0,4,5,0],
           [0,3,1,0]]

    assert is_equal(erase(M), M_e)
    assert is_equal(erase(N), N_e)
    print("Successfuly passed: erase test")
    print(".")


    

    
