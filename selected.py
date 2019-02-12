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


"""
   group anagrams
   learning: sorted(word) gives a list, which is unhashable, hence can not be used as a key of dictionary
             we need to convert this list back to string using join
             strings are immutable and hence hashable

   To DO: find reference to what is hashable and what is not hashable
"""
from collections import defaultdict

def group_anagrams(words):
    anagrams = defaultdict(list)
    for word in words:
        anagrams["".join(sorted(word))].append(word)
    return anagrams

def is_inside(x, Y):
    for y in Y:
        if is_equal(x, y):
            return True
    return False

def test_group_anagrams():
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    grouped = group_anagrams(words)
    grouped_list = [grouped[ana] for ana in grouped]
    expected_grouped_list = [["ate","eat","tea"],
                        ["nat","tan"],
                        ["bat"]]
    print(grouped_list)
    for x in grouped_list:
        if not is_inside(x, expected_grouped_list):
            raise Exception("Test Failed")    
        
    print("Successfuly passed: grouped anagram test")
    print(".")

    
"""
    longest substring without repeting characters
"""        

def longest_nonrep(s):
    if not s:
        return 0
    
    last_seen = {s[0]: 0}
    
    begin = 0
    end = 0
    max_nonrep = 1
    
    while begin < len(s) and end < len(s):  
        if s[end] in last_seen:
            begin = last_seen[s[end]] + 1
        last_seen[s[end]] = end
        end = end + 1
        max_nonrep = max(max_nonrep, end-begin)
    return max_nonrep

def test_longest_nonrep():
    s = "abcabcbb"
    assert longest_nonrep(s) == 3
    s = "bbbbb"
    assert longest_nonrep(s) == 1
    s = "pwwkew"
    assert longest_nonrep(s) == 3
    print("Successfuly passed: longest nonrep")
    print(".")

"""
    increasing triplet subsequence
"""

_infinity = 99999999
def get_prefix_mins(A):
    prefix_mins = {-1: _infinity}
    for i in range(len(A)):
        prefix_mins[i] = min(A[i], prefix_mins[i-1])
    return prefix_mins

def get_suffix_maxs(A):
    suffix_maxs = {len(A): -_infinity}
    for i in range(len(A)-1, -1, -1):
        suffix_maxs[i] = max(A[i], suffix_maxs[i+1])
    return suffix_maxs

def contains_increasing_triplet(A):
    prefix_mins = get_prefix_mins(A)
    suffix_maxs = get_suffix_maxs(A)

    for i in range(len(A)):
        if prefix_mins[i-1] < A[i] < suffix_maxs[i+1]:
            return True
    return False

"""
    space efficient
"""
def increasing_triplet(A):
    if len(A) < 3:
        return False

    chain = [A[0]]
    for i in range(1, len(A)):
        if len(chain) == 1:
            if A[i] > chain[-1]:
                chain.append(A[i])
            else:
                chain[-1] = A[i]
        elif len(chain) == 2:
            if A[i] > chain[-1]:
                return True # triplet found
            else:
                if A[i] > chain[-2]:
                    chain[-1] = A[i]
                else:
                    chain[-2] = A[i]
    return False
                    
def test_increasing_triplet():
    A = [1,3,2,8,4]
    B = [5,4,6,2,1]
    assert contains_increasing_triplet(A) == True
    assert contains_increasing_triplet(B) == False
    assert increasing_triplet(A) == True
    assert increasing_triplet(B) == False
    
    print("Successfuly passed: increasing triplet")

"""
    letter combinations of the phone number
"""
digit_to_letters = {
        '1' : [],
        '2' : ['a', 'b', 'c'],
        '3' : ['d', 'e', 'f'],
        '4' : ['g', 'h', 'i'],
        '5' : ['j', 'k', 'l'],
        '6' : ['m', 'n', 'o'],
        '7' : ['p', 'q', 'r', 's'],
        '8' : ['t', 'u', 'v'],
        '9' : ['w', 'x', 'y', 'z']
        }

def add_extensions(number, prefix, output):
    if len(prefix) == len(number):
        output.append(prefix)
        return output

    next_digit = number[len(prefix)]

    for letter in digit_to_letters[next_digit]:
        new_prefix = prefix + letter
        add_extensions(number, new_prefix, output)

    return output    

        
def phone_number(number):
   
    output = []
    prefix = ""
    add_extensions(number, prefix, output)
    return output

def test_phone_number():
    number = '23'
    output = ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]

    assert is_equal(phone_number(number), output)
    print(phone_number(number))
    print("Successfuly passed: phone number test")



"""
    binary tree inorder traversal
"""

class Node:
    def __init__(self, root_value, left=None, right=None):
        self.root_value = root_value
        self.left = left
        self.right = right

        
def inorder(tree):
    if not tree:
        return []

    return inorder(tree.left) + [tree.root_value] + inorder(tree.right)


def test_inorder():
    tree = Node(1, Node(2, None, None), Node(3, None, None))

    assert is_equal(inorder(tree), [2, 1, 3])
    print(inorder(tree))
    print("Successfuly passed: test inorder")


"""
  intersection of two linked lists
"""

class linkedListNode:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next_node = next_node

    def show(self):
        current = self
        output = []
        while current:
            output.append(current.value)
            current = current.next_node
        return output

def intersection(A, B):
    values = {}

    current = A

    while current:
        values[current.value] = current
        current = current.next_node

    intersection = []

    current = B
    while current:
        if current.value in values:
            intersection.append(current.value)
        current = current.next_node

    return intersection     



def test_intersection():
    A = linkedListNode(1, linkedListNode(2, None))
    B = linkedListNode(2, linkedListNode(4, None))

    assert is_equal(intersection(A, B), [2])

    print("Successfuly passed: test intersection")


"""
    Add two numbers: INCOMPLETE
"""

def add_carry(number, carry):
    if carry == 0:
        return number
    
    current = number
    new_number = None
    while current:
        total = current.value + carry
        last_digit =  total % 10
        carry = (total - last_digit) // 10
        new_number = linkedListNode(last_digit, new_number)
        current = current.next_node

    if carry:    
        new_number = linkedListNode(carry, new_number)    
    return new_number

def test_add_carry():
    number = linkedListNode(8, linkedListNode(7, None))
    print(add_carry(number, 5).show())
    assert is_equal(add_carry(number, 5).show(), [8, 3])
    print("Successfully passed: test_add_carry")

def add_numbers(A, B):
    if not B:
        return A
    if not A:
        return B

    last_a = A.value
    last_b = B.value

    total = last_a + last_b

    last_digit = total % 10
    carry = (total - last_digit) // 10

    
    tail = add_numbers(A.next_node, B.next_node)
    addition = linkedListNode(last_digit, add_carry(tail, carry))
    return addition

def test_add_numbers():
    """
        is_equal function seems to be buggy
        it outputs True when comparing [0,1,0] and [1,0,0]
    """

    A = linkedListNode(8, None)
    B = linkedListNode(2, None)
    print(add_numbers(A, B).show())
    assert (is_equal(add_numbers(A, B).show(), [1, 0]))

    A = linkedListNode(8, linkedListNode(7, None))
    B = linkedListNode(2, linkedListNode(2, None))
    assert (is_equal(add_numbers(A, B).show(), [1,0,0]))
    print(add_numbers(A, B).show())
    print("Successfuly passed: test_add_numbers")
    
"""
    subsets
"""

def add_extensions(S, subsets, prefix):
    if len(prefix) == len(S):
        A = [S[i] for i in range(len(S)) if prefix[i] == 1]
        subsets.append(set(A))
        return subsets

    for b in range(2):
        prefix.append(b)
        add_extensions(S, subsets, prefix)
        prefix.pop(-1)
    return subsets    
           
def generate_subsets(S):
    subsets = []
    add_extensions(S, subsets, prefix=[])
    return subsets

def test_generate_subsets():
    subsets_0 = generate_subsets([])
    subsets_1 = generate_subsets([1])
    subsets_2 = generate_subsets([1,2])

    print(subsets_0)
    #assert subsets_0 == [[]]
    print(subsets_1)
    #assert is_equal(subsets_1, [[],[1]])
    print(subsets_2)
    #assert is_equal(subsets_2, [[],[1],[2],[1,2]])

    

    
        
