"""
    References
    ----------
    - https://leetcode.com/explore/featured/card/top-interview-questions-easy/
    - https://vim.fandom.com/wiki/Search_and_replace
    - https://www.freecodecamp.org/news/10-important-git-commands-that-every-developer-should-know/
"""

"""
    Array
"""

"""
    remove duplicates from sorted array 
    - start 16 37
    - finish coding 17 00
    - finish debugging 17 06
    - added test cases and some practice with vi for search and replace 17 20
"""

def remove_duplicates(numbers):
    if len(numbers) < 2:
        return len(numbers)

    write_index = 1
    read_index = 1
    while read_index < len(numbers):
        if numbers[read_index] != numbers[write_index-1]:
            numbers[write_index] = numbers[read_index]
            write_index = write_index + 1
            read_index = read_index + 1
        else:
            read_index = read_index + 1
    return write_index

def test_remove_duplicates():
    # test remove_duplicates
    numbers = [1,2,3]
    assert remove_duplicates(numbers) == 3

    numbers = [1,1,1]
    assert remove_duplicates(numbers) == 1

    numbers = []
    assert remove_duplicates(numbers) == 0

    numbers = [1]
    assert remove_duplicates(numbers) == 1

    numbers = [1,1,2,2,3,3]
    assert remove_duplicates(numbers) == 3

    print('.')

    return True

"""
    rotate array
    - start 17:58
    - finish coding 18:20
    - finish testing 18:38 
"""

def gcd(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    if a == 1:
        return 1
    if b == 1:
        return 1
    if a == b:
        return a
    if b > a:
        return gcd(b, a)

    return gcd(b, a % b)

def rotate_array(numbers, k):
    if len(numbers) < 2:
        return numbers

    n = len(numbers)
    k = k % n

    if k == 0:
        return numbers

    for i in range(gcd(n, k)):
        current = numbers[i]
        for j in range(n//gcd(n, k)):
            temp = numbers[(i + (j+1)*k)%n]
            numbers[(i+(j+1)*k)%n] = current
            current = temp

    return numbers

def test_rotate_array():
    numbers = [1,2,3]
    k = 1
    
    assert rotate_array(numbers, k) == [3,1,2]

    numbers = [1,2,3,4,5,6]
    k = 3

    assert rotate_array(numbers, k) == [4,5,6,1,2,3]

    print('.')
    return True

"""
    single number: find unique number that is not duplicated
    - start 15:30
    - finish coding and testing 15:36 
"""
def single_number(numbers):
    x = 0
    for number in numbers:
        x = x^number
    return x

def test_single_number():
    numbers = [1,1,2]
    assert single_number(numbers) == 2

    numbers = [1,1,2,2,3]
    assert single_number(numbers) == 3

    print('.')
    return True

"""
    plus one
    start 15:40
    finish coding 15:48 
    finish coding 15:51
"""

def plus_one(digits):
    carry = 1
    for i in range(len(digits)-1, -1, -1):
        if digits[i] < 9:
            digits[i] = digits[i] + 1
            carry = 0
            break
        else:
            digits[i] = 0

    if carry == 1:
        return [1] + digits

    return digits


def test_plus_one():
    digits = [0]
    assert plus_one(digits) == [1]

    digits = [1,2,3]
    assert plus_one(digits) == [1,2,4]

    digits = [9]
    assert plus_one(digits) == [1,0]

    digits = [9,9]
    assert plus_one(digits) == [1,0,0]

    print('.')
    return True


"""
    two sum
    start: 15:55
    finish coding: 16:02
    finish testing: 16:07
"""

def two_sum(numbers, target):
    if len(numbers) < 2:
        return []

    seen = {}
    for i, number in enumerate(numbers):
        if target - number in seen:
            return [seen[target-number], i]
        seen[number] = i

    return []

def test_two_sum():
    numbers = [1,2,3]
    target = 4
    assert two_sum(numbers, target) == [0,2]

    numbers = [3,3,3]
    target = 6
    assert two_sum(numbers, target) == [0,1]

    numbers = [2,3,4]
    target = 8
    assert two_sum(numbers, target) == []

    print('.')
    return True

"""
    rotate image
    start 17:11
    finish coding 17:31
    finish testing 17:33
"""
def rotate_layer(matrix, top, left, bottom, right):
    for i in range(right-left):
        temp = matrix[top][left+i]
        matrix[top][left+i] = matrix[bottom-i][left]
        matrix[bottom-i][left] = matrix[bottom][right-i]
        matrix[bottom][right-i] = matrix[top+i][right]
        matrix[top+i][right] = temp
    return matrix

def rotate_image(matrix):
    top, left = 0,0
    bottom, right = len(matrix)-1, len(matrix)-1

    while top < bottom and left < right:
        rotate_layer(matrix, top, left, bottom, right)
        top = top + 1
        bottom = bottom - 1
        left = left + 1
        right = right - 1
    return matrix
    
def test_rotate_image():
    matrix = [1]
    assert rotate_image(matrix) == [1]

    matrix = [[1,2],[3,4]]
    assert rotate_image(matrix) == [[3,1],[4,2]]

    print('.')
    return True

"""
    buy and sell
    start 17 46
    finish coding 17 48
    finish testing 17:54
"""
def buy_and_sell_many(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] - prices[i-1] > 0:
            profit = profit + prices[i] - prices[i-1]
    return profit

def test_buy_and_sell_many():
    prices = [7,1,5,3,6,4]
    assert buy_and_sell_many(prices) == 7

    prices = [1,2,3,4,5]
    assert buy_and_sell_many(prices) == 4

    prices = [7,6,4,3,1]
    assert buy_and_sell_many(prices) == 0

    print('.')
    return True

"""
    contains duplicate
    start: 17 58
    finish coding: 17:59
    finish testing 18:01
"""
def contains_duplicate(numbers):
    seen = set()
    for number in numbers:
        if number in seen:
            return True
        seen.add(number)
    return False

def test_contains_duplicate():
    numbers = [1,2,3,1]
    assert contains_duplicate(numbers) == True

    numbers = [1,2,3,4]
    assert contains_duplicate(numbers) == False

    numbers = [1,1,1,3,3,4]
    assert contains_duplicate(numbers) == True

    numbers = []
    assert contains_duplicate(numbers) == False

    print('.')
    return True

"""
    array intersection
    start : 18:10
    finish testing 18:20
"""
from collections import Counter
def array_intersection(A, B):
    counts_A = Counter(A)
    counts_B = Counter(B)

    C = []
    for a in counts_A:
        if a in counts_B:
            for i in range(min(counts_A[a], counts_B[a])):
                    C.append(a)

    return C


def test_array_intersection():
    A = [1,2,2,1]
    B = [2,2]
    assert array_intersection(A, B) == [2,2]

    A = [4,9,5]
    B = [9,4,9,8,4]
    assert array_intersection(A, B) == [4,9]

    print('.')
    return True


"""
    move zeros to the end
    start 14:38
    finish coding 14:45
    finish testing 14:46
"""

def move_zeros(numbers):
    write_index = 0
    read_index = 0

    while read_index < len(numbers):
        if numbers[read_index] == 0:
            read_index = read_index + 1
        else:
            numbers[write_index] = numbers[read_index]
            write_index = write_index + 1
            read_index = read_index + 1

    for i in range(write_index, len(numbers)):
        numbers[i] = 0
    return numbers

def test_move_zeros():
    numbers = [0,1,0,3,12]
    assert move_zeros(numbers) == [1,3,12,0,0]

    numbers = [1,2,3]
    assert move_zeros(numbers) == [1,2,3]

    print('.')
    return True

"""
    valid sudoku
    start 14:53
    finish coding 15:12
    finish testing 15:20
"""
def valid_sudoku(board):
    # board is 9 x 9
    n = len(board)
    m = len(board[0])
    if m != 9 or n != 9:
        return False

    rows = [board[i] for i in range(n)]

    columns = []
    for j in range(n):
        columns.append([board[i][j] for i in range(n)])
    
    boxes = []
    for a in range(3):
        for b in range(3):
            box = []
            for i in range(9):
                for j in range(9):
                    if i // 3 == a and j // 3 == b:
                        box.append(board[i][j])
            boxes.append(box)
    
    # each row must contain digits 1-9 without repetition
    for i in range(n):
        digits = [x for x in rows[i] if x != '.']
        for digit in digits:
            if not digit.isdigit():
                return False
            if not (1 <= int(digit) <= 9):
                return False
        if not len(digits) == len(set(digits)):
            return False

    # each column must contain digits 1-9 without repetition
    for j in range(n):
        digits = [x for x in columns[i] if x != '.']
        for digit in digits:
            if not digit.isdigit():
                return False
            if not (1 <= int(digit) <= 9):
                return False
        if not len(digits) == len(set(digits)):
            return False

    # each of the 3x3 sub-boxes must contain digits 1-9 without repetition
    for i in range(n):
        digits = [x for x in boxes[i] if x != '.']
        for digit in digits:
            if not digit.isdigit():
                return False
            if not (1 <= int(digit) <= 9):
                return False
        if not len(digits) == len(set(digits)):
            return False

    return True

def test_valid_sudoku():
    board = [["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]

    assert valid_sudoku(board) == True

    board = [["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]

    assert valid_sudoku(board) == False

    print('.')
    return  True

"""
    reverse string
    start 10:02
    finish coding and testing 10:06
"""
def reverse_string(chars):
    i = 0
    j = len(chars)-1

    while i < j:
        temp = chars[i]
        chars[i] = chars[j]
        chars[j] = temp
        i = i + 1
        j = j - 1

    return chars

def test_reverse_string():
    chars = ['h','e','l','l','o']
    assert reverse_string(chars) == ['o','l','l','e','h']

    chars = list("Hannah")
    assert reverse_string(chars) == list("hannaH")

    print('.')
    return True

"""
    first unique char
    start 10:10
    finish coding 10:15
    finish testing 10 17
"""
def first_unique_char(s):
    counts = Counter(s)

    for i in range(len(s)):
        if counts[s[i]] == 1:
            return i

    return -1

def test_first_unique_char():
    s = "leetcode"
    assert first_unique_char(s) == 0

    s = "loveleetcode"
    assert first_unique_char(s) == 2

    print('.')
    return True

"""
    valid palindrome
    start 15:13
    finish coding 15:16 
    finish testing 15:20
"""
def valid_palindrome(s):
    i = 0
    j = len(s)-1
    while i < j:
        if not s[i].isalpha():
            i = i + 1
        elif not s[j].isalpha():
            j = j - 1
        elif not s[i].lower() == s[j].lower():
            return False
        else:
            i = i + 1
            j = j - 1
    return True

def test_valid_palindrome():
    s = "A man, a plan, a canal: Panama"
    assert valid_palindrome(s) == True

    s = "race a car"
    assert valid_palindrome(s) == False

    print('.')
    return True

"""
    index of
    start 21:33
    finish coding and testing 21:41 
"""
def is_matching(needle, haystack, i):
    if i + len(needle) > len(haystack):
        return False

    for j in range(len(needle)):
        if not haystack[i+j] == needle[j]:
            return False
    return True

def index_of(needle, haystack):
    if not needle:
        return 0

    for i in range(len(haystack)-len(needle)):
        if is_matching(needle, haystack, i):
            return i
    return -1

def test_index_of():
    needle = 'll'
    haystack = 'hello'
    assert index_of(needle, haystack) == 2

    needle = 'bba'
    haystack = 'aaaa'
    assert index_of(needle, haystack) == -1

    needle = ''
    haystac = ''
    assert index_of(needle, haystack) == 0

    print('.')
    return True

"""
    longest common prefix
    start 21:45
    finish: 21:55
"""
def longest_common_prefix(words):
    if not words:
        return ""

    n = min([len(word) for word in words])

    for i in range(n):
        for j in range(1, len(words)):
            if not words[j][i] == words[0][i]:
                return words[0][:i]
    return words[0][:n]

def test_longest_common_prefix():
    words = ["flower", "flow", "flight"]
    assert longest_common_prefix(words) == "fl"

    words = ["dog", "racecar", "car"]
    assert longest_common_prefix(words) == ""
    
    print('.')
    return True

"""
    reverse integer
    start 22:02
    finish: 22:12 
"""

def reverse_integer(x):
    if x == 0:
        return 0
    if x < 0:
        return -reverse_integer(-x)

    s = str(x)
    s_rev = s[::-1]

    answer = 0
    for i in range(len(s_rev)):
        if not s_rev[i] == 0:
            answer = int(s_rev[i:])
            break

    return answer

def test_reverse_integer():
    x = 123
    assert reverse_integer(x) == 321

    x = -123
    assert reverse_integer(x) == -321

    x = 120
    assert reverse_integer(x) == 21

    x = 0
    assert reverse_integer(x) == 0

    print('.')
    return True

"""
    valid anagram
    start 22:14
    finish coding and testing 22:19 
"""
def valid_anagram(s, t):
    counts_s = Counter(s)
    counts_t = Counter(t)

    for char in counts_s:
        if not char in counts_t:
            return False
        if not counts_s[char] == counts_t[char]:
            return False

    for char in counts_t:
        if not char in counts_s:
            return False
        if not counts_s[char] == counts_t[char]:
            return False
    return True

def test_valid_anagram():
    s = "anagram"
    t = "nagaram"
    assert valid_anagram(s, t) == True

    s = "rat"
    t = "car"
    assert valid_anagram(s, t) == False
    
    print('.')
    return True

"""
    string to integer
    start 22:26
    finish 22:54
"""
def a_to_i(s):
    s = s.lstrip(' ')
    if not s:
        return 0

    if not s[0].isdigit() and not s[0] in ['+', '-']:
        return 0

    n = 1
    for i in range(1, len(s)):
        if not s[i].isdigit():
            break
        n = n + 1

    if s[0].isdigit():
        return int(s[:n])
    elif s[0] == '+':
        return int(s[1:n])
    elif s[0] == '-':
        return -int(s[1:n])
    else:
        return 0

def test_a_to_i():
    s = '42'
    assert a_to_i(s) == 42

    s = "   -42"
    assert a_to_i(s) == -42

    s = "4193 with words"
    assert a_to_i(s) == 4193

    s = "words and 987"
    assert a_to_i(s) == 0

    print(".")
    return True

def update(word):
    output = []
    count = 1
    char = word[0]
    for i in range(1, len(word)):
        if word[i] == char:
            count = count + 1
        else:
            output.append("{}{}".format(count, char))
            count = 1
            char = word[i]

    output.append("{}{}".format(count, char))

    return ''.join(output)

"""
    count and say
    start 10 20
    finish 10 26
"""
def count_and_say(n):
    answer = '1'
    for i in range(1, n):
        answer = update(answer)
    return answer

def test_count_and_say():
    n = 1
    assert count_and_say(1) == '1'

    n = 4
    assert count_and_say(4) == '1211'

    print('.')
    return True

"""
    delete node
"""
class LinkedList:
    def __init__(self, value, next_node=None):
        self.val = value
        self.next_node = next_node

def delete_node(linked_list_node):
    current = linked_list_node
    prev = None
    while current.next_node.next_node:
        current.value = current.next_node.value
        prev = current
        current = current.next_node

    prev.next_node = None
    return linked_list_node

def reverse(head):
    if not head:
        return head
    if not head.next_node:
        return head

    prev = None
    current = head
    while current:
        temp = current.next_node
        current.next_node = prev
        prev = current
        current = temp

    return prev

def is_palindrome(head):
    if not head:
        return True
    if not head.next_node:
        return True

    # get length
    n = 0
    current = head
    while current:
        n = n + 1
        current = current.next_node

    m = n//2

    current = head
    for i in range(m-1):
        current = current.next_node

    
    if n % 2 == 1:
        right_half = current.next_node.next_node
        left_half = head
        current.next_node = None
    else:
        right_half = current.next_node
        left_half = head
        current.next_node = None

    right_half_reversed = reverse(right_half)

    current_left = head
    current_right = right_half_reversed

    while current_left and current_right:
        if not current_left.value == current_right.value:
            return False
        current_left = current_left.next_node
        current_right = current_right.next_node

    return True

def merge(A, B):
    write_A = len(A)-1
    read_A = len(A)-len(B)-1
    read_B = len(B)-1

    while read_A >= 0 and read_B >=0:
        if A[read_A] > B[read_B]:
            A[write_A] = A[read_A]
            write_A = write_A - 1
            read_A = read_A - 1
        else:
            A[write_A] = B[read_B]
            write_A = write_A - 1
            read_B = read_B - 1

    if read_A > 0:
        while read_A >= 0:
            A[write_A] = A[read_A]
            write_A = write_A - 1
            read_A = read_A - 1
    if read_B > 0:
        while read_A >=0:
            A[write_A] = B[read_B]
            write_A = write_A - 1
            read_B = read_B - 1
     
    return A

def test_merge():
    A = [1,2,3,0,0,0]
    B = [2,5,6]

    assert merge(A, B) == [1,2,2,3,5,6]
    print('.')
    return True

def climbing_stairs(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    lag_2 = 1
    lag_1 = 1

    for i in range(2, n+1):
        temp = lag_1
        lag_1 = lag_1 + lag_2
        lag_2 = temp

    return lag_1


def test_climbing_stairs():
    n = 2
    assert climbing_stairs(n) == 2

    n = 3
    assert climbing_stairs(n) == 3

    print('.')
    return True

def maximum_subarray(numbers):
    """
        table : dict
        table[i] := maximum sum of a contiguous subarray of numbers[:i] ending in numbers[i-1]
        returns max{table[i]: 0<= i <=n)}
    """
    table = {}
    table[0] = 0

    for i in range(1, len(numbers)+1):
        if table[i-1] < 0:
            table[i] = numbers[i]
        else:
            table[i] = table[i-1] + numbers[i]

    return max([table[i] for i in table if i != 0])

def test_maximum_subarray():
    numbers = [-2,1,-3,4,-1,2,1-5,4]
    assert maximum_subarray(numbers) == 6

    numbers = [1]
    assert maximum_subarray(numbers) == 1

    numbers = [0]
    assert maximum_subarray(numbers) == 0
    
    numbers = [-1]
    assert maximum_subarray(numbers) == -1

"""
    buy and sell once
    start 1138
    finish 1150
"""

def buy_and_sell_once(prices):
    if not prices:
        return 0

    min_price = prices[0]
    max_profit = 0

    for i in range(1, len(prices)):
        min_price = min(min_price, prices[i])
        max_profit = max(max_profit, prices[i]-min_price)

    return max_profit

def test_buy_and_sell_once():
    prices = [7,1,5,3,6,4]
    assert buy_and_sell_once(prices) == 5

    prices = [7,6,4,3,1]
    assert buy_and_sell_once(prices) == 0

    print('.')
    return True
        



"""
    first bad version
    start: 1120
    finish: 1135 
"""
def is_bad_version(x, bad):
    return (x >= bad)

def first_bad_version(n, bad):
    answer = n+1

    begin = 1
    end = n

    while begin <= end:
        mid = (begin + end)//2
        if is_bad_version(mid, bad):
            answer = mid
            end = mid - 1
        else:
            begin = mid + 1

    return answer

def test_first_bad_version():
    n = 5
    bad = 4
    assert first_bad_version(n, bad) == 4

    n = 1
    bad = 1
    assert first_bad_version(n, bad) == 1
    
    print('.')
    return True

"""
    3 sum
    start 11 03
    finish coding 11 17
    finish testing 11 25

    time : O(n^2 log n)
    space: O(1)
"""

def add_zero_triplets(numbers, i, zero_triplets):
    a = numbers[i]

    j = i + 1
    k = len(numbers)-1

    while j < k:
        if numbers[i] + numbers[j] + numbers[k] == 0:
            zero_triplets.add(tuple(sorted([numbers[i], numbers[j], numbers[k]])))
            j = j + 1
            k = k - 1
        elif numbers[i] + numbers[i] + numbers[k] < 0:
            j = j + 1
        else: # numbers[i] + numbers[j] + numbers[k] > 0
            k = k - 1

    return zero_triplets

def three_sum(numbers):
    numbers.sort()

    zero_triplets = set()

    for i in range(len(numbers)):
        add_zero_triplets(numbers, i, zero_triplets)

    return zero_triplets

def test_three_sum():
    numbers = [-1,0,1,2,-1,-4]
    assert three_sum(numbers) == {(-1,-1,2),(-1,0,1)}

    print('.')
    return True

"""
    group anagrams
    start : 11 28
    finish coding : 11 38
    finish testing: 12 01
"""
from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)

    for word in words:
        hash_word = ''.join(sorted(list(word)))
        groups[hash_word].append(word)

    return list(groups.values())
    

def test_group_anagrams():
    words = ["eat","tea","tan","ate","nat","bat"]
    assert sorted([sorted(group) for group in group_anagrams(words)]) ==  [["ate","eat","tea"],["bat"],["nat","tan"]]

    words = [""]
    assert group_anagrams(words) == [[""]]

    words = ["a"]
    assert group_anagrams([["a"]])


"""
    longest palindromic substring
    start : 1201
    finish coding: 1220
    finish testing: 12 46
    time: O(n^2)
    space: O(n) if we count storing the substring (can be made O(1) if we return the indices of the substring start and end)


"""
def longest_palindromic_substring(word):
    longest_len = 0
    longest_substring = ""

    for i in range(len(word)):
        # find longest palindromic substring of odd length with i in center
        current_len = 0
        while i - current_len - 1 >= 0 and i + current_len + 1 < len(word) and word[i-current_len-1] == word[i + current_len + 1]:
            current_len = current_len + 1
      
        if 2*current_len + 1  > longest_len:
            longest_len = 2*current_len + 1
            longest_substring = word[i-current_len: i + current_len + 1]

        # find longest palindromic substring of even length with i as left of center
        current_len = 0
        while i-current_len >= 0 and i+current_len+1 < len(word) and word[i-current_len] == word[i+current_len+1]:
            current_len = current_len + 1

        if 2*current_len > longest_len:
           longest_len = 2*current_len
           longest_substring = word[i-current_len+1: i + current_len + 1]
        
    return longest_substring

def test_longest_palindromic_substring():
    word = "babad"
    assert longest_palindromic_substring(word) in ["bab", "aba"]

    word = "cbbd"
    assert longest_palindromic_substring(word) == "bb"

    word = "a"
    assert longest_palindromic_substring(word) == 'a'

    word = "ac"
    assert longest_palindromic_substring(word) == 'a'

    print('.')
    return True
"""
    permutations
    start 1028
    finish writing test cases 1038
    finish coding 1046
    finish testing 1052
"""
def get_all_permutations(numbers):
    if not numbers:
        return [[]]

    permutations = []
    for i in range(len(numbers)):
        tail_numbers = [numbers[j] for j in range(len(numbers)) if j != i]
        tail_permutations = get_all_permutations(tail_numbers)
        for tail_permutation in tail_permutations:
            permutations.append([numbers[i]] + tail_permutation)
    return permutations

def test_get_all_permutations():
    def to_set(permutations):
        return set([tuple(permutation) for permutation in permutations])

    numbers = [1]
    answer = get_all_permutations(numbers)
    assert to_set(answer) == to_set([[1]])
    
    numbers = [1,2]
    answer = get_all_permutations(numbers)
    assert to_set(answer) == to_set([[1,2],[2,1]])

    numbers = [1,2,3]
    answer = get_all_permutations(numbers)
    assert to_set(answer) == to_set([[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])
    
    print('.')
    return True

"""
    evaluate reverse polish
    start 1100
    finish writing tests 1111
    finish coding 1118
    finish testing 1139

"""

def apply_op(op, a, b):
    a = int(a)
    b = int(b)
    if op == '*':
        return a * b
    if op == '/':
        if a // b >= 0:
            return a // b
        else:
            return - (-a // b)
    if op == '+':
        return a + b
    if op == '-':
        return a - b

def is_number(symbol):
    if not symbol:
        return False
    if symbol[0] == '-':
        return is_number(symbol[1:])
    return symbol.isnumeric()

def evaluate_reverse_polish(expression):
    stack = []
    for symbol in expression:
        if is_number(symbol):
            stack.append(int(symbol))
        else: # symbol is an op
            op = symbol
            b = stack.pop()
            a = stack.pop()
            stack.append(apply_op(op, a, b))
    return stack[-1]

def test_evaluate_reverse_polish():
    expression = ["2", "1", "+", "3", "*"]
    answer = evaluate_reverse_polish(expression)
    assert answer == 9

    expression = ["4", "13", "5", "/", "+"] 
    answer = evaluate_reverse_polish(expression) 
    assert answer == 6

    expression = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
    answer = evaluate_reverse_polish(expression) 
    assert answer == 22
    print('.')
    return True


"""
    generate parentheses
    start 1258
    finish coding 1316
    finish testing 1326
"""


def add_balanced_extensions(output, prefix, open_count, close_count, n):
    if open_count < close_count:
        return
    if open_count > n:
        return

    if open_count + close_count == 2*n:
        output.add(''.join(prefix))
        return 

    if open_count == close_count:
        prefix.append("(")
        add_balanced_extensions(output, prefix, open_count+1, close_count, n)
        prefix.pop()
    else: # open_count > close_count
        prefix.append("(")
        add_balanced_extensions(output, prefix, open_count+1, close_count, n)
        prefix.pop()

        prefix.append(")")
        add_balanced_extensions(output, prefix, open_count, close_count+1, n)
        prefix.pop()
    return 
    
def generate_parentheses(n):
    if n == 0:
        return [""]

    output = set()
    add_balanced_extensions(output, [], 0, 0, n)

    return output

def test_generate_parentheses():
    n = 3
    answer = generate_parentheses(n)
    assert set(answer) == set(["((()))","(()())","(())()","()(())","()()()"])

    n = 1
    answer = generate_parentheses(n)
    assert set(answer) == set(["()"])

    print('.')
    return True

"""
    merge intervals
    start 1535
    finish writing tests started coding 1540
    finish testing 1546
"""

def merge_intervals(intervals):
    if not intervals:
        return intervals

    intervals.sort(key=lambda x:x[0]) # sort by start time

    output = [intervals[0]]
    for i in range(1, len(intervals)):
        prev_start = output[-1][0]
        prev_end = output[-1][1]
        current_start = intervals[i][0]
        current_end = intervals[i][1]

        if current_start <= prev_end:
            output.pop()
            output.append([prev_start, current_end])
        else:
            output.append([current_start, current_end])
    return output

def test_merge_intervals():
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    assert merge_intervals(intervals) == [[1,6],[8,10],[15,18]]

    intervals = [[1,4],[4,5]]
    assert merge_intervals(intervals) == [[1,5]]


    print('.')
    return True

"""
    factorial trailing zeroes
    start 1550
    finish writing tests: 1554
    finish coding brute force: 1559
    finish testing brute force: 1600
    found the answer to followup question: 1610
    finish coding optimal solution: 1615
"""

def get_largest_expo(x, n):
    largest_expo = 0
    while n % x == 0:
        largest_expo = largest_expo + 1
        n = n // x
    return largest_expo

def factorial_trailing_zeroes_brute_force(n):
    two_expo = 0
    five_expo = 0

    for i in range(1, n+1):
        two_expo = two_expo + get_largest_expo(2, i)
        five_expo = five_expo + get_largest_expo(5, i)

    return min(two_expo, five_expo)

def get_largest_expo_fact(x, n):
    largest_expo_fact = 0
    while x <= n:
        largest_expo_fact = largest_expo_fact + n // x
        n = n // x
    return largest_expo_fact

def factorial_trailing_zeroes(n):
    two_expo_fact = get_largest_expo_fact(2, n)
    five_expo_fact = get_largest_expo_fact(5, n)
    return min(two_expo_fact, five_expo_fact)

def test_factorial_trailing_zeroes():
    n = 3
    assert factorial_trailing_zeroes(n) == 0

    n = 5
    assert factorial_trailing_zeroes(n) == 1

    n = 0
    assert factorial_trailing_zeroes(n) == 0
    print('.')
    return True

    
"""
    generate subsets
    start 1135
    finish coding recursive: 1142
    finish testing 1145
    finish coding space efficient version: 1157

"""

def add_extensions(prefix, numbers, output):
    if len(prefix) == len(numbers):
        output.append([numbers[i] for i in range(len(numbers)) if prefix[i]==1])
        return 

    prefix.append(0)
    add_extensions(prefix, numbers, output)
    prefix.pop()

    prefix.append(1)
    add_extensions(prefix, numbers, output)
    prefix.pop()
    return

    
def generate_subsets(numbers):
    output = []
    prefix = []
    add_extensions(prefix, numbers, output)
    return output

def generate_subsets_brute_force(numbers):
    if not numbers:
        return [[]]

    output = []
    tail_subsets = generate_subsets(numbers[:-1])
    for tail_subset in tail_subsets:
        output.append(tail_subset)
        output.append(tail_subset + [numbers[-1]])
    return output

def test_generate_subsets():
    def to_set(A):
        return set([tuple(a) for a in A])

    numbers = [1,2,3]
    answer = generate_subsets(numbers)
    assert to_set(answer) == to_set([[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]])

    numbers = [0]
    answer = generate_subsets(numbers)
    assert to_set(answer) == to_set([[],[0]])
    print('.')
    return True

"""
    top k most frequent
    start: 1200
    finished nlogn : 1208

"""

from collections import Counter
from heapq import heapify, heappush, heappop
def top_k_most_freq(numbers, k):
    counts = Counter(numbers)
    heap = []
    heapify([])
    for number in counts:
        heappush(heap, (counts[number], number))
        if len(heap) > k:
            heappop(heap)

    heap.sort(reverse=True)
    return [x[1] for x in heap]

def top_k_most_freq_nlogn(numbers, k):
    counts = Counter(numbers)
    output = sorted([x for x in counts], key=lambda x:counts[x], reverse=True)
    return output[:k]

def test_top_k_most_freq():
    numbers = [1,1,1,2,2,3]
    k = 2
    answer = top_k_most_freq(numbers, k)
    print(answer)
    assert top_k_most_freq(numbers, k) == [1,2]

    numbers = [1]
    k = 1
    assert top_k_most_freq(numbers, k) == [1]
    print('.')

"""
    bfs:
    start 1555
    end 1603
"""
from collections import deque
def bfs(G, s):
    if not s in G:
        return {}

    queue = deque([])
    distances = {s: 0}

    while queue:
        u = queue.popleft()
        for v in G[u]:
            if not v in distances:
                distances[v] = distances[u] + 1
                queue.append(v)

    return distances

"""
    dfs
    start: 1605
    end: 1615
"""
import math
_undefined = math.inf

def get_next_unvisited_neighbor(u, G, visits):
    for v in G[u]:
        if not v in visits:
            return v
    return None

def dfs(G, s):
    stack = [s]
    t = 0
    visits = {s: [0, _undefined]}

    while stack:
        t = t + 1
        u = stack[-1]
        v = get_next_unvisited_neighbor(u, G, visits)
        if not v:
            visits[v] = [t, _undefined]
        else:
            visits[u][1] = t
            stack.pop()
    return visits

"""
    topo sort
"""
def topo_sort_quadratic(G):
    if not G:
        return []

    output = []
    while G:
        sink = get_next_sink(G)
        output.append(sink)

        for v in G:
            G[v].remove(sink)
        del G[sink]

    return output
"""
    word search
    start : 1028
    finish writing test: 1037
    finish coding: 1046
    finish testing: 1103
"""
def get_next_cells(prefix, prefix_set, board, word):
    i, j = prefix[-1][0], prefix[-1][1]

    m = len(board)
    n = len(board[0])

    next_cells = []
    candidates = [(i+1,j), (i-1,j), (i,j+1),(i,j-1)]
    for candidate in candidates:
        a,b = candidate
        if not (a,b) in prefix_set and 0<= a < m and 0<= b < n and board[a][b]==word[len(prefix)]:
            next_cells.append((a,b))

    return next_cells
    
def search_extension(prefix, prefix_set, board, word):
    if len(prefix) == len(word):
        return True
    
    next_cells = get_next_cells(prefix, prefix_set, board, word)

    for next_cell in next_cells:
        prefix.append(next_cell)
        prefix_set.add(next_cell)
        if search_extension(prefix, prefix_set, board, word):
            return True
        prefix.pop()
        prefix_set.discard(next_cell)
    return False

def word_search(board, word):
    if not board:
        return False

    m = len(board)
    n = len(board[0])

    for i in range(m):
        for j in range(n):
            prefix = [(i,j)]
            prefix_set = set(prefix)
            if search_extension(prefix, prefix_set, board, word):
                return True
    return False

def test_word_search():
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    word = "ABCCED"
    assert word_search(board, word) == True

    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    word = "SEE" 
    assert word_search(board, word) == True

    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    word = "ABCB"
    assert word_search(board, word) == False
    print('.')
    return True


def binary_left_search(target, numbers):
    begin_index = 0
    end_index = len(numbers)-1

    left_search_index = -1
    while begin_index <= end_index:
        mid_index = (begin_index + end_index) // 2
        if numbers[mid_index] == target:
            left_search_index = mid_index
            end_index = mid_index-1
        elif numbers[mid_index] > target:
            end_index = mid_index-1
        else: # numbers[mid_index] < target
            begin_index = mid_index + 1
    return left_search_index

def binary_left_search_djn(key, nums):
    """search for left most position of key in sorted array nums[start:stop]"""
    if len(nums) == 0:
        return -1
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if key <= nums[mid]:
            if nums[mid] == key and (mid == 0 or nums[mid - 1] < key):
                return mid
            right = mid
        else:
            left = mid + 1
    return -1

def test_binary_left_search():
    key = 1
    nums = [1,1,1]
    assert binary_left_search(1, nums) == 0

    key = 2
    nums = [1,1,1]
    assert binary_left_search(2, nums) == -1

    print('.')
    return True

if __name__ == '__main__':
    # array
    print("array: easy")
    test_remove_duplicates()
    test_rotate_array()
    test_single_number()
    test_plus_one()
    test_two_sum()
    test_rotate_image()
    test_buy_and_sell_many()
    test_contains_duplicate()
    test_array_intersection()
    test_move_zeros()
    test_valid_sudoku()
    print("array: medium")
    test_three_sum()
    test_group_anagrams()
    test_longest_palindromic_substring()

    # string
    print("string")
    test_reverse_string()
    test_first_unique_char()
    test_valid_palindrome()
    test_index_of()
    test_longest_common_prefix()
    test_reverse_integer()
    test_valid_anagram()
    test_a_to_i()
    test_count_and_say()
    
    # DP
    print("dynamic programming")
    test_climbing_stairs()
    test_buy_and_sell_once()

    # sorting and searching
    print("sorting and searching: easy")
    test_merge()
    test_first_bad_version()
    test_binary_left_search()

    print("sorting and searching: medium")
    test_merge_intervals()
    test_top_k_most_freq()
   
    # backtracking
    print("backtracking: medium")
    test_get_all_permutations()
    test_generate_parentheses()
    test_generate_subsets()
    test_word_search()

    # math
    print("math: medium")
    test_factorial_trailing_zeroes()

    # others
    print("other: medium")
    test_evaluate_reverse_polish()

   
