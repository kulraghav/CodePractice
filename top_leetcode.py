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

if __name__ == '__main__':
    # array
    print("array")
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

    # string
    print("string")
    test_reverse_string()
    test_first_unique_char()
    test_valid_palindrome()
    test_index_of()
    test_longest_common_prefix()


