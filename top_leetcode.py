"""
    References
    ----------
    - https://leetcode.com/explore/featured/card/top-interview-questions-easy/
    - https://vim.fandom.com/wiki/Search_and_replace
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

if __name__ == '__main__':
    test_remove_duplicates()
    test_rotate_array()
    test_single_number()
    test_plus_one()
    test_two_sum()
