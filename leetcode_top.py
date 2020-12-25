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



if __name__ == '__main__':
    test_remove_duplicates()
    test_rotate_array()

