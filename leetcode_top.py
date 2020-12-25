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


if __name__ == '__main__':

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
