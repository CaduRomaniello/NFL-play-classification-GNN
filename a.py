def calc_sq_rt(x, nums, begin, end):
    if (end - begin + 1) == 2:
        begin_sq = nums[begin] * nums[begin]
        end_sq = nums[end] * nums[end]

        if end_sq == x:
            return nums[end]
        if begin_sq == x:
            return nums[begin]
        if begin_sq < x and end_sq > x:
            return nums[begin]

    middle = (end + begin) // 2
    if nums[middle] * nums[middle] > x:
        return calc_sq_rt(x, nums, begin, middle)
    else:
        return calc_sq_rt(x, nums, middle, end)


def mySqrt(x: int) -> int:
    nums = [i for i in range(1, x)]
    return calc_sq_rt(x, nums, 0, len(nums) - 1)

print(mySqrt(8))  # Output: 2
print(mySqrt(4))  # Output: 2
print(mySqrt(9))  # Output: 3  
print(mySqrt(16))  # Output: 4
print(mySqrt(25))  # Output: 5
print(mySqrt(35))  # Output: 6