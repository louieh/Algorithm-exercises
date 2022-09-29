## LeetCode - Binary Search

[toc]

https://leetcode.com/discuss/general-discussion/786126/python-powerful-ultimate-binary-search-template-solved-many-problems

Template

```python
def binary_search(array) -> int:
    def condition(value) -> bool:
        pass

    left, right = min(search_space), max(search_space) # could be [0, n], [1, n] etc. Depends on problem
    while left < right:
        mid = left + (right - left) // 2
        if condition(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

What's really nice of this template is that, for most of the binary search problems, **we only need to modify three parts after copy-pasting this template, and never need to worry about corner cases and bugs in code any more**:

- Correctly initialize the boundary variables `left` and `right` to specify search space. Only one rule: set up the boundary to **include all possible elements**;
- Decide return value. Is it `return left` or `return left - 1`? Remember this: **after exiting the while loop, `left` is the minimal k satisfying the `condition` function**;
- Design the `condition` function. This is the most difficult and most beautiful part. Needs lots of practice.



Template I

```python
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # End Condition: left > right
    return -1
```

检查每个元素，直到最后只剩下一个元素，检查后 `while loop` 结束。

 

Template II

```python
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid

    # Post-processing:
    # End Condition: left == right
    if left != len(nums) and nums[left] == target:
        return left
    return -1
```

用于检查当前元素和其右边元素，可保证检查空间中至少有两个元素，当 `left == right` 时，循环跳出。跳出循环后添加对最后一个元素进行检查。

template III

```python
def binarySearch(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return -1

    left, right = 0, len(nums) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid
        else:
            right = mid

    # Post-processing:
    # End Condition: left + 1 == right
    if nums[left] == target: return left
    if nums[right] == target: return right
    return -1
```



### 33. Search in Rotated Sorted Array

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
      # 思考 findpeak 函数原理，边界条件，start 是可以等于 mid + 1 的，为什么？start 为什么不能 <= end，mid 为什么可以是直接 (start + end) // 2
        
        def findpeak(nums):
            start = 0
            end = len(nums) - 1
            while start < end:
                mid = start + (end - start) // 2
                if nums[mid] > nums[mid+1]:
                    return mid
                if nums[mid] > nums[end]:
                    start = mid
                else:
                    end = mid
            return -1
        
        def binsearch(start, end, nums, target):
            while start <= end:
                mid = start + (end - start) // 2
                if nums[mid] == target:
                    return mid
                if target > nums[mid]:
                    start = mid + 1
                else:
                    end = mid - 1
            return -1
        
        if not nums:
            return -1
        index = findpeak(nums)
        if index == -1:
            return binsearch(0, len(nums)-1, nums, target)
        res = binsearch(0, index, nums, target)
        if res == -1:
            res = binsearch(index+1, len(nums)-1, nums, target)
        return res
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        # 1,2,3,4,5,6,7
        # 5,6,7,1,2,3,4
        def find_peak():
            left, right = 0, len(nums)-1
            while left < right:
                mid = left + (right - left) // 2
                if nums[mid] > nums[right]:
                    if nums[mid] > nums[mid+1]:
                        return mid
                    else:
                        left = mid + 1
                else:
                    right = mid
            return left
        def binary_search(left, right):
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] == target:
                    return mid
                if nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return -1
        peak_index = find_peak()
        res = binary_search(0, peak_index)
        return res if res != -1 else binary_search(peak_index+1, len(nums)-1)
```



### 34. Find First and Last Position of Element in Sorted Array

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1,-1]
        
        res_l, res_r = -1, -1
        # left
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                if mid == 0:
                    res_l = 0
                    break
                else:
                    if nums[mid-1] != target:
                        res_l = mid
                        break
                    else:
                        right = mid - 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        if res_l == -1 and nums[left] != target:
            return [-1, -1]
        elif res_l == -1 and nums[left] == target:
            res_l = left
        if res_l == len(nums) - 1 or nums[res_l+1] != target:
            return [res_l, res_l]
        
        # right
        left, right = res_l + 1, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                if mid == len(nums) - 1:
                    return [res_l, len(nums)-1]
                else:
                    if nums[mid+1] != target:
                        return [res_l, mid]
                    else:
                        left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return [res_l, left]
```

```python
class Solution:
    # returns leftmost (or rightmost) index at which `target` should be inserted in sorted
    # array `nums` via binary search.
    def extreme_insertion_index(self, nums, target, left):
        lo = 0
        hi = len(nums)

        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] > target or (left and target == nums[mid]): # 通过当nums[mid] == target时left=mid还是right=mid来控制向左走还是向右走
                hi = mid
            else:
                lo = mid+1

        return lo


    def searchRange(self, nums, target):
        left_idx = self.extreme_insertion_index(nums, target, True)

        # assert that `left_idx` is within the array bounds and that `target`
        # is actually in `nums`.
        if left_idx == len(nums) or nums[left_idx] != target:
            return [-1, -1]

        return [left_idx, self.extreme_insertion_index(nums, target, False)-1]
```



### 35. Search Insert Position

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid
            else:
                left = mid + 1

        return left
```

notice that the input `target` might be larger than all elements in `nums` and therefore needs to placed at the end of the array. That's why we should initialize `right = len(nums)` instead of `right = len(nums) - 1`.

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return left
```



### 69. Sqrt(x)

```python
class Solution:
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        low = 0
        height = x
        while True:
            mid = (height - low) / 2 + low
            if int(mid * mid) == x:
                break
            if mid * mid > x:
                height = mid
            if mid * mid < x:
                low = mid

        return int(mid)
```

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        left, right = 0, x+1
        
        while left < right:
            mid = left + (right - left) // 2
            mid_2 = mid * mid
            if mid_2 == x:
                return mid
            elif mid_2 > x:
                right = mid
            else:
                left = mid + 1
        return left - 1
```



### 81. Search in Rotated Sorted Array II

```python
# 有问题， 有一个case过不去
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        if not nums:
            return False
        def peed_num(i, j):
            low = i
            high = j
            while low < high:
                mid = low + (high - low) // 2
                if nums[mid] > nums[high]:
                    if nums[mid] > nums[mid+1]:
                        return mid
                    low = mid + 1
                else:
                    high = mid
            return low
        
        def binary_search(i, j):
            low = i
            high = j
            while low <= high:
                mid = low + (high - low) // 2
                if nums[mid] == target:
                    return True
                elif nums[mid] > target:
                    high = mid - 1
                else:
                    low  = mid + 1
            return False
        
        peek = peed_num(0, len(nums)-1)
        return binary_search(0, peek) or binary_search(peek+1, len(nums)-1)
```

```python
# 此方法同样适用于没有重复元素的情况，也就是81题
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        
        left, right = 0, len(nums)-1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return True
            
            # [4,5,6,6,7,0,1,2,4,4]
            # If we know for sure left side is sorted or right side is unsorted
            elif nums[mid] > nums[left] or nums[mid] > nums[right]:
                if nums[mid] > target and target >= nums[left]:
                    right = mid - 1
                else:
                    left = mid + 1
            
            # If we know for sure right side is sorted or left side is unsorted
            elif nums[mid] < nums[right] or nums[mid] < nums[left]:
                if nums[mid] < target and target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            
            else:
            #If we get here, that means nums[start] == nums[mid] == nums[end], then shifting out
            #any of the two sides won't change the result but can help remove duplicate from
            #consideration, here we just use end-- but left++ works too
                right -= 1
        return False
```



### 153. Find Minimum in Rotated Sorted Array

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if not nums:
            return
        
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                if nums[mid] > nums[mid+1]:
                    return nums[mid+1]
                left = mid + 1 # 此处+1/+2均可以，+2有可能会直接到最小值，但这时候最小值就会变为第一个，同样可以到最后退出循环时被选中。此处加与不加均可以，可能时间会有所不同，根据不同取值。
            else:
                right = mid
        return nums[left]
```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right-left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]
```



### 154. Find Minimum in Rotated Sorted Array II

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if not nums:
            return
        
        low = 0
        high = len(nums) - 1
        while low < high:
            mid = low + (high - low) // 2
            if nums[mid] == nums[high]:
                high -= 1 	# 相等的时候high-=1
            elif nums[mid] > nums[high]:
                if nums[mid] > nums[mid+1]:
                    return nums[mid+1]
                else:
                    low = mid + 1
            else:
                high = mid
        return nums[low]
```

https://leetcode.com/articles/find-minimum-in-rotated-sorted-array-ii/



### 162. Find Peak Element

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        for i in range(len(nums)-1):
            if nums[i] > nums[i+1]:
                return i
        return len(nums)-1
```

```python
 class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        def search(l, r):
            if l == r:
                return l
            
            mid = (l + r) // 2
            if nums[mid] > nums[mid+1]:
                return search(l, mid)
            else:
                return search(mid+1, r)
        
        return search(0, len(nums)-1)
```



### 167. Two Sum II - Input array is sorted

```python
# binary Search
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        if not numbers or len(numbers)==1:
            return[]
        if numbers[0] > target:
            return[]
        high = len(numbers)-1
        while numbers[high] + numbers[0] > target:
            high-=1
            
            
        def binarySearch(low, high, target, target_index):
            while low <= high:
                mid = (high-low)//2+low
                if numbers[mid] == target and mid != target_index:
                    return mid
                elif numbers[mid] > target:
                    high = mid-1
                else:
                    low = mid+1
            return -1
        for i in range(high+1):
            ans = binarySearch(0, high, target-numbers[i], i)
            if ans != -1:
                return [i+1, ans+1]
        return []
```

```python
# two pointers
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        if not numbers:
            return []
        
        low = 0
        high = len(numbers)-1
        
        while low < high:
            if numbers[low] + numbers[high] == target:
                return [low+1, high+1]
            elif numbers[low] + numbers[high] > target:
                high-=1
            else:
                low+=1
        return []
```



### 270. Closest Binary Search Tree Value

```python
class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        if not root:
            return
        import sys
        self.res = [sys.maxsize, None]
        def helper(root):
            if not root:
                return
            diff = abs(root.val - target)
            if diff < self.res[0]:
                self.res[0] = diff
                self.res[1] = root.val
            if target > root.val:
                return helper(root.right)
            else:
                return helper(root.left)
        helper(root)
        return self.res[1]
```



### 278. First Bad Version

```python
# 思考二分的时候 while 循环何时写条件何时写True，目测是肯定能找到的情况下可以写True，但是写True是否对时间有影响，也就是说有没有可能可以提前退出的。或者说考虑在边界改变的情况下，有没有可能后边界已经超过前边界的时候仍然没有找到，不过不太可能出现这种情况如果可能有目标数的话。
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        first = 1
        last = n
        while True:
            mid = first + (last - first) // 2
            if not isBadVersion(mid):
                if isBadVersion(mid+1):
                    return mid+1
                first = mid + 1
            else:
                if mid == 1:
                    return 1
                last = mid
```

```python
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        first = 1
        last = n
        while first < last:
            mid = first + (last - first) // 2
            if not isBadVersion(mid):
                if isBadVersion(mid+1):
                    return mid+1
                first = mid + 2
            else:
                last = mid
        return first
```

```java
public int firstBadVersion(int n) {
    int left = 1;
    int right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (isBadVersion(mid)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
  	// 跳出循环之前会剩下[good, bad]，跳出循环时left会指向bad所以已经是符合condition的最小值了所以不用再次检查left
}
```

```python
class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                if mid == 1 or not isBadVersion(mid-1):
                    return mid
                else:
                    right = mid - 1
            else:
                left = mid + 1
        return left
```



### 279. Perfect Squares

```python
class Solution:
    def numSquares(self, n: int) -> int:
        if n < 2:
            return n
        
        sq_list = []
        i = 1
        while i**2 <= n:
            sq_list.append(i**2)
            i += 1
        
        check_list = {n}
        ans = 0
        while check_list:
            ans += 1
            temp = set()
            for each in check_list:
                for sq in sq_list:
                    if each == sq:
                        return ans
                    if each < sq:
                        break
                    temp.add(each - sq)
            check_list = temp
        return ans
```

用n分别减小于它本身的完全平方数，得到的差再去减小于它的完全平方数，直到遇到完全平方数为止。



### 367. Valid Perfect Square

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        
        low = 2
        high = num // 2
        while low <= high:
            mid = low + (high - low) // 2
            mid2 = mid * mid
            if mid2 == num:
                return True
            if mid2 > num:
                high = mid - 1
            else:
                low = mid + 1
        return False
```



### 374. Guess Number Higher or Lower

```python
class Solution:
    def guessNumber(self, n: int) -> int:
        left = 1
        right = n
        while True:
            mid = left + (right - left) // 2
            guess_num = guess(mid)
            if guess_num == 0:
                return mid
            elif guess_num == -1:
                right = mid - 1
            else:
                left = mid + 1
```

```python
class Solution:
    def guessNumber(self, n: int) -> int:
        
        left, right = 1, n
        while left <= right:
            mid = left + (right - left) // 2
            res = guess(mid)
            if res == 0:
                return mid
            elif res == -1:
                right = mid - 1
            else:
                left = mid + 1
```



### 378. Kth Smallest Element in a Sorted Matrix

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        
        def enough(x):
            res = 0
            j = len(matrix[0]) - 1
            for i in range(len(matrix)):
                while j >= 0 and matrix[i][j] > x:
                    j -= 1
                res += j+1
            return res >= k
        
        left, right = matrix[0][0], matrix[len(matrix)-1][len(matrix[0])-1]
        while left < right:
            mid = left + (right - left) // 2
            if enough(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/85173/Share-my-thoughts-and-Clean-Java-Code



### 410. Split Array Largest Sum

```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        def feasible(mid):
            total = 0
            counter = 1
            for num in nums:
                total += num
                if total > mid:
                    total = num
                    counter += 1
                    if counter > m:
                        return False
            return True
        
        left, right = max(nums), sum(nums)
        while left < right:
            mid = left + (right - left) // 2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

similar to LC 1011 and LC 875



### 436. Find Right Interval

```python
class Solution:
    def findRightInterval(self, intervals: List[List[int]]) -> List[int]:
        if len(intervals) == 1:
            return [-1]
        ans = []
        start_dict = {interval[0]: index for index, interval in enumerate(intervals)}
        start_list = sorted([interval[0] for interval in intervals])
        
        def binary_search(num):
            left, right = 0, len(start_list)-1
            if num > start_list[right]:
                return -1
            if num < start_list[left]:
                return 0
            while left < right:
                mid = left + (right - left) // 2
                if start_list[mid] == num:
                    return start_dict[start_list[mid]]
                if num < start_list[mid]:
                    right = mid
                elif num > start_list[mid]:
                    left = mid + 1
            return start_dict[start_list[left]]
        
        return [binary_search(interval[1]) for interval in intervals]
```



### 441. Arranging Coins

```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        sum_, i = 0, 1
        while True:
            sum_ += i
            if sum_ > n:
                return i - 1
            i += 1
```

```python
# binary search
class Solution:
    def arrangeCoins(self, n: int) -> int:
        left, right = 1, n
        
        while left <= right:
            mid = left + (right - left) // 2
            temp = (1+mid)*mid//2
            if temp == n:
                return mid
            elif temp > n:
                right = mid - 1
            else:
                left = mid + 1
        return right
```

```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        remain = n
        i = 1
        
        while remain >= i:
            remain -= i
            i += 1
        
        return i - 1
```

```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        
        left, right = 0, n
        while left <= right:
            mid = left + (right - left) // 2
            cur = (1 + mid) * mid / 2
            if cur == n:
                return mid
            if cur > n:
                right = mid - 1
            else:
                left = mid + 1
        
        return right
```

The reason we have to return `right` in the binary search approach in case `curr == n` never happens:

In the last iteration of the loop, `left == right == k`. `curr` calculated with this `k` will just barely overshoot or undershoot `n`.
If `k` is too small, `left` will be incremented, but with the new value of `left`, `curr` will overshoot `n`, so we want `left - 1` which is `right` since that's the max integer that would cause `curr` to undershoot `n`.
If `k` is too big, then `right` will be decremented, so `right` will now be the max integer that satisfies the inequality.

最后返回right的原因是，当循环进行到最后的时候left==right，mid也等于left和right，如果计算结果比n小，那么left会加一，这样最后计算结果会大于n，那我们想要的是最后left-1也就是right，如果计算结果比n大，那么正好我们需要mid-1也就是right，所以最后返回right



### 540. Single Element in a Sorted Array

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans ^= num
        return ans
```

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        left, right = 0, len(nums)-1
        
        while left <= right and left < len(nums) and right >= 0:
            mid = left + (right - left) // 2
            
            if (mid-1 >= 0 and nums[mid-1] == nums[mid]) or (mid+1 < len(nums) and nums[mid+1] == nums[mid]):
                currlen = right - left
                if currlen // 2 % 2 == 0:
                    if nums[mid-1] == nums[mid]:
                        right = mid - 2
                    else:
                        left = mid + 2
                else:
                    if nums[mid-1] == nums[mid]:
                        left = mid + 1
                    else:
                        right = mid - 1
            else:
                return nums[mid]
        return nums[left]
```
https://leetcode.com/problems/single-element-in-a-sorted-array/discuss/100733/Java-Binary-Search-with-Detailed-Explanation

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        left, right = 0, len(nums) - 1
        while left < right:
            mid  = left + (right - left) // 2
            if mid % 2 == 0 and nums[mid] == nums[mid+1] or mid % 2 == 1 and nums[mid] == nums[mid-1]:
                left = mid + 1
            else:
                right = mid
        return nums[left]
```

https://leetcode.com/problems/single-element-in-a-sorted-array/discuss/627921/Java-or-C%2B%2B-or-Python3-or-Easy-explanation-or-O(logn)-or-O(1)

```shell
EXPLANATION:-
Suppose array is [1, 1, 2, 2, 3, 3, 4, 5, 5]
we can observe that for each pair, 
first element takes even position and second element takes odd position
for example, 1 is appeared as a pair,
so it takes 0 and 1 positions. similarly for all the pairs also.

this pattern will be missed when single element is appeared in the array.

From these points, we can implement algorithm.
1. Take left and right pointers . 
    left points to start of list. right points to end of the list.
2. find mid.
    if mid is even, then it's duplicate should be in next index.
	or if mid is odd, then it's duplicate  should be in previous index.
	check these two conditions, 
	if any of the conditions is satisfied,
	then pattern is not missed, 
	so check in next half of the array. i.e, left = mid + 1
	if condition is not satisfied, then the pattern is missed.
	so, single number must be before mid.
	so, update end to mid.
3. At last return the nums[left]

Time: -  O(logN)
space:-  O(1)

IF YOU  HAVE ANY DOUBTS, FEEL FREE TO ASK
IF YOU UNDERSTAND, DON'T FORGET TO UPVOTE.
```



### 658. Find K Closest Elements

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        if not arr:
            return
        
        def binary_search():
            left, right = 0, len(arr) - 1
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid] == x:
                    return mid
                elif arr[mid] > x:
                    right = mid - 1
                else:
                    left = mid + 1
            return left
        
        if x <= arr[0]:
            temp = -(len(arr)-k) # 避免a[:-0]的情况
            if temp == 0:
                return arr[:]
            else:
                return arr[:temp]
        elif x >= arr[-1]:
            return arr[(len(arr)-k):]
        else:
            index = binary_search()
            low = max(0, index-k)
            high = min(len(arr)-1, index+k)
            while (high - low > k - 1):
                if x - arr[low] <= arr[high] - x:
                    high -= 1
                elif x - arr[low] > arr[high] - x:
                    low += 1
            return arr[low: high+1]
```

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        
        def find_index():
            left, right = 0, len(arr) - 1

            while left <= right:
                mid = left + (right - left) // 2
                if arr[mid] == x:
                    return mid, True
                elif arr[mid] > x:
                    right = mid - 1
                else:
                    left = mid + 1
            return left, False
        
        # 找中点
        index, exist = find_index()
        
        # print(f"index: {index}, exist: {exist}")
        # 设置左右索引
        if exist:
            res = [arr[index]]
            k -= 1
            l = r = index
        else:
            res = []
            l, r = index, index - 1
        
        # 向左， 向右遍历，比较左右元素与x差值，添加小的到结果中
        while k:
            if l - 1 >= 0:
                l_diff = abs(arr[l - 1] - x)
            else:
                l_diff = sys.maxsize
            
            if r + 1 <= len(arr) - 1:
                r_diff = abs(arr[r + 1] - x)
            else:
                r_diff = sys.maxsize
            
            
            if l_diff <= r_diff:
                l -= 1
                res.insert(0, arr[l])
            else:
                r += 1
                res.append(arr[r])
            
            k -= 1
        
        return res
```

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
      
      # 最后结果肯定是连续的
      # 设置最左最右两个指针，比较两边与x差值，向里缩小直到左右长度等于k
        
        left, right = 0, len(arr) - 1
        
        while right - left >= k:
            if abs(arr[left] - x) > abs(arr[right] - x):
                left += 1
            else:
                right -= 1
        
        return [arr[i] for i in range(left, right+1)]
```



### 668. Kth Smallest Number in Multiplication Table

```python
class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        
        def enough(num):
            total = 0
            for i in range(1, m+1):
                count = min(num//i, n)
                if count == 0:
                    break
                total += count
            return total >= k
        
        left, right = 1, m*n
        
        while left < right:
            mid = left + (right - left) // 2
            if enough(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

Enough function判断是否至少有k个数小于num，二分查找符合条件的最小的num即第k小

enough函数中min(num//i, n)因为乘法表每行按比例递增，第一行是1的倍数，第二行是2的倍数，第三行是3的倍数...每一行中用num//这行的倍数等于此行中有多少个数是小于等于num的，例如num=7，在第三行中[3,6,9...] 7//3=2，原理是第三行第一个数表示1个三，第二个数2个三，第三个数3个三等，那么判断num中有几个三则等于有几个数是小于等于它的。

count==0表示num比下面都小于是直接跳出。

```python
class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        
        def count(x): # 是否至少有k个元素小于等于x
            res = 0
            for i in range(1, m+1):
                res += min(x//i, n)
            return res >= k
        
        left, right = 1, m*n
        while left < right:
            mid = left + (right - left) // 2
            if count(mid):
                right = mid
            else:
                left = mid + 1
        return left
```



### 702. Search in a Sorted Array of Unknown Size

```python
class Solution:
    def search(self, reader, target):
        """
        :type reader: ArrayReader
        :type target: int
        :rtype: int
        """
        if target < reader.get(0):
            return -1
        # get range first
        low = 0
        high = 1
        while True:
            if reader.get(high) >= target and reader.get(low) <= target:
                break
            else:
                low = high
                high *= 2
        
        while low <= high:
            mid = low + (high - low) // 2
            if target == reader.get(mid):
                return mid
            elif target > reader.get(mid):
                low = mid + 1
            else:
                high = mid - 1
        return -1
```



### 704. Binary Search

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target: return mid
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1 
```



### 719. Find K-th Smallest Pair Distance

```python
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        
        def enough(distance):
            count = slow = fast = 0
            while slow < len(nums) or fast < len(nums):
                while fast < len(nums) and nums[fast] - nums[slow] <= distance:
                    fast += 1
                count += fast - slow - 1
                slow += 1
            return count >= k
        
        nums.sort()
        left, right = 0, nums[-1] - nums[0]
        while left < right:
            mid = left + (right - left) // 2
            if enough(mid):
                right = mid
            else:
                left = mid + 1
        return left
```



### 744. Find Smallest Letter Greater Than Target

```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        for each in letters:
            if each > target:
                return each
        return letters[0]
```

```python
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        low = 0
        high = len(letters) - 1
        while low < high:
            mid = low + (high - low) // 2
            if letters[mid] == target:
                if letters[mid+1] > target:
                    return letters[mid+1]
                else:
                    low = mid + 1
            elif letters[mid] < target:
                low = mid + 1
            else:
                if letters[mid-1] <= target:
                    return letters[mid]
                else:
                    high = mid - 1
        if letters[low] > target:
            return letters[low]
        else:
            return letters[0]
```



### 852. Peak Index in a Mountain Array

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        for i in range(len(arr)-1):
            if arr[i] > arr[i+1]:
                return i
```

```python
class Solution:
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        left, right = 0, len(arr) - 1
        
        while left < right:
            mid = left + (right - left) // 2
            if arr[mid] < arr[mid+1]:
                left = mid + 1
            else:
                right = mid
        return left
```



### 875. Koko Eating Bananas

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        def feasible(speed):
            return sum(math.ceil(pile/speed) for pile in piles) <= H
        
        left, right = 1, max(piles)
        while left < right:
            mid = left + (right - left) // 2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

similar to LC 1011 and LC 410



### 878. Nth Magical Number

```python
class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        
        def lcm(aa, bb):
            return aa * bb // math.gcd(aa, bb)
        
        def enough(mid):
            return mid // a + mid // b - mid // lcm(a, b) >= n
        
        # left, right = min(a, b), 10**10
        left, right = min(a,b), n*min(a,b)
        while left < right:
            mid = left + (right - left) // 2
            if enough(mid):
                right = mid
            else:
                left = mid + 1
        return left % (10**9+7)
```

similar to LC 1201

f(x) = mid // a + mid // b - mid // lcm(a, b) 至于为什么下面有证明

两个数字的最大公约数等于两数乘积除以他们的最小公倍数也就是lcm(a,b) = a*b//gcd(a,b)

**Intuition**

The number of magical numbers less than or equal to x*x* is a monotone increasing function in x*x*, so we can binary search for the answer.

**Algorithm**

Say L = \text{lcm}(A, B)*L*=lcm(*A*,*B*), the *least common multiple* of A*A* and B*B*; and let f(x)*f*(*x*) be the number of magical numbers less than or equal to x*x*. A well known result says that L = \frac{A * B}{\text{gcd}(A, B)}*L*=gcd(*A*,*B*)*A*∗*B*, and that we can calculate the function \gcdgcd. For more information on least common multiples and greatest common divisors, please visit [Wikipedia - Lowest Common Multiple](https://en.wikipedia.org/wiki/Least_common_multiple).

Then f(x) = \lfloor \frac{x}{A} \rfloor + \lfloor \frac{x}{B} \rfloor - \lfloor \frac{x}{L} \rfloor*f*(*x*)=⌊*A**x*⌋+⌊*B**x*⌋−⌊*L**x*⌋. Why? There are \lfloor \frac{x}{A} \rfloor⌊*A**x*⌋ numbers A, 2A, 3A, \cdots*A*,2*A*,3*A*,⋯ that are divisible by A*A*, there are \lfloor \frac{x}{B} \rfloor⌊*B**x*⌋ numbers divisible by B*B*, and we need to subtract the \lfloor \frac{x}{L} \rfloor⌊*L**x*⌋ numbers divisible by A*A* and B*B* that we double counted.

Finally, the answer must be between 00 and N * \min(A, B)*N*∗min(*A*,*B*).
Without loss of generality, suppose A \geq B*A*≥*B*, so that it remains to show

\lfloor \frac{N * \min(A, B)}{A} \rfloor + \lfloor \frac{N * \min(A, B)}{B} \rfloor - \lfloor \frac{N * \min(A, B)}{\text{lcm}(A, B)} \rfloor \geq N⌊*A**N*∗min(*A*,*B*)⌋+⌊*B**N*∗min(*A*,*B*)⌋−⌊lcm(*A*,*B*)*N*∗min(*A*,*B*)⌋≥*N*

\Leftrightarrow \lfloor \frac{N*A}{A} \rfloor + \lfloor \frac{N*A}{B} \rfloor - \lfloor \frac{N*A*\gcd(A, B)}{A*B} \rfloor \geq N⇔⌊*A**N*∗*A*⌋+⌊*B**N*∗*A*⌋−⌊*A*∗*B**N*∗*A*∗gcd(*A*,*B*)⌋≥*N*

\Leftrightarrow \lfloor \frac{N*A}{B} \rfloor \geq \lfloor \frac{N*\gcd(A, B)}{B} \rfloor⇔⌊*B**N*∗*A*⌋≥⌊*B**N*∗gcd(*A*,*B*)⌋

\Leftrightarrow A \geq \gcd(A, B)⇔*A*≥gcd(*A*,*B*)

as desired.



### 1011. Capacity To Ship Packages Within D Days

```python
class Solution:
    def shipWithinDays(self, weights: List[int], D: int) -> int:
        
        def canShip(capacity):
            days, total = 1, 0
            for weight in weights:
                total += weight
                if total > capacity:
                    days += 1
                    total = weight
                    if days > D:
                        return False
            return True
        
        left, right = max(weights), sum(weights)
        while left < right:
            mid = left + (right - left) // 2
            if canShip(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

similar to LC 875 and LC 410



### 1201. Ugly Number III

```python
class Solution:
    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        
        def lcm(aa, bb):
            return aa * bb // math.gcd(aa, bb)
        
        def enough(num):
            return num//a + num//b + num//c - num//lcm(a, b) - num//lcm(a, c) - num//lcm(b, c) + num//lcm(a, lcm(b, c)) >= n
        
        left, right = 1, 10**10
        while left < right:
            mid = left + (right - left) // 2
            if enough(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

https://leetcode.com/problems/ugly-number-iii/discuss/387780/JavaC%2B%2B-Binary-Search-with-Venn-Diagram-Explain-Math-Formula



### 1283. Find the Smallest Divisor Given a Threshold

```python
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        
        def reasible(mid):
            ans = 0
            for num in nums:
                ans += math.ceil(num/mid)
            return ans <= threshold
        
        left, right = 1, max(nums)
        while left < right:
            mid = left + (right - left) // 2
            if reasible(mid):
                right = mid
            else:
                left = mid + 1
        return left
```



### 1482. Minimum Number of Days to Make m Bouquets

```python
class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        if m * k > len(bloomDay):
            return -1
        
        def feasible(days):
            flowers_now = bouquets = 0
            for each in bloomDay:
                if each > days:
                    flowers_now = 0
                else:
                    bouquets += (flowers_now+1) // k
                    flowers_now = (flowers_now+1) % k
            return bouquets >= m
         # (flowers_now+1)//k 先把花数量增加一再除以k看目前可以组成几个花束
         # flowers_now = (flowers_now+1)%k 更新花数量，现增加一再余k，因为组成花束后便不可用所以重新计算
         # if each < days: flowers_now = 0 因为组成花束的花必须相邻所以如果遇到不相邻则将花数量重置为0
        
        left, right = 1, max(bloomDay)
        while left < right:
            mid = left + (right - left) // 2
            if feasible(mid):
                right = mid
            else:
                left = mid + 1
        return left
```

