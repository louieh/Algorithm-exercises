## LeetCode - Binary Search

[toc]

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
}
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

