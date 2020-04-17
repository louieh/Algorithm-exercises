## LeetCode - Array

[toc]

### 26. Remove Duplicates from Sorted Array

```python
# two pointers
# 将循环变量作为fast
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        slow = 0
        
        for fast in range(1, len(nums)):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]
        return slow+1
```

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        slow, fast = 0, 1
        temp = nums[0]
        while fast < len(nums):
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]
            fast += 1
        return slow + 1
```



### 27. Remove Element

```python
class Solution:
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        num = 0
        last_num = 1
        number = len(nums)
        for i in range(number):
            if nums[i] == val:
                num += 1
                while number-last_num != i:
                    if nums[number-last_num] != val:
                        temp = nums[i]
                        nums[i] = nums[number-last_num]
                        nums[number-last_num] = temp
                        last_num += 1
                        break
                    else:
                        num += 1
                        last_num += 1
            if number-last_num == i:
                break
                    
        return len(nums)-num
```

```python
# 10/5/2019
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        if not nums:
            return 0
        
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        return k
```



### 88. Merge Sorted Array

```python
class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        if n == 0:
            #nums1[:] = nums1 + nums2
            nums1 += nums2
        else:
            nums1[:] = nums1[:-n] + nums2
        temp_num = 0
        for i in range(len(nums1)):
            for j in range(len(nums1)-i):
                if j+1 == len(nums1):
                    break
                if nums1[j] > nums1[j+1]:
                    temp_num = nums1[j]
                    nums1[j] = nums1[j+1]
                    nums1[j+1] = temp_num
```

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i, j, k = m-1, n-1, m+n-1
        
        while i >= 0 and j >= 0:
            if nums1[i] >= nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        while j >= 0:
            nums1[k] = nums2[j]
            k -= 1
            j -= 1
```



### 485. Max Consecutive Ones

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        ans = 0
        ans_temp = 0
        for each in nums:
            if each == 1:
                ans_temp += 1
            else:
                if ans < ans_temp:
                    ans = ans_temp
                ans_temp = 0
        if ans < ans_temp:
            ans = ans_temp
        return ans
```

```python
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        ans = 0
        temp_max = 0
        
        for each in nums:
            if each == 1:
                temp_max += 1
            else:
                ans = max(ans, temp_max)
                temp_max = 0
        ans = max(ans, temp_max)
        return ans
```



### 941. Valid Mountain Array

```python
class Solution:
    def validMountainArray(self, A: List[int]) -> bool:
        if len(A) < 3:
            return False
        if A[0] > A[1]:
            return False
        peak = None
        for i in range(1, len(A)):
            if peak is not None:
                if A[i] >= A[i-1]:
                    return False
            else:
                if A[i] == A[i-1]:
                    return False
                if A[i] < A[i-1]:
                    peak = i-1
        return True if peak is not None else False
```

```python
class Solution:
    def validMountainArray(self, A: List[int]) -> bool:
        if len(A) < 3:
            return False
        
        if A[1] <= A[0]:
            return False
        if A[-1] >= A[-2]:
            return False
        
        for i in range(1, len(A)):
            if A[i] == A[i-1]:
                return False
            if A[i] < A[i-1]:
                break
        
        for j in range(i, len(A)-1):
            if A[j] <= A[j+1]:
                return False
        return True
```



### 977. Squares of a Sorted Array

```python
class Solution:
    def sortedSquares(self, A: List[int]) -> List[int]:
        if A[0] >= 0:
            return [each**2 for each in A]
        if A[-1] <= 0:
            return [each**2 for each in A][::-1]
        
        for i in range(len(A)):
            if A[i] >= 0:
                break
        A = [each**2 for each in A]
        A1, A2 = A[:i][::-1], A[i:]
        
        def merge(A1, A2):
            if A1[-1] <= A2[0]:
                return A1 + A2
            elif A2[-1] <= A1[0]:
                return A2 + A1
            else:
                ans = []
                i = j = 0
                
                while i < len(A1) and j < len(A2):
                    if A1[i] <= A2[j]:
                        ans.append(A1[i])
                        i += 1
                    else:
                        ans.append(A2[j])
                        j += 1
                
                while i < len(A1):
                    ans.append(A1[i])
                    i += 1
                while j < len(A2):
                    ans.append(A2[j])
                    j += 1
                return ans
            
        return merge(A1, A2)
```



### 1010. Pairs of Songs With Total Durations Divisible by 60

```python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        if not time:
            return 0
        
        from collections import Counter
        temp_dict = Counter()
        ans = 0
        for each in time:
            if each % 60 == 0 and 0 in temp_dict:
                ans += temp_dict[0]
            elif 60 - each % 60 in temp_dict:
                ans += temp_dict[60 - each % 60]
            temp_dict[each % 60] += 1
        return ans
```



### 1089. Duplicate Zeros

```python
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        num_zero_remain = 0  # 保留的0的个数
        length_ = len(arr) - 1  # 长度-1，以便下面length_-num_zero_remain可直接得到index
        for i in range(len(arr)):
            if length_ - num_zero_remain < i:  # 当i超过
                break
            if arr[i] == 0:  # 边界条件，当应保留的最后位置正好等于0时，这个0不计算在应保留的0个数中，把此0直接放到数组最后一个，并且最后位置向前挪1: length_-1
                if i == length_ - num_zero_remain:
                    arr[-1] = 0
                    length_ -= 1
                    break
                num_zero_remain += 1
        
        last_index = length_ - num_zero_remain  # 最后位置的index
        for i in range(last_index, -1, -1):  # 从后向前遍历，此时num_zero_remain相当于与实际位置的间距，当有0出现，两个位置置0并且间距缩短1，也就是num_zero_remain-1
            if arr[i] == 0:
                arr[i+num_zero_remain] = 0
                num_zero_remain -= 1
                arr[i+num_zero_remain] = 0
            else:
                arr[i+num_zero_remain] = arr[i]
```



### 1295. Find Numbers with Even Number of Digits

```python
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        return len([each for each in nums if len(str(each)) % 2 == 0])
```



### 1299. Replace Elements with Greatest Element on Right Side

```python
class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        if not arr:
            return
        
        ans = [-1]
        temp_max = -sys.maxsize
        for i in range(len(arr)-1, 0, -1):
            temp_max = max(temp_max, arr[i])
            ans.append(temp_max)
        return list(reversed(ans))
```



### 1346. Check If N and Its Double Exist

```python
class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        if not arr:
            return False
        
        temp_set = set()
        
        for each in arr:
            if each * 2 in temp_set:
                return True
            if each % 2 == 0 and each // 2 in temp_set:
                return True
            temp_set.add(each)
        return False
```
