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



### 41. First Missing Positive

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i, val in enumerate(nums):
            if val <= 0 or val > n:
                nums[i] = n+1
        for i, val in enumerate(nums):
            index = abs(val)
            if index > n:
                continue
            index -= 1
            if nums[index] > 0:
                nums[index] = -1 * nums[index]
        for i, val in enumerate(nums):
            if val > 0:
                return i + 1
        return n+1
```

https://leetcode.com/problems/first-missing-positive/discuss/17214/Java-simple-solution-with-documentation



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



### 134. Gas Station

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        sumGas = sumCost = tank = start = 0
        for i in range(len(gas)):
            sumGas += gas[i]
            sumCost += cost[i]
            tank += gas[i] - cost[i]
            if tank < 0:
                start = i + 1
                tank = 0
        if sumGas >= sumCost:
            return start
        else:
            return -1
```

几个facts：汽油总数大于等于消耗总数时是有答案的，反之没有答案。遍历过程中累积计算汽油与消耗，当发现小于零时将起始节点设置为下一节点，所以从A到B后发现累积小于零了那么把起始节点设置为B+1，累积清零，所以其中隐含着A和B中间的所有节点都无法到B。上面结论都需要证明。同样比较疑惑的是为什么把B+1设置为起始节点后，当总消耗小于总汽油量后直接返回B+1。



### 274. H-Index

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        count = [0]*(len(citations)+1)
        
        for each in citations:
            if each > len(citations):
                count[-1] += 1
            else:
                count[each] += 1
        
        cite_count_now = len(citations)
        total_paper_num = 0
        while cite_count_now >= 0:
            total_paper_num += count[cite_count_now]
            if total_paper_num >= cite_count_now:
                return cite_count_now
            cite_count_now -= 1
        return 0
```

```java
public int hIndex(int[] citations) {
    int len = citations.length;
    int[] count = new int[len + 1];
    
    for (int c: citations)
        if (c > len) 
            count[len]++;
        else 
            count[c]++;
    
    
    int total = 0;
    for (int i = len; i >= 0; i--) {
        total += count[i];
        if (total >= i)
            return i;
    }
    
    return 0;
}
```

先记录不同引用数paper的数量，大于数组长度的引用数记在最后，其余的记录在引用数为index，之后从最后一个位置开始遍历累加paper数，当paper数量大于等于当前引用数时返回当前引用数。



### 414. Third Maximum Number

```python
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        a = b = c = -sys.maxsize
        
        for num in nums:
            if num in {a, b, c}:
                continue
            if num > a:
                a, b, c = num, a, b
            elif num > b:
                a, b, c = a, num, b
            elif num > c:
                a, b, c = a, b, num
        
        return c if c != -sys.maxsize else max(a, b)
```



### 435. Non-overlapping Intervals

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) <= 1:
            return 0
        
        temp = sorted(intervals, key=lambda k: k[1])
        ans = 1
        end = temp[0][1]
        for i in range(1, len(temp)):
            if temp[i][0] >= end:
                end = temp[i][1]
                ans += 1
        return len(temp)-ans
```



### 442. Find All Duplicates in an Array

```python
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        ans = []
        
        for each in nums:
            if nums[abs(each)-1] < 0:
                ans.append(abs(each))
            else:
                nums[abs(each)-1] *= -1
        return ans
```



### 448. Find All Numbers Disappeared in an Array

```python
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = -abs(nums[index])
        return [i + 1 for i, val in enumerate(nums) if val > 0]
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



### 532. K-diff Pairs in an Array

```python
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        if k < 0:
            return 0
        res = 0
        from collections import Counter
        c = Counter(nums)
        for key, v in c.items():
            if k == 0 and v > 1:
                res += 1
            elif k != 0 and key+k in c:
                res += 1
        return res
```



### 560. Subarray Sum Equals K

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        temp_dict = collections.Counter()
        temp_dict[0] = 1
        ans = 0
        sum_ = 0
        for num in nums:
            sum_ += num
            if sum_ - k in temp_dict:
                ans += temp_dict[sum_-k]
            temp_dict[sum_] += 1
        return ans
```

累加每个数字并将结果存到dict中，也就是说每加一个数字就记录到目前为止和的个数。每次判断sum-k是否在dict中，也就是说sum-k的结果我们已经在之前遇到过，那么说明存在子列和等于k

例如：

[ 3 2 7 1 6 ] k = 10
from index 0 ... index 3 sum = 13
map:
3 : 1
5: 1
12: 1
when it comes to 13 the code check if 13 -10 = 3 in the map (我们遇到过3也就是说将现在的和减3便得到k)
well it is in the map then that means we found sub array that sums to 10 which from index 1 to index 3 ==> [ 2 7 1 ]



### 565. Array Nesting

```python
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        seen = set()
        res = 0
        for i in range(len(nums)):
            if nums[i] not in seen:
                temp_res = 0
                start = nums[i]
                while True:
                    start = nums[start]
                    seen.add(start)
                    temp_res += 1
                    if start == nums[i]:
                        break
                res = max(res, temp_res)
        return res
```



### 905. Sort Array By Parity

```python
class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        odd_list = []
        even_list = []
        
        for each in A:
            if each % 2 == 0:
                even_list.append(each)
            else:
                odd_list.append(each)
        return even_list + odd_list
```

```python
class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        i, j = 0, len(A)-1
        
        while i < j:
            if A[i] % 2 > A[j] % 2:
                A[i], A[j] = A[j], A[i]
            if A[i] % 2 == 0:
                i += 1
            if A[j] % 2 != 0:
                j -= 1
        return A
```



### 922. Sort Array By Parity II

```python
class Solution:
    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        odd_list = []
        even_list = []
        ans = []
        for each in A:
            if each % 2 == 0:
                even_list.append(each)
            else:
                odd_list.append(each)
        for i in range(len(A)):
            if i % 2 == 0:
                ans.append(even_list.pop())
            else:
                ans.append(odd_list.pop())
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



### 969. Pancake Sorting

```python
class Solution:
    def pancakeSort(self, A: List[int]) -> List[int]:
        ans = []
        
        for i in range(len(A), 1, -1):
            # i:= len(A) -> 2
            max_i = A.index(i)
            ans.extend([max_i+1, i])
            A = A[:max_i:-1] + A[:max_i]
        return ans
```

大意思想：先找到最大值然后从0到最大值做反转使得最大值到第一个位置，再把整个数组反转使得最大值到最后一个位置，这样则放置好了最大值，继续操作A[:len(A)-1]

代码大意：因为数组中数字是小于等于数组长度且不相同的，也就是全排列，所以for循环从len(A)开始到2截止。

`A.index(i)` 找到最大值的index，`[max_i+1,i]` 两次翻转，`A[:max_i:-1]+A[:max_i]` 重新组合A，也就是最大值后面的翻转+最大值前面，正好排出了最大值。

Python 切片：注意此处切片操作，[​a:/b:c]第三个数为step也就是步长，默认为1，也就是走一步取一个值。当步长大于零时，开始位置a需在结束位置b的左边，从左到右依步长取值，不包括结束位置。当步长小于零时，开始位置a需在结束位置b的右边，从右到左依步长取值，不包括结束位置。

开始位置a和结束位置b可以省略，当步长为正时，如果省略开始位置a那么a默认为最左位置，当步长为负时，如果省略开始位置a那么a默认为最右位置。



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



### 986. Interval List Intersections

```python
class Solution:
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        i = j = 0
        ans = []
        while i < len(A) and j < len(B):
            if A[i][0] <= B[j][1] and A[i][0] >= B[j][0] or B[j][0] <= A[i][1] and B[j][0] >= A[i][0]:
                ans.append([max(A[i][0], B[j][0]), min(A[i][1], B[j][1])])
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return ans
```

```python
class Solution:
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        i = j = 0
        ans = []
        while i < len(A) and j < len(B):
            if A[i][0] <= B[j][1] and A[i][0] >= B[j][0] or B[j][0] <= A[i][1] and B[j][0] >= A[i][0]:
                ans.append([max(A[i][0], B[j][0]), min(A[i][1], B[j][1])])
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return ans
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



### 1051. Height Checker

```python
class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        return sum(i != j for i, j in zip(heights, sorted(heights)))
```

此题有些问题，题目问最少交换多少次使得数组生序排列。例如：[1,1,4,2,1,3] 交换 4-1 和 4-3 两次便可以完成。但是答案为3次。



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



### 1288. Remove Covered Intervals

```python
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 1:
            return 1
        intervals.sort()
        
        cur = intervals[0]
        res = len(intervals)
        for i in range(1, len(intervals)):
            start, end = intervals[i]
            if start == cur[0] and end >= cur[1]:
                cur = intervals[i]
                res -= 1
            elif start > cur[0] and end <= cur[1]:
                res -= 1
            else:
                cur = intervals[i]
        return res
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
