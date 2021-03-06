## LeetCode - Array

[toc]

### 4. Median of Two Sorted Arrays

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        res_index = (len(nums1) + len(nums2)) // 2
        if not nums1 or not nums2:
            new_nums = nums1 or nums2
        else:
            new_nums = []
            if nums2[0] >= nums1[len(nums1)-1]:
                new_nums = nums1 + nums2
            elif nums1[0] >= nums2[len(nums2)-1]:
                new_nums = nums2 + nums1
        if new_nums:
            if (len(nums1) + len(nums2)) % 2 == 0:
                return (new_nums[res_index]+new_nums[res_index-1])/2
            else:
                return new_nums[res_index]

        nums1_index = 0
        nums2_index = 0
        
        for i in range(len(nums1)+len(nums2)):
            if nums1_index <= len(nums1)-1 and nums2_index <= len(nums2)-1:
                if nums1[nums1_index] <= nums2[nums2_index]:
                    new_nums.append(nums1[nums1_index])
                    nums1_index += 1
                else:
                    new_nums.append(nums2[nums2_index])
                    nums2_index += 1
            else:
                if nums1_index <= len(nums1)-1:
                    new_nums.append(nums1[nums1_index])
                    nums1_index += 1
                elif nums2_index <= len(nums2)-1:
                    new_nums.append(nums2[nums2_index])
                    nums2_index += 1
            if i == res_index:
                if (len(nums1) + len(nums2)) % 2 == 0:
                    return (new_nums[res_index]+new_nums[res_index-1])/2
                else:
                    return new_nums[res_index]
```



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



### 61. Rotate List

```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head or k == 0:
            return head
        
        temp_move = temp_l = head
        
        # get length of linked list
        length = 0
        while temp_l:
            length += 1
            temp_l = temp_l.next
        
        # 计算倒数第几个
        num = k % length
        if num == 0:
            return head
        
        # 将指针挪到上面那个数的前一个, 并取出后面那个数作为新头且断开与后面那个数的链接
        for i in range(length-num-1):
            temp_move = temp_move.next
        new_head = temp3 = temp_move.next
        temp_move.next = None
        
        # 将结尾与原来的头链接
        while temp3 and temp3.next:
            temp3 = temp3.next
        temp3.next = head
        
        return new_head
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



### 152. Maximum Product Subarray

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        min_, max_, ans = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            max_temp = max(max(max_*nums[i], min_*nums[i]), nums[i])
            min_temp = min(min(max_*nums[i], min_*nums[i]), nums[i])
            ans = max(ans, max_temp)
            max_ = max_temp
            min_ = min_temp
        return ans
```



### 169. Majority Element

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        from collections import Counter
        
        c = Counter(nums)
        for k,v in c.items():
            if v > len(nums) // 2:
                return k
```

```python
class Solution:
    def majorityElement(self, nums):
        nums.sort()
        return nums[len(nums)//2]
```

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        
        def partition(left, right):
            small = left - 1
            for i in range(left, right):
                if nums[i] < nums[right]:
                    small += 1
                    if small != i:
                        nums[small], nums[i] = nums[i], nums[small]
            small += 1
            nums[small], nums[right] = nums[right], nums[small]
            return small
        
        left, right = 0, len(nums)-1
        mid = left + (right - left) // 2
        index = partition(left, right)
        while mid != index:
            if index > mid:
                right = index - 1
                index = partition(left, right)
            else:
                left = index + 1
                index = partition(left, right)
        return nums[mid]
```

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        can, count = 0, 0
        for num in nums:
            if count == 0:
                can, count = num, 1
            elif can == num:
                count += 1
            else:
                count -= 1
        return can
```



### 187. Repeated DNA Sequences

```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        if len(s) < 10: return []
        seen, repeat = set(), set()
        for i in range(len(s)+1):
            if s[i:i+10] not in seen:
                seen.add(s[i:i+10])
            else:
                repeat.add(s[i:i+10])
        return list(repeat)
```



### 189. Rotate Array

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums:
            return
        k = k % len(nums);
        count = 0
        for i in range(len(nums)):
            last_starter = i
            index = i
            temp1 = nums[index]
            temp2 = None
            while 1:
                wait_index = (index + k) % len(nums)
                temp2 = nums[wait_index]
                nums[wait_index] = temp1
                index = wait_index
                temp1 = temp2
                count += 1
                if count == len(nums):
                    return
                if wait_index == last_starter:
                    break
```

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k %= len(nums)
        if k == 0: return nums
        
        start = count = 0
        while count < len(nums):
            cur_index, prev_num = start, nums[start]
            while 1:
                next_index = (cur_index + k) % len(nums)
                nums[next_index], prev_num = prev_num, nums[next_index]
                cur_index = next_index
                count += 1
                if cur_index == start: break
            start += 1
        return nums
```



### 215. Kth Largest Element in an Array

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        if not nums:
            return
        
        def swap(a, b, nums=nums):
            if a == b:
                return
            temp = nums[a]
            nums[a] = nums[b]
            nums[b] = temp
        
        def findKthLargest_tool(nums, left, right, k):
            if left > right:
                return -1
            
            import random
            pivot = int((right - left + 1) * random.random()) + left
            swap(pivot, right)
            c = left
            for i in range(left, right):
                if nums[i] >= nums[right]:
                    swap(i, c)
                    c += 1
            swap(c, right)
            
            if c == k-1:
                return nums[c]
            elif c > k-1:
                return findKthLargest_tool(nums, left, c-1, k)
            return findKthLargest_tool(nums, c+1, right, k)
        
        return findKthLargest_tool(nums, 0, len(nums)-1, k)
```

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(left, right):
            large = left - 1
            for i in range(left, right):
                if nums[i] > nums[right]:
                    large += 1
                    if large != i:
                        nums[large], nums[i] = nums[i], nums[large]
            large += 1
            nums[large], nums[right] = nums[right], nums[large]
            return large
        
        left, right = 0, len(nums) - 1
        index = partition(left, right)
        while index != k - 1:
            if index > k - 1:
                right = index - 1
            else:
                left = index + 1
            index = partition(left, right)
        return nums[index]
```



### 229. Majority Element II

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        return [each for each, val in collections.Counter(nums).items() if val > len(nums)//3]
```

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        can1, count1 = 0, 0
        can2, count2 = 1, 0
        for num in nums:
            if can1 == num:
                count1 += 1
            elif can2 == num:
                count2 += 1
            elif count1 == 0:
                can1, count1 = num, 1
            elif count2 == 0:
                can2, count2 = num, 1
            else:
                count1 -= 1
                count2 -= 1
        return [each for each in (can1, can2) if nums.count(each) > len(nums) // 3]
```



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



### 349. Intersection of Two Arrays

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1_set = set(nums1)
        nums2_set = set(nums2)
        res = []
        for each in nums1_set:
            if each in nums2_set:
                res.append(each)
        return res
```

```js
This is a Facebook interview question.
They ask for the intersection, which has a trivial solution using a hash or a set.

Then they ask you to solve it under these constraints:
O(n) time and O(1) space (the resulting array of intersections is not taken into consideration).
You are told the lists are sorted.

Cases to take into consideration include:
duplicates, negative values, single value lists, 0's, and empty list arguments.
Other considerations might include
sparse arrays.

function intersections(l1, l2) {
    l1.sort((a, b) => a - b) // assume sorted
    l2.sort((a, b) => a - b) // assume sorted
    const intersections = []
    let l = 0, r = 0;
    while ((l2[l] && l1[r]) !== undefined) {
       const left = l1[r], right = l2[l];
        if (right === left) {
            intersections.push(right)
            while (left === l1[r]) r++;
            while (right === l2[l]) l++;
            continue;
        }
        if (right > left) while (left === l1[r]) r++;
         else while (right === l2[l]) l++;
        
    }
    return intersections;
}
```



### 350. Intersection of Two Arrays II

```python
# 2 years ago since 3/22/2020
class Solution(object):
    def intersect(self, nums1, nums2):
        finArray = []
        for each in nums1:
            if each in nums2:
                finArray.append(each)
                nums2.remove(each)
        return finArray
        
```

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        from collections import Counter
        
        nums1Dict = Counter(nums1)
        res = []
        for each in nums2:
            if each in nums1Dict and nums1Dict[each] != 0:
                res.append(each)
                nums1Dict[each] -= 1
        return res
```



### 390. Elimination Game

```java
class Solution {
    public int lastRemaining(int n) {
        int head = 1;
        int remain = n;
        int step = 1;
        boolean left = true;
        while (remain > 1) {
            if (left || remain % 2 != 0) {
                head += step;
            }
            remain /= 2;
            step *= 2;
            left = !left;
        }
        return head;
    }
}
```

https://leetcode.com/problems/elimination-game/discuss/87119/JAVA%3A-Easiest-solution-O(logN)-with-explanation



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



### 456. 132 Pattern

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[j] > nums[i]:
                    for k in range(j+1, len(nums)):
                        if nums[k] < nums[j] and nums[k] > nums[i]:
                            return True
        return False
```

```python
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        if len(nums) < 3: return False
        stack = []
        i = len(nums) - 1
        s3 = -sys.maxsize
        while i >= 0:
            if nums[i] < s3: return True
            while stack and nums[i] > stack[-1]:
                s3 = stack.pop()
            stack.append(nums[i])
            i -= 1
        return False
```

https://leetcode.com/problems/132-pattern/discuss/94071/Single-pass-C%2B%2B-O(n)-space-and-time-solution-(8-lines)-with-detailed-explanation.

我们要找s1<s3<s2，从后向前遍历插入栈，如果遇到元素(s2)大于栈顶元素则pop出栈顶元素作为s3，此时s2的index是小于s3的但其值是大于s3的(s3<s2)，那么下面如果遇到有元素小于s3我们便找到了s1，满足条件s1<s3<s2



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



### 525. Contiguous Array

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        temp_dict = dict()
        count = ans = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                count -= 1
            else:
                count += 1
            if count == 0:
                ans = max(ans, i+1)
            if count not in temp_dict:
                temp_dict[count] = i
            else:
                ans = max(ans, i - temp_dict[count])
        return ans
```

遇0减一遇1加一，并将此数作为key，index作为值存入字典，如果遇到相同的数组说明有相等的0和1出现，用当前index值键字典中相同数字的index得到长度。
如果遇到数字为0，说明从0到目前index为止的0和1数量相同



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



### 605. Can Place Flowers

```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        counter = collections.Counter(flowerbed)
        if n > math.ceil(counter.get(0, 0) / 2): return False
        if counter.get(1) == 0: return n == math.ceil(counter.get(0) / 2)
        
        i = res = 0
        while i < len(flowerbed):
            if flowerbed[i] == 0 and (i == len(flowerbed)-1 or flowerbed[i+1] == 0):
                res += 1
                i += 2
                continue
            if flowerbed[i] == 1:
                i += 2
                continue
            i += 1
            
        return res >= n
```



### 697. Degree of an Array

```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        count = collections.Counter(nums)
        degrees = []
        max_degree = max(count.values())
        for key, val in count.items():
            if val == max_degree:
                degrees.append(key)
        ans = sys.maxsize
        start = end = None
        for each in degrees:
            i = 0
            while True:
                if nums[i] == each:
                    start = i
                    break
                i += 1
            i = len(nums)-1
            while True:
                if nums[i] == each:
                    end = i
                    break
                i -= 1
            ans = min(ans, end-start+1)
        return ans
```

```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        left, right = {}, {}
        from collections import defaultdict
        count = defaultdict(int)
        for i in range(len(nums)):
            if nums[i] not in left:
                left[nums[i]] = i
            right[nums[i]] = i
            count[nums[i]] += 1
        
        max_degree = max(count.values())
        ans = sys.maxsize
        for key, value in count.items():
            if value == max_degree:
                ans = min(ans, right[key]-left[key]+1)
        return ans
```



### 845. Longest Mountain in Array

```python
class Solution:
    def longestMountain(self, A: List[int]) -> int:
        
        res = cur = 0
        
        while cur < len(A):
            temp = cur
            if temp < len(A)-1 and A[temp] < A[temp+1]:
                while temp < len(A)-1 and A[temp] < A[temp+1]:
                    temp += 1
                
                while temp < len(A)-1 and A[temp] > A[temp+1]:
                    temp += 1
                    res = max(res, temp-cur+1)
            cur = max(cur+1, temp)
        return res
```

https://leetcode.com/problems/longest-mountain-in-array/solution/



### 846. Hand of Straights

```python
class Solution:
    def isNStraightHand(self, hand: List[int], W: int) -> bool:
        counter = collections.Counter(hand)
        
        def del_counter(num):
            if counter[num] == 1:
                counter.pop(num)
                return
            counter[num] -= 1
        
        while counter:
            min_num = min(counter.keys())
            del_counter(min_num)
            for num in range(min_num+1, min_num+W):
                if num not in counter: return False
                del_counter(num)
        return True
```

same as 1296



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



### 912. Sort an Array

```python
# QuickSort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        
        def sort_method(i, j):
            oi = i
            pivot = nums[i]
            i += 1
            while True:
                while i < j and nums[i] < pivot:
                    i += 1
                while i <= j and nums[j] >= pivot:
                    j -= 1
                if i >= j: break
                nums[i], nums[j] = nums[j], nums[i]
            nums[oi], nums[j] = nums[j], nums[oi]
            return j
        
        def quick_sort(i, j):
            if i >= j: return
            
            k = random.randint(i, j)
            nums[i], nums[k] = nums[k], nums[i]
            mid = sort_method(i, j)
            quick_sort(i, mid)
            quick_sort(mid+1, j)
        
        quick_sort(0, len(nums)-1)
        return nums
```

```python
# QuickSort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def sort(i, j):
            pivot = nums[i]
            slow = i
            for fast in range(i+1, j+1):
                if nums[fast] < pivot:
                    slow += 1
                    nums[slow], nums[fast] = nums[fast], nums[slow]
            nums[i], nums[slow] = nums[slow], nums[i]
            return slow
        
        def partition(i, j):
            if i >= j: return
            mid = sort(i, j)
            partition(i, mid-1)
            partition(mid+1, j)
            
        
        partition(0, len(nums)-1)
        return nums
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



### 950. Reveal Cards In Increasing Order

```python
class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        deck.sort()
        index_q = collections.deque(range(len(deck)))
        res = [None] * len(deck)
        
        for card in deck:
            res[index_q.popleft()] = card
            if index_q:
                index_q.append(index_q.popleft())
        return res
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



### 1007. Minimum Domino Rotations For Equal Row

```python
class Solution:
    def minDominoRotations(self, A: List[int], B: List[int]) -> int:
        countA = [0] * 7
        countB = [0] * 7
        same = [0] * 7
        
        for i in range(len(A)):
            countA[A[i]] += 1
            countB[B[i]] += 1
            if A[i] == B[i]: same[A[i]] += 1
        
        for i in range(1, 7):
            if countA[i] + countB[i] - same[i] == len(A):
                return len(A) - max(countA[i], countB[i])
        return -1
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



### 1054. Distant Barcodes

```python
class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        res = [None] * len(barcodes)
        counter_temp = collections.Counter(barcodes)
        counter = sorted(counter_temp.items(), key=lambda x:(x[1], x[0]), reverse=True)
        index = 0
        for barcode, count in counter:
            for i in range(count):
                res[index] = barcode
                index += 2
                if index >= len(barcodes):
                    index = 1
        return res
```

注意Python字典的排序，另外Counter对象有most_common()方法，返回tuple列表



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



### 1217. Minimum Cost to Move Chips to The Same Position

```python
class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        odd = even = 0
        for each in position:
            if each % 2 == 0:
                even += 1
            else:
                odd += 1
        
        return min(odd, even)
```

我们可以用0cost将偶数位硬币移动到0，用0cost将奇数位硬币移动到1，之后将0移动到1或将1移动到0，选择较少的即可。



### 1295. Find Numbers with Even Number of Digits

```python
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        return len([each for each in nums if len(str(each)) % 2 == 0])
```



###  1296. Divide Array in Sets of K Consecutive Numbers

```python
class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        counter = collections.Counter(nums)
        
        def del_counter(num):
            if counter[num] == 1:
                counter.pop(num)
                return
            counter[num] -= 1
        
        while counter:
            min_num = min(counter.keys())
            del_counter(min_num)
            for num in range(min_num+1, min_num+k):
                if num not in counter: return False
                del_counter(num)
        return True
```

```python
class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        counter = collections.Counter(nums)
        
        for num in sorted(counter):
            if counter[num] > 0:
                for i in range(k)[::-1]:
                    counter[num+i] -= counter[num]
                    if counter[num+i] < 0:
                        return False
                    
        return True
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



### 1456. Maximum Number of Vowels in a Substring of Given Length

```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        v = {'a', 'e', 'i', 'o', 'u'}
        cur = len([i for i in range(k) if s[i] in v])
        res, left, right = cur, 0,  k - 1
        while right < len(s)-1:
            right += 1
            cur += s[right] in v
            cur -= s[left] in v
            left += 1
            res = max(res, cur)
        return res
```

```python
    def maxVowels(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        ans = cnt = 0
        for i, c in enumerate(s):
            if c in vowels:
                cnt += 1
            if i >= k and s[i - k] in vowels:
                cnt -= 1
            ans  = max(cnt, ans)
        return ans  
```

two pointers