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



### 31. Next Permutation

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = None
        l = None
        for i in range(len(nums)-2, -1, -1):
            if nums[i] < nums[i+1]:
                k = i
                break
        if k is None:
            left, right = 0, len(nums)-1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        else:
            for i in range(len(nums)-1, -1, -1):
                if nums[i] > nums[k]:
                    l = i
                    break
            nums[k], nums[l] = nums[l], nums[k]
            nums[k+1:] = nums[k+1:][::-1]
```

According to [Wikipedia](https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order), a man named Narayana Pandita presented the following simple algorithm to solve this problem in the 14th century.

1. Find the largest index `k` such that `nums[k] < nums[k + 1]`. If no such index exists, just reverse `nums` and done.
2. Find the largest index `l > k` such that `nums[k] < nums[l]`.
3. Swap `nums[k]` and `nums[l]`.
4. Reverse the sub-array `nums[k + 1:]`.

![next-permutation-algorithm](https://www.nayuki.io/res/next-lexicographical-permutation-algorithm/next-permutation-algorithm.svg)



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



### 42. Trapping Rain Water

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        res = 0
        if not height: return res
        
        left_max = [0] * len(height)
        left_max[0] = height[0]
        for i in range(1, len(height)):
            left_max[i] = max(height[i], left_max[i-1])
        
        right_max = [0] * len(height)
        right_max[-1] = height[-1]
        for i in range(len(height)-2, -1, -1):
            right_max[i] = max(height[i], right_max[i+1])
        
        
        for i in range(1, len(height)-1):
            res += (min(left_max[i], right_max[i]) - height[i])
        
        return res
```

**Intuition**

In brute force, we iterate over the left and right parts again and again just to find the highest bar size upto that index. But, this could be stored. Voila, dynamic programming.

The concept is illustrated as shown:

![Screen Shot 2021-10-24 at 23.50.14](https://leetcode.com/problems/trapping-rain-water/Figures/42/trapping_rain_water.png)



**Algorithm**

- Find maximum height of bar from the left end upto an index i in the array left_max.

- Find maximum height of bar from the right end upto an index i in the array right_max.

- Iterate over the height

  array and update ans:

  - Add min(left_max[*i*],right_max[*i*])−height[*i*] to ans

similar question 238，1769



### 53. Maximum Subarray

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        local_min = -2147483648
        global_min = -2147483648
        
        for each in nums:
            local_min = max(each, each+local_min)
            global_min = max(local_min, global_min)
        
        return global_min
```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        global_min = nums[0]
        
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i]+nums[i-1])
            global_min = max(nums[i], global_min)
        return global_min
```



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



### 80. Remove Duplicates from Sorted Array II

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        counter = collections.Counter()
        res = 0
        for i, num in enumerate(nums):
            if counter[num] < 2:
                nums[res] = num
                res += 1
                counter[num] += 1
        return res
```



### 84. Largest Rectangle in Histogram

```python
# TLE
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        left_less_list = [0] * len(heights)
        right_less_list = [0] * len(heights)
        left_less_list[0] = -1
        right_less_list[-1] = len(heights)
        
        for i in range(1, len(heights)):
            p = i - 1
            while p >= 0 and heights[p] >= heights[i]:
                p -= 1 # 可优化 p = left_less_list[p]
            left_less_list[i] = p
        
        for i in range(len(heights)-2, -1, -1):
            p = i + 1
            while p <= len(heights) - 1 and heights[p] >= heights[i]:
                p += 1 # 可优化 p = right_less_list[p]
            right_less_list[i] = p
        
        res = 0
        
        for i in range(len(heights)):
            res = max(res, heights[i] * (right_less_list[i] - left_less_list[i] - 1))
        
        return res
```

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        left_less_list = [0] * len(heights)
        right_less_list = [0] * len(heights)
        left_less_list[0] = -1
        right_less_list[-1] = len(heights)
        
        for i in range(1, len(heights)):
            p = i - 1
            while p >= 0 and heights[p] >= heights[i]:
                p = left_less_list[p]
            left_less_list[i] = p
        
        for i in range(len(heights)-2, -1, -1):
            p = i + 1
            while p <= len(heights) - 1 and heights[p] >= heights[i]:
                p = right_less_list[p]
            right_less_list[i] = p
        
        res = 0
        
        for i in range(len(heights)):
            res = max(res, heights[i] * (right_less_list[i] - left_less_list[i] - 1))
        
        return res
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



### 126. Word Ladder II

```python
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        word_set = set(wordList)
        word_set.add(beginWord)
        distance = {beginWord: 0}
        node_neighbor = collections.defaultdict(list)
        solution = []
        res = []
        
        def bfs():
            q = collections.deque()
            q.append(beginWord)
            while q:
                count = len(q)
                cur = q.popleft()
                cur_distance = distance.get(cur)
                neighbors = get_neighbors(cur)
                for neighbor in neighbors:
                    node_neighbor[cur].append(neighbor)
                    if neighbor not in distance:
                        distance[neighbor] = cur_distance + 1
                        q.append(neighbor)
        
        # def bfs():
        #     q = collections.deque()
        #     q.append(beginWord)
        #     while q:
        #         count = len(q)
        #         found_end = False
        #         for i in range(count):
        #             cur = q.popleft()
        #             cur_distance = distance.get(cur)
        #             neighbors = get_neighbors(cur)
        #             for neighbor in neighbors:
        #                 node_neighbor[cur].append(neighbor)
        #                 if neighbor not in distance:
        #                     distance[neighbor] = cur_distance + 1
        #                     if neighbor == endWord:
        #                         found_end = True
        #                     else:
        #                         q.append(neighbor)
        #             if found_end:
        #                 break
        
        def get_neighbors(node):
            res = []
            node_list = list(node)
            for c in string.ascii_lowercase:
                for i in range(len(node)):
                    if node[i] == c: continue
                    old_c = node[i]
                    node_list[i] = c
                    if "".join(node_list) in word_set:
                        res.append("".join(node_list))
                    node_list[i] = old_c
            return res
        
        def dfs(cur):
            solution.append(cur)
            if cur == endWord:
                res.append(solution.copy())
            else:
                if node_neighbor.get(cur):
                    for neighbor in node_neighbor.get(cur):
                        if distance[neighbor] == distance[cur] + 1:
                            dfs(neighbor)
            solution.pop()
        
        bfs()
        dfs(beginWord)
        return res
```

https://leetcode.com/problems/word-ladder-ii/discuss/40475/My-concise-JAVA-solution-based-on-BFS-and-DFS



### 127. Word Ladder

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        word_set = set(wordList)
        word_set.add(beginWord)
        distance = {beginWord: 0}
        node_neighbor = collections.defaultdict(list)
        
        def bfs():
            q = collections.deque()
            q.append(beginWord)
            while q:
                count = len(q)
                cur = q.popleft()
                cur_distance = distance.get(cur)
                neighbors = get_neighbors(cur)
                for neighbor in neighbors:
                    node_neighbor[cur].append(neighbor)
                    if neighbor not in distance:
                        distance[neighbor] = cur_distance + 1
                        q.append(neighbor)
        
        def get_neighbors(node):
            res = []
            node_list = list(node)
            for c in string.ascii_lowercase:
                for i in range(len(node)):
                    if node[i] == c: continue
                    old_c = node[i]
                    node_list[i] = c
                    if "".join(node_list) in word_set:
                        res.append("".join(node_list))
                    node_list[i] = old_c
            return res
        
        bfs()
        return distance[endWord] + 1 if endWord in distance else 0
```

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        word_set = set(wordList)
        word_set.add(beginWord)
        distance = {beginWord: 0}
        node_neighbor = collections.defaultdict(list)
        
        def bfs():
            q = collections.deque()
            q.append(beginWord)
            while q:
                count = len(q)
                cur = q.popleft()
                cur_distance = distance.get(cur)
                neighbors = get_neighbors(cur)
                for neighbor in neighbors:
                    node_neighbor[cur].append(neighbor)
                    if neighbor not in distance:
                        distance[neighbor] = cur_distance + 1
                        if neighbor == endWord:
                            return distance[neighbor] + 1
                        q.append(neighbor)
        
        def get_neighbors(node):
            res = []
            node_list = list(node)
            for c in string.ascii_lowercase:
                for i in range(len(node)):
                    if node[i] == c: continue
                    old_c = node[i]
                    node_list[i] = c
                    if "".join(node_list) in word_set:
                        res.append("".join(node_list))
                    node_list[i] = old_c
            return res
        
        res = bfs()
        return res if res else 0
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

> @daxianji007
>
> I have thought for a long time and got two ideas:
>
> - If car starts at A and can not reach B. Any station between A and B
>   can not reach B.(B is the first station that A can not reach.) 当从A到达B时剩余小于0时，那么A和B之间的所有点都没法到达B，所以直接把起点设置为B的下一个点。
> - If the total number of gas is bigger than the total number of cost. There must be a solution. 如果总剩余小于那么没有答案。



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

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k %= len(nums)
        if k == 0: return
        nums1 = nums[-k:] + nums[:len(nums)-k]
        for i in range(len(nums)):
            nums[i] = nums1[i]
```



### 215. Kth Largest Element in an Array

```python
# ETL
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
# ETL
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

```python
# ETL
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:

        def partition(left, right):
            pivot = left - 1
            for i in range(left, right):
                if nums[i] > nums[right]:
                    pivot += 1
                    if pivot != i:
                        nums[pivot], nums[i] = nums[i], nums[pivot]
            pivot += 1
            nums[pivot], nums[right] = nums[right], nums[pivot]
            return pivot

        left, right = 0, len(nums) - 1
        pivot = partition(left, right)

        while pivot != k - 1:
            if pivot > k - 1:
                right = pivot - 1
            else:
                left = pivot + 1
            pivot = partition(left, right)
        
        return nums[pivot]
```

类似快排中的 partition，以 right 为分割点，将大于它的数字放到左边，小于它的数字放右边，其中 pivot 是中间分割点，初始化为 left - 1，每遇到一个大于的数字就将 pivot + 1，如果 pivot 不等于当前位置也就是 i，那么将 pivot 数字与 i 数字交换，因为有可能 i 是要比 pivot 移动的快的，pivot 只是大于 nums[right] 的分割点，最后将 pivot + 1 后与 nums[right] 交换，返回 pivot。

分割后看返回的 pivot 是不是等于 k - 1，也就是当前分割点是不是第 k 大，不是的话调整左右继续分割。



类似于上面快排的方法，但是每次构建新的list

```python
class Solution:
    def findKthLargest(self, nums, k):
        def quick_select(nums, k):
            pivot = random.choice(nums)
            left, mid, right = [], [], []

            for num in nums:
                if num > pivot:
                    left.append(num)
                elif num < pivot:
                    right.append(num)
                else:
                    mid.append(num)
            
            if k <= len(left):
                return quick_select(left, k)
            
            if len(left) + len(mid) < k:
                return quick_select(right, k - len(left) - len(mid))
            
            return pivot
        
        return quick_select(nums, k)
```



使用堆，保持小根堆大小是k

```python
class Solution:
    def findKthLargest(self, nums, k):
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        
        return heap[0]
```

```python
# 先全部插入堆中，然后pop出len(nums)-k个元素，让堆中只有k个元素
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
        
        for _ in range(len(heap) - k):
            heapq.heappop(heap)
        
        return heap[0]
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



### 238. Product of Array Except Self

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        left_accu = [1]
        right_accu = [1]
        
        for i in range(len(nums)-1):
            left_accu.append(nums[i] * left_accu[i])
        i = len(nums)-1
        while i > 0:
            right_accu.insert(0, nums[i] * right_accu[0])
            i -= 1
        
        ans = []
        for i in range(len(nums)):
            ans.append(left_accu[i] * right_accu[i])
        
        return ans
```

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        left_accu = [1]
        
        for i in range(len(nums)-1):
            left_accu.append(left_accu[i] * nums[i])
        
        ans = [1] * len(nums)
        temp = 1
        for i in range(len(nums)-1, -1, -1):
            ans[i] = temp * left_accu[i]
            temp *= nums[i]
        return ans
```

合并后两个for循环。

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        leftRes = rightRes = 1
        res = [1] * len(nums)
        
        for i in range(1, len(nums)):
            leftRes *= nums[i-1]
            res[i] = leftRes
        
        for i in range(len(nums)-2, -1, -1):
            rightRes *= nums[i+1]
            res[i] *= rightRes
        
        return res
```

same as question 1769, similar question 42



### 268. Missing Number

```python
class Solution:
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        if nums[-1] != len(nums):
            return len(nums)
        
        for i in range(len(nums)):
            if i != nums[i]:
                return i
```

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        sum_should_be = (1 + len(nums)) * len(nums) // 2
        
        return sum_should_be - sum(nums)
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



### 287. Find the Duplicate Number

```python
# 自认为很漂亮的方法，原理是因为有n+1个数且只有一个数字重复，并且每个数字的取值范围在[1, n], 所以只要把每个数字方法它该放的位置，也就是说把1放到0, 2放到1, 3放到2, 4放到3...最后一个数字便是重复的数字。当然在遍历过程中如果遇到在该位置上已经有了正确的数字直接返回该数字
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        if not nums:
            return
        i = 0
        while i <= len(nums)-2:
            if nums[i] == i+1:
                i += 1
                continue
            if nums[nums[i]-1] == nums[i]:
                return nums[i]
            else:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        return nums[len(nums)-1]
```

```python
# Floyd's Tortoise and Hare (Cycle Detection)
# same as Linked List Cycle II 
class Solution:
    def findDuplicate(self, nums):
        # Find the intersection point of the two runners.
        tortoise = nums[0]
        hare = nums[0]
        while True:
            tortoise = nums[tortoise]
            hare = nums[nums[hare]]
            if tortoise == hare:
                break
        
        # Find the "entrance" to the cycle.
        ptr1 = nums[0]
        ptr2 = tortoise
        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]
        
        return ptr1
```

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        for i, num in enumerate(nums):
            nums[abs(num)] *= -1
            if nums[abs(num)] > 0:
                return abs(num)
```



### 318. Maximum Product of Word Lengths

```python
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        set_list = [set(word) for word in words]
        len_list = [len(word) for word in words]
        
        res = 0
        
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                if not set_list[i] &  set_list[j]:
                    res = max(res, len_list[i] * len_list[j])
        
        return res
```

```java
	public static int maxProduct(String[] words) {
    if (words == null || words.length == 0)
      return 0;
    int len = words.length;
    int[] value = new int[len];
    for (int i = 0; i < len; i++) {
      String tmp = words[i];
      value[i] = 0;
      for (int j = 0; j < tmp.length(); j++) {
        value[i] |= 1 << (tmp.charAt(j) - 'a');
      }
    }
    int maxProduct = 0;
    for (int i = 0; i < len; i++)
      for (int j = i + 1; j < len; j++) {
        if ((value[i] & value[j]) == 0 && (words[i].length() * words[j].length() > maxProduct))
          maxProduct = words[i].length() * words[j].length();
      }
    return maxProduct;
}
```



### 341. Flatten Nested List Iterator

```python
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def isInteger(self) -> bool:
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        """
#
#    def getInteger(self) -> int:
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        """
#
#    def getList(self) -> [NestedInteger]:
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        """

class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self._list = []
        self.flatten(nestedList)
        self._index = 0
    
    def flatten(self, nestedList):
        for each in nestedList:
            if each.isInteger():
                self._list.append(each.getInteger())
            else:
                self.flatten(each.getList())
    
    def next(self) -> int:
        res = self._list[self._index]
        self._index += 1
        return res
    
    def hasNext(self) -> bool:
        return self._index <= len(self._list) - 1
         

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
```



### 344. Reverse String

```java
class Solution {
    public void reverseString(char[] s) {
        if (s.length == 0) return;
        int low = 0;
        int high = s.length-1;
        while (low < high){
            char temp = s[low];
            s[low] = s[high];
            s[high] = temp;
            low++;
            high--;
        }
    }
}
```

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def reverseString_helper(i1, i2):
            if i1 < i2:
                s[i1], s[i2] = s[i2], s[i1]
                reverseString_helper(i1+1, i2-1)
        
        reverseString_helper(0, len(s)-1)
```

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
```



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



### 413. Arithmetic Slices

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        @cache
        def count(n):
            if n < 3: return 0
            return n - 2 + count(n - 1)
        
        prev = None
        res = 0
        cur = 2
        
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i-1]
            if diff == prev:
                cur += 1
            else:
                res += count(cur)
                cur = 2
            prev = diff
        return res + count(cur)
```

给定数组长度计算子数组个数：count_of_subarrays = 1 + (len - 3) + count_of_subarrays(len-1)

https://leetcode.com/problems/arithmetic-slices/discuss/1814595/Python3-CACHE-()-Explained



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



### 443. String Compression

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        cur, count = 0, 1
        cur_char = chars[0]
        for i in range(1, len(chars)):
            if chars[i] == chars[i-1]:
                count += 1
            elif count == 1:
                chars[cur] = cur_char
                cur += 1
                cur_char = chars[i]
            else:
                count_str = str(count)
                chars[cur] = cur_char
                for each in count_str:
                    cur += 1
                    chars[cur] = each
                cur += 1
                cur_char = chars[i]
                count = 1
        chars[cur] = cur_char
        if count != 1:
            count_str = str(count)
            for each in count_str:
                cur += 1
                chars[cur] = each
        return cur + 1
```

```python
class Solution:
    def compress(self, chars: List[str]) -> int:
        # chars = ["a","a","a","b","b","a","a"]
        cur = 0
        i = 0
        while i < len(chars):
            count = 0
            curChar = chars[i]
            while i < len(chars) and chars[i] == curChar:
                i += 1
                count += 1
            chars[cur] = curChar
            cur += 1
            if count != 1:
                for each in str(count):
                    chars[cur] = each
                    cur += 1
        return cur
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



### 451. Sort Characters By Frequency

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        counter_list = sorted([(k, v) for k, v in collections.Counter(s).items()], key=lambda x: x[1], reverse=True)
        return "".join([k*v for k, v in counter_list])
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

我们要找s1<s3<s2，从后向前遍历插入栈，如果遇到元素(s2)大于栈顶元素则pop出栈顶元素作为s3，此时s2的index是小于s3的但其值是大于s3的(s3<s2)，那么下面如果遇到有元素小于s3我们便找到了s1，满足条件s1<s3<s2. 持续向前遍历会保证s3是最大的



### 472. Concatenated Words

```python
# TLE
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words_set = set(words)
        
        def dfs(word):
            for i in range(1, len(word)):
                prefix = word[:i]
                postfix = word[i:]
                if prefix in words_set and postfix in words_set:
                    return True
                if prefix in words_set and dfs(postfix):
                    return True
                if postfix in words_set and dfs(prefix):
                    return True
            return False
        
        res = []
        
        for word in words:
            if dfs(word):
                res.append(word)
        return res
```

```python
# TLE
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words_set = set(words)
        dp = {}
        def dfs(word):
            if word in dp:
                return dp[word]
            dp[word] = False
            for i in range(1, len(word)):
                prefix = word[:i]
                postfix = word[i:]
                if prefix in words_set and postfix in words_set:
                    dp[word] = True
                if prefix in words_set and dfs(postfix):
                    dp[word] = True
                if postfix in words_set and dfs(prefix):
                    dp[word] = True
            return dp[word]
        
        return [word for word in words if dfs(word)]
```

```python
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        trie = {}
        for word in words:
            root = trie
            for c in word:
                if c not in root:
                    root[c] = {}
                root = root[c]
            root["$"] = True

        @lru_cache(None)
        def find(word):
            if not word: return True
            root = trie
            res = False
            for i, c in enumerate(word):
                if c not in root:
                    break
                root = root[c]
                if "$" in root:
                    res |= find(word[i+1:])
                    if res:
                        break
            return res
        
        res = []
        for word in words:
            root = trie
            for i, c in enumerate(word):
                root = root[c]
                if "$" in root:
                    if i != len(word) - 1 and find(word[i+1:]):
                        res.append(word)
                        break
        return res
```

加lru_cache装饰器过了



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

遇0减一遇1加一，并将此数作为key，index作为值存入字典，如果遇到相同的数组说明有相等的0和1出现（因为从上一个点到当前点的count没有变说明这段距离中的0和1的数量一致），用当前index值键字典中相同数字的index得到长度。
如果遇到数字为0，说明从0到目前index为止的0和1数量相同



### 532. K-diff Pairs in an Array

```python
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
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



### 581. Shortest Unsorted Continuous Subarray

```python
# TLE
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        left, right = len(nums) - 1, 0
        
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] > nums[j]:
                    left = min(left, i)
                    right = max(right, j)
        return 0 if left >= right else right - left + 1
```

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        
        nums_sorted = sorted(nums)
        left = right = 0
        for i in range(len(nums)):
            if nums[i] != nums_sorted[i]:
                left = i
                break
        for i in range(len(nums)-1, -1, -1):
            if nums[i] != nums_sorted[i]:
                right = i
                break
        return right - left + 1 if right > left else 0
```

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        stack = []
        left, right = len(nums) - 1, 0
        
        for i in range(len(nums)):
            while stack and nums[stack[-1]] > nums[i]:
                left = min(left, stack.pop())
            stack.append(i)
        
        stack = []
        
        for i in range(len(nums)-1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                right = max(right, stack.pop())
            stack.append(i)
        
        return right - left + 1 if right > left else 0
```

https://leetcode.com/problems/shortest-unsorted-continuous-subarray/solution/



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



### 611. Valid Triangle Number

```python
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        res = 0
        nums = [num for num in nums if num != 0]
        nums.sort()
        
        for i in range(len(nums)-2):
            k = i + 2
            for j in range(i+1, len(nums)-1):
                while k < len(nums) and nums[i] + nums[j] > nums[k]:
                    k += 1
                res += k - j - 1
        return res
```



### 645. Set Mismatch

```python
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        res = []
        temp = [None]*len(nums)
        
        for num in nums:
            if temp[num-1] is not None:
                res.append(num)
            temp[num-1] = num
        for i, v in enumerate(temp):
            if v is None:
                res.append(i+1)
        return res
```

```python
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        dup = miss = -1
        
        for i, v in enumerate(nums):
            if nums[abs(v)-1] < 0:
                dup = abs(v)
            else:
                nums[abs(v)-1] *= -1

        for i in range(len(nums)):
            if nums[i] > 0:
                miss = i + 1
                break
        return [dup, miss]
```



### 674. Longest Continuous Increasing Subsequence

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        ans = temp = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                temp += 1
                ans = max(ans, temp)
            else:
                temp = 1
        return ans
```



### 680. Valid Palindrome II

```python
# 07/14/2019
class Solution:
    def validPalindrome(self, s: str) -> bool:
        # eabbade
        # edabbae

        pos = 0
        neg = 0

        for i in range(len(s)):
            if i > len(s)//2-1:
                return True
            if s[i+pos] != s[len(s)-1-i-neg]:
                if pos+neg !=0:
                    return False
                if s[i+1] == s[len(s)-1-i]:
                    if i+2 < len(s):
                        if s[i+2] == s[len(s)-1-i-1]:
                            pos = 1
                    else:
                        pos = 1
                elif s[i] == s[len(s)-1-i-1] and s[i+1] == s[len(s)-1-i-2]:
                    neg = 1
                else:
                    return False
```

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        
        def helper(left, right):
            while left < right:
                if s[left] != s[right]: return False
                left += 1
                right -= 1
            return True
        
        left, right = 0, len(s) - 1
        while left < right :
            if s[left] != s[right]:
                return helper(left+1, right) or helper(left, right-1)
            left += 1
            right -= 1
        return True
```

>[GeneBelcher](https://leetcode.com/GeneBelcher) Here is the solution I gave on my Meta phone screen and passed. Follow up question was asked to explain how I will extend this to work for n changes. My answer was to have a sub-function that compares left and right and calls recursively with a counter.
>
>```python
>class Solution(object):
>    def validPalindrome(self, s):
>        left = 0
>        right = len(s) - 1
>        
>        while left < right:
>            if s[left] != s[right]:
>                one = s[left:right]
>                two = s[left+1:right+1]
>                return one == one[::-1] or two == two[::-1]
>            left += 1
>            right -= 1
>            
>        return True
>```



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



### 769. Max Chunks To Make Sorted

```python
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        res = 0
        cur_max = 0
        for i, num in enumerate(arr):
            cur_max = max(cur_max, num)
            if cur_max == i:
                res += 1
        return res
```

因为数组元素是0-n-1的排列，所以如果某一个位置的index等于到该位置为止的最大值，则说明该位置之前是可以通过排序还原成原顺序的。



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



### 849. Maximize Distance to Closest Person

```python
class Solution:
    def maxDistToClosest(self, seats: List[int]) -> int:
        left_index = res = 0
        for right_index, seat in enumerate(seats):
            if seat == 1:
                if left_index == 0 and seats[0] == 0:
                    distance = right_index
                else:
                    distance = (right_index - left_index) // 2
                res = max(res, distance)
                left_index = right_index
        if seats[-1] == 0:
            res = max(res, len(seats) - 1 - left_index)
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

```python
# One Pass
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1
        
        while left < right:
            if nums[left] % 2 != 0:
                nums[left], nums[right] = nums[right], nums[left]
                right -= 1
            else:
                left += 1
        return nums
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



### 923. 3Sum With Multiplicity

```python
class Solution:
    def threeSumMulti(self, arr: List[int], target: int) -> int:
        arr.sort()
        cnt = Counter(arr)  # obtain the number of instances of each number
        res, i, l = 0, 0, len(arr)
        while i < l:  # in replacement of the for-loop, so that we can increment i by more than 1
            j, k = i, l-1  # j should be the leftmost index, hence j=i instead of j=i+1
            while j < k:  # i <= j < k; arr[i] <= arr[j] <= arr[k]
                if arr[i]+arr[j]+arr[k] < target:
                    j += cnt[arr[j]]
                elif arr[i]+arr[j]+arr[k] > target:
                    k -= cnt[arr[k]]
                else:  # arr[i]+arr[j]+arr[k] == target
                    if arr[i] != arr[j] != arr[k]:  # Case 1: All the numbers are different
                        res += cnt[arr[i]]*cnt[arr[j]]*cnt[arr[k]]
                    elif arr[i] == arr[j] != arr[k]:  # Case 2: The smaller two numbers are the same
                        res += cnt[arr[i]]*(cnt[arr[i]]-1)*cnt[arr[k]]//2  # math.comb(cnt[arr[i]], 2)*cnt[arr[k]]
                    elif arr[i] != arr[j] == arr[k]:  # Case 3: The larger two numbers are the same
                        res += cnt[arr[i]]*cnt[arr[j]]*(cnt[arr[j]]-1)//2  # math.comb(cnt[arr[j]], 2)*cnt[arr[i]]
                    else:  # Case 4: All the numbers are the same
                        res += cnt[arr[i]]*(cnt[arr[i]]-1)*(cnt[arr[i]]-2)//6  # math.comb(cnt[arr[i]], 3)
					# Shift pointers by the number of instances of the number
                    j += cnt[arr[j]]
                    k -= cnt[arr[k]]
            i += cnt[arr[i]]  # Shift pointer by the number of instances of the number
        return res%1000000007
```

https://leetcode.com/problems/3sum-with-multiplicity/discuss/1918718/Python-3Sum-Approach-with-Explanation

I initialised it to `i` because I wanted `j` to be the leftmost index of all the numbers in `arr` that are equal to `arr[j]`. This is because if I add `cnt[arr[j]]`, I am guaranteed that the new `j` is the leftmost index of all the numbers in `arr` that are equal to the new `arr[j]`. For example:

```
target = 6
arr: 1 1 1 2 2 3 3 4
     i
     j             k  --> valid, j += cnt[1].
           j     k
```

For cases where there is only one instance of `arr[i]`, i.e. not a valid tuple, the code will detect that `arr[i] == arr[j]` and hence use the formula `cnt[arr[i]] * (cnt[arr[i]]-1) // 2`, which equates to zero. Hence, that case will not be counted into the final result.

Of course, if this is unintuitive to you, you can initialise `j = i+1` and increment `j` based on conditionals: `j += cnt[arr[j]]-(j == i+1)`. Hope this helps you :)



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



### [962. Maximum Width Ramp](https://leetcode.com/problems/maximum-width-ramp/)

```python
# TLE
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        max_width = left = 0
        while (start := left + max_width + 1) < len(nums):
            for each in range(start, len(nums)):
                if nums[each] >= nums[left]:
                    max_width = max(max_width, each-left)
            left += 1
        return max_width
```

```python
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        index = [i for i in range(len(nums))]
        index.sort(key=lambda i: (nums[i], i))
        min_index = len(nums)
        max_width = 0
        for i in index:
            max_width = max(max_width, i-min_index)
            min_index = min(min_index, i)
        return max_width
```

将index按值从小到大排序，得到的index列表后面值比前面大这样只要找到后面index与前端index最大差值。

遍历这个index列表，记录最小的index值，同时将当前遍历到的index与迄今为止见到的最小的index做差，这个差值的最大值便是结果。

```python
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        
        max_width = 0
        
        stack = []
        for i in range(len(nums)):
            if not stack or nums[i] <= nums[stack[-1]]:
                stack.append(i)
        
        for j in range(len(nums)-1, -1, -1):
            if not stack: break
            while stack and nums[j] >= nums[stack[-1]]:
                max_width = max(max_width, j - stack[-1])
                stack.pop()
        
        return max_width
```

没有完全理解，大致意思是，构造一个单减stack，该列表中包含最小值。然后从原列表最后向前遍历，用当前元素大于等于stack最后元素那么计算它们index差值，然后pop出stack中最后元素，因为随着遍历进行，不可能有更大距离了，所以pop出去。

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

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        if nums[0] >= 0 or nums[-1] <= 0:
            res = [x**2 for x in nums]
            if nums[-1] <= 0:
                return res[::-1]
            return res
        
        left = right = 0
        for i, num in enumerate(nums):
            if num >= 0:
                left = i - 1
                right = i
                break
        res = []
        while left >= 0 and right < len(nums):
            if -nums[left] <= nums[right]:
                res.append(nums[left])
                left -= 1
            else:
                res.append(nums[right])
                right += 1
        while left >= 0:
            res.append(nums[left])
            left -= 1
        while right < len(nums):
            res.append(nums[right])
            right += 1
        return [x**2 for x in res]
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

```python
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        temp_dict = defaultdict(int)
        ans = 0
        for each in time:
            if each % 60 == 0:
                ans += temp_dict[0]
            else:
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



### 1200. Minimum Absolute Difference

```python
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        arr.sort()
        min_abs = 10**6
        for i in range(len(arr)-1):
            temp_abs = abs(arr[i] - arr[i+1])
            min_abs = min(min_abs, temp_abs)
        res = []
        for i in range(len(arr)-1):
            temp_abs = abs(arr[i] - arr[i+1])
            if temp_abs == min_abs:
                res.append([arr[i], arr[i+1]])
        return res
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



### 1239. Maximum Length of a Concatenated String with Unique Characters

```python
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        temp = [set()]
        for each in arr:
            each_set = set(each)
            if len(each_set) != len(each): continue
            for each1 in temp[:]:
                if each_set & each1: continue
                temp.append(each_set | each1)
        return max([len(each) for each in temp])
```



### 1282. Group the People Given the Group Size They Belong To

```python
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        res = []
        from collections import defaultdict
        temp = defaultdict(list)
        
        for i, n in enumerate(groupSizes):
            temp[n].append(i)
        
        for groupSize, nemberList in temp.items():
            if groupSize == len(nemberList):
                res.append(nemberList)
            else:
                for i in range(0, len(nemberList), groupSize):
                    res.append(nemberList[i:i+groupSize])
        return res
```



### [1287. Element Appearing More Than 25% In Sorted Array](https://leetcode.com/problems/element-appearing-more-than-25-in-sorted-array/)

```go
func findSpecialInteger(arr []int) int {
    length := len(arr)
    curr, currCount := arr[0], 1
    if length == 1 {
        return curr
    }
    for _, num := range arr[1:] {
        if num == curr {
            currCount += 1
        } else {
            curr = num
            currCount = 1
        }
        if currCount > length / 4 {
            return curr
        }
    }
    return -1
}
```



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



### 1306. Jump Game III

```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        if 0 <= start < len(arr) and arr[start] >= 0:
            arr[start] = -arr[start]
            return arr[start] == 0 or self.canReach(arr, start + arr[start]) or self.canReach(arr, start - arr[start])
        return False
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



### 1385. Find the Distance Value Between Two Arrays

```python
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        i = j = 0
        arr1.sort()
        arr2.sort()
        res = 0
        
        while i < len(arr1) and j < len(arr2):
            if arr1[i] > arr2[j]:
                if arr1[i] - arr2[j] > d:
                    j += 1
                else:
                    i += 1
            else:
                if arr2[j] - arr1[i] > d:
                    res += 1
                    i += 1
                else:
                    i += 1
        return res + (len(arr1) - i)
```

```
# Some remarks on how to interpret this algorithm.
#
# Each branch of the nested if-else statement will lead you to a single conclusion about your
# current configuration of pointers regarding two questions:
# 1. does the i-th element of arr1 sastisfies distance condition or not -- if not we drop i-th
# element, i.e. ignore augmenting distance counter and advance the pointer
# 2. is the j-th element of arr2 neccessary for comparisons with current or next elements of
# arr1 -- if not we advance the j pointer
#
# The concluding correction accounts for the tail of arr1 in the case when its values are greater
# than all of the arr2. I need it because my algorithm for the sake of simplicity and its
# correctness assumes that there will be always a concluding element of arr2 that is greater
# that any elmeent of arr1. You can see on the test sets it is not always the case, therefore is
# the correction.
```



### 1413. Minimum Value to Get Positive Step by Step Sum

```python
class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        accum = [nums[0]]
        for i in range(1, len(nums)):
            accum.append(accum[-1]+nums[i])
        _min = min(accum)
        return 1-_min if _min < 0 else 1
```



### 1423. Maximum Points You Can Obtain from Cards

```python
class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        front_sum = [0]
        behind_sum = [0]
        
        for each in cardPoints:
            front_sum.append(each + front_sum[-1])
        for each in cardPoints[::-1]:
            behind_sum.append(each + behind_sum[-1])
        
        all_combin = [front_sum[i] + behind_sum[k-i] for i in range(k+1)]
        
        return max(all_combin)
```

https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/discuss/597825/Simple-Clean-Intuitive-Explanation-with-Visualization



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



### 1502. Can Make Arithmetic Progression From Sequence

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        min_val, max_val = min(arr), max(arr)
        n = len(arr)
        if min_val == max_val: return True
        if (max_val - min_val) % (n - 1): return False # 最大最小值的差小于等于总数
        if len(arr) != len(set(arr)): return False

        diff = (max_val - min_val) // (n - 1)
        for each in arr:
            if (each - min_val) % diff: return False
        return True
```

如果一个序列相邻数字差是相同的，那么最大值最小值的差一定是相邻数字差的倍数，这个倍数是数字个数减一。例如：1，3，5，7，9，diff = (9 - 1) / 4 = 2

几个提前判断条件：

* 如果最大最小值一样，那么说明序列中所有数字一样，直接返回True
* 如果最大最小值差余(n-1)不为0，说明diff不能被(n-1)整除，则找不到整数diff，直接返回False
* 如果序列中有相同的数字直接返回False，如果第一条是False的话

下面遍历每个数字，判断每个数字与最小值的差是不是diff的倍数，不是则返回False

时间复杂度：O(n)

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        arr.sort()
        diff = arr[1] - arr[0]
        for i in range(2, len(arr)):
            if arr[i] - arr[i-1] != diff: return False
        return True
```

可以直接对序列排序，判断相邻数字差是否相同。

时间复杂度：O(nlogN)



### 1658. Minimum Operations to Reduce X to Zero

```python
# 没太懂这个方法为什么不行
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        left, right = 0, len(nums) - 1
        res = 0
        while x != 0:
            if left > right: return -1
            if nums[left] > x and nums[right] > x: return -1
            if nums[left] > x:
                x -= nums[right]
                right -= 1
            elif nums[right] > x:
                x -= nums[left]
                left += 1
            elif nums[left] >= nums[right]:
                x -= nums[left]
                left += 1
            else:
                x -= nums[right]
                right -= 1
            res += 1
            print(res)
        
        return res
```

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        target = sum(nums) - x
        if target == 0: return len(nums)
        res = -sys.maxsize
        cur_sum = 0
        sum_dict = {0: -1}
        for i, num in enumerate(nums):
            cur_sum += num
            if cur_sum - target in sum_dict:
                res = max(res, i - sum_dict[cur_sum-target])
            sum_dict[cur_sum] = i
        
        return -1 if res == -sys.maxsize else len(nums) - res
```

**Key Notes:**

- We could use dfs+memo or BFS, but they are too slow and will TLE (?)

- If it exists an answer, then it means we have **a subarray in the middle of original array whose sum is == totalSum - x**

- If we want to minimize our operations, then we should

  maximize the length of the middle subarray.

  - Then the qeustion becomes: *Find the Longest Subarray with Sum Equals to TotalSum - X*
  - We could simply use Map + Prefix Sum to get it!

![1](https://assets.leetcode.com/users/images/bf560734-2107-4a1b-811a-f3dd6d54c6e6_1605413025.6626496.png)

```java
int target = -x;
for (int num : nums) target += num;

if (target == 0) return nums.length;  // since all elements are positive, we have to take all of them

Map<Integer, Integer> map = new HashMap<>();
map.put(0, -1);
int sum = 0;
int res = Integer.MIN_VALUE;

for (int i = 0; i < nums.length; ++i) {

	sum += nums[i];
	if (map.containsKey(sum - target)) {
		res = Math.max(res, i - map.get(sum - target));
	}

    // no need to check containsKey since sum is unique
	map.put(sum, i);
}

return res == Integer.MIN_VALUE ? -1 : nums.length - res;
```



### 1710. Maximum Units on a Truck

```python
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        boxTypes.sort(key=lambda x: x[1], reverse=True)
        res = 0
        for num, units in boxTypes:
            if truckSize < num:
                res += units * truckSize
                break
            else:
                res += num * units
                truckSize -= num
        return res
```



### 1769. Minimum Number of Operations to Move All Balls to Each Box

```python
class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        tempSet = set()
        res = []
        
        for i, box in enumerate(boxes):
            if box == '1':
                tempSet.add(i)
        
        for i, box in enumerate(boxes):
            tempSum = 0
            for each1 in tempSet:
                if each1 == i: continue
                tempSum += abs(each1 - i)
            res.append(tempSum)
        
        return res
```

先把所有1的index存起来，之后遍历每个box，同时便利1的index，计算移动到这个box所需的步骤

```python
class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        
        res = [0] * len(boxes)
        
        leftCount = leftCost = rightCount = rightCost = 0
        
        # left:
        for i in range(1, len(boxes)):
            if boxes[i-1] == '1':
                leftCount += 1
            leftCost += leftCount # each step move to right, the cost increases by # of 1s on the left
            res[i] = leftCost
        # right
        for i in range(len(boxes)-2, -1, -1):
            if boxes[i+1] == '1':
                rightCount += 1
            rightCost += rightCount
            res[i] += rightCost
        return res
```

https://leetcode.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/discuss/1075895/Easy-Python-beats-100-time-and-space

Similar to **238. Product of Array Except Self** and **42. Trapping Rain Water**.
For each index, the cost to move all boxes to it is sum of the cost `leftCost` to move all left boxes to it, and the cost `rightCost` to move all right boxes to it.

- `leftCost` for all indexes can be calculted using a single pass from left to right.
- `rightCost` for all indexes can be calculted using a single pass from right to left.

same as question 238，similar question 42



### 1877. Minimize Maximum Pair Sum in Array

```python
class Solution:
    def minPairSum(self, nums: List[int]) -> int:
        nums.sort()
        left, right = 0, len(nums)-1
        res = 0
        while left < right:
            res = max(res, nums[left]+nums[right])
            left += 1
            right -= 1
        return res
```



### 2090. K Radius Subarray Averages

```python
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        if len(nums) < k * 2 + 1:
            return [-1] * len(nums)

        res = []

        cur_sum = None
        for i in range(k, len(nums)-k):
            if cur_sum is None:
                cur_sum = sum(nums[i-k:i+k+1])
            else:
                cur_sum -= nums[i-k-1]
                cur_sum += nums[i+k]
            res.append(cur_sum // (k * 2 + 1))

        return [-1] * k + res + [-1] * k
```



### 2279. Maximum Bags With Full Capacity of Rocks

```python
class Solution:
    def maximumBags(self, capacity: List[int], rocks: List[int], additionalRocks: int) -> int:
        remain_cap = [capacity[i] - rocks[i] for i in range(len(rocks))]
        remain_cap.sort()
        
        res = 0
        
        for each_remain in remain_cap:
            if additionalRocks > 0 and additionalRocks >= each_remain:
                res += 1
                additionalRocks -= each_remain
            else:
                break
        
        return res
```



### 2280. Minimum Lines to Represent a Line Chart

```python
class Solution:
    def minimumLines(self, stockPrices: List[List[int]]) -> int:
        if len(stockPrices) < 2:
            return 0
        
        stockPrices.sort(key=lambda x: (x[0], x[1]))
        
        fir, sec = stockPrices[0], stockPrices[1]
        cur_slope = (sec[1] - fir[1]) / (sec[0] - fir[0])
        cur_node = fir
        
        res = 1
        
        for i in range(2, len(stockPrices)):
            x, y = stockPrices[i]
            slope = (y - cur_node[1]) / (x - cur_node[0])
            if slope == cur_slope: continue
            else:
                res += 1
                cur_node = stockPrices[i-1]
                cur_slope = (y - cur_node[1]) / (x - cur_node[0])
        
        return res
```

```python
class Solution:
    
    def minimumLines(self, stockPrices: List[List[int]]) -> int:
        from fractions import Fraction
        if len(stockPrices) < 2:
            return 0
        
        stockPrices.sort(key=lambda x: (x[0], x[1]))
        
        fir, sec = stockPrices[0], stockPrices[1]
        cur_slope = Fraction(sec[1] - fir[1], sec[0] - fir[0])
        cur_node = fir
        
        res = 1
        
        for i in range(2, len(stockPrices)):
            x, y = stockPrices[i]
            slope = Fraction(y - cur_node[1], x - cur_node[0])
            if slope == cur_slope: continue
            else:
                res += 1
                cur_node = stockPrices[i-1]
                cur_slope = Fraction(y - cur_node[1], x - cur_node[0])
        
        return res
```

第一个方法`[[1,1],[500000000,499999999],[1000000000,999999998]]`这个case过不去，应该是精度的问题



### 2293. Min Max Game

```python
class Solution:
    def minMaxGame(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        
        while len(nums) > 1:
            temp = []
            for i, num in enumerate(nums):
                if i % 2 == 0:
                    if len(temp) % 2 == 0:
                        temp.append(min(nums[i], nums[i+1]))
                    else:
                        temp.append(max(nums[i], nums[i+1]))
            nums = temp
        return nums[0]
```



### 2294. Partition Array Such That Maximum Difference Is K

```python
class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        res = prev = 0
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] - nums[prev] <= k: continue
            res += 1
            prev = i
        return res + 1
```

原题目要求：A **subsequence** is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

可以对数组排序，直觉上是因为任何子数组都可以按原顺序被 pick 出来。



### 2296. Design a Text Editor

```python
class TextEditor:

    def __init__(self):
        self.text = ""
        self.nums = 0
        self.cursor = 0
    
    def print_status(self):
        print("self.test: ", self.text)
        print("self.nums: ", self.nums)
        print("self.cursor: ", self.cursor)
        print("-----------------------------")

    def addText(self, text: str) -> None:
        self.text = self.text[:self.cursor] + text + self.text[self.cursor:]
        self.cursor += len(text)
        self.nums += len(text)

    def deleteText(self, k: int) -> int:
        res = min(self.cursor, k)
        self.text = self.text[:self.cursor - res] + self.text[self.cursor:]
        self.nums -= res
        self.cursor -= res
        return res

    def cursorLeft(self, k: int) -> str:
        self.cursor -= min(self.cursor, k)
        return self.text[self.cursor-min(10, self.cursor):self.cursor]

    def cursorRight(self, k: int) -> str:
        self.cursor += min(self.nums-self.cursor, k)
        return self.text[self.cursor-min(10, self.cursor):self.cursor]


# Your TextEditor object will be instantiated and called as such:
# obj = TextEditor()
# obj.addText(text)
# param_2 = obj.deleteText(k)
# param_3 = obj.cursorLeft(k)
# param_4 = obj.cursorRight(k)
```

