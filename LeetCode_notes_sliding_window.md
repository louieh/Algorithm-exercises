## LeetCode - Sliding Window

[toc]

### 209. Minimum Size Subarray Sum

```python
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        for i in range(1, len(nums)+1): # 遍历长度 (1-6)
            for j in range(len(nums)):  # 遍历起始点 (0-5)
                if j + i - 1 >= len(nums):
                    break
                sum_temp = 0
                for k in range(i): # 从 j 开始遍历 i 长度
                    sum_temp += nums[k+j]
                if sum_temp >= s:
                    return i
        return 0
```

$O(n^3)$ 超时。将数组中各个位的累积和存在另一个数组中，这样时间复杂度缩小到 $O(n^2)$

```java
class Solution {
    public int minSubArrayLen(int s, int[] nums) {
        if (nums.length == 0)
            return 0;
        
        for(int i=0; i<nums.length; i++){
            if (nums[i] >= s)
                return 1;
        }
        
        int[] dest = nums.clone();
        
        for(int i=2; i<=nums.length; i++){ // 遍历长度
            for(int j=0; j<nums.length; j++){ // 遍历起始点
                if(j+i-1 >= nums.length)
                    break;
                if (dest[j] + nums[j+i-1] >= s)
                    return i;
                else{
                    dest[j] += nums[j+i-1];
                }
            }
        }
        return 0;
    }
}
```

Java $O(n^2)$ accepted

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        
        for each in nums:
            if each >= s:
                return 1

        nums_sum_temp = nums.copy()
        
        for i in range(2, len(nums)+1): # 遍历长度
            for j in range(len(nums)):  # 遍历起始点
                if j + i - 1 >= len(nums):
                    break
                if nums_sum_temp[j] + nums[j+i-1] >= s:
                    return i
                else:
                    nums_sum_temp[j] += nums[j+i-1]
        return 0
```

Python $O(n^2)$ 超时

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        sums = 0
        left = 0
        ans = len(nums) + 1
        for i in range(len(nums)):
            sums += nums[i]
            while (sums >= s):
                ans = min(ans, i + 1 - left)
                sums -= nums[left]
                left += 1
        if ans < len(nums) + 1:
            return ans
        else:
            return 0
```

$O(n)$ 先一直向前加，加到和>=s后再从左边开始向右减，然后再从刚刚加到的位置开始。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:

        ans = sys.maxsize

        left = right = 0
        _sum = nums[0]

        while left <= right:
            if _sum >= target:
                if right - left + 1 == 1:
                    return 1
                ans = min(ans, right - left + 1)
                _sum -= nums[left]
                left += 1
            else:
                right += 1
                if right == len(nums): break
                _sum += nums[right]
        
        return 0 if ans == sys.maxsize else ans
```

与上面的方法相同，维护一个窗口，当窗口中元素和大于等于target的时候记录长度然后将左边往前移动一个，当窗口中元素小于target的时候将窗口右边往前移动一个，过程中有两个跳出条件：1. 当窗口元素和大于等于target且长度为1时直接返回1，因为没可能再小了，2. 当窗口和小于target且右边到达边界直接跳出，因为当前窗口已经小于target且无法再增大。

该算法时间最差时间复杂度应该是每个元素遍历2次则2N -> O(2n)

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        
        left = _sum = 0
        ans = sys.maxsize

        for right, num in enumerate(nums):
            
            _sum += num

            while _sum >= target:
                ans = min(ans, right - left + 1)
                _sum -= nums[left]
                left += 1
        
        return ans if ans != sys.maxsize else 0

```

逻辑与代码结构同2024，424

均是外循环为右边界，循环内使用while或if判断当前窗口是否valid，从而控制左边界，根据题目选择在while或if代码块中更新ans还是在外更新，当前题目是在while循环中，2024与424是在外更新。



### 239. Sliding Window Maximum

```python
# TLE
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        
        max_num = max(nums[:k])
        ans = [max_num]

        delete_index = 0
        for i in range(k, len(nums)):
            if nums[delete_index] == max_num:
                max_num = max(nums[delete_index+1:delete_index+1+k])
                ans.append(max_num)
                continue
            delete_index += 1
            if nums[i] > max_num:
                max_num = nums[i]
            ans.append(max_num)
        return ans
```

```python
# TLE
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:

        return [max(nums[i: i+k]) for i in range(len(nums) - k + 1)]
```

```python
# Monotonic Deque
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        res = []

        for i in range(k):
            while dq and nums[i] > nums[dq[-1]]:
                dq.pop()
            dq.append(i)
        res.append(nums[dq[0]])

        for i in range(k, len(nums)):
            if dq[0] == i - k: # 保持窗口大小为 k
                dq.popleft()
            while dq and nums[i] > nums[dq[-1]]:
                dq.pop()
            dq.append(i)
            res.append(nums[dq[0]])
        
        return res
```

```python
# Monotonic Deque
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import deque
        dq = deque()
        res = []
        
        for i,val in enumerate(nums):
            while dq and dq[0] < i - k + 1:
                dq.popleft()
            while dq and nums[dq[-1]] < val:
                dq.pop()
            dq.append(i)
            if i >= k-1:
                res.append(nums[dq[0]])
        return res         
```

https://leetcode.com/problems/sliding-window-maximum/editorial/

维护一个长度为 k 的单调递减的队列，先将前三个元素插入队列插入的同时维护其单调递减，最后队列中第一个元素是当前窗口的最大值。

之后从 index=k 开始向后遍历，首先看队列中第一个元素是不是等于 i - k，其目的是维持窗口大小为 k，之后写入一个新元素，同时维护单调递减性，写入后当前队列第一元素即为当前窗口最大值。

其实可以不用把第一个窗口单独拿出来，下面方法是合并在一起的。

#### Complexity Analysis

Here *n* is the size of `nums`.

- Time complexity: *O*(*n*).
  - At first glance, it may look like the time complexity of this algorithm should be O(n2)O(n^2)*O*(*n*2), because there is a nested while loop inside the for loop. However, each element can only be added to the deque once, which means the deque is limited to nn*n* pushes. Every iteration of the while loop uses `1` pop, which means the while loop will not iterate more than nn*n* times in total, across all iterations of the for loop.
  - An easier way to think about this is that in the worst case, every element will be pushed and popped once. This gives a time complexity of O(n)*O*(2⋅*n*)=*O*(*n*).
- Space complexity: O*(*k).
  - The size of the deque can grow a maximum up to a size of *k*.

这个方法时间复杂度是 O(n)，有点不是太理解，当前的理解是 while 循环中操作是 O(1) 的复杂度可忽略不计。



### 424. Longest Repeating Character Replacement

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        start = maxSame = ans = 0
        count = dict()
        
        for end in range(len(s)):
            count[s[end]] = count.get(s[end], 0) + 1
            # maxSame = max(maxSame, count[s[end]]) 没太懂这个为什么work
            maxSame = max(count.values()) # 感觉这个更 make sence 一点
            if end - start + 1 - maxSame > k:
                count[s[start]] -= 1
                start += 1
            ans = max(ans, end-start+1)
        return ans
```

**`maxCount` may be invalid at some points, but this doesn't matter, because it was valid earlier in the string, and all that matters is finding the max window that occurred \*anywhere\* in the string**. Additionally, it will expand ***if and only if\*** enough repeating characters appear in the window to make it expand. So whenever it expands, it's a valid expansion.

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:

        counter = collections.Counter()

        ans = left = 0

        for right, c in enumerate(s):
            counter[c] += 1
            if right - left + 1 - max(counter.values()) > k:
                counter[s[left]] -= 1
                left += 1
            else:
                ans = max(ans, right - left + 1)

        return ans 
```

这里窗口不valid的条件是字符串转换的次数大于k，也就是将字符串中所有字母都转变为出现次数最多的那个字母的次数大于k，也就是窗口长度减出现最多的字母次数大于k

逻辑与代码结构同209，2024



### 438. Find All Anagrams in a String

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        from collections import defaultdict
        
        res = []
        
        if len(p) > len(s):
            return res
        
        def to_dict(s: str):
            res = defaultdict(int)
            for each in s:
                res[each] += 1
            return res
        
        if len(p) == len(s):
            return [0] if to_dict(p) == to_dict(s) else res
        
        left, right = 0, len(p)-1
        p_dict = to_dict(p)
        s_dict = defaultdict(int)
        
        while right < len(s)-1:
            if not s_dict:
                s_dict = to_dict(s[left:right+1])
            else:
                s_dict[s[left]] -= 1
                if s_dict[s[left]] == 0:
                    s_dict.pop(s[left])
                left += 1
                right += 1
                s_dict[s[right]] += 1
            if s_dict == p_dict:
                res.append(left)
        return res
```

same as question 567

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        res = []
        
        if len(p) > len(s):
            return res
        from collections import Counter
        if len(p) == len(s):
            return [0] if Counter(s) == Counter(p) else res
        p_dict = Counter(p)
        s_dict = None
        left, right = 0, len(p) - 1
        while right < len(s) - 1:
            if s_dict is None:
                s_dict = Counter(s[left: right+1])
            else:
                s_dict[s[left]] -= 1
                if s_dict[s[left]] == 0:
                    s_dict.pop(s[left])
                left += 1
                right += 1
                s_dict[s[right]] += 1
            if s_dict == p_dict:
                res.append(left)
        
        return res
```



### 567. Permutation in String

```python
# TLE
# 直接算s1的所有全排列然后依次去查是否在s2中，超时
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        permutation = []
        s1_list = list(s1)
        s1_list.sort()
        
        def backtrack(tempList, used):
            if len(tempList) == len(s1):
                permutation.append("".join(tempList))
            else:
                for i, num in enumerate(s1_list):
                    if used[i] or i > 0 and s1_list[i] == s1_list[i-1] and not used[i-1]:
                        continue
                    used[i] = True
                    tempList.append(num)
                    backtrack(tempList.copy(), used.copy())
                    tempList.pop()
                    used[i] = False
        backtrack([], [False]*len(s1))
        
        for each in permutation:
            if each in s2:
                return True
        return False
```

```java
// TLE
// 排序
// The idea behind this approach is that one string will be a permutation of another string only if both of them contain the same characters the same number of times. One string xx is a permutation of other string yy only if sorted(x)=sorted(y)sorted(x)=sorted(y).

public class Solution {
    public boolean checkInclusion(String s1, String s2) {
        s1 = sort(s1);
        for (int i = 0; i <= s2.length() - s1.length(); i++) {
            if (s1.equals(sort(s2.substring(i, i + s1.length()))))
                return true;
        }
        return false;
    }
    
    public String sort(String s) {
        char[] t = s.toCharArray();
        Arrays.sort(t);
        return new String(t);
    }
}
```

```java
// TLE
// hashmap
// 基本思想就是设置一个与s1一样长的窗口在s2上滑动，同时比较s1的字典和s2的字典是否一致
public class Solution {
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length())
            return false;
        HashMap < Character, Integer > s1map = new HashMap<> ();
        
        for (int i = 0; i < s1.length(); i++)
            s1map.put(s1.charAt(i), s1map.getOrDefault(s1.charAt(i), 0) + 1);
        
        for (int i = 0; i <= s2.length() - s1.length(); i++) {
            HashMap <Character, Integer> s2map = new HashMap<> ();
            for (int j = 0; j < s1.length(); j++) {
                s2map.put(s2.charAt(i + j), s2map.getOrDefault(s2.charAt(i + j), 0) + 1);
            }
            if (matches(s1map, s2map))
                return true;
        }
        return false;
    }
    
    public boolean matches(HashMap <Character, Integer> s1map, HashMap <Character, Integer> s2map) {
        for (char key: s1map.keySet()) {
            if (s1map.get(key) - s2map.getOrDefault(key, -1) != 0)
                return false;
        }
        return true;
    }
```

```python
# 还是上面的hashmap方法Python通过
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        
        if len(s1) > len(s2):
            return False
        
        from collections import defaultdict
        
        def to_dict(s: str):
            res = defaultdict(int)
            for each in s:
                res[each] += 1
            return res
        
        s1_dict = to_dict(s1)
        left, right = 0, len(s1)
        while right <= len(s2):
            s2_dict = to_dict(s2[left:right])
            if s2_dict == s1_dict:
                return True
            left += 1
            right += 1
        return False
```

```python
# 还是上面的hashmap方法，每次移动s2窗口的时候不重新构造字典，而是只对左右两端字母进行更新
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        
        if len(s1) > len(s2):
            return False
        
        from collections import defaultdict
        
        def to_dict(s: str):
            res = defaultdict(int)
            for each in s:
                res[each] += 1
            return res
        
        if len(s1) == len(s2):
            return to_dict(s1) == to_dict(s2)
        
        s1_dict = to_dict(s1)
        s2_dict = defaultdict(int)
        
        left, right = 0, len(s1)-1
        while right < len(s2)-1:
            if not s2_dict:
                s2_dict = to_dict(s2[left:right+1])
            else:
                s2_dict[s2[left]] -= 1
                if s2_dict[s2[left]] == 0:
                    s2_dict.pop(s2[left])
                left += 1
                right += 1
                s2_dict[s2[right]] += 1
            if s2_dict == s1_dict:
                return True
        return False
```

same as question 438



### 713. Subarray Product Less Than K

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1: return 0
        left = res = 0
        prod = 1
        for right, val in enumerate(nums):
            prod *= val
            while prod >= k:
                prod //= nums[left]
                left += 1
            res += right - left + 1
        return res
```



### 992. Subarrays with K Different Integers

```python
class Solution:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        temp = dict()
        count = res = start = 0
        for i, val in enumerate(A):
            if val not in temp:
                temp[val] = 1
                count = 0
            else:
                temp[val] += 1
            while len(temp) == K:
                temp[A[start]] -= 1
                if temp[A[start]] == 0:
                    temp.pop(A[start])
                start += 1
                count += 1
            res += count
        return res
```

Try to use the thought of 1248 but it not work, something wrong.

```python
class Solution:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        
        def atMost(K):
            temp = dict()
            res = start = 0
            for i, val in enumerate(A):
                if val not in temp:
                    temp[val] = 1
                    K -= 1
                else:
                    temp[val] += 1
                while K < 0:
                    temp[A[start]] -= 1
                    if temp[A[start]] == 0:
                        temp.pop(A[start])
                        K += 1 
                    start += 1
                res += i - start + 1
            return res
        return atMost(K) - atMost(K-1)
```



### 1004. Max Consecutive Ones III

```python
class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        
        zero_num = start = ans = 0
        for i, val in enumerate(A):
            if val == 0:
                zero_num += 1
            while zero_num > K:
                if A[start] == 0:
                    zero_num -= 1
                start += 1
            ans = max(ans, i-start+1)
        return ans
```

```python
    def longestOnes(self, A, K):
        i = 0
        for j in xrange(len(A)):
            K -= 1 - A[j]
            if K < 0:
                K += 1 - A[i]
                i += 1
        return j - i + 1
```

还没有完全懂这个写法，目前理解是当 i到j 达到最大宽度后，当j继续向前移动时如果遇到超出0个数的A[j] 那么 i 也同时向前移动，也就是说保持这个宽度不变向前移动，最后得到的i，j并不是准确的位置但是i，j的宽度是最大的。

https://leetcode.com/problems/max-consecutive-ones-iii/discuss/247564/JavaC%2B%2BPython-Sliding-Window



### 1248. Count Number of Nice Subarrays

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        ans = count = start = 0
        for i, val in enumerate(nums):
            if val & 1:
                k -= 1
                count = 0
            while k == 0:
                k += nums[start] & 1
                start += 1
                count += 1
            ans += count
        return ans
```

窗口右边向右移动同时k减去遇到的奇数个数，当k==0时也就是窗口中有k个奇数时，左边开始向右移动，移动一个代表一个subarray，ans+1同时k加上遇到的奇数个数，当k!=0时停止增长ans，因为此时窗口中奇数个数小于k，窗口右边继续向右移动，重复上面步骤。

```python
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        
        def atMost(k):
            ans = count = start = 0
            for i, val in enumerate(nums):
                if val & 1:
                    k -= 1
                while k < 0:
                    k += nums[start] & 1
                    start += 1
                ans += i - start + 1
            return ans
        return atMost(k) - atMost(k-1)
```



### 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

```python
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        from collections import deque
        maxq = deque()
        minq = deque()
        i = 0
        for num in nums:
            while maxq and maxq[-1] < num:
                maxq.pop()
            while minq and minq[-1] > num:
                minq.pop()
            
            maxq.append(num)
            minq.append(num)
            
            if maxq[0] - minq[0] > limit:
                if nums[i] == maxq[0]: maxq.popleft()
                if nums[i] == minq[0]: minq.popleft()
                i += 1
        return len(nums) - i
```



### 1695. Maximum Erasure Value

```python
# TLE same idea as problem 3
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        temp_dict = {}
        temp_sum = start = 0
        for i, num in enumerate(nums):
            if num in temp_dict and temp_dict[num] >= start:
                temp_sum = max(temp_sum, sum(nums[start:i]))
                start = temp_dict[num] + 1
            temp_dict[num] = i
        
        return max(temp_sum, sum(nums[start:]))
```

```python
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        left = right = 0
        res = 0
        temp_set = set()
        cur_sum = 0
        
        while right < len(nums):
            if nums[right] not in temp_set:
                cur_sum += nums[right]
                temp_set.add(nums[right])
                right += 1
                res = max(res, cur_sum)
            else:
                cur_sum -= nums[left]
                temp_set.remove(nums[left])
                left += 1
        return res
```



### 2024. Maximize the Confusion of an Exam

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:

        counter = collections.Counter()

        ans = left = 0

        for right, c in enumerate(s):
            counter[c] += 1
            if right - left + 1 - max(counter.values()) > k:
                counter[s[left]] -= 1
                left += 1
            else:
                ans = max(ans, right - left + 1)

        return ans 
```

这里重要的点是窗口是否valid条件也就是左边界调整条件是：当窗口中T和F的数量均大于k

算法大致是：初始化一个计数器，右边每向右移动一格便向计数器中增加该元素，之后判断窗口是否valid，不是的话将左边向右移动一格，相应的将对应元素从计数器中数量减少1，否则更新ans。

逻辑与代码结构同209，424

重要点均是判断窗口是否满足条件，不满足则将右边界右移，不同的点是这里窗口不valid的时候不用while循环右移左边界，因为这里只有两个元素，左边界右移一次窗口立即变valid。
