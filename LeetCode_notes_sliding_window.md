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



### 239. Sliding Window Maximum

```python
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
                delete_index += 1
                continue
            delete_index += 1
            if nums[i] > max_num:
                max_num = nums[i]
            ans.append(max_num)
        return ans
```

```python
# deque
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
        # 0, 1, 2, 3, 4, 5
```



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

