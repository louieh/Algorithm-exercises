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

