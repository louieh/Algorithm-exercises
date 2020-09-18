## LeetCode - Sliding Window

[toc]

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