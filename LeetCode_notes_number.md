## LeetCode - Number

[toc]

### 1015. Smallest Integer Divisible by K

```python
class Solution:
    def smallestRepunitDivByK(self, k: int) -> int:
        rem = 1
        length = 1
        for i in range(k):
            if rem % k != 0:
                N = rem * 10 + 1
                rem = N % k
                length += 1
            else:
                return length
        return -1
```

https://leetcode.com/problems/smallest-integer-divisible-by-k/solution/



### 1291. Sequential Digits

```python
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        ans = []
        if low == high: return ans
        
        for i in range(1, 9):
            next = num = i
            while num <= high and next < 10:
                if num >= low: ans.append(num)
                next += 1
                num = num * 10 + next
        return sorted(ans)
```

