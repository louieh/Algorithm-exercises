## LeetCode - Number

[toc]

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

