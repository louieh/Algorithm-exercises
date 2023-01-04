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



### 2244. Minimum Rounds to Complete All Tasks

```python
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        counter = collections.Counter(tasks)
        
        res = 0

        for num in counter.values():
            if num == 1:
                return -1
            elif num % 3 == 0:
                res += num // 3
            elif num % 3 == 1:
                res += num // 3 - 1 + 2
            elif num % 3 == 2:
                res += num // 3 + 1
            elif num % 2 == 0:
                res += num // 2

        return res
```

这个题实际上是问一个数字怎么用2或3组成，3的个数尽量最大。我们知道除1之外所有自然数都可以用几个2和几个3表示，因为一个数字除以3的余数可能是1或2，如果余2的话那么加一个2即可，如果余1的话，可以减少一个3，把这个3补充到余下来的1组成4，也就是加两个2即可。

所以可以根据以上逻辑计算，先判断是否可以整除3，最后再判断是否可以整除2。

https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/solutions/1955622/java-c-python-sum-up-freq-2-3/

@lee215答案中把所有情况总结为 `(freq + 2) / 3` 不太能理解为什么可以这样
