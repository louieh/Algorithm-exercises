## LeetCode - Number

[toc]

### 974. Subarray Sums Divisible by K

```python
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        temp_sum = temp_mod = 0
        res = 0

        mod_dict = defaultdict(int)
        mod_dict[0] = 1

        for num in nums:
            temp_sum += num
            temp_mod = temp_sum % k
            res += mod_dict[temp_mod]
            mod_dict[temp_mod] += 1
        
        return res
```

https://leetcode.com/problems/subarray-sums-divisible-by-k/solutions/217979/pictured-explanation-python-o-n-clean-solution-8-lines/

Running Sum[i]%K == Running Sum[j]%k that means we have sum(i,j) which is divisible by K.

从左开始依次累加，每累加到一个点时，计算当时的累加值%k的结果，当两个点的累加值%k的结果相同时，这两个点之间的数字和可以被k整除，证明：

A % k = x --> A = n1*k + x
B % k = x --> B = n2*k + x
(A-B) = n1*k + x - n2*k - x = (n1-n2)*k
It means (A-B) % k = 0

根据以上理论，我们记录每个累加值%k的结果个数即可。



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
