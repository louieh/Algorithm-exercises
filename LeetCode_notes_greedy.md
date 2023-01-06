## LeetCode - Greedy

[toc]

### 11. Container With Most Water

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l = 0
        r = len(height) - 1
        ans = 0
        
        while l < r:
            ans = max(ans, (r - l) * min(height[l], height[r]))
            if height[l] > height[r]:
                r -= 1
            else:
                l += 1
        return ans
```

https://leetcode.com/problems/container-with-most-water/discuss/6099/Yet-another-way-to-see-what-happens-in-the-O(n)-algorithm



### 45. Jump Game II

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = 0
        curFarest = 0
        curPos = 0
        for i in range(len(nums)-1):
            curFarest = max(curFarest, i+nums[i])
            if i == curPos:
                jumps += 1
                curPos = curFarest
        return jumps
```

https://leetcode.com/problems/jump-game-ii/discuss/18014/Concise-O(n)-one-loop-JAVA-solution-based-on-Greedy

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) <= 1: return 0
        
        left, right = 0, nums[0]
        times = 1

        while right < len(nums) - 1:
            times += 1
            nxt = max(each + nums[each] for each in range(left, right + 1))
            left, right = right, nxt
        
        return times
```

The idea is to maintain two pointers `left` and `right`, where left initialy set to be `0` and `right` set to be `nums[0]`. So points between `0` and `nums[0]` are the ones you can reach by using just 1 jump.
Next, we want to find points I can reach using 2 jumps, so our new `left` will be set equal to `right`, and our new `right` will be set equal to the farest point we can reach by `two` jumps. which is:
`right = max(i + nums[i] for i in range(left, right + 1)`

第二个方法的逻辑是，先设置一个窗口，左边是0，右边是一步可达最大index也就是nums[0]. 下面我们要找第二个窗口位置，也就是，左边是一步可达最远的点nums[0]，右边是在第一步范围内可达最远的位置，也就是 `max(each + nums[each] for each in range(left, right + 1))`. 以此类推向前移动窗口，每移动一个窗口步数增加一，直到窗口覆盖最后一个点 `right > len(nums) - 1`。

第一个方法逻辑相同，不过寻找最右节点是在遍历过程中，而没有单独写循环。



### 55. Jump Game

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        last_good_pos = len(nums) - 1
        for i in range(len(nums)-2, -1, -1):
            if i + nums[i] >= last_good_pos:
                last_good_pos = i
        return last_good_pos == 0
```

先将最后一个index设置为目标点last_good_pos，然后从倒数第二个index向前遍历，过程中判断从当前点能否到达目标点，也就是 i + nums[i] >= last_good_pos，如果可以那么把当前点设置为新目标点。存在可能中间有的点无法到达目标点，那么目标点不变，过跳该点继续向前推进，判断下一个位置可否达到。



### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i in range(len(prices)-1):
            ans = max(ans, max(prices[i+1:])-prices[i])
        return ans
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        min_price = sys.maxsize
        for each in prices:
            if each < min_price:
                min_price = each
            elif each - min_price > ans:
                ans = each - min_price
        return ans
```

```python
# 从后向前遍历，过程中保存最大值，同时用之前的最大值减当前元素为最后结果
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        temp = prices[-1]
        for i in range(len(prices)-2, -1, -1):
            ans = max(ans, temp - prices[i])
            temp = max(temp, prices[i])
        return ans
```



### 122. Best Time to Buy and Sell Stock II

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = low = high = i = 0
        while i < len(prices) - 1:
            while i < len(prices) - 1 and prices[i] >= prices[i + 1]:
                i += 1
            low = prices[i]
            while i < len(prices) - 1 and prices[i] <= prices[i + 1]:
                i += 1
            high = prices[i]
            ans += (high - low)
        return ans
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                ans += prices[i] - prices[i-1]
        return ans
```

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/solutions/127712/best-time-to-buy-and-sell-stock-ii/?orderBy=most_votes



### 1833. Maximum Ice Cream Bars

```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        costs.sort()
        res, remain = 0, coins
        for each in costs:
            if each > remain: return res
            res += 1
            remain -= each
        return res
```

