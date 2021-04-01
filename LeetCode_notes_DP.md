## LeetCode - Dynamic programming

[toc]

### 55. Jump Game

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        last_good_pos = len(nums) - 1
        for i in range(len(nums)-1, -1, -1):
            if i + nums[i] >= last_good_pos:
                last_good_pos = i
        return last_good_pos == 0
```



### 62. Unique Paths

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        grid = []
        for i in range(m):
            grid.append([0]*n)
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    grid[i][j] = 1
                else:
                    grid[i][j] = grid[i-1][j] + grid[i][j-1]
        return grid[m-1][n-1]
```



### 64. Minimum Path Sum

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        
        for i in range(1, len(grid[0])):
            grid[0][i] += grid[0][i-1]
        for i in range(1, len(grid)):
            grid[i][0] += grid[i-1][0]
            
        for row in range(1, len(grid)):
            for col in range(1, len(grid[0])):
                grid[row][col] += min(grid[row-1][col], grid[row][col-1])
        return grid[-1][-1]
```



### 70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        cache = dict()
        
        def climbStairs_helper(i, n):
            if i in cache:
                return cache[i]
            
            if i > n:
                return 0
            if i == n:
                return 1
            temp = climbStairs_helper(i+1, n) + climbStairs_helper(i+2, n)
            cache[i] = temp
            return temp
        
        return climbStairs_helper(0, n)
```



### 198. House Robber

```python
# TLE
class Solution:
    def rob(self, nums: List[int]) -> int:
        
        def rob_helper(i):
            if i < 0:
                return 0
            return max(nums[i]+rob_helper(i-2), rob_helper(i-1))
        
        return rob_helper(len(nums)-1)
```

```python
# with memo
class Solution:
    def rob(self, nums: List[int]) -> int:
        
        rem = dict()
        
        def rob_helper(i):
            if i < 0:
                return 0
            if i in rem:
                return rem[i]
            rem_val = max(nums[i]+rob_helper(i-2), rob_helper(i-1))
            rem[i] = rem_val
            return rem_val
        
        return rob_helper(len(nums)-1)
```

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums: return 0
        memo = [None] * (len(nums)+1)
        memo[0], memo[1] = 0, nums[0]
        for i in range(1, len(nums)):
            memo[i+1] = max(memo[i], memo[i-1]+nums[i])
        return memo[-1]
```

https://leetcode.com/problems/house-robber/discuss/156523/From-good-to-great.-How-to-approach-most-of-DP-problems.



### 213. House Robber II

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        
        def rob_helper(nums):
            prev = curr = 0
            for num in nums:
                prev, curr = curr, max(curr, prev + num)
            return curr
        
        return max(nums[0] + rob_helper(nums[2:-1]), rob_helper(nums[1:]))
```



### 337. House Robber III

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        def helper(root, temp_dict):
            if not root:
                return 0
            if root in temp_dict:
                return temp_dict.get(root)
            
            val = 0
            
            if root.left:
                val += helper(root.left.left, temp_dict) + helper(root.left.right, temp_dict)
            if root.right:
                val += helper(root.right.left, temp_dict) + helper(root.right.right, temp_dict)
            
            val = max(root.val + val, helper(root.left, temp_dict) + helper(root.right, temp_dict))
            temp_dict[root] = val
            return val
        
        return helper(root, {})
```



### 714. Best Time to Buy and Sell Stock with Transaction Fee

```python
class Solution(object):
    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
        cash, hold = 0, -prices[0]
        
        for i in range(1, len(prices)):
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, cash - prices[i])
        return cash
```



### 1143. Longest Common Subsequence

```python
# longest common subsequence using recursion: LTE
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        def helper(str1, str2, index1, index2):
            if index1 < 0 or index2 < 0:
                return 0
            if str1[index1] == str2[index2]:
                return 1 + helper(str1, str2, index1-1, index2-1)
            else:
                return max(helper(str1, str2, index1-1, index2), helper(str1, str2, index1, index2-1))
        return helper(text1, text2, len(text1)-1, len(text2)-1)
```

```python
# longest common subsequence recursion with memorization or top-down approach: Accept
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        def helper(str1, str2, index1, index2, dp_dict):
            if index1 < 0 or index2 < 0:
                return 0
            if (index1, index2) in dp_dict:
                return dp_dict[(index1, index2)]
            if str1[index1] == str2[index2]:
                return 1 + helper(str1, str2, index1-1, index2-1, dp_dict)
            else:
                dp_dict[(index1, index2)] = max(helper(str1, str2, index1-1, index2, dp_dict), helper(str1, str2, index1, index2-1, dp_dict))
                return dp_dict[(index1, index2)]
        return helper(text1, text2, len(text1)-1, len(text2)-1, {})
```

```python
# longest common Subsequence using bottom-up approach using the 2-D array
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = []
        for i in range(len(text1)+1):
            dp.append([None] * (len(text2)+1))
                      
        for i in range(len(text1)+1):
            for j in range(len(text2)+1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[len(text1)][len(text2)]
```

https://leetcode.com/problems/longest-common-subsequence/discuss/398711/ALL-4-ways-Recursion-greater-Top-down-greaterBottom-Up-greater-Efficient-Solution-O(N)-including-VIDEO-TUTORIAL



### 1262. Greatest Sum Divisible by Three

```python
class Solution:
    def maxSumDivThree(self, nums: List[int]) -> int:
        dp = [0, 0, 0]
        for num in nums:
            dp_next = dp.copy()
            for each in dp:
                tempSum = each + num
                index = tempSum % 3
                dp_next[index] = max(tempSum, dp_next[index])
            dp = dp_next
        return dp[0]
```

