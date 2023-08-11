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

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        
        rows, cols = m, n
        dp = [[None] * cols for _ in range(rows)]
        
        for row in range(rows):
            for col in range(cols):
                if not row or not col:
                    dp[row][col] = 1
                else:
                    dp[row][col] = dp[row-1][col] + dp[row][col-1]
        
        return dp[rows-1][cols-1]
```



### 63. Unique Paths II

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        rows, cols = len(obstacleGrid), len(obstacleGrid[0])
        if obstacleGrid[0][0] or obstacleGrid[rows-1][cols-1]: return 0
        
        dp = []
        for _ in range(rows):
            dp.append([None] * cols)
        dp[0][0] = 1
        
        for i in range(1, cols):
            dp[0][i] = dp[0][i-1] if obstacleGrid[0][i] == 0 else 0
        
        for i in range(1, rows):
            dp[i][0] = dp[i-1][0] if obstacleGrid[i][0] == 0 else 0
        
        for row in range(1, rows):
            for col in range(1, cols):
                dp[row][col] = dp[row-1][col] + dp[row][col-1] if obstacleGrid[row][col] == 0 else 0
        
        return dp[rows-1][cols-1]
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
# 自顶向下
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

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 1: return 1
    		return self.climbStairs(n - 1) + self.climbStairs(n - 2) # 因为既可以从n-1层爬上来也可以从n-2层爬上来  
```

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        ans = [0] * (n+1)
        ans[1] = 1
        ans[2] = 2
        for i in range(3, len(ans)):
            ans[i] = ans[i-1] + ans[i-2]
        return ans[n]
```

```python
# 自底向上
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        a, b = 1, 2
        for _ in range(n - 2):
            a, b = b, a + b
        return b
```



### 91. Decode Ways

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        dp = [0] * (len(s) + 1)
        dp[0] = 1
        dp[1] = 0 if s[0] == "0" else 1
        
        for i in range(2, len(s)+1):
            if 0 < int(s[i-1:i]) <= 9:
                dp[i] += dp[i-1]
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
        
        return dp[len(s)]
```



### 96. Unique Binary Search Trees

```python
class Solution:
    def numTrees(self, n: int) -> int:
        from collections import defaultdict
        mem = defaultdict(int)
        def helper(n):
            if n <= 1: return 1
            if n in mem:
                return mem[n]
            for i in range(1, n+1):
                mem[n] += helper(i-1) * helper(n-i)
            return mem[n]
        return helper(n)
```

https://leetcode.com/problems/unique-binary-search-trees/discuss/1565543/C%2B%2BPython-5-Easy-Solutions-w-Explanation-or-Optimization-from-Brute-Force-to-DP-to-Catalan-O(N)



### 120. Triangle

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = triangle[-1]
        
        for layer in range(len(triangle)-2, -1, -1):
            for i in range(layer+1):
                dp[i] = triangle[layer][i] + min(dp[i], dp[i+1])
        
        return dp[0]
```

This problem is quite well-formed in my opinion. The triangle has a tree-like structure, which would lead people to think about traversal algorithms such as DFS. However, if you look closely, you would notice that the adjacent nodes always share a 'branch'. In other word, there are **overlapping subproblems**. Also, suppose x and y are 'children' of k. Once minimum paths from x and y to the bottom are known, the minimum path starting from k can be decided in O(1), that is **optimal substructure**. Therefore, dynamic programming would be the best solution to this problem in terms of time complexity.

What I like about this problem even more is that the difference between 'top-down' and 'bottom-up' DP can be 'literally' pictured in the input triangle. For 'top-down' DP, starting from the node on the very top, we recursively find the minimum path sum of each node. When a path sum is calculated, we store it in an array (memoization); the next time we need to calculate the path sum of the same node, just retrieve it from the array. However, you will need a cache that is at least the same size as the input triangle itself to store the pathsum, which takes O(N^2) space. With some clever thinking, it might be possible to release some of the memory that will never be used after a particular point, but the order of the nodes being processed is not straightforwardly seen in a recursive solution, so deciding which part of the cache to discard can be a hard job.

'Bottom-up' DP, on the other hand, is very straightforward: we start from the nodes on the bottom row; the min pathsums for these nodes are the values of the nodes themselves. From there, the min pathsum at the ith node on the kth row would be the lesser of the pathsums of its two children plus the value of itself, i.e.:

```
minpath[k][i] = min( minpath[k+1][i], minpath[k+1][i+1]) + triangle[k][i];
```

Or even better, since the row minpath[k+1] would be useless after minpath[k] is computed, we can simply set minpath as a 1D array, and iteratively update itself:

```
For the kth level:
minpath[i] = min( minpath[i], minpath[i+1]) + triangle[k][i]; 
```

Thus, we have the following solution

```c++
int minimumTotal(vector<vector<int> > &triangle) {
    int n = triangle.size();
    vector<int> minlen(triangle.back());
    for (int layer = n-2; layer >= 0; layer--) // For each layer
    {
        for (int i = 0; i <= layer; i++) // Check its every 'node'
        {
            // Find the lesser of its two children, and sum the current value in the triangle with it.
            minlen[i] = min(minlen[i], minlen[i+1]) + triangle[layer][i]; 
        }
    }
    return minlen[0];
}
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



### 300. Longest Increasing Subsequence

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        dp = [1] * len(nums)
        
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j] and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
        
        return max(dp)
```

https://leetcode.com/problems/longest-increasing-subsequence/discuss/1326308/C%2B%2BPython-DP-Binary-Search-BIT-Solutions-Picture-explain-O(NlogN)

对于dp方法，首先定义一个长度为n的一维数组，并且填充1，dp[i]表示截止到 i 的最大升序子串长度。

两层循环，外循环从0到n-1遍历，内循环从0到i-1，每向前移动一个i，就从0开始比较到i，看nums[i] 是否大于 nums[j]，如果大于看dp[i] 是否小于 dp[j] + 1，如果是则更新dp[i]，最后返回dp最大的

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        res = []
        
        for num in nums:
            if not res or res[-1] < num:
                res.append(num)
            else:
                index = bisect_left(res, num)
                res[index] = num
        return len(res)
```

**Solution 2: Greedy with Binary Search**

- Let's construct the idea from following example.
- Consider the example nums = [2, 6, 8, 3, 4, 5, 1], let's try to build the increasing subsequences starting with an empty one: sub1 = []
  1. Let pick the first element, `sub1 = [2]`.
  2. `6` is greater than previous number, `sub1 = [2, 6]`
  3. `8` is greater than previous number, `sub1 = [2, 6, 8]`
  4. `3` is less than previous number, we can't extend the subsequence `sub1`, but we must keep `3` because in the future there may have the longest subsequence start with `[2, 3]`, `sub1 = [2, 6, 8], sub2 = [2, 3]`.
  5. With `4`, we can't extend `sub1`, but we can extend `sub2`, so `sub1 = [2, 6, 8], sub2 = [2, 3, 4]`.
  6. With `5`, we can't extend `sub1`, but we can extend `sub2`, so `sub1 = [2, 6, 8], sub2 = [2, 3, 4, 5]`.
  7. With `1`, we can't extend neighter `sub1` nor `sub2`, but we need to keep `1`, so `sub1 = [2, 6, 8], sub2 = [2, 3, 4, 5], sub3 = [1]`.
  8. Finally, length of longest increase subsequence = `len(sub2)` = 4.
- In the above steps, we need to keep different `sub` arrays (`sub1`, `sub2`..., `subk`) which causes poor performance. But we notice that we can just keep one `sub` array, when new number `x` is not greater than the last element of the subsequence `sub`, we do binary search to find the smallest element >= `x` in `sub`, and replace with number `x`.
- Let's run that example nums = [2, 6, 8, 3, 4, 5, 1] again:
  1. Let pick the first element, `sub = [2]`.
  2. `6` is greater than previous number, `sub = [2, 6]`
  3. `8` is greater than previous number, `sub = [2, 6, 8]`
  4. `3` is less than previous number, so we can't extend the subsequence `sub`. We need to find the smallest number >= `3` in `sub`, it's `6`. Then we overwrite it, now `sub = [2, 3, 8]`.
  5. `4` is less than previous number, so we can't extend the subsequence `sub`. We overwrite `8` by `4`, so `sub = [2, 3, 4]`.
  6. `5` is greater than previous number, `sub = [2, 3, 4, 5]`.
  7. `1` is less than previous number, so we can't extend the subsequence `sub`. We overwrite `2` by `1`, so `sub = [1, 3, 4, 5]`.
  8. Finally, length of longest increase subsequence = `len(sub)` = 4.



### 322. Coin Change

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [-1] * (amount + 1)
        dp[0] = 0
        
        for i in range(1, amount+1):
            min_coin = 2**31 - 1
            for coin in coins:
                if coin <= i and dp[i-coin] < min_coin:
                    min_coin = dp[i-coin] + 1
            dp[i] = min_coin
        return dp[amount] if dp[amount] != 2**31 - 1 else -1
```

https://mp.weixin.qq.com/s/thn3WGARmfiVc3G70PlTdQ

我们定义一个长度为amount+1的一维数组，dp[i]为amount为i时的最优解，dp[0] = 0

对于每个amount，遍历coins，对于小于amount的硬币，我们可求当前硬币对于当前amount的解为dp[i-coin]+1，最优解为所有小于amount的硬币的解的最小值，dp最后便为最后的解。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [None] * (amount + 1)
        dp[0] = 0
        
        for each_amount in range(1, amount+1):
            each_min_coins = sys.maxsize
            for coin in coins:
                if coin <= each_amount and dp[each_amount-coin] < each_min_coins:
                    each_min_coins = dp[each_amount-coin] + 1
            dp[each_amount] = each_min_coins
        
        return dp[-1] if dp[-1] < sys.maxsize else -1
```

我们定义一个长度为amount+1的一维数组，dp[i]为兑换数量为i时需要的最少硬币，所以当i=0时，需要0个硬币，dp[amount] = 兑换amount时需要的最少硬币，该值为最终答案。

从1开始遍历每个amount，记录一个当前需要最小硬币数量的变量，再遍历每个coin，如果当前coin大于当前要兑换的amount，那没有可能兑换则跳过，如果当前coin小于当前要兑换的amount，我们要看兑换amount-coin所需最小硬币数量是否小于当前需要最小硬币数量，如果小于则更新当前需要最小硬币数量的变量为dp[each_amount-coin] + 1，兑换amount-coin所需最小硬币数量纪录在dp中，dp[each_amount-coin]，这便是自底向上的dp。



### 329. Longest Increasing Path in a Matrix

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        rows, cols = len(matrix), len(matrix[0])
        dp = []
        for _ in range(rows):
            dp.append(([0] * cols).copy())
        
        def dfs(row, col):
            if not dp[row][col]:
                val = matrix[row][col]
                dp_val = 1 + max(
                    dfs(row-1, col) if row >= 1 and val > matrix[row-1][col] else 0,
                    dfs(row+1, col) if row < rows - 1 and val > matrix[row+1][col] else 0,
                    dfs(row, col-1) if col >= 1 and val > matrix[row][col-1] else 0,
                    dfs(row, col+1) if col < cols - 1 and val > matrix[row][col+1] else 0
                )
                dp[row][col] = dp_val
            return dp[row][col]
        
        res = -sys.maxsize
        for row in range(rows):
            for col in range(cols):
                res = max(res, dfs(row, col))
        
        return res
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



### 354. Russian Doll Envelopes

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        LIS = []
        size = 0
        for (w, h) in envelopes:
            if not LIS or h > LIS[-1]:
                LIS.append(h)
                size += 1
            else:
                l, r = 0, size
                while l < r:
                    m = l + (r - l) // 2
                    if LIS[m] < h:
                        l = m + 1
                    else:
                        r = m
                LIS[l] = h
        return size
```



### 368. Largest Divisible Subset

```python
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        temp = [None] * len(nums)
        prev = [None] * len(nums)
        
        nums.sort()
        _max = 0
        index = -1
        for i in range(len(nums)):
            temp[i] = 1
            prev[i] = -1
            for j in range(i-1, -1, -1):
                if nums[i] % nums[j] == 0 and temp[j] + 1 > temp[i]:
                    temp[i] = temp[j] + 1
                    prev[i] = j
            if temp[i] > _max:
                _max = temp[i]
                index = i
        res = []
        while index != -1:
            res.append(nums[index])
            index = prev[index]
        return res
```

https://leetcode.com/problems/largest-divisible-subset/discuss/684677/3-STEPS-c%2B%2B-or-python-or-java-dp-PEN-PAPER-DIAGRAM

不太明白



### 416. Partition Equal Subset Sum

```python
# TLE
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        
        all_sum = sum(nums)
        temp_mask = 1 << len(nums)

        for i in range(2**len(nums)):
            mask = bin(i|temp_mask)[3:]
            temp_sum = 0
            for j, val in enumerate(mask):
                if val == '1':
                    temp_sum += nums[j]
            if temp_sum == all_sum - temp_sum:
                return True
        return False
```

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        all_sum = sum(nums)
        if all_sum % 2 == 1: return False
        all_sum //= 2
        
        dp = []
        for i in range(len(nums)+1):
            dp.append([False] * (all_sum + 1))
        
        for i in range(len(nums)+1):
            dp[i][0] = True
        
        for i in range(1, len(nums)+1):
            for j in range(1, (all_sum+1)):
                if nums[i-1] > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
        
        return dp[len(nums)][all_sum]
```

https://leetcode.com/problems/partition-equal-subset-sum/discuss/90592/01-knapsack-detailed-explanation

https://leetcode-cn.com/problems/partition-equal-subset-sum/solution/fen-ge-deng-he-zi-ji-by-leetcode-solution/

例：[1, 5, 11, 5]

| -    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | o    | x    | x    | x    | x    | x    | x    | x    | x    | x    | x    | x    |
| 1    | o    | o    | x    | x    | x    | x    | x    | x    | x    | x    | x    | x    |
| 5    | o    | o    | x    | x    | x    | o    | o    | x    | x    | x    | x    | x    |
| 11   | o    | o    | x    | x    | x    | o    | o    | x    | x    | x    | x    | o    |
| 5    | o    | o    | x    | x    | x    | o    | o    | x    | x    | x    | o    | o    |

此题为0/1背包问题变形，每个数字有两个选择，要么选要么不选，最后求和为数组和的一半。定义dp[i] [j] 为截止到地 i 个元素是否有和为 j 的组合。

我们定义数组的时候多加一行一列，第一行全部为 False 除了00，第一列全部为 True

状态转移方程：当当前数字大于 j 的时候也就是 nums[i-1] > j: 不能选当前数字，那么当前状态便由之前的状态决定也就是 dp[i-1] [j]

否则 dp[i] [j] = dp[i-1] [j] || dp[i-1] [j-nums[i]]



对于典型的0/1背包问题：

例：背包容量10

物品：

|        | a    | b    | c    | d    | e    |
| ------ | ---- | ---- | ---- | ---- | ---- |
| value  | 6    | 3    | 5    | 4    | 6    |
| weight | 2    | 2    | 6    | 5    | 4    |

定义dp[i] [j]为截止到 i 物品，背包容量为 j 的最大价值，我们多加一行一列

| v    | w    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 6    | 2    | 0    | 0    | 6    | 6    | 6    | 6    | 6    | 6    | 6    | 6    | 6    |
| 3    | 2    | 0    | 0    | 6    | 6    | 9    | 9    | 9    | 9    | 9    | 9    | 9    |
| 5    | 6    | 0    | 0    | 6    | 6    | 9    | 9    | 9    | 9    | 11   | 11   | 11   |
| 4    | 5    | 0    | 0    | 6    | 6    | 9    | 9    | 9    | 10   | 11   | 13   | 13   |
| 6    | 4    | 0    | 0    | 6    | 6    | 9    | 9    | 12   | 12   | 15   | 15   | 15   |

状态转移方程：如果该物品大小大于当前背包容量，则不能选，则当前状态取决于前面的状态也就是dp[i-1] [j]

否则当前状态为不选和选之中的最大值，选的话价值为前面一个价值加当前物品价值，也就是dp[i-1] [j-nums[i-1]]



### 474. Ones and Zeroes

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        counter = []
        for each in strs:
            counter.append((Counter(each)["0"], Counter(each)["1"]))
        
        @lru_cache(None)
        def helper(index, nums0, nums1):
            if index >= len(strs) or nums0 == 0 and nums1 == 0:
                return 0
            
            option1 = helper(index+1, nums0, nums1)
            if nums0 >= counter[index][0] and nums1 >= counter[index][1]:
                option2 = 1 + helper(index+1, nums0-counter[index][0], nums1-counter[index][1])
                return max(option1, option2)
            return option1
        
        return helper(0, m, n)
```

https://leetcode.com/problems/ones-and-zeroes/discuss/814077/Dedicated-to-Beginners



### [518. Coin Change II](https://leetcode.com/problems/coin-change-ii/)

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:

        memo = {}

        def helper(i, amount):
            if amount == 0:
                return 1
            
            if i >= len(coins):
                return 0

            if memo.get((i, amount)) is not None:
                return memo[(i, amount)]
            
            if coins[i] > amount:
                memo[(i, amount)] = helper(i + 1, amount)
            else:
                memo[(i, amount)] = helper(i + 1, amount) + helper(i, amount - coins[i])

            return memo[(i, amount)]
        
        return helper(0, amount)

```

https://leetcode.com/problems/coin-change-ii/editorial/ 官方答案写的不错，简单易懂

Top-down dp with memo

将问题分解成子问题，子问题有两个参数，一个是 coins 数量，一个是额度，也就是 helper 函数参数。

helper 函数参数 i 表示当前 coins index，amount 表示当前剩余额度，其中两种递归情况：

1. 当前 index 的硬币数额大于当前剩余额度，意味着不能使用当前硬币，于是跳过当前硬币，i 加 1，amount 不变 `helper(i + 1, amount)`
2. 否则如果可以使用当前硬币，那么当前有两种选择，一个是不使用当前硬币也就是和上面 1 情况一样，一个是使用当前硬币，那么 i 不变，amount 减去当前硬币数量 `helper(i, amount - coins[i])`

因为有很多重复的计算，所以我们在过程中将子问题结果缓存在字典里，可以想答案中一样使用二维数组但也可以简单使用字典（tuple 作为key），我们最终要计算的是使用全部硬币和全部额度：`helper(0, amount)` 。

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        dp = [[0] * (amount + 1) for _ in range(n + 1)]
        for i in range(n):
            dp[i][0] = 1
        
        # print(f"dp: {dp}")
        # 硬币数组长度是行数
        # amount 时列数
        # 遍历是从数组左下方，倒数第二行正数第二列开始，向右上方
        # dp[0][amount] 是答案
        
        for i in range(n - 1, -1, -1):
            for j in range(1, amount + 1):
                if coins[i] > j:
                    dp[i][j] = dp[i + 1][j]
                else:
                    dp[i][j] = dp[i + 1][j] + dp[i][j - coins[i]]
        
        return dp[0][amount]
```

Down-top dp



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
            hold = max(hold, cash - prices[i]) # 与上一行可以交换顺序
        return cash
```

At the end of the `i`-th day, we maintain `cash`, the maximum profit we could have if we did not have a share of stock, and `hold`, the maximum profit we could have if we owned a share of stock.

To transition from the `i`-th day to the `i+1`-th day, we either sell our stock `cash = max(cash, hold + prices[i] - fee)` or buy a stock `hold = max(hold, cash - prices[i])`. At the end, we want to return `cash`. We can transform `cash` first without using temporary variables because selling and buying on the same day can't be better than just continuing to hold the stock.

@ [vegito2002](https://leetcode.com/vegito2002/) For future readers:
If I am holding a share after today, then either I am just continuing holding the share I had yesterday, or that I held no share yesterday, but bought in one share today: `hold = max(hold, cash - prices[i])`
If I am not holding a share after today, then either I did not hold a share yesterday, or that I held a share yesterday but I decided to sell it out today: `cash = max(cash, hold + prices[i] - fee)`.
Make sure `fee` is only incurred once.

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        
        # maximum profit that can be obtained without holding the stock in the first i days
        free = [None] * len(prices)
        free[0] = 0

        # maximum profit that can be obtained holding the stock in the first i days
        hold = [None] * len(prices)
        hold[0] = -prices[0]

        for i in range(1, len(prices)):
            price = prices[i]
            free[i] = max(free[i-1], hold[i-1] + price - fee) # 对于 free 数组，要在第 i 天卖掉，则卖掉的价格需要获取前一天的持有利润也就是hold[i-1] + price - fee
            hold[i] = max(hold[i-1], free[i-1] - price) # 对于 hold 数组，要在第 i 天买，则买的价格需要获取前一天未持有利润也就是free[i-1] - price
        
        return free[-1]
```

很好的解答：https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/editorial/



### 746. Min Cost Climbing Stairs

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        @cache
        def dp(i):
            if i <= 1: return 0
            return min(dp(i-1) + cost[i - 1], dp(i - 2) + cost[i - 2])
        return dp(len(cost))
```

函数dp为爬到第i层最小代价是多少

当在第0层或第1层不需要爬所以代价是0，否则有两个选择要么从 i - 1 层爬上来（到顶层），同时付出代价 cost[i - 1]，要么从 i - 2 层爬上来，同时付出代价 cost[i - 2]，取其中最小。



### 790. Domino and Tromino Tiling

```python
class Solution:
    def numTilings(self, n: int) -> int:
        @cache
        def helper(i, previous_gap):
            if i > n: return 0
            if i == n: return not previous_gap
            if previous_gap:
                return helper(i+1, False) + helper(i+1, True)
            return helper(i+1, False) + helper(i+2, False) + 2*helper(i+2, True)
        return helper(0, False) % 1_000_000_007
```

https://leetcode.com/problems/domino-and-tromino-tiling/discuss/1620975/C%2B%2BPython-Simple-Solution-w-Images-and-Explanation-or-Optimization-from-Brute-Force-to-DP

![9f0fa40d-874c-45dd-88fb-a6d964026e69_1639105504.9259145](https://assets.leetcode.com/users/images/9f0fa40d-874c-45dd-88fb-a6d964026e69_1639105504.9259145.png)

![9a096afe-85f9-4358-a179-5563f960fc37_1639115832.0220907](https://assets.leetcode.com/users/images/9a096afe-85f9-4358-a179-5563f960fc37_1639115832.0220907.png)

题目大意为现在有两种类型的图形，想拼成2*n的矩形，也就是两行n列，求有多少种拼法。

对于这两种图形，答案中给出了6种情况，下面分别分析了每种情况的拼法。

对于前面没有空隙的时候：

放置第一种，不会产生空隙，向前走一列，所以solve(i+1, previous_gap=False)

放置第二种，不会产生空隙，向前走两列，solve(i+2, previous_gap=False)

放置第三种或第四种，会产生空隙，向前走两步，solve(i+2, previous_gap=True)

对于前面有空隙的情况，也就是之前放置了第三种或第四种：

放置第五种或第六种，正好填补好空隙所以不再有空隙，向前走一步，solve(i+1, previous_gap=False)

放置第二种，虽然填补了之前的空隙但是有产生了新的空隙，向前走一步，solve(i+1, previous_gap=True)



### 920. Number of Music Playlists

```python
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        MOD = 10**9 + 7

        dp = [[0 for _ in range(n + 1)] for _ in range(goal + 1)]
        dp[0][0] = 1

        for i in range(1, goal+1):
            for j in range(1, n+1):
                if j > i: continue
                scenario1 = dp[i-1][j-1] * (n - (j - 1))
                scenario2 = dp[i-1][j] * (j - k) if j > k else 0
                dp[i][j] = (scenario1 + scenario2) % MOD
        
        return dp[goal][n]
```

https://leetcode.com/problems/number-of-music-playlists/editorial/

有 `n` 首歌，创建一个有 `goal` 首歌的播放列表，其中 `goal > n`，要求是1. `n` 首歌每首至少播放一遍 2. 重复播放下一首歌的条件是已经播放了 `k` 首其他的歌，问有多少种播放列表可能。

创建一个二维数组`dp` ，有 `goal + 1` 行，`n + 1` 列，`dp[i][j]` 表示当 `goal=i & n=j` 时有多少种方案，显然这个二维数组中有一半也就是 `j > i` 时是空的，因为播放列表歌曲数量小于歌曲总数时，不可能将每首歌都播放。

此时我们从 `dp[0][0]` 开始向下遍历，每个格的可能数量是两种情况之和：

1. 歌曲总数加一 `j+1` 且播放列表中歌曲数加一 `i+1`，这种情况下因为总共有 `n` 首歌，已经添加了 `j-1` 首歌，所有还可能添加 `n-(j-1)` 首歌，每首歌都对应之前的 `dp[i-1][j-1]` 中可能，所以最终是 `dp[i-1][j-1] * (n-(j-1))`
2. 歌曲总数不变，播放列表中歌曲数加一 `i+1`，这种情况下因为有 `j` 首歌曲可以再次播放，但是题目要求需要播放了 `k` 首其他歌曲，所以当 `j > k` 时才有可能，而且有 `j - k` 首歌可以再次播放，每首歌都对应之前的 `dp[i-1][j]` 种可能，所以最终是 `dp[i-1][j] * (j - k)` 

对于 `dp[i][j]` 位置我们需要把上面两种可能相加，最后返回 `dp[goal][n]`



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

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        memo = [[0] * len(text2) for _ in range(len(text1))]
        temp = False
        for i in range(len(text2)):
            if temp:
                memo[0][i] = 1
            elif text1[0] == text2[i]:
                temp = True
                memo[0][i] = 1
        temp = False
        for i in range(len(text1)):
            if temp:
                memo[i][0] = 1
            elif text1[i] == text2[0]:
                temp = True
                memo[i][0] = 1
        for i in range(1, len(text1)):
            for j in range(1, len(text2)):
                if text1[i] == text2[j]:
                    memo[i][j] = memo[i-1][j-1] + 1
                else:
                    memo[i][j] = max(memo[i][j-1], memo[i-1][j])
        return memo[len(text1)-1][len(text2)-1]
```



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



### 1626. Best Team With No Conflicts

```python
class Solution:
    def bestTeamScore(self, scores: List[int], ages: List[int]) -> int:
        
        age_score_list = sorted([(ages[i], scores[i]) for i in range(len(ages))])

        res = 0
        dp = [each[1] for each in age_score_list]

        for i in range(len(ages)):
            for j in range(i):
                if age_score_list[i][1] >= age_score_list[j][1]:
                    dp[i] = max(dp[i], dp[j] + age_score_list[i][1])
            res = max(res, dp[i])
        
        return res
```

https://leetcode.com/problems/best-team-with-no-conflicts/solutions/2886659/best-team-with-no-conflicts/

### Approach 2: Bottom-Up Dynamic Programming

#### Intuition

If we observe closely, after sorting the list of pairs (age, score) by age, we need to find the highest sum of a non-decreasing subsequence of scores in the list. This is because after sorting, the list has the ages in ascending order, and in order to be non-conflicting, the score also has to be in non-decreasing order. Therefore we need to find the largest sum of scores in any non-decreasing subsequence of scores in the list of pairs. This is a typical dynamic programming problem very similar to [[309\] Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/).

Similar to the previous approach, we will first sort the pairs in ascending order of age and then by score. Then we will iterate over the players from left to right. For each player, we will try to find the previous player it could be paired with. We will iterate over the players on the left and find the pairing that provides the maximum score for this player. The maximum score of any player will be the answer.

#### Algorithm

1. Store the ages and scores of all the players in the list `ageScorePair`.
2. Sort the list `ageScorePair` in ascending order of age and then in ascending order of score.
3. Initialize the array `dp` of size `N`. The `dp[i]` represents the maximum score possible by taking `ith` player and possible players before it. All values initially will be equal to the score of individual players.
4. Iterate over players from `0` to `N - 1` for each player at index `i`
   - Iterate over the players on the left, i.e., from `0` to `i - 1`. For each such player, `j`, check if the score of the `ith` player is greater than or equal to the `jth` player's score. If it is, we can add the total score of the `jth` player (`dp[j]`) to the score of the `ith` player and update the maximum score of the `ith` player `dp[i]` accordingly.
5. Store the maximum of all `dp[i]` in the variable `answer`.
6. Return `answer`.
