## LeetCode - Binary Search

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

