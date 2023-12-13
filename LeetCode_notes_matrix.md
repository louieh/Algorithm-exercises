## LeetCode - Matrix

[toc]

### Rotate Matrix Elements

```python
"""
Input:
1    2    3    4    
5    6    7    8
9    10   11   12
13   14   15   16
Output:
5    1    2    3
9    10   6    4
13   11   7    8
14   15   16   12
"""

# Function to rotate a matrix 
def rotateMatrix(mat): 
  
    if not len(mat): 
        return
      
    """ 
        top : starting row index 
        bottom : ending row index 
        left : starting column index 
        right : ending column index 
    """
  
    top = 0
    bottom = len(mat)-1
  
    left = 0
    right = len(mat[0])-1
  
    while left < right and top < bottom: 
  
        # Store the first element of next row, 
        # this element will replace first element of 
        # current row 
        prev = mat[top+1][left] 
  
        # Move elements of top row one step right 
        for i in range(left, right+1): 
            curr = mat[top][i] 
            mat[top][i] = prev 
            prev = curr 
  
        top += 1
  
        # Move elements of rightmost column one step downwards 
        for i in range(top, bottom+1): 
            curr = mat[i][right] 
            mat[i][right] = prev 
            prev = curr 
  
        right -= 1
  
        # Move elements of bottom row one step left 
        for i in range(right, left-1, -1): 
            curr = mat[bottom][i] 
            mat[bottom][i] = prev 
            prev = curr 
  
        bottom -= 1
  
        # Move elements of leftmost column one step upwards 
        for i in range(bottom, top-1, -1): 
            curr = mat[i][left] 
            mat[i][left] = prev 
            prev = curr 
  
        left += 1
  
    return mat
```

### Rotate matrix among Diagnals

```python
"""
Input:
[[1, 2, 3, 4, 5],
 [6, 7, 8, 9, 10],
 [11, 12, 13, 14, 15],
 [16, 17, 18, 19, 20],
 [21, 22, 23, 24, 25]]
 Output:
 [[1, 6, 2, 3, 5],
 [11, 7, 12, 9, 4],
 [16, 18, 13, 8, 10],
 [22, 17, 14, 19, 15],
 [21, 23, 24, 20, 25]]
"""
up = left = 0
down = len(matrix)-1
right = len(matrix[0]) - 1
In [95]: while left < right and up < down:
    ...:     prev = matrix[up+1][left]
    ...:     for i in range(left, right+1):
    ...:         if up == i or (up + i) == (len(matrix)-1):
    ...:             continue
    ...:         current = matrix[up][i]
    ...:         matrix[up][i] = prev
    ...:         prev = current
    ...:     up += 1
    ...:     for i in range(up, down+1):
    ...:         if i == right or (right + i) == (len(matrix)-1):
    ...:             continue
    ...:         current = matrix[i][right]
    ...:         matrix[i][right] = prev
    ...:         prev = current
    ...:     right -= 1
    ...:     for i in range(right, left-1, -1):
    ...:         if i == down or (down + i) == (len(matrix)-1):
    ...:             continue
    ...:         current = matrix[down][i]
    ...:         matrix[down][i] = prev
    ...:         prev = current
    ...:     down -= 1
    ...:     for i in range(down, up-1, -1):
    ...:         if i == left or (left + i) == (len(matrix)-1):
    ...:             continue
    ...:         current = matrix[i][left]
    ...:         matrix[i][left] = prev
    ...:         prev = current
    ...:     left += 1
```



### 48. Rotate Image

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        matrix[:] = matrix[::-1]
        for i in range(len(matrix)):
            for j in range(i+1, len(matrix[0])):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```
https://leetcode.com/problems/rotate-image/discuss/18872/A-common-method-to-rotate-the-image



### 54. Spiral Matrix

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return[]
        ans = []
        
        if len(matrix) == 1:
            return matrix[0]
        if len(matrix[0]) == 1:
            for each in matrix:
                ans.append(each[0])
            return ans
        
        up_limit = 0
        down_limit = len(matrix)-1
        right_limit = len(matrix[0])-1
        left_limit = 0
        
        row = 0
        col = 0
        
        direction = 'r'
        
        for i in range(len(matrix)*len(matrix[0])):
            if direction == 'r':
                ans.append(matrix[row][col])
                col += 1
                if col == right_limit:
                    direction = 'd'
                    up_limit += 1

            elif direction == 'd':
                ans.append(matrix[row][col])
                row += 1
                if row == down_limit:
                    direction = 'l'
                    right_limit -= 1

            elif direction == 'l':
                ans.append(matrix[row][col])
                col -= 1
                if col == left_limit:
                    direction = 'u'
                    down_limit -= 1

            elif direction == 'u':
                ans.append(matrix[row][col])
                row -= 1
                if row == up_limit:
                    direction = 'r'
                    left_limit += 1

        return ans
```



```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix:
            return []
        
        up = 0
        down = len(matrix)-1
        left = 0
        right = len(matrix[0])-1
        
        ans = []
        count = 0
        num = len(matrix) * len(matrix[0])
        # while left <= right or up <= down:
        while 1:
            for i in range(left, right+1):
                ans.append(matrix[up][i])
                count += 1
            if count == num:
                break
            up += 1
            for i in range(up, down+1):
                ans.append(matrix[i][right])
                count += 1
            if count == num:
                break
            right -= 1
            for i in range(right, left-1, -1):
                ans.append(matrix[down][i])
                count += 1
            if count == num:
                break
            down -= 1
            for i in range(down, up-1, -1):
                ans.append(matrix[i][left])
                count += 1
            if count == num:
                break
            left += 1
        return ans
```



### 59. Spiral Matrix II

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        if not n:
            return [[]]
        
        ans = []
        for i in range(n):
            ans.append([0]*n)
        
        up_limit = 0
        down_limit = n - 1
        right_limit = n - 1
        left_limit = 0
        
        row = 0
        col = 0
        
        direction = 'right'
        
        for i in range(n*n):
            ans[row][col] = i + 1
            if direction == 'right':
                if col == right_limit:
                    direction = 'down'
                    up_limit += 1
                    row += 1
                else:
                    col += 1
            elif direction == 'down':
                if row == down_limit:
                    direction = 'left'
                    right_limit -= 1
                    col -= 1
                else:
                    row += 1
            elif direction == 'left':
                if col == left_limit:
                    direction = 'up'
                    down_limit -= 1
                    row -= 1
                else:
                    col -= 1
            elif direction == 'up':
                if row == up_limit:
                    direction = 'right'
                    left_limit += 1
                    col += 1
                else:
                    row -= 1
            
        return ans
```

```py
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        grid = []
        row = [None] * n
        for _ in range(n):
            grid.append(row.copy())
        
        i, row, col = 0, 0, 0 
        while i < n * n:
            
            # left -> right
            while col <= n - 1 and grid[row][col] is None:
                i += 1
                grid[row][col] = i
                col += 1
            col -= 1
            row += 1
            
            # up -> down
            while row <= n - 1 and grid[row][col] is None:
                i += 1
                grid[row][col] = i
                row += 1
            row -= 1
            col -= 1
            
            # left <- right
            while col >= 0 and grid[row][col] is None:
                i += 1
                grid[row][col] = i
                col -= 1
            col += 1
            row -= 1
            
            # up <- down
            while row >= 0 and grid[row][col] is None:
                i += 1
                grid[row][col] = i
                row -= 1
            row += 1
            col += 1
            
        return grid
```



### 74. Search a 2D Matrix

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix) == 0 or len(matrix[0]) == 0: return False
        rows, cols = len(matrix), len(matrix[0])
        if target < matrix[0][0] or target > matrix[rows-1][cols-1]: return False
        
        def index_to_matrix(mid):
            if rows == 1: return 0, mid
            if cols == 1: return mid, 0
            if (mid + 1) % cols == 0:
                return (mid + 1) // cols - 1, cols - 1
            return (mid + 1) // cols, (mid + 1) % cols - 1
        
        left, right = 0, rows*cols-1
        while left <= right:
            mid = left + (right - left) // 2
            row_i, col_i = index_to_matrix(mid)
            num = matrix[row_i][col_i]
            if num == target: return True
            if target > num:
                left = mid + 1
            else:
                right = mid - 1
        return False
```

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix) == 0 or len(matrix[0]) == 0: return False
        rows, cols = len(matrix), len(matrix[0])
        if target < matrix[0][0] or target > matrix[rows-1][cols-1]: return False

        left, right = 0, rows*cols-1
        while left <= right:
            mid = left + (right - left) // 2
            num = matrix[mid//cols][mid%cols]
            if num == target: return True
            if target > num:
                left = mid + 1
            else:
                right = mid - 1
        return False
```

二分，主要问题是将index转换为matrix坐标`matrix[mid/cols][mid%cols]`

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        rows, cols = len(matrix), len(matrix[0])
        left, right = 1, rows * cols

        while left <= right:
            mid = left + (right - left) // 2
            row = mid // cols if mid % cols != 0 else mid // cols - 1
            col = mid - row * cols - 1
            num = matrix[row][col]
            if num == target:
                return True
            if num > target:
                right = mid - 1
            else:
                left = mid + 1
        
        return False
```



### 79. Word Search

```python
# TLE
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if len(word) > len(board) * len(board[0]):
            return False
        
        def dfs(row, col, index):
            if index == len(word) - 1:
                return True
            temp = board[row][col]
            board[row][col] = '#'
            u, d, l, r = False, False, False, False
            if col > 0 and board[row][col-1] == word[index+1]:
                l = dfs(row, col-1, index+1)
            if col < len(board[0])-1 and board[row][col+1] == word[index+1]:
                r = dfs(row, col+1, index+1)
            if row > 0 and board[row-1][col] == word[index+1]:
                u = dfs(row-1, col, index+1)
            if row < len(board)-1 and board[row+1][col] == word[index+1]:
                d = dfs(row+1, col, index+1)
            board[row][col] = temp
            return u or d or l or r
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    if dfs(i, j, 0):
                        return True
        return False
```

```python
# accept
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if len(word) > len(board) * len(board[0]):
            return False
        
        def dfs(row, col, index):
            if row < 0 or row >= len(board) or col < 0 or col >= len(board[0]) or board[row][col] != word[index]:
                return False
            
            if index == len(word) - 1:
                return True
            
            temp = board[row][col]
            board[row][col] = '#'
            res = dfs(row, col-1, index+1) or dfs(row, col+1, index+1) or dfs(row-1, col, index+1) or dfs(row+1, col, index+1)
            board[row][col] = temp
            return res
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    if dfs(i, j, 0):
                        return True
        return False
```

if 语句太耗时了？？？

https://leetcode.com/problems/word-search/discuss/27660/Python-dfs-solution-with-comments.



### 85. Maximal Rectangle

```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        rows = len(matrix)
        if not rows: return 0
        cols = len(matrix[0])
        if not cols: return 0
        
        height = [0] * (cols+1)
        res = 0
        for i in range(rows):
            stack = []
            for j in range(cols+1):
                if j < cols:
                    if matrix[i][j] == '1':
                        height[j] += 1
                    else:
                        height[j] = 0
                
                while stack and height[stack[-1]] >= height[j]:
                    h = height[stack.pop()]
                    if stack:
                        w = j - stack[-1] - 1
                    else:
                        w = j
                    res = max(res, h * w)
                
                stack.append(j)
        
        return res
```

没懂



### 130. Surrounded Regions

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        q = []
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if (i == 0 or j == 0 or i == len(board)-1 or j == len(board[0])-1) and board[i][j] == 'O':
                    q.append((i, j))
        
        while q:
            i, j = q.pop()
            if 0 <= i and i < len(board) and 0 <= j and j <len(board[0]) and board[i][j] == 'O':
                board[i][j] = 'S'
                q.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'S':
                    board[i][j] = 'O'
                elif board[i][j] == 'O':
                    board[i][j] = 'X'
```

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        
        rows, cols = len(board), len(board[0])
        
        def capture(row, col):
            if row >= 0 and row <= rows-1 and col >= 0 and col <= cols-1 and board[row][col] == 'O':
                board[row][col] = 'T'
                capture(row+1, col)
                capture(row-1, col)
                capture(row, col+1)
                capture(row, col-1)
        
        for row in range(rows):
            for col in range(cols):
                if row == 0 or row == rows-1 or col == 0 or col == cols-1:
                    if board[row][col] == 'O':
                        capture(row, col)
        
        for row in range(rows):
            for col in range(cols):
                if board[row][col] == 'O':
                    board[row][col] = 'X'
                if board[row][col] == 'T':
                    board[row][col] = 'O'
```



### 200. Number of Islands

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0
        
        def delete_island(row, col, direction):
            grid[row][col] = '0'
            if direction != 'r' and col > 0 and grid[row][col-1] == '1':
                delete_island(row, col-1, 'l')
            if direction != 'l' and col < len(grid[0])-1 and grid[row][col+1] == '1':
                delete_island(row, col+1, 'r')
            if direction != 'd' and row > 0 and grid[row-1][col] == '1':
                delete_island(row-1, col, 'u')
            if direction != 'u' and row < len(grid)-1 and grid[row+1][col] == '1':
                delete_island(row+1, col, 'd')
        
        island = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    island += 1
                    delete_island(i, j, 'r')
        return island
```

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]: return 0
        
        rows, cols = len(grid), len(grid[0])
        res = 0
        def dfs(row, col):
            grid[row][col] = "0"
            if col > 0 and grid[row][col-1] == "1":
                dfs(row, col-1)
            if col < cols-1 and grid[row][col+1] == "1":
                dfs(row, col+1)
            if row > 0 and grid[row-1][col] == "1":
                dfs(row-1, col)
            if row < rows-1 and grid[row+1][col] == "1":
                dfs(row+1, col)
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == "1":
                    dfs(i, j)
                    res += 1
        return res
```



### 240. Search a 2D Matrix II

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        for i in range(len(matrix)):
            low = i * len(matrix[0]) + 1
            high = low + len(matrix[0]) - 1

            while low <= high:
                mid = (high-low)//2+low
                if mid % len(matrix[0]) == 0:
                    index_num = matrix[mid//len(matrix[0])-1][len(matrix[0])-1]
                else:
                    index_num = matrix[mid//len(matrix[0])][mid%len(matrix[0])-1]

                if index_num == target:
                    return True
                if index_num > target:
                    high = mid - 1
                else:
                    low = mid + 1

        return False
```

```python
# O(m+n)
# 从右上角开始，如果target小的话向左走，如果target大的话向下走
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        
        row = 0
        col = len(matrix[0])-1
        
        while col >= 0 and row <= len(matrix)-1:
            if target == matrix[row][col]:
                return True
            elif target > matrix[row][col]:
                row += 1
            else:
                col -= 1
        return False
```



### 289. Game of Life

```python
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        change_list = []
        
        def test_change(i, j):
            upper_left = board[i-1][j-1] if i > 0 and j > 0 else 0
            upper = board[i-1][j] if i > 0 else 0
            upper_right = board[i-1][j+1] if i > 0 and j < len(board[0]) - 1 else 0
            right = board[i][j+1] if j < len(board[0]) - 1 else 0
            lower_right = board[i+1][j+1] if i < len(board) - 1 and j < len(board[0]) - 1 else 0
            lower = board[i+1][j] if i < len(board) - 1 else 0
            lower_left = board[i+1][j-1] if i < len(board) - 1 and j > 0 else 0
            left = board[i][j-1] if j > 0 else 0
            live_neighbors = upper_left + upper + upper_right + right + lower_right + lower + lower_left + left
            if board[i][j] == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    return 0
            else:
                if live_neighbors == 3:
                    return 1
            return None
        
        for row in range(len(board)):
            for col in range(len(board[0])):
                res = test_change(row, col)
                if res is not None:
                    change_list.append((row, col, res))
        
        for row, col, res in change_list:
            board[row][col] = res
```



### 419.Battleships in a Board

```python
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        if not board or not board[0]:
            return 0
        
        ans = 0
        
        def delete_(row, col, direction):
            board[row][col] = '.'
            if direction != 'r' and col > 0 and board[row][col-1] == 'X':
                delete_(row, col-1, 'l')
            if direction != 'l' and col < len(board[0])-1 and board[row][col+1] == 'X':
                delete_(row, col+1, 'r')
            if direction != 'u' and row < len(board)-1 and board[row+1][col] == 'X':
                delete_(row+1, col, 'd')
            if direction != 'd' and row > 0 and board[row-1][col] == 'X':
                delete_(row-1, col, 'u')
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'X':
                    ans += 1
                    delete_(i, j, 'r')
        return ans
```

和数岛那个题一毛一样。



### 463. Island Perimeter

```python
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        res = 0
        rows = len(grid)
        cols = len(grid[0])
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1:
                    if row == 0 or grid[row-1][col] == 0:
                        res += 1
                    if row == rows - 1 or grid[row+1][col] == 0:
                        res += 1
                    if col == 0 or grid[row][col-1] == 0:
                        res += 1
                    if col == cols - 1 or grid[row][col+1] == 0:
                        res += 1
        return res
```



### 542. 01 Matrix

```python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        def compute_distance(obj1, obj2):
            return abs(obj1[0]-obj2[0]) + abs(obj1[1]-obj2[1])
        
        zero_list = []
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] == 0:
                    zero_list.append((row, col))
        
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] != 0:
                    min_dis = 2**23-1
                    for each in zero_list:
                        temp_distance = compute_distance(each, (row, col))
                        if temp_distance < min_dis:
                            min_dis = temp_distance
                    matrix[row][col] = min_dis
        return matrix
```

Brute force 超时

```python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        from collections import deque
        Q = deque([])
        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    Q.append((i,j,0))
                else:
                    matrix[i][j] = 2**32-1
        
        while Q:
            row, col, step = Q.popleft()
            
            if row > 0 and matrix[row-1][col] > step+1:
                matrix[row-1][col] = step + 1
                Q.append((row-1, col, step+1))
            if row < len(matrix)-1 and matrix[row+1][col] > step+1:
                matrix[row+1][col] = step+1
                Q.append((row+1, col, step+1))
            if col > 0 and matrix[row][col-1] > step+1:
                matrix[row][col-1] = step+1
                Q.append((row, col-1, step+1))
            if col < len(matrix[0])-1 and matrix[row][col+1] > step+1:
                matrix[row][col+1] = step+1
                Q.append((row, col+1, step+1))
        return matrix
```

从每个0开始向四周做bfs，松弛操作，可松弛的加入队列。

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:

        dq = deque([])
        rows, cols = len(mat), len(mat[0])

        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    dq.append((i, j, 0))
                else:
                    mat[i][j] = sys.maxsize
        
        while dq:
            row, col, step = dq.popleft()

            # up
            if row > 0 and mat[row-1][col] > step + 1:
                mat[row-1][col] = step + 1
                dq.append((row-1, col, step+1))
            # down
            if row < rows - 1 and mat[row+1][col] > step + 1:
                mat[row+1][col] = step + 1
                dq.append((row+1, col, step+1))
            # left
            if col > 0 and mat[row][col-1] > step + 1:
                mat[row][col-1] = step + 1
                dq.append((row, col-1, step+1))
            # right
            if col < cols - 1 and mat[row][col+1] > step + 1:
                mat[row][col+1] = step + 1
                dq.append((row, col+1, step+1))
        
        return mat
```

将 1 所在位置标记为最大值，是肯定要在之后松弛的。从每个 0 所在位置为起始位置开始向四周松弛，每个 0 互不影响是因为松弛是最终到最小值，所以比如第一个 0 将某个位置松弛到一个值后，另一个 0 为起点到该位置的距离更小那么会更新该位置的值。



### 695. Max Area of Island

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def bfs(i, j, direction):
            nonlocal count
            if grid[i][j] == 1:
                count += 1 
                grid[i][j] = 0
            if direction != 'left' and j < len(grid[0])-1 and grid[i][j+1] != 0:
                bfs(i, j+1, 'right')
            if direction != 'right' and j > 0 and grid[i][j-1] != 0:
                bfs(i, j-1, 'left')
            if direction != 'up' and i < len(grid)-1 and grid[i+1][j] != 0:
                bfs(i+1, j, 'down')
            if direction != 'down' and i > 0 and grid[i-1][j] != 0:
                bfs(i-1, j, 'up')
        max_count = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                count = 0
                if grid[row][col] == 1:
                    bfs(row, col, 'right')
                    max_count = max(max_count, count)
        return max_count
```



### 733. Flood Fill

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if image[sr][sc] == newColor:
            return image
        
        def floodFill_helper(image, sr, sc, oldColor, newColor, direction):
            image[sr][sc] = newColor
            if direction != 'r' and sc > 0 and image[sr][sc-1] == oldColor:
                floodFill_helper(image, sr, sc-1, oldColor, newColor, 'l')
            if direction != 'l' and sc < len(image[0]) - 1 and image[sr][sc+1] == oldColor:
                floodFill_helper(image, sr, sc+1, oldColor, newColor, 'r')
            if direction != 'u' and sr > 0 and image[sr-1][sc] == oldColor:
                floodFill_helper(image, sr-1, sc, oldColor, newColor, 'd')
            if direction != 'd' and sr < len(image) - 1 and image[sr+1][sc] == oldColor:
                floodFill_helper(image, sr+1, sc, oldColor, newColor, 'u')
                    
                    
        oldColor = image[sr][sc]
        floodFill_helper(image, sr, sc, oldColor, newColor, 'r')
        floodFill_helper(image, sr, sc, oldColor, newColor, 'l')
        return image
```

和数岛那个题类似方法。

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if image[sr][sc] == newColor:
            return image
        
        def floodFill_helper(image, sr, sc, oldColor, newColor):
            if sr >= 0 and sr <= len(image) - 1 and sc >= 0 and sc <= len(image[0]) - 1:
                if image[sr][sc] == oldColor:
                    image[sr][sc] = newColor
                    floodFill_helper(image, sr, sc-1, oldColor, newColor)
                    floodFill_helper(image, sr, sc+1, oldColor, newColor)
                    floodFill_helper(image, sr-1, sc, oldColor, newColor)
                    floodFill_helper(image, sr+1, sc, oldColor, newColor)
                    
        floodFill_helper(image, sr, sc, image[sr][sc], newColor)
        return image
```

直接向上下左右四个方向遍历，遇到与oldColor不等就停止。

```go
func doFill(_image *[][]int, row int, col int, d string, color int, oriColor int) {
        if row < 0 || row >= len(*_image) || col < 0 || col >= len((*_image)[0]) || (*_image)[row][col] != oriColor {
            return
        }
        (*_image)[row][col] = color
        if d == "up" {
            doFill(_image, row, col-1, "right", color, oriColor)
            doFill(_image, row, col+1, "left", color, oriColor)
            doFill(_image, row+1, col, "up", color, oriColor)
        }
        if d == "down" {
            doFill(_image, row, col-1, "right", color, oriColor)
            doFill(_image, row, col+1, "left", color, oriColor)
            doFill(_image, row-1, col, "down", color, oriColor)
        }
        if d == "left" {
            doFill(_image, row+1, col, "up", color, oriColor)
            doFill(_image, row-1, col, "down", color, oriColor)
            doFill(_image, row, col+1, "left", color, oriColor)
        }
        if d == "right" {
            doFill(_image, row+1, col, "up", color, oriColor)
            doFill(_image, row-1, col, "down", color, oriColor)
            doFill(_image, row, col-1, "right", color, oriColor)
        }
    }

func floodFill(image [][]int, sr int, sc int, color int) [][]int {
    if image[sr][sc] == color {
        return image
    }
    oriColor := image[sr][sc]

    image[sr][sc] = color
    doFill(&image, sr-1, sc, "down", color, oriColor)
    doFill(&image, sr+1, sc, "up", color, oriColor)
    doFill(&image, sr, sc-1, "right", color, oriColor)
    doFill(&image, sr, sc+1, "left", color, oriColor)
    return image
}
```

```go
func doFill(_image *[][]int, row int, col int, color int, oriColor int) {
	if row < 0 || row >= len(*_image) || col < 0 || col >= len((*_image)[0]) || (*_image)[row][col] != oriColor {
		return
	}
	(*_image)[row][col] = color
	doFill(_image, row, col-1, color, oriColor)
	doFill(_image, row, col+1, color, oriColor)
	doFill(_image, row+1, col, color, oriColor)
	doFill(_image, row-1, col, color, oriColor)
}

func floodFill(image [][]int, sr int, sc int, color int) [][]int {
	if image[sr][sc] == color {
		return image
	}
	oriColor := image[sr][sc]
	doFill(&image, sr, sc, color, oriColor)
	return image
}

```

```go
func doFill(_image [][]int, row int, col int, color int, oriColor int) {
	if row < 0 || row >= len(_image) || col < 0 || col >= len(_image[0]) || _image[row][col] != oriColor {
		return
	}
	_image[row][col] = color
	doFill(_image, row, col-1, color, oriColor)
	doFill(_image, row, col+1, color, oriColor)
	doFill(_image, row+1, col, color, oriColor)
	doFill(_image, row-1, col, color, oriColor)
}

func floodFill(image [][]int, sr int, sc int, color int) [][]int {
	if image[sr][sc] == color {
		return image
	}
	oriColor := image[sr][sc]
	doFill(image, sr, sc, color, oriColor)
	return image
}

```



### 861. Score After Flipping Matrix

```python
class Solution:
    def matrixScore(self, A: List[List[int]]) -> int:
        rows, cols = len(A), len(A[0])
        res = (1 << cols - 1) * rows
        for col in range(1, cols):
            num_same_as_first = sum(A[row][col] == A[row][0] for row in range(rows))
            res += max(num_same_as_first, rows-num_same_as_first) * (1 << cols - 1 - col)
        return res
```

第4行：先默认将第一列置1，计算结果但不操作矩阵。

第6行：之后从第二列开始遍历每一列，计算此列元素中该元素与所在行第一个数相同的个数，`max(num_same_as_first, rows-num_same_as_first)` 是该列可变1的最大个数。此处不用修改第一列就可以判断是因为计算的是和该元素所在行第一个数相同的个数，所以第一个数变为1后该元素也会变为1，会和第一个元素保持一致。

第7行，之后便可以按照每列最多1的个数根据该列index计算累加结果。



### 867. Transpose Matrix

```python
class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        rows, cols = len(matrix), len(matrix[0])
        return [[matrix[row][col] for row in range(rows)] for col in range(cols)]
```



### 944. Delete Columns to Make Sorted

```python
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        res = 0

        if len(strs) == 1: return 0

        for i in range(len(strs[0])):
            for j in range(1, len(strs)):
                if strs[j][i] < strs[j-1][i]:
                    res += 1
                    break

        return res        
```



### 1020. Number of Enclaves

```python
class Solution:
    def numEnclaves(self, A: List[List[int]]) -> int:
        
        rows, cols = len(A), len(A[0])
        
        def fill(row, col):
            if row < 0 or col < 0 or row >= rows or col >= cols or A[row][col] == 0:
                return 0
            A[row][col] = 0
            return 1 + fill(row+1, col) + fill(row-1, col) + fill(row, col+1) + fill(row, col-1)
        
        for row in range(rows):
            for col in range(cols):
                if row == 0 or col == 0 or row == rows-1 or col == cols-1:
                    fill(row, col)
        
        res = 0
        
        for row in range(rows):
            for col in range(cols):
                if A[row][col] == 1:
                    res += fill(row, col)
        
        return res
```

Similar as 1254, 只是最后在填充的时候记录数量，注意fill函数递归记述方法。



### 1091. Shortest Path in Binary Matrix

```python
# TLE
# dfs
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        res = sys.maxsize
        rows, cols = len(grid), len(grid[0])
        
        def dfs(row, col, path_l):
            if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col] in [1, -1]:
                return
            nonlocal res
            if row == rows - 1 and col == cols - 1:
                res = min(res, path_l)
                return

            prev_v = grid[row][col]
            grid[row][col] = -1
            dfs(row-1, col-1, path_l+1)
            dfs(row-1, col, path_l+1)
            dfs(row-1, col+1, path_l+1)
            dfs(row, col+1, path_l+1)
            dfs(row+1, col+1, path_l+1)
            dfs(row+1, col, path_l+1)
            dfs(row+1, col-1, path_l+1)
            dfs(row, col-1, path_l+1)
            grid[row][col] = prev_v
        
        dfs(0, 0, 1)
        
        return res if res != sys.maxsize else -1
```

```python
# Accept
# bfs
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        
        q = [(0, 0, 1)]
        
        for row, col, d in q:
            if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col]:
                continue
            if row == rows - 1 and col == cols - 1:
                return d
            grid[row][col] = 1
            for row_offset, col_offset in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
                new_row, new_col = row + row_offset, col + col_offset
                q.append([new_row, new_col, d+1])
        
        return -1
```



### 1254. Number of Closed Islands

```python
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        
        rows, cols = len(grid), len(grid[0])
        
        def fill(row, col):
            if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col] == 1:
                return
            grid[row][col] = 1
            fill(row+1, col)
            fill(row-1, col)
            fill(row, col-1)
            fill(row, col+1)
        
        
        for row in range(rows):
            for col in range(cols):
                if row == 0 or col == 0 or row == len(grid) - 1 or col == len (grid[0]) - 1:
                    fill(row, col)
        
        res = 0
        
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 0:
                    res += 1
                    fill(row, col)
        
        return res
```

先从四周的0开始填充1，遇到1停止。填充完之后会剩下中间的岛，再遍历一次，遇到岛(0)岛数量+1并将其填充。



### 1260. Shift 2D Grid

```python
class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        rows, cols = len(grid), len(grid[0])
        k %= rows * cols
        
        def move_one_step(grid):
            last_one = grid[-1][-1]
            for row in range(rows):
                this_one = grid[row][-1]
                for col in range(cols-2, -1, -1):
                    grid[row][col+1] = grid[row][col]
                grid[row][0] = last_one
                last_one = this_one
        for _ in range(k):
            move_one_step(grid)
        
        return grid
```



### 1329. Sort the Matrix Diagonally

```python
class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        rows, cols = len(mat), len(mat[0])
        
        for col in range(cols-1):
            temp = []
            row_, col_ = 0, col
            while col_ < cols and row_ < rows:
                temp.append(mat[row_][col_])
                row_ += 1
                col_ += 1
            temp.sort(reverse=True)
            row_, col_ = 0, col
            while col_ < cols and row_ < rows:
                mat[row_][col_] = temp.pop()
                row_ += 1
                col_ += 1
        
        
        for row in range(1, rows-1):
            temp = []
            row_, col_ = row, 0
            while row_ < rows and col_ < cols:
                temp.append(mat[row_][col_])
                row_ += 1
                col_ += 1
            temp.sort(reverse=True)
            row_, col_ = row, 0
            while row_ < rows and col_ < cols:
                mat[row_][col_] = temp.pop()
                row_ += 1
                col_ += 1
        
        return mat
```



### 1337. The K Weakest Rows in a Matrix

```python
class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        temp = []
        for index, row in enumerate(mat):
            num_of_1 = 0
            for each in row:
                if each == 0:
                    break
                num_of_1 += 1
            temp.append((num_of_1, index))
        temp.sort()
        return [each[1] for each in temp[:k]]
```

```python
# 二分找1的个数
class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        
        def numOfOne(row):
            left, right = 0, len(row) - 1
            while left < right:
                mid = left + (right - left) // 2
                if row[mid] == 1:
                    if mid != len(row)-1 and row[mid+1] == 1:
                        left = mid + 1
                    else:
                        return mid + 1
                else:
                    right = mid - 1
            return left + 1 if row[left] == 1 else 0
        
        temp = []
        for index, row in enumerate(mat):
            temp.append((numOfOne(row), index))
        temp.sort()
        return [each[1] for each in temp[:k]]
```

```python
class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        # [0,0,0,0]
        # [1,1,1,1]
        # [1,1,0,0]
        # 注意这个二分法，虽然 right = mid，但是right也在变动，不会产生死循环
        def numOfOne(row):
            left, right = 0, len(row)
            while left < right:
                mid = left + (right - left) // 2
                if row[mid] == 1:
                    left = mid + 1
                else:
                    right = mid
            return left
        
        temp = []
        for index, row in enumerate(mat):
            temp.append((numOfOne(row), index))
        temp.sort()
        return [each[1] for each in temp[:k]]
```



### 1351. Count Negative Numbers in a Sorted Matrix

```python
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        r, c, ans = rows-1, 0, 0
        
        while r >= 0 and c < cols:
            if grid[r][c] < 0:
                ans += cols - c
                r -= 1
            else:
                c += 1
        return ans
```

```python
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        res = 0
        rows, cols = len(grid), len(grid[0])

        for row in range(rows-1, -1, -1):
            if grid[row][cols-1] >= 0: break
            res += 1
            for col in range(cols-2, -1, -1):
                if grid[row][col] >= 0: break
                res += 1
        
        return res
```



### 1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts

```python
class Solution:
    def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
        horizontalCuts.sort()
        verticalCuts.sort()
        
        max_h = max(horizontalCuts[0], h - horizontalCuts[-1])
        max_v = max(verticalCuts[0], w - verticalCuts[-1])
        
        for i in range(1, len(horizontalCuts)):
            max_h = max(max_h, horizontalCuts[i] - horizontalCuts[i-1])
        for i in range(1, len(verticalCuts)):
            max_v = max(max_v, verticalCuts[i] - verticalCuts[i-1])
        
        return max_h * max_v % (10**9 + 7)
```

求横切的最大长度，在求纵切最大长度，乘积为最大面积



### 1905. Count Sub Islands

```python
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        
        res = 0
        
        rows, cols = len(grid2), len(grid2[0])
        
        def dfs(row, col):
            if row >= 0 and row <= rows-1 and col >=0 and col <= cols-1 and grid2[row][col] == 1:
                grid2[row][col] = 0
                dfs(row+1, col)
                dfs(row-1, col)
                dfs(row, col+1)
                dfs(row, col-1)
        
        for row in range(rows):
            for col in range(cols):
                if grid1[row][col] == 0 and grid2[row][col] == 1:
                    dfs(row, col)
        
        for row in range(rows):
            for col in range(cols):
                if grid2[row][col] == 1:
                    dfs(row, col)
                    res += 1
        return res
```

首先dfs填充grid2是1而grid1中是0的格子，因为这样的岛屿不可能是grid1的子集，之后数grid2中岛屿数量就可以了
