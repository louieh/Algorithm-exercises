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

### **Rotate matrix among Diagnals** 

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

