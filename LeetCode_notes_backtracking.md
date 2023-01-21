## Leetcode - backtracking

[toc]

### 17. Letter Combinations of a Phone Number

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits: return []
        res = []
        digits_dict = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }
        
        def backtrack(i, curStr):
            if len(curStr) == len(digits):
                res.append(curStr)
                return
            for c in digits_dict[digits[i]]:
                backtrack(i+1, curStr+c)
        
        backtrack(0, "")
        return res
```



### 22. Generate Parentheses

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        
        ans = []
        
        def backtrack(tempString, op, cl):
            if len(tempString) == n * 2:
                ans.append(tempString)
                return
            if op < n:
                backtrack(tempString+'(', op+1, cl)
            if cl < op:
                backtrack(tempString+')', op, cl+1)
        
        backtrack("", 0, 0)
        return ans
```



### 39. Combination Sum

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(tempList, remain, start):
            if remain < 0:
                return
            elif remain == 0:
                ans.append(tempList)
            else:
                for i in range(start, len(candidates)):
                    tempList.append(candidates[i])
                    backtrack(tempList.copy(), remain-candidates[i], i)
                    tempList.pop()
        
        ans = []
        backtrack([], target, 0)
        return ans
```

![Scannable文档创建于2020年8月13日 00_06_59](/Users/hanluyi/Downloads/other_Python_ex/leetcode/Scannable文档创建于2020年8月13日 00_06_59.png)

https://leetcode.com/problems/combination-sum/discuss/16502/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partitioning)



### 40. Combination Sum II

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(tempList, remain, start):
            if remain < 0:
                return
            elif remain == 0:
                ans.append(tempList)
            else:
                for i in range(start, len(candidates)):
                    if i > start and candidates[i] == candidates[i-1]:
                        continue
                    tempList.append(candidates[i])
                    backtrack(tempList.copy(), remain-candidates[i], i+1)
                    tempList.pop()
        
        ans = []
        candidates.sort()
        backtrack([], target, 0)
        return ans
```



### 46. Permutations 

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return []
        
        ans = []
        
        def permute_tool(nums, left, right):
            if left == right:
                ans.append(nums[::])
            else:
                for i in range(left, right+1):
                    nums[i], nums[left] = nums[left], nums[i]
                    permute_tool(nums, left+1, right)
                    nums[i], nums[left] = nums[left], nums[i]
        
        permute_tool(nums, 0, len(nums)-1)
        return ans
```

注意最后append的时候要重新复制一遍数组，改变数据地址。

https://blog.csdn.net/zhoufen12345/article/details/53560099

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        
        ans = []
        
        def backtrack(tempList):
            if len(tempList) == len(nums):
                ans.append(tempList)
            else:
                for num in nums:
                    if num in tempList:
                        continue
                    tempList.append(num)
                    backtrack(tempList.copy())
                    tempList.pop()
        
        backtrack([])
        return ans
```



### 47. Permutations II

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        
        ans = []
        nums.sort()
        def backtrack(tempList, used):
            if len(tempList) == len(nums):
                ans.append(tempList)
            else:
                for i, num in enumerate(nums):
                    if used[i] or i > 0 and nums[i] == nums[i-1] and not used[i-1]: continue
                    tempList.append(num)
                    used[i] = True
                    backtrack(tempList.copy(), used.copy())
                    used[i] = False
                    tempList.pop()
        
        backtrack([], [False]*len(nums))
        return ans
```

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        
        def backtrack(tempList, counter):
            if len(tempList) == len(nums):
                res.append(tempList)
                return
            
            for num in counter:
                if counter[num] > 0:
                    counter[num] -= 1
                    tempList.append(num)
                    backtrack(tempList.copy(), counter)
                    tempList.pop()
                    counter[num] += 1
        backtrack([], Counter(nums))
        return res
```



### 51. N-Queens

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        Q = [['.'] * n for _ in range(n)]
        def helper(row):
            if row == n:
                res.append(["".join(each) for each in Q])
                return
            for col in range(n):
                if valid(row, col):
                    Q[row][col] = "Q"
                    helper(row+1)
                    Q[row][col] = "."
        
        def valid(row, col):
            # col
            for i in range(row):
                if Q[i][col] == "Q":
                    return False
            # 90 C
            i, j = row-1, col-1
            while i >= 0 and j >= 0:
                if Q[i][j] == "Q":
                    return False
                i -= 1
                j -= 1
            
            # 135 C
            i, j = row-1, col+1
            while i >= 0 and j < n:
                if Q[i][j] == "Q":
                    return False
                i -= 1
                j += 1
            return True
        
        helper(0)
        return res
```

https://leetcode.com/problems/n-queens/discuss/19808/Accepted-4ms-c%2B%2B-solution-use-backtracking-and-bitmask-easy-understand.



### 77. Combinations

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        
        ans = []
        nums = [i for i in range(1, n+1)]
        def backtrack(tempList, start):
            if len(tempList) == k:
                ans.append(tempList)
            else:
                for i in range(start, len(nums)):
                    tempList.append(nums[i])
                    backtrack(tempList.copy(), i+1)
                    tempList.pop()
        backtrack([], 0)
        return ans
```



### 78. Subsets

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = [[]]
        
        for num in nums:
            output += [curr + [num] for curr in output]
        
        return output
```

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nth_bit = 1 << len(nums)
        ans = []
        for i in range(2**len(nums)):
            bitmask = bin(i|nth_bit)[3:]
            ans.append([nums[j] for j in range(len(nums)) if bitmask[j] == '1'])
        return ans
```

熟记：获取长度为n的所有可能二进制数：

```python
# | 按位或
nth_bit = 1 << n
for i in range(2**n):
    # generate bitmask, from 0..00 to 1..11
    bitmask = bin(i | nth_bit)[3:]
```

```python
for i in range(2**n, 2**(n + 1)):
    # generate bitmask, from 0..00 to 1..11
    bitmask = bin(i)[3:]
```

```python
# backtrack 模版
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        
        ans = []
        
        def backtrack(tempList, start):
            ans.append(tempList)
            for i in range(start, len(nums)):
                tempList.append(nums[i])
                backtrack(tempList.copy(), i + 1)
                tempList.pop()
        
        backtrack([], 0)
        return ans
```



### 90. Subsets II

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        
        ans = []
        nums.sort()
        def backtrack(tempList, start):
            ans.append(tempList)
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i-1]: continue
                tempList.append(nums[i])
                backtrack(tempList.copy(), i + 1)
                tempList.pop()
        
        backtrack([], 0)
        return ans
```

![Scannable文档创建于2020年8月17日 02_39_51](/Users/hanluyi/Downloads/other_Python_ex/leetcode/Scannable文档创建于2020年8月17日 02_39_51.png)



### 131. Palindrome Partitioning

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        
        ans = []
        
        def backtrack(tempList, start):
            if start == len(s):
                ans.append(tempList)
            else:
                for i in range(start, len(s)):
                    if isPalindrome(start, i):
                        tempList.append(s[start:i+1])
                        backtrack(tempList.copy(), i+1)
                        tempList.pop()
        
        def isPalindrome(left, right):
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        backtrack([], 0)
        return ans
```



### 216. Combination Sum III

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        nums = [i for i in range(1, 10)]
        ans = []
        def backtrack(tempList, remain, start):
            if len(tempList) == k:
                if remain == 0:
                    ans.append(tempList)
                return
            else:
                for i in range(start, len(nums)):
                    tempList.append(nums[i])
                    backtrack(tempList.copy(), remain-nums[i], i+1)
                    tempList.pop()
        
        backtrack([], n, 0)
        return ans
```



### 491. Non-decreasing Subsequences

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        
        res = []

        def backtrack(start, temp_list):
            if len(temp_list) >= 2:
                res.append(temp_list.copy())
            used = set()
            for i in range(start, len(nums)):
                if nums[i] in used: continue
                if temp_list and nums[i] < temp_list[-1]: continue
                used.add(nums[i])
                temp_list.append(nums[i])
                backtrack(i+1, temp_list)
                temp_list.pop()
        
        backtrack(0, [])
        return res
```

跟 Permutations II 类似



### 949. Largest Time for Given Digits

```python
class Solution:
    def largestTimeFromDigits(self, A: List[int]) -> str:
        ans = ""
        self.hour_max, self.mini_max = -1, -1
        
        A.sort()
        def backtrack(tempList, used):
            if len(tempList) == len(A):
                hour_temp = tempList[0]*10+tempList[1]
                mini_temp = tempList[2]*10+tempList[3]
                if hour_temp <= 23 and mini_temp <= 59:
                    if hour_temp > self.hour_max:
                        self.hour_max = hour_temp
                        self.mini_max = mini_temp
                    elif hour_temp == self.hour_max and mini_temp > self.mini_max:
                        self.mini_max = mini_temp
                
            else:
                for i, num in enumerate(A):
                    if used[i] or i > 0 and A[i] == A[i-1] and not used[i-1]:
                        continue
                    used[i] = True
                    tempList.append(num)
                    backtrack(tempList.copy(), used.copy())
                    used[i] = False
                    tempList.pop()
        backtrack([], [False]*len(A))
        
        if self.hour_max == -1 and self.mini_max == -1:
            return ans
        if len(str(self.hour_max)) == 1:
            self.hour_max = "0" + str(self.hour_max)
        if len(str(self.mini_max)) == 1:
            self.mini_max = "0" + str(self.mini_max)
        return str(self.hour_max) + ":" + str(self.mini_max)
```



### 980. Unique Paths III

```python
class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        
        ans = remain = init_row = init_col = 0
        
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] >= 0:
                    remain += 1
                if grid[row][col] == 1:
                    init_row = row
                    init_col = col
        
        
        def backtrack(row, col, remain):
            nonlocal ans
            if grid[row][col] == 2 and remain == 1:
                ans += 1
                return
            
            ori = grid[row][col]
            grid[row][col] = -2
            remain -= 1
            
            for row_, col_ in [(0,1), (0,-1), (1,0), (-1,0)]:
                next_row = row + row_
                next_col = col + col_
                if not (0 <= next_row <= len(grid)-1 and 0 <= next_col <= len(grid[0])-1) or grid[next_row][next_col] < 0:
                    continue
                backtrack(next_row, next_col, remain)
            
            grid[row][col] = ori
        
        backtrack(init_row, init_col, remain)
        return ans
```

https://leetcode.com/problems/unique-paths-iii/solution/



### 1079. Letter Tile Possibilities

```python
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        if len(tiles) == 1:
            return 1
        
        count = [0] * 26
        for each in tiles:
            count[ord(each)-65] += 1
        
        def backtrack(count):
            ans = 0
            for i in range(26):
                if count[i] == 0: continue
                ans += 1
                count[i] -= 1
                ans += backtrack(count)
                count[i] += 1
            return ans
        return backtrack(count)
```



### 1219. Path with Maximum Gold

```python
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        
        
        self.res = self.cur = 0
        
        def dfs(row, col):
            ori = grid[row][col]
            self.cur += grid[row][col]
            grid[row][col] = -1
            
            for _row, _col in [(0,1), (0,-1), (1,0), (-1,0)]:
                next_row = row + _row
                next_col = col + _col
                if not (0 <= next_row <= len(grid)-1 and 0 <= next_col <= len(grid[0])-1) or grid[next_row][next_col] == -1 or grid[next_row][next_col] == 0:
                    continue
                dfs(next_row, next_col)
            self.res = max(self.res, self.cur)
            self.cur -= ori
            grid[row][col] = ori
        
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] > 0:
                    dfs(row, col)
                    self.cur = 0
        
        return self.res
```



### 1286. Iterator for Combination

```python
class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        nth_bit = 1 << len(characters)
        self.bitmask = []
        self.index = -1
        self.characters = characters
        for i in range(2**len(characters)):
            bitmask = bin(i|nth_bit)[3:]
            num_1 = 0
            for each in bitmask:
                if each == '1':
                    num_1 += 1
                    if num_1 > combinationLength:
                        break
            if num_1 == combinationLength:
                self.bitmask.append(bitmask)
        self.bitmask = self.bitmask[::-1]
        

    def next(self) -> str:
        self.index += 1
        bitmask = self.bitmask[self.index]
        temp = [self.characters[index] for index, bit in enumerate(bitmask) if bit == '1']
        return ("").join(temp)
    
    def hasNext(self) -> bool:
        return self.index + 1 < len(self.bitmask)
```



### 1641. Count Sorted Vowel Strings

```python
# TLE
class Solution:
    def countVowelStrings(self, n: int) -> int:
        
        nums = ['a', 'e', 'i', 'o', 'u']
        nums_dict = {
            'a': 0,
            "e": 1,
            "i": 2,
            "o": 3,
            "u": 4
        }
        res = []
        
        def backtrack(tempList, start):
            if len(tempList) == n:
                res.append(tempList)
                return
            for i in range(start, len(nums)):
                tempList.append(nums[i])
                backtrack(tempList.copy(), nums_dict[nums[i]])
                tempList.pop()
        
        backtrack([], 0)
        return len(res)
```

