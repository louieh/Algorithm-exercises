## Leetcode - backtracking

[toc]

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
            print(tempList)
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



### 77. Combinations

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        
        ans = []
        nums = [i for i in range(1, n+1)]
        def backtrack(tempList, start):
            if len(tempList) == k:
                ans.append(tempList)
            elif start == len(nums):
                return
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

