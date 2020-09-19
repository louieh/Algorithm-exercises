## LeetCode - String

[toc]

### 3. Longest Substring Without Repeating Characters

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        if s == " ":
            return 1
        
        if len(s) == 1:
            return 1
        
        def helper(s, max_length):
            i = 0
            temp_list = []
            while i <= len(s)-1:
                if s[i] not in temp_list:
                    temp_list.append(s[i])
                    i += 1
                    if len(temp_list) > max_length:
                        max_length = len(temp_list)
                else:
                    if len(temp_list) > max_length:
                        max_length = len(temp_list)
                    return helper(s[s.index(s[i])+1:], max_length)
            return max_length
        
        return helper(s, 0)
```

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = 0
        temp = set()
        ans = i = j = 0
        
        while i < len(s) and j < len(s):
            if s[j] not in temp:
                temp.add(s[j])
                j += 1
                ans = max(ans, j-i)
            else:
                temp.remove(s[i])
                i += 1
        return ans
```

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        temp = []
        ans = 0
        for c in s:
            if c in temp:
                temp = temp[temp.index(c)+1:]
            temp.append(c)
            ans = max(ans, len(temp))
        return ans
```



### 14. Longest Common Prefix

```python
class Solution:
    def longestCommonPrefix(self, strs):
        if len(strs) == 0:
            return ""
        min_length = len(strs[0])
        min_str = strs[0]
        for each_str in strs[1:]:
            if len(each_str) < min_length:
                min_length = len(each_str)
                min_str = each_str

        for i in range(0, len(min_str)):
            for each in strs:
                if min_str[i] != each[i]:
                    if len(min_str[:i]) > 0:
                        return min_str[:i]
                    else:
                        return ""
        return min_str
```

```python
# 10/3/2019
class Solution:
     def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs or not strs[0]:
            return ""
        
        min_length = len(strs[0])
        for each in strs:
            if len(each) < min_length:
                min_length = len(each)
        
        if min_length == 0:
            return ""
        
        for i in range(min_length):
            temp = strs[0][i]
            for each in strs:
                if each[i] != temp:
                    return strs[0][:i]
        return strs[0][:i+1]
```



### 53. Maximum Subarray

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        local_min = -2147483648
        global_min = -2147483648
        
        for each in nums:
            local_min = max(each, each+local_min)
            global_min = max(local_min, global_min)
        
        return global_min
```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        global_min = nums[0]
        
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i]+nums[i-1])
            global_min = max(nums[i], global_min)
        return global_min
```



### 139. Word Break

```python
# 超时
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        self.ans = False
        
        def dfs(s):
            if not s or self.ans:
                return
            for word in wordDict:
                if self.ans:
                    return
                if not s.startswith(word):
                    continue
                if len(word) == len(s):
                    self.ans = True
                    return
                else:
                    dfs(s[len(word):])
        dfs(s)
        return self.ans
```

```python
# DP
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        ans = [False] * (len(s) + 1)
        ans[0] = True
        
        for i in range(1, len(s)+1):
            for j in range(i):
                if ans[j] and s[j:i] in wordDict:
                    ans[i] = True
                    break
        return ans[len(s)]
```

https://leetcode.com/problems/word-break/discuss/43790/Java-implementation-using-DP-in-two-ways



### 140. Word Break II

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        
        def dfs(s, memo):
            if s in memo:
                return memo[s]
            if not s:
                return []
            
            res = []
            for word in wordDict:
                if not s.startswith(word):
                    continue
                if len(s) == len(word):
                    res.append(word)
                else:
                    rest = dfs(s[len(word):], memo)
                    for each in rest:
                        item = word + " " + each
                        res.append(item)
            
            memo[s] = res
            return res
        return dfs(s, {})
```
https://leetcode.com/problems/word-break-ii/discuss/44311/Python-easy-to-understand-solution
还没完全懂



### 238. Product of Array Except Self

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        left_accu = [1]
        right_accu = [1]
        
        for i in range(len(nums)-1):
            left_accu.append(nums[i] * left_accu[i])
        i = len(nums)-1
        while i > 0:
            right_accu.insert(0, nums[i] * right_accu[0])
            i -= 1
        
        ans = []
        for i in range(len(nums)):
            ans.append(left_accu[i] * right_accu[i])
        
        return ans
```

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        left_accu = [1]
        
        for i in range(len(nums)-1):
            left_accu.append(left_accu[i] * nums[i])
        
        ans = [1] * len(nums)
        temp = 1
        for i in range(len(nums)-1, -1, -1):
            ans[i] = temp * left_accu[i]
            temp *= nums[i]
        return ans
```

合并后两个for循环。



### 424. Longest Repeating Character Replacement

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        start = maxSame = ans = 0
        count = dict()
        
        for end in range(len(s)):
            count[s[end]] = count.get(s[end], 0) + 1
            maxSame = max(maxSame, count[s[end]])
            if end - start + 1 - maxSame > k:
                count[s[start]] -= 1
                start += 1
            ans = max(ans, end-start+1)
        return ans
```



### 678. Valid Parenthesis String

```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        minc, maxc = 0, 0
        for each in s:
            if each == "(":
                minc += 1
                maxc += 1
            if each == ")":
                minc -= 1
                maxc -= 1
            if each == "*":
                maxc += 1
                minc -= 1
            if maxc < 0:
                return False
            minc = max(minc, 0)
        return minc == 0
```
记录待匹配的 ( 的数量
minc 将 * 当做 ) 如果最后 minc > 0 说明把 * 当做 ) 的情况下仍有 ( 剩下
maxc 将 * 当做 ( 如果 maxc < 0 说明把 * 当做 ( 的情况下仍有过多的 ) 出现



### 763. Partition Labels

```python
class Solution:
    def tool(self, S):
        max_index = {c: i for i, c in enumerate(S)}
        max_now = 0
        i = 0
        for each in S:
            if max_index[S[i]] >= max_now:
                max_now = max_index[S[i]]
            if i == max_now:
                return S[:i+1], S[i+1:]
            i += 1
                
            
    
    def partitionLabels(self, S: str) -> List[int]:
        if not S:
            return []
        ans = []
        while S:
            S_, S = self.tool(S)
            ans.append(len(S_))
        return ans
```

```python
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        if not S:
            return []
        
        ans = []
        
        temp_dict = {}
        i = len(S) - 1
        while i >= 0:
            if S[i] not in temp_dict:
                temp_dict[S[i]] = i
            i -= 1
        
        def far(start_index, end_index, far_index):
            if end_index == len(S) - 1:
                ans.append(end_index)
                return
            if start_index == end_index:
                ans.append(far_index)
                far(end_index + 1, temp_dict[S[end_index + 1]], temp_dict[S[end_index + 1]])
            else:
                changed = False
                for i in range(start_index + 1, end_index):
                    if temp_dict[S[i]] > far_index:
                        far_index = temp_dict[S[i]]
                        changed = True
                if changed:
                    far(temp_dict[S[start_index]], far_index, far_index)
                else:
                    ans.append(far_index)
                    far(end_index + 1, temp_dict[S[end_index + 1]], temp_dict[S[end_index + 1]])
        
        far(0, temp_dict[S[0]], temp_dict[S[0]])
        
        i = len(ans) - 1
        while i >= 0:
            if i == 0:
                ans[i] += 1
            else:
                ans[i] = ans[i] - ans[i-1]
            i -= 1
        return ans
```

```python
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        last = {c: i for i, c in enumerate(S)}
        j = anchor = 0
        ans = []
        for i, c in enumerate(S):
            j = max(j, last[c])
            if i == j:
                ans.append(i - anchor + 1)
                anchor = i + 1
            
        return ans
```

