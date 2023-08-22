## LeetCode - String

[toc]

https://leetcode.com/discuss/interview-question/2001789/collections-of-important-string-questions-pattern

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

```python
# 这个方法好
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start = res = 0
        temp = dict()
        for i, v in enumerate(s):
            if v in temp and temp[v] >= start:
                start = temp[v] + 1
            else:
                res = max(res, i-start+1)
            temp[v] = i
        return res
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        temp_dict = {}
        res = start = 0
        for i, c in enumerate(s):
            if c in temp_dict and temp_dict[c] >= start:
                res = max(res, i-start)
                start = temp_dict[c] + 1
            temp_dict[c] = i
        
        return max(res, len(s)-1-start+1)
```

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s: return 0
        left = 0
        temp = {}
        res = 0
        for right, val in enumerate(s):
            if val in temp:
                left_old = left
                left = temp[val] + 1
                for i in range(left_old, left):
                    temp.pop(s[i])
            temp[val] = right
            res = max(res, right-left+1)
        return res
```

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        res = 0
        cur_set = set()
        start = 0
        for index, c in enumerate(s):
            if c not in cur_set:
                cur_set.add(c)
                res = max(res, len(cur_set))
            else:
                for i in range(start, index):
                    if s[i] != c:
                        cur_set.remove(s[i])
                    else:
                        start = i + 1
                        break
        return res
```

```python
# 这个方法好 two points same as problem 1695
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        res = cur_len = 0
        left = right = 0
        S = set()
        
        while right < len(s):
            if s[right] not in S:
                cur_len += 1
                S.add(s[right])
                right += 1
                res = max(res, cur_len)
            else:
                cur_len -= 1
                S.remove(s[left])
                left += 1
        return res
```



### 5. Longest Palindromic Substring

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) < 2: return s
        maxLength = 0
        res = None
        def isPalindrome(left, right):
            nonlocal res
            nonlocal maxLength
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            if right-left+1 > maxLength:
                maxLength = right - left + 1
                res = s[left+1:right]
        
        for i in range(len(s)):
            isPalindrome(i, i)
            isPalindrome(i, i+1)
        
        return res
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



### 139. Word Break

```python
# BFS - AC
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        words = set(wordDict)
        queue = deque([0])
        seen = set()
        
        while queue:
            start = queue.popleft()
            if start == len(s):
                return True
            
            for end in range(start + 1, len(s) + 1):
                if end in seen:
                    continue
                
                if s[start:end] in words:
                    queue.append(end)
                    seen.add(end)
                
        return False
```

https://leetcode.com/problems/word-break/editorial/

将 s 中每个字符看作图中点，wordDict 中字符串看左边，连通的两个点是字符串首尾字符，起始点是 s 首字母，求能否到达终点也就是 s 尾字母。之所以是 BFS 是因为在一次遍历中将所有当前点的可到达点均加入队列。for 循环的作用是判断当前节点（start）的所有可到达点，加入队列。`start == len(s)` 作用是判断当前节点（start）是否到达了终点。

```python
# DFS - 超时
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

该方法之所以是 DFS 是因为较上面方法，每遇到一个可到达点就从该节点进入递归，从该节点重新遍历所有可到达点。

例如：`s = "aaaaaaa" wordDict = ["aaa","aaaa"]`

第一次循环会发现两个可达点，一个从index=3处截断，一个从index=4处截断，因为是 DFS 所以在第一处可达点直接进入下一次递归，第二次递归中 s 变为 `aaaa` ，依旧判断可达点，还是两个，一个是index=3，一个从index=4，在index=3处截断后 s 变为 `a` 进入下一次递归，没有找到任何可达点而返回，在index=4处满足了 `len(word) == len(s)` 条件而返回。此时递归回到第一次index=4截断处进入下一次递归...

```python
# DP - AC
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

初始化数组 ans 表示每个点是否可达，在遍历每个节点时，回顾之前所有可达点看该可达点到当前节点的字符串`s[j:i]` 是否在 wordDict 中，也就是看当前节点是否可达，最后返回数组最有一个点是否可达即可。



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



### 389. Find the Difference

```python
class Solution:
    def findTheDifference(self, s, t):
        nums = list(s + t)
        nums.sort()
        for i in range(len(nums)):
            if i % 2 == 0:  # even
                if i + 1 == len(nums):
                    return nums[i]
                if nums[i] != nums[i + 1]:
                    return nums[i]
            else:
                if nums[i] != nums[i - 1]:
                    return nums[i]
```

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        s_counter = Counter(s)
        t_counter = Counter(t)
        for c, num in t_counter.items():
            if c not in s_counter or num != s_counter[c]:
                return c
```



### 434. Number of Segments in a String

```python
class Solution:
    def countSegments(self, s: str) -> int:
        if not s: return 0
        res = 0
        temp = False if s[0] == ' ' else True
        for i in range(1, len(s)):
            if s[i] == ' ' and temp is True:
                res += 1
                temp = False
                continue
            if s[i] != ' ' and temp is False:
                temp = True
        return res if temp is False else res + 1
```

Similar as 1446



### 459. Repeated Substring Pattern

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return s in (s + s)[1:-1]
```

https://leetcode.com/problems/repeated-substring-pattern/editorial/

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n  = len(s)

        for i in range(1, n // 2 + 1):
            if n % i == 0:
                if s[:i] * (n // i) == s:
                    return True
        return False
```

第二个方法容易理解一下，我们知道如果要满足题目要求的子串的长度肯定是总字符串长度的约数，且假设子串是a 则满足 `s == a * (len(s) // len(a))`， 并且子串是 s 的前缀，所以遍历所有 s 长度的约数判断能不能组合成原字符串。注意除自己之外的最大约数是 `n // 2`



### 520. Detect Capital

```python
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        return word.islower() or word.isupper() or word[0].isupper() and word[1:].islower()
```



### 647. Palindromic Substrings

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        
        res = 0
        
        def helper(left, right):
            nonlocal res
            while left >= 0 and right <= len(s) - 1 and s[left] == s[right]:
                res += 1
                left -= 1
                right += 1
        
        for i in range(len(s)):
            helper(i, i)
            helper(i, i+1)
        
        return res
```

https://leetcode.com/problems/palindromic-substrings/discuss/105688/Very-Simple-Java-Solution-with-Detail-Explanation

A very easy explanation with an example
Lets take a string "aabaa"

**Step 1:** Start a for loop to point at every single character from where we will trace the palindrome string.
checkPalindrome(s,i,i); //To check the palindrome of odd length palindromic sub-string
checkPalindrome(s,i,i+1); //To check the palindrome of even length palindromic sub-string

**Step 2:** From each character of the string, we will keep checking if the sub-string is a palindrome and increment the palindrome count. To check the palindrome, keep checking the left and right of the character if it is same or not.

First loop:

![Alt text](https://discuss.leetcode.com/assets/uploads/files/1500788789821-300147d3-e98e-4977-83f1-9eb8213a485e-image.png)

Palindrome: a (Count=1)

![Alt text](https://discuss.leetcode.com/assets/uploads/files/1500788808273-fec1dec5-ab5f-44cf-8dbd-eb2780e8d65f-image.png)

Palindrome: aa (Count=2)



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



### 844. Backspace String Compare

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        
        def helper(text):
            res = []
            for c in text:
                if c != "#":
                    res.append(c)
                elif res:
                    res.pop()
            return "".join(res)
        
        return helper(s) == helper(t)
```

可以使用two pointer从末尾遍历两个字符串，遇到`#`则向前移动两格。



### 859. Buddy Strings

```python
class Solution:
    def buddyStrings(self, A: str, B: str) -> bool:
        if len(A) != len(B) or A == "" or B == "": return False
        if A == B: return len(A) > len(set(A))
        
        diff_index = [(a, b) for a, b in zip(A, B) if a != b]
        if len(diff_index) == 2 and diff_index[0] != diff_index[1][::-1] or len(diff_index) > 2 or len(diff_index) == 1:
            return False
        return True
```



### 916. Word Subsets

```python
# TLE
class Solution:
    def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:
        from collections import Counter
        B_counter_list = [Counter(each) for each in B]
        res = []
        for each in A:
            counterA = Counter(each)
            flag1 = True
            for counterB in B_counter_list:
                flag2 = True
                for k, v in counterB.items():
                    if k not in counterA or v > counterA.get(k):
                        flag2 = False
                        break
                if flag2 is False:
                    flag1 = False
                    break
            if flag1 is True:
                res.append(each)
        return res
```

```python
class Solution:
    def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:
        from collections import Counter
        counterB = dict()
        for each in B:
            for k,v in Counter(each).items():
                if k not in counterB:
                    counterB[k] = v
                else:
                    counterB[k] = max(counterB[k], v)
        res = []
        for each in A:
            counterA = Counter(each)
            flag1 = True
            for k, v in counterB.items():
                if k not in counterA or v > counterA.get(k):
                    flag1 = False
                    break
            if flag1 is True:
                res.append(each)
        return res
```

```python
class Solution:
    def wordSubsets(self, A: List[str], B: List[str]) -> List[str]:
        
        def counter(word):
            temp = [0] * 26
            for c in word:
                temp[ord(c) - ord('a')] += 1
            return temp
        
        counterB = [0] * 26
        for word in B:
            counterB = [max(each) for each in zip(counter(word), counterB)]
        
        res = []
        
        for each in A:
            if all(a >= b for a, b in zip(counter(each), counterB)):
                res.append(each)
        return res
```



### 1446. Consecutive Characters

```python
class Solution:
    def maxPower(self, s: str) -> int:
        if len(s) == 1: return 1
        cur_c = s[0]
        cur_num = max_num = 1
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                cur_num += 1
                max_num = max(max_num, cur_num)
            else:
                cur_c = s[i]
                cur_num = 1
        
        return max_num
```

```python
class Solution:
    def maxPower(self, s: str) -> int:
        res = 1
        if len(s) == 1: return res
        
        cur_c = s[0]
        cur_num = 1
        for c in s[1:]:
            if c == cur_c:
                cur_num += 1
            else:
                res = max(res, cur_num)
                cur_c = c
                cur_num = 1
                
        return max(res, cur_num)
```

Similar as 434



### 1573. Number of Ways to Split a String

```python
class Solution:
    def numWays(self, s: str) -> int:
        counter = collections.Counter(s)
        num1 = counter.get('1')
        if num1 is None: return (len(s)-2) * (len(s)-1) // 2 % (10 ** 9 + 7)
        if num1 % 3 != 0: return 0
        
        firstCut = secondCut = 0
        num1Temp = 0
        for each in s:
            if each == '1':
                num1Temp += 1
            if num1Temp == num1 // 3:
                firstCut += 1
            if num1Temp == num1 // 3 * 2:
                secondCut += 1
        return firstCut * secondCut % (10 ** 9 + 7)
```



### 2278. Percentage of Letter in String

```python
class Solution:
    def percentageLetter(self, s: str, letter: str) -> int:
        counter = Counter(s)
        if letter not in counter: return 0
        return int(counter.get(letter) / len(s) * 100)
```



### 2288. Apply Discount to Prices

```python
class Solution:
    def discountPrices(self, sentence: str, discount: int) -> str:
        
        sen_list = sentence.split(" ")
        
        def is_price(word):
            if not word.startswith("$"):
                return False, 0
            try:
                res = float(word[1:])
                return True, res
            except:
                return False, 0
        
        for i, word in enumerate(sen_list):
            flag, num = is_price(word)
            if flag:
                new_word = "$" + format((num - num * (discount / 100)), '.2f')
                sen_list[i] = new_word
        
        return " ".join(sen_list)
```

