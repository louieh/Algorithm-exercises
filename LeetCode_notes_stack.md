## LeetCode - Stack

[toc]

### 155. Min Stack

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.min_stack = []
        self.mini_list = []
        self.mini_num = 2**32-1
        

    def push(self, x: int) -> None:
        if x <= self.mini_num:
            self.mini_num = x
            self.mini_list.append(x)
        self.min_stack.append(x)

    def pop(self) -> None:
        if self.min_stack.pop() == self.mini_num:
            self.mini_list.pop()
            if self.mini_list:
                self.mini_num = self.mini_list[-1]
            else:
                self.mini_num = 2**32-1

    def top(self) -> int:
        return self.min_stack[-1]

    def getMin(self) -> int:
        return self.mini_num
```



### 227. Basic Calculator II

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack, s_list = [], []
        sy = {'+', '-', '*', '/'}
        sy_first = {'*', '/'}
        sy_second = {'+', '-'}
        temp = ''
        for i, val in enumerate(s):
            if val == ' ': continue
            if val not in sy:
                temp += val
            else:
                s_list.append(int(temp))
                temp = ''
                s_list.append(val)
        s_list.append(int(temp))
        
        i = 0
        while i < len(s_list):
            if s_list[i] not in sy:
                stack.append(s_list[i])
                i += 1
            elif s_list[i] in sy_second:
                stack.extend([s_list[i], s_list[i+1]])
                i += 2
            elif s_list[i] in sy_first:
                a = stack.pop()
                b = s_list[i+1]
                temp = a * b if s_list[i] == '*' else a // b
                stack.append(temp)
                i += 2
        stack = stack[::-1]
        while len(stack) > 1:
            a = stack.pop()
            sy_temp = stack.pop()
            b = stack.pop()
            temp = a + b if sy_temp == '+' else a - b
            stack.append(temp)
        return int(stack[0])
```



### 316. Remove Duplicate Letters

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        visited = [False] * 26
        temp = [0] * 26
        for c in s:
            temp[ord(c)-ord('a')] += 1
        stack = []
        
        for c in s:
            index = ord(c) - ord('a')
            temp[index] -= 1
            if visited[index] is True: continue
            while stack and c < stack[-1] and temp[ord(stack[-1])-ord('a')] != 0:
                visited[ord(stack.pop())-ord('a')] = False
            visited[ord(c)-ord('a')] = True
            stack.append(c)
        return "".join(stack)
```

same as 1081



### 394. Decode String

```python
class Solution:
    def decodeString(self, s: str) -> str:
        ans = ""
        temp_ans = ""
        temp_num = ""
        stack = []
        
        import string
        for i in range(len(s)-1, -1, -1):
            if s[i] in string.digits:
                temp_num = s[i] + temp_num
            else:
                if temp_ans and temp_num:
                    temp = int(temp_num) * temp_ans
                    if stack:
                        stack[-1] = temp + stack[-1]
                    else:
                        ans = temp + ans
                    temp_ans = ''
                    temp_num = ''
                if s[i] in string.ascii_letters:
                    if not stack:
                        ans = s[i] + ans
                    else:
                        stack[-1] = s[i] + stack[-1]
                elif s[i] == ']':
                    stack.append('')
                elif s[i] == '[':
                    temp_ans = stack.pop()
        if temp_ans and temp_num:
            ans = int(temp_num) * temp_ans + ans
        
        return ans
```

初始化一个空temp_ans用来存放临时ans，初始化一个空temp_num用来存放临时个数，初始化一个空stack

从后向前遍历字符串，如果是数字则添加到temp_num中，不是数字的话先判断temp_num和temp_ans有没有东西，如果有的话相乘的结果添加到ans或stack[-1]中，这取决于stack是否为空，非空说明有嵌套。

再判断如果是字母添加到ans或stack[-1]中，这取决于stack是否为空，

如果是 ] 则添加一个空字符串到stack，

如果是 [ 说明一个完整字符串遍历完整，stack.pop() 取出最后一个字符串放到 temp_ans中。

最后循环完成后，再检查temp_ans中是否有东西需要加到ans



### 402. Remove K Digits

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        if k == len(num):
            return "0"
        
        stack  = []
        i = 0
        while i < len(num):
            # whenever meet a digit which is less than the previous digit, discard the previous one
            while k > 0 and stack and num[i] < stack[-1]:
                stack.pop()
                k -= 1
            stack.append(num[i])
            i += 1
        
        while k > 0:
            stack.pop()
            k -= 1
        
        print(stack)
        
        for index, val in enumerate(stack):
            if val != "0":
                return ("").join(stack[index:])
        return "0"
```

https://leetcode.com/problems/remove-k-digits/discuss/88708/Straightforward-Java-Solution-Using-Stack



### 496. Next Greater Element I

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums2_dict = {val: index for index, val in enumerate(nums2)}
        ans, stack = [-1] * len(nums2), []
        for i in range(len(nums2)):
            while stack and nums2[stack[-1]] < nums2[i]:
                ans[stack.pop()] = nums2[i]
            stack.append(i)
        return [ans[nums2_dict.get(num)] for num in nums1]
```



### 503. Next Greater Element II

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        ans_len = len(nums)
        ans = [-1] * ans_len * 2
        nums += nums
        stack = []
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                ans[stack.pop()] = nums[i]
            stack.append(i)
        return ans[:ans_len]
```

借鉴https://leetcode.com/problems/sum-of-subarray-minimums/discuss/178876/stack-solution-with-very-detailed-explanation-step-by-step

```python
    def nextGreaterElements(self, A):
        stack, res = [], [-1] * len(A)
        for i in range(len(A)) * 2:
            while stack and (A[stack[-1]] < A[i]):
                res[stack.pop()] = A[i]
            stack.append(i)
        return res
```

```java
public class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 2 * nums.length - 1; i >= 0; --i) {
            while (!stack.empty() && nums[stack.peek()] <= nums[i % nums.length]) {
                stack.pop();
            }
            res[i % nums.length] = stack.empty() ? -1 : nums[stack.peek()];
            stack.push(i % nums.length);
        }
        return res;
    }
}
```



### 735. Asteroid Collision

```python
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        
        stack = []
        
        for each in asteroids:
            while True:
                if not stack or each * stack[-1] > 0 or (stack[-1] < 0 and each > 0):
                    stack.append(each)
                    break
                else:
                    if abs(each) == abs(stack[-1]) or abs(each) > abs(stack[-1]):
                        temp = stack.pop()
                        if abs(each) == abs(temp): break
                    else: break
        return stack
```



### 739. Daily Temperatures

```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        if not T:
            return []
        
        ans = [0] * len(T)
        stack = []
        
        for i in range(len(T)-1, -1, -1):
            while stack and T[i] >= T[stack[-1]]:
                stack.pop()
            if stack:
                ans[i] = stack[-1] - i
            
            stack.append(i)
        return ans
```

向stack中添加数的时候保证升序，不是升序的话pop，是升序的话，升序距离就是当前元素stack[-1]-index

similar as linked list 1019. Next Greater Node In Linked List



### 856. Score of Parentheses

```python
class Solution:
    def scoreOfParentheses(self, S: str) -> int:
        stack = [0]
        
        for each in S:
            if each == "(":
                stack.append(0)
            else:
                temp = stack.pop()
                if temp == 0:
                    stack[-1] += 1
                else:
                    stack[-1] += 2 * temp
        return stack[-1]
```

没有完全懂

```python
class Solution(object):
    def scoreOfParentheses(self, S):
        ans = bal = 0
        for i, x in enumerate(S):
            if x == '(':
                bal += 1
            else:
                bal -= 1
                if S[i-1] == '(':
                    ans += 1 << bal
        return ans
```

https://leetcode.com/articles/score-of-parentheses/



### 901. Online Stock Span

```python
class StockSpanner:

    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        res = 1
        while self.stack and self.stack[-1][0] <= price:
            res += self.stack.pop()[1]
        self.stack.append((price, res))
        return res
```



### 946. Validate Stack Sequences

```py
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        i = j = 0
        while i < len(pushed) and j < len(popped):
            if not stack or stack[-1] != popped[j]:
                while i < len(pushed) and (not stack or stack[-1] != popped[j]):
                    stack.append(pushed[i])
                    i += 1
            else:
                while j < len(popped) and stack and stack[-1] == popped[j]:
                    stack.pop()
                    j += 1
        return stack == popped[j:][::-1]
```



### 1047. Remove All Adjacent Duplicates In String

```python
class Solution:
    def removeDuplicates(self, S: str) -> str:
        stack = []
        
        for each in S:
            if stack and stack[-1] == each:
                stack.pop()
            else:
                stack.append(each)
        return ("").join(stack)
```



### 1081. Smallest Subsequence of Distinct Characters

```python
class Solution:
    def smallestSubsequence(self, s: str) -> str:
        visited = [False] * 26
        temp = [0] * 26
        for c in s:
            temp[ord(c)-ord('a')] += 1
        stack = []
        
        for c in s:
            index = ord(c) - ord('a')
            temp[index] -= 1
            if visited[index] is True: continue
            while stack and c < stack[-1] and temp[ord(stack[-1])-ord('a')] != 0:
                visited[ord(stack.pop())-ord('a')] = False
            visited[ord(c)-ord('a')] = True
            stack.append(c)
        return "".join(stack)
```

same as 316



### 1249. Minimum Remove to Make Valid Parentheses

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        s_list = list(s)
        stack = []
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            elif s[i] == ')':
                if stack:
                    stack.pop()
                else:
                    s_list[i] = None
        for each in stack:
            s_list[each] = None
        return ("").join([s_list[i] for i in range(len(s)) if s_list[i] is not None])
```

遍历字符串，遇到 `(` 则将index放入stack中，遇到 `)` 且 stack 不为空则pop，如果为空，说明这是多余的 `)` 则将该位置打标记用来最后删除。遍历完成后，如果stack不为空，则说明这是多余的 `(`. 返回最后的字符串时候不返回stack中的多余的 `(` 也不返回字符串中已经打标记的多余的 `)`

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        s_list = list(s)
        stack = []
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i+1)
            elif s[i] == ')':
                if stack and stack[-1] >= 0:
                    stack.pop()
                else:
                    stack.append(-(i+1))
        for each in stack:
            if each >= 0:
                s_list[each-1] = None
            else:
                s_list[-each-1] = None
        return ("").join([s_list[i] for i in range(len(s)) if s_list[i] is not None])
```



### 1381. Design a Stack With Increment Operation

```python
class CustomStack:

    def __init__(self, maxSize: int):
        self._stack = []
        self.maxSize = maxSize
        self.size_now = 0

    def push(self, x: int) -> None:
        if self.size_now < self.maxSize:
            self._stack.append(x)
            self.size_now += 1

    def pop(self) -> int:
        if self.size_now > 0:
            res = self._stack.pop()
            self.size_now -= 1
            return res
        return -1

    def increment(self, k: int, val: int) -> None:
        for i in range(min(k, self.size_now)):
            self._stack[i] += val
```

