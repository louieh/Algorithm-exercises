## LeetCode - Stack

[toc]

### 20. Valid Parentheses

```python
class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        temp = [];
        for each in s:
            if each in ['(','[','{']:
                temp.append(each)
            elif each in [')',']','}']:
                if not temp:
                    return False
                poptemp = temp.pop()
                if poptemp == '(' and each != ')' or poptemp == '[' and each != ']' or poptemp == '{' and each != '}':
                    return False
        if not temp:
            return True
        else:
             return False
```

20 和 921 两道关于合法括号的题目，原理相似，921 难度为中等反而更简单些。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        
        for c in s:
            if c in ['[', '(', '{']:
                stack.append(c)
            else:
                if not stack or c == ']' and stack[-1] != '[' or c == '}' and stack[-1] != '{' or c == ')' and stack[-1] != '(':
                    return False
                stack.pop()
        return not stack
```



### 32. Longest Valid Parentheses

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        if len(s) <= 1: return 0
        
        stack = [-1]
        res = 0
        for index, each in enumerate(s):
            if each == "(":
                stack.append(index)
            else:
                stack.pop()
                if not stack:
                    stack.append(index)
                else:
                    res = max(res, index - stack[-1])
        
        return res
                    
```

https://leetcode.com/problems/longest-valid-parentheses/solution/

Instead of finding every possible string and checking its validity, we can make use of a stack while scanning the given string to:

1. Check if the string scanned so far is valid.
2. Find the length of the longest valid string.

In order to do so, we start by pushing -1 onto the stack. For every ‘(’ encountered, we push its index onto the stack.

For every ‘)’ encountered, we pop the topmost element. Then, the length of the currently encountered valid string of parentheses will be the difference between the current element's index and the top element of the stack.

If, while popping the element, the stack becomes empty, we will push the current element's index onto the stack. In this way, we can continue to calculate the length of the valid substrings and return the length of the longest valid string at the end.

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        left = right = 0
        res = 0
        
        # left -> right
        for i in range(len(s)):
            if s[i] == "(":
                left += 1
            else:
                right += 1
            if left == right:
                res = max(res, left * 2)
            elif right > left:
                left = right = 0
        
        left = right = 0
        # left <- right
        for i in range(len(s)-1, -1, -1):
            if s[i] == "(":
                left += 1
            else:
                right += 1
            if left == right:
                res = max(res, left * 2)
            elif left > right:
                left = right = 0
        
        return res
```

In this approach, we make use of two counters left and right. First, we start traversing the string from the left towards the right and for every ‘(’ encountered, we increment the left counter and for every ‘)’ encountered, we increment the right counter. Whenever left becomes equal to right, we calculate the length of the current valid string and keep track of maximum length substring found so far. If right becomes greater than left we reset left and right to 00.

Next, we start traversing the string from right to left and similar procedure is applied.

不使用额外空间，先从左向右遍历，遍历过程中计算 '(' 和 ')' 个数，如果相等则得到暂时匹配的长度，如果右侧个数大于左侧，则左右括号个数均置零。再从右向左遍历，同样的逻辑，只是此时当左括号个数大于右侧时，左右个数置零。



### 71. Simplify Path

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        path_list = path.split('/')
        
        stack = []
        
        for each in path_list:
            if not each or each == '.': continue
            if each == '..':
                if stack: stack.pop()
                continue
            stack.append(each)
        return '/' + '/'.join(stack)
```



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



### 224. Basic Calculator

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        number = 0
        result = 0
        sign = 1
        # number, +, -, (, )
        for c in s:
            if c.isdigit():
                number = number * 10 + int(c)
            elif c == "+":
                result += sign * number
                number = 0
                sign = 1
            elif c == "-":
                result += sign * number
                number = 0
                sign = -1
            elif c == "(":
                stack.append(result)
                stack.append(sign)
                result = 0
                sign = 1
            elif c == ")":
                result += sign * number
                number = 0
                # sign = 1
                result *= stack.pop()
                result += stack.pop()
        if number != 0:
            result += sign * number
        return result
```

https://leetcode.com/problems/basic-calculator/discuss/62361/Iterative-Java-solution-with-stack



### 225. Implement Stack using Queues

```python
import queue

class MyStack:
    
    def __init__(self):
        self.q1 = queue.Queue()
        self.q2 = queue.Queue()

    def push(self, x: int) -> None:
        if not self.q2.empty():
            self.q2.put(x)
        else:
            self.q1.put(x)
    
    def _helper(self):
        if not self.q2.empty():
            while self.q2.qsize() != 1:
                self.q1.put(self.q2.get())
            return self.q2.get()
        elif not self.q1.empty():
            while self.q1.qsize() != 1:
                self.q2.put(self.q1.get())
            return self.q1.get()
        else:
            return None

    def pop(self) -> int:
        return self._helper()

    def top(self) -> int:
        res = self._helper()
        if not self.q1.empty():
            self.q1.put(res)
        else:
            self.q2.put(res)
        return res

    def empty(self) -> bool:
        return self.q1.empty() and self.q2.empty()


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
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

```python
class Solution:
    def decodeString(self, s: str) -> str:
        
        def helper(ss):
            res = ""
            i = 0
            while i < len(ss):
                if ss[i] in string.ascii_lowercase:
                    res += ss[i]
                    i += 1
                    continue
                if ss[i] in string.digits:
                    num = 0
                    while ss[i] != "[":
                        num = num * 10 + int(ss[i])
                        i += 1
                    i += 1
                    temp = ""
                    num_of_left = 0
                    while ss[i] != "]" or num_of_left != 0:
                        if ss[i] == "[":
                            num_of_left += 1
                        if ss[i] == "]":
                            num_of_left -= 1
                        temp += ss[i]
                        i += 1
                    res += helper(temp) * num
                    i += 1
            return res
        return helper(s)
```

```python
class Solution(object):
    def decodeString(self, s):
        stack = []; curNum = 0; curString = ''
        for c in s:
            if c == '[':
                stack.append(curString)
                stack.append(curNum)
                curString = ''
                curNum = 0
            elif c == ']':
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num*curString
            elif c.isdigit():
                curNum = curNum*10 + int(c)
            else:
                curString += c
        return curString
```



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
#def nextGreaterElements(self, A):
#    stack, res = [], [-1] * len(A)
#    for i in range(len(A)) * 2:
#      while stack and (A[stack[-1]] < A[i]):
#        res[stack.pop()] = A[i]
#        stack.append(i)
#        return res
# 正序
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        res = [-1] * len(nums)
        stack = []
        for i in range(len(nums)*2):
            index = i % len(nums)
            while stack and nums[index] > nums[stack[-1]]:
                res[stack.pop()] = nums[index]
            stack.append(index)
        return res
```

```java
// 倒序
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

```python
# 所以求previous的时候我们可以使用正序遍历，求next的时候我们可以使用倒序遍历，但是无论怎样遍历何种问题都会有方法解决
stack = []
nums = [4,6,9,2,1]
res = [0] * len(nums)
# previous less
for i in range(len(nums)):
	  while stack and nums[stack[-1]] > nums[i]:
        stack.pop()
    if stack:
  			res[i] = nums[stack[-1]]
    stack.append(i)

# next less
for i in range(len(nums)):
  	while stack and nums[i] < nums[stack[-1]]:
      	res[stack.pop()] = nums[i]
    stack.append(i)

# previous greater
for i in range(len(nums)):
  	while stack and nums[stack[-1]] < nums[i]:
      	stack.pop()
    if stack:
      	res[i] = nums[stack[-1]]
    stack.append(i)

# next greater
for i in range(len(nums)):
  	while stack and nums[i] > nums[stack[-1]]:
      	res[stack.pop()] = nums[i]
    stack.append(i)
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

向stack中添加数的时候保证升序，不是升序的话pop，是升序的话，升序距离就是当前元素stack[-1]-index。

将数组从末尾开始向stock中插，每次插入前，把stack中小于当前元素的栈顶元素pop出去，这样最后stack中剩下的便是单增的，也就是栈顶元素在遍历过程中保持最小值，下面的正序遍历则是保持最大值。

无论是从前向后遍历还是从后向前遍历，stack都是用于暂时存放已遍历过的元素。

similar as linked list 1019. Next Greater Node In Linked List

```python
# 正序遍历
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []
        for i in range(len(temperatures)):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                stack_last = stack.pop()
                res[stack_last] = i - stack_last
            stack.append(i)
        return res
```



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



### 921. Minimum Add to Make Parentheses Valid

```python
class Solution:
    def minAddToMakeValid(self, S: str) -> int:
        if not S:
            return 0
        
        temp = []
        for each in S:
            if each == '(':
                temp.append(each)
            elif each == ')':
                if temp and temp[-1] == '(':
                    temp.pop()
                else:
                    temp.append(each)
        return len(temp)
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



### 1209. Remove All Adjacent Duplicates in String II

```python
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        
        temp = []
        cur_c = s[0]
        cur_count = 1
        for c in s[1:]:
            if c != cur_c:
                temp.append((cur_c, cur_count))
                cur_c = c
                cur_count = 1
            else:
                cur_count += 1
        temp.append((cur_c, cur_count))
        
        stack = []
        for cur_c, cur_count in temp:
            cur_count %= k
            if not cur_count: continue
            if not stack or stack[-1][0] != cur_c:
                stack.append((cur_c, cur_count))
            else:
                _, prev_count = stack.pop()
                cur_count = (cur_count + prev_count) % k
                if not cur_count: continue
                else:
                    stack.append((cur_c, cur_count))
                
        res = ""
        for cur_c, cur_count in stack:
            res += cur_c * cur_count
        return res
```



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



### 2289. Steps to Make Array Non-decreasing

```python
class Solution:
    def totalSteps(self, nums: List[int]) -> int:
        res = 0
        dp = [0] * len(nums)
        stack = []
        for i in range(len(nums)-1, -1, -1):
            while stack and nums[i] > nums[stack[-1]]:
                stack_top = stack.pop()
                dp[i] = max(dp[i] + 1, dp[stack_top])
                res = max(res, dp[i])
            stack.append(i)
        
        return res
```

https://leetcode.com/problems/steps-to-make-array-non-decreasing/discuss/2085864/JavaC%2B%2BPython-Stack-%2B-DP-%2B-Explanation

本质上就是使用 stack 求升序序列，在求的过程中记录最大 pop 的数量

