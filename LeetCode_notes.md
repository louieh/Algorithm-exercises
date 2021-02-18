## LeetCode - Node

[toc]

### 1.Two Sum

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        temp_dict = dict()
        
        for i in range(len(nums)):
            c = target - nums[i]
            if c in temp_dict:
                return [temp_dict[c], i]
            else:
                temp_dict[nums[i]] = i
```

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        temp = dict()
        for i, val in enumerate(nums):
            c = target - val
            if c in temp:
                return [temp[c], i]
            temp[val] = i
```



### cs6301 final Question 7

Given an array of integers, and x. Provide an algorithm to find how many pairs of elements of the array sum to x. For example, if A = {3, 3, 4, 5, 3, 5, 4} them `howMany(A, 8)` return 7. RT should be `O(nlogn)` or better.

 ```python
    def howMany(A, target):
        temp_dict = dict()
        ans = 0
        # A = {3, 3, 4, 5, 3, 5, 4} target = 8
        for i in range(len(A)):
            c = target - A[i]
            if c in temp_dict:
                ans += temp_dict[c]
            if A[i] in temp_dict:
                temp_dict[A[i]] += 1
            else:
                temp_dict[A[i]] = 1
        return ans
 ```



### 2. Add Two Numbers

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 and not l2:
            return
        if not l1 or not l2:
            return l1 or l2
        
        temp = 0
        dummy = ans = ListNode(0)
        
        while l1 or l2:
            l1_num = l1.val if l1 else 0
            l2_num = l2.val if l2 else 0
            temp_result = l1_num + l2_num + temp
            temp = temp_result // 10
            dummy.next = ListNode(temp_result % 10)
            l1 = l1.next if l1 else l1
            l2 = l2.next if l2 else l2
            dummy = dummy.next
        
#         while l1:
#             temp_result = (l1 or l2).val + temp
#             temp = temp_result // 10
#             dummy.next = ListNode(temp_result % 10)
            
#             dummy = dummy.next
        
#         while l2:
#             temp_result = l2.val + temp
#             temp = temp_result // 10
#             dummy.next = ListNode(temp_result % 10)
#             l2 = l2.next
#             dummy = dummy.next
        
        if temp:
            dummy.next = ListNode(temp)
        
        return ans.next
```

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return l1 or l2
        
        res = head = ListNode(0)
        plus = 0
        while l1 and l2:
            temp = l1.val + l2.val + plus
            head.next = ListNode(temp % 10)
            plus = temp // 10
            l1 = l1.next
            l2 = l2.next
            head = head.next
        l = l1 or l2
        while l:
            temp = l.val + plus
            head.next = ListNode(temp % 10)
            plus = temp // 10
            l = l.next
            head = head.next
        if plus:
            head.next = ListNode(plus)
        return res.next
```



### 6. ZigZag Conversion

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s
        
        ans = [[] for i in range(min(numRows, len(s)))]
        
        cur_row, down = 0, True
        for c in s:
            ans[cur_row].append(c)
            if cur_row == len(ans) - 1 and down is True:
                cur_row -= 1
                down = False
                continue
            elif cur_row == 0 and down is False:
                cur_row += 1
                down = True
                continue
            if down:
                cur_row += 1
            else:
                cur_row -= 1
        
        return "".join(["".join(row) for row in ans])
```



### 7. Reverse Integer

```c++
class Solution {
    public:
    int reverse(int x){
        //x=1999999999 考虑溢出的问题
        long long ans = 0;
        const int maxint = 0x7fffffff;//2147483647
        const int minint = 0x80000000;//-2147483648
        while( x!=0 )
        {
            ans = ans*10+(x%10);
            x/=10; //x的各位变成ans最高位
        }
        //判断溢出
        if( ans<minint || ans>maxint )
        {
            ans=0;
        }
        return ans;
    }
};
```

```python
# 10/25/2019
class Solution:
    def reverse(self, x):
        if not x:
            return x
        max_num = 2147483647
        min_num = -2147483648
        if x < 0:
            x = -x
            sign = -1
        else:
            sign = 1
        
        ans = 0
        
        while x != 0:
            ans = ans * 10 + x % 10
            x //= 10
        
        ans = ans if sign == 1 else -ans
        
        import sys
        if ans > max_num or ans < min_num:
            return 0
        else:
            return ans
```



### 8. String to Ineger (atoi)

```c++
Class Solution{
    public:
    int myAtoi(string str){
        
        const int maxint = 0x7fffffff;
        const int minint = 0x80000000;
        
        long long ans = 0;
        bool flag = false;
        int st = 0;
         while (st<str.length() && str[st]==' ')
         {
             st++;
         }
        if (st<str.length() && str[st]=='+')
        {
            st++;
        }
        else
        {
            if (st<str.length() && str[st]=='-')
            {
                flag=true;
                st++;
            }
        }
        for(int i=st;i<str.length();i++)
        {
            if(str[i]<='9' && str[i]>='0')
            {
                ans = ans*10+str[i]-'0';
            }
            else
            {
                break;
            }
        }
        if (flag) ans = -ans;
        if (ans>maxint) ans = manint;
        if (ans<minint) ans = minint;
        return ans;
    }
}
```

```python
# 10/25/2019
class Solution:
    def myAtoi(self, str: str) -> int:
        if not str:
            return 0
        
        max_num = 2147483647
        min_num = -2147483648
        ans = 0
        begin = False
        sign = 1
        for i in range(len(str)):
            if str[i] == ' ':
                if not begin:
                    continue
                else:
                    break
            elif str[i] == '+' or str[i] == '-':
                if not begin:
                    sign = -1 if str[i] == '-' else 1
                    begin = True
                    continue
                else:
                    break
            elif (ord(str[i]) < 48 or ord(str[i]) > 57) and not begin:
                return 0
            elif (ord(str[i]) < 48 or ord(str[i]) > 57) and begin:
                break
            begin = True
            ans = ans * 10 + (ord(str[i]) - 48)
        
        ans = ans if sign == 1 else -ans
        
        if ans > max_num:
            return max_num
        elif ans < min_num:
            return min_num
        return ans
```



### 11. Container With Most Water

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l = 0
        r = len(height) - 1
        ans = 0
        
        while l < r:
            ans = max(ans, (r - l) * min(height[l], height[r]))
            if height[l] > height[r]:
                r -= 1
            else:
                l += 1
        return ans
```



### 12. Integer to Roman

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        res = ''
        
        def helper(res, c4, c9, c5, c1, temp):
            if temp == 5:
                return res + c5
            elif temp > 5:
                if temp == 9:
                    return res + c9
                else:
                    res += c5
                    for i in range(temp-5):
                        res += c1
                    return res
            else:
                if temp == 4:
                    return res + c4
                else:
                    for i in range(temp):
                        res += c1
                    return res
        
        if num >= 1000:
            temp = num // 1000
            num %= 1000
            for i in range(temp):
                res += 'M'
                
        if num >= 100:
            temp = num // 100
            num %= 100
            res = helper(res, 'CD', 'CM', 'D', 'C', temp)
            
        if num >= 10:
            temp = num // 10
            num %= 10
            res = helper(res, 'XL', 'XC', 'L', 'X', temp)

        if num >= 1:
            temp = num
            res = helper(res, 'IV', 'IX', 'V', 'I', temp)

        return res
            
```



### 15. 3Sum

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = []
        
        processed = []
        
        for i in range(len(nums)):
            if nums[i] in processed:
                continue
            used = {}
            for j in range(i+1, len(nums)):
                target = -(nums[i]+nums[j])
                if not target in used:
                    if not nums[j] in used:
                        used[nums[j]] = False
                elif not used[target] and not nums[j] in processed and not target in processed:
                    ans.append([nums[i], nums[j], target])
                    used[target] = True
            processed.append(nums[i])
        return ans
```

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        nums.sort()
        ans = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum > 0:
                    right -= 1
                elif sum < 0:
                    left += 1
                else:
                    ans.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left+1] == nums[left]:
                        left += 1
                    while left < right and nums[right-1] == nums[right]:
                        right -= 1
                    left += 1
                    right -= 1
        return ans
```



### 16. 3Sum Closest

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        diff = sys.maxsize
        ans = None
        for i in range(len(nums)):
            left, right = i+1, len(nums)-1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if abs(target - sum) < diff:
                    diff = abs(target - sum)
                    ans = sum
                if sum < target:
                    left += 1
                elif sum > target:
                    right -= 1
                else:
                    break
            if diff == 0:
                break
        return ans
```



### 18. 4Sum

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:

        def twoSum(tempList, target):
            ans = []
            left, right = 0, len(tempList)-1
            while left < right:
                sum = tempList[left] + tempList[right]
                if sum < target or (left > 0 and tempList[left] == tempList[left-1]):
                    left += 1
                elif sum > target or (right < len(tempList)-1 and tempList[right] == tempList[right+1]):
                    right -= 1
                else:
                    ans.append([tempList[left], tempList[right]])
                    left += 1
                    right -= 1
            return ans
        
        def kSum(tempList, target, k):
            ans = []
            if len(tempList) == 0 or tempList[0] * k > target or tempList[-1] * k < target:
                return ans
            if k == 2:
                return twoSum(tempList, target)
            for i in range(len(tempList)):
                if i > 0 and tempList[i] == tempList[i-1]:
                    continue
                for each in kSum(tempList[i+1:], target-tempList[i], k-1):
                    ans.append([tempList[i]] + each)
            return ans
        
        nums.sort()
        return kSum(nums, target, 4)
```



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



### 28. Implement strStr()

```python
class Solution:
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle not in haystack:
            return -1
        elif needle == "":
            return 0
        else:
            for i in range(0,len(haystack)-len(needle)+1):
                prefect = 0
                for j in range(0,len(needle)):
                    if needle[j] != haystack[i+j]:
                        prefect = 0
                        break
                    else:
                        prefect = 1
                if prefect == 1:
                    return i
                    break
```

```python
# 10/3/2019
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not haystack and needle:
            return -1
        if not haystack or not needle:
            return 0
        if len(needle) > len(haystack):
            return -1

        for i in range(len(haystack)):
            ifans = True
            if haystack[i] == needle[0]:
                if len(haystack) - i - 1 < len(needle) - 1:
                    return -1
                for j in range(len(needle) - 1):
                    if haystack[i + j + 1] != needle[j + 1]:
                        if i + j + 2 <= len(haystack) - 1:
                            ifans = False
                            break
                        else:
                            return -1
                if ifans:
                    return i
        return -1
```



### 36. Valid Sudoku

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        if not board:
            return False
        temp = set()
        for row in range(len(board)):
            for col in range(len(board[0])):
                num = board[row][col]
                if num != '.':
                    if num+'row'+str(row) not in temp and num+'col'+str(col) not in temp and num+'box'+str(row//3)+str(col//3) not in temp:
                        temp.add(num+'row'+str(row))
                        temp.add(num+'col'+str(col))
                        temp.add(num+'box'+str(row//3)+str(col//3))
                    else:
                        return False
        return True
```

分别encode成`1row1 or 1col1 or 1box00`



### 49. Group Anagrams

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import defaultdict
        res = defaultdict(list)
        
        for each in strs:
            res[tuple(sorted(each))].append(each)
        return res.values()
```

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        from collections import defaultdict
        ans = defaultdict(list)
        for each in strs:
            count = [0] * 26
            for c in each:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(each)
        return ans.values()
```



### 50. Pow(x, n)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        
        def myPow_helper(x, n, temp):
            if n == 1:
                return x * temp
            return myPow_helper(x, n-1, x*temp)
        
        if n > 0:
            return myPow_helper(x, n, 1)
        else:
            return 1/myPow_helper(x, -n, 1)
```

recursionError: maximum recursion depth exceeded in comparison

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        
        def myPow_helper(x, n):
            if n == 1:
                return x
            if n == 0:
                return 1
            
            temp = myPow_helper(x, n//2)
            
            if n % 2 == 0:
                return temp * temp
            else:
                return temp * temp * x
        
        if n > 0:
            return myPow_helper(x, n)
        else:
            return 1/myPow_helper(x, -n)
```



### 58. Length of Last Word

```python
# 05/02/2018 11:18
class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        if s == " ":
            return 0
        else:
            mark = False
            s_list = s.split(" ")
            for each in s_list[::-1]:
                if each:
                    return len(each)
                    mark = True
                    break
            if mark != True:
                return 0
```

```python
# 9/15/2020
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if not s or s == "":
            return 0
        return len(s.strip().split(" ")[-1])
```



### 66. Plus one

```python
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        temp_text = ""
        for each in digits:
            temp_text += str(each)
            
        temp_text = str(int(temp_text) + 1)
        
        temp_return = []
        for each in temp_text:
            temp_return.append(int(each))
        return temp_return
```

```java
// 9/28/2019
class Solution {
    public int[] plusOne(int[] digits) {
        
        if(digits.length == 0){
            return null;
        }
        
        int wait_plus = 0;
        
        for(int i=digits.length-1; i>=0; i--){
            int temp = i == digits.length-1 ? digits[i] + wait_plus + 1 : digits[i] + wait_plus;
            int digit_ten = temp / 10;
            int digit_one = temp % 10;
            digits[i] = digit_one;
            wait_plus = digit_ten;
        }
        if (wait_plus != 0){
            int[] ans = new int[digits.length+1];
            ans[0] = wait_plus;
            for(int i=1; i<ans.length; i++){
                ans[i] = digits[i-1];
            }
            return ans;
        }
        else{
            return digits;
        }
        
    }
}
```



### 67. Add Binary

```python
class Solution:
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if len(a) > len(b):
            for i in range(len(a)-len(b)):
                b = "0" + b
        if len(b) > len(a):
            for i in range(len(b)-len(a)):
                a = "0" + a

        re_a = a[::-1]
        re_b = b[::-1]
        is_Carry = 0 #是否有进位
        return_num_str = ""
        for i in range(len(a)):
            if int(re_a[i]) + int(re_b[i]) + is_Carry == 3: #1+1+1
                #digit = 1 #个位数字
                return_num_str += "1"
                is_Carry = 1
            if int(re_a[i]) + int(re_b[i]) + is_Carry == 2: #1+1+0
                #digit = 0
                return_num_str += "0"
                is_Carry = 1
            if int(re_a[i]) + int(re_b[i]) + is_Carry == 0: #0+0+0
                #digit = 0
                return_num_str += "0"
                is_Carry = 0
            if int(re_a[i]) + int(re_b[i]) + is_Carry == 1: #1+0+0
                #digit = 1
                return_num_str += "1"
                is_Carry = 0
        if is_Carry == 1:
            return_num_str += "1"

        return_num = return_num_str[::-1]
        return return_num

```

```python
# 10/3/2019
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        temp = 0
        ans = ''
        
        l_min = min(len(a), len(b))
        for i in range(l_min):
            num = int(a[len(a)-1-i]) + int(b[len(b)-1-i]) + temp
            if num == 2:
                temp = 1
                ans = '0' + ans
            elif num == 3:
                temp = 1
                ans = '1' + ans
            else:
                temp = 0
                ans = str(num) + ans
        
        l = len(a) - len(b)
        i = abs(l)
        if l < 0:
            while i-1>=0:
                num = int(b[i-1]) + temp
                if num == 2:
                    temp = 1
                    ans = '0' + ans
                elif num == 3:
                    temp = 1
                    ans = '1' + ans
                else:
                    temp = 0
                    ans = str(num) + ans
                i -= 1
        elif l > 0:
            while i-1>=0:
                num = int(a[i-1]) + temp
                if num == 2:
                    temp = 1
                    ans = '0' + ans
                elif num == 3:
                    temp = 1
                    ans = '1' + ans
                else:
                    temp = 0
                    ans = str(num) + ans
                i -= 1
        
        if temp == 1:
            ans = '1' + ans
        return ans
```



### 73. Set Matrix Zeroes

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        def set_zero(row, col):
            # row
            for i in range(len(matrix[0])):
                matrix[row][i] = 0
            # col
            for i in range(len(matrix)):
                matrix[i][col] = 0
                
        zero_list = []
        
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] == 0:
                    zero_list.append([row, col])
        for each in zero_list:
            set_zero(each[0], each[1])
        return matrix
```

```python
# Time Complexity: O(M×N) where M and N are the number of rows and columns respectively.
# Space Complexity: O(M+N).
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        R = len(matrix)
        C = len(matrix[0])
        rows, cols = set(), set()

        # Essentially, we mark the rows and columns that are to be made zero
        for i in range(R):
            for j in range(C):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)

        # Iterate over the array once again and using the rows and cols sets, update the elements
        for i in range(R):
            for j in range(C):
                if i in rows or j in cols:
                    matrix[i][j] = 0
```

```python
# Space Complexity : O(1)
# Time Complexity : O((M×N)×(M+N))
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        def set_zero(row, col, val):
            # row
            for i in range(len(matrix[0])):
                if matrix[row][i] != 0:
                    matrix[row][i] = val
            # col
            for i in range(len(matrix)):
                if matrix[i][col] != 0:
                    matrix[i][col] = val
        
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] == 0:
                    set_zero(row, col, '&')
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] == '&':
                    matrix[row][col] = 0
        return matrix
```

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        first_row = first_col = False
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col] == 0:
                    matrix[row][0] = 0
                    matrix[0][col] = 0
                    if row == 0:
                        first_row = True
                    if col == 0:
                        first_col = True
        
        for row in range(1, len(matrix)):
            if matrix[row][0] == 0:
                for i in range(1, len(matrix[0])):
                    matrix[row][i] = 0
        for col in range(1, len(matrix[0])):
            if matrix[0][col] == 0:
                for i in range(1, len(matrix)):
                    matrix[i][col] = 0
        if first_row:
            for i in range(len(matrix[0])):
                matrix[0][i] = 0
        if first_col:
            for i in range(len(matrix)):
                matrix[i][0] = 0
        return matrix
```



### 75. Sort Colors

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        cur_index = 0
        for i in range(3):
            for j in range(len(nums)):
                if nums[j] == i:
                    nums[cur_index], nums[j] = nums[j], nums[cur_index]
                    cur_index += 1
        return nums
```

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        p0 = i = 0
        p2 = len(nums) - 1
        
        while i <= p2:
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                p0 += 1
                i += 1
            elif nums[i] == 2:
                nums[i], nums[p2] = nums[p2], nums[i]
                p2 -= 1
            else:
                i += 1
        return nums
```



### 118. Pascal's Triangle

```python
class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        ans = []
        for i in range(1,numRows+1):
            row_list = []
            if i == 1:
                row_list = [1]
                ans.append(row_list)
                continue
            elif i == 2:
                row_list = [1,1]
                ans.append(row_list)
                continue
            else:
                row_list.append(1)
                for j in range(i-2):
                    row_list.append(ans[i-2][j]+ans[i-2][j+1])
                row_list.append(1)
                ans.append(row_list)
        
        
        return ans
```

```python
# 10/2/2019
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if not numRows:
            return []
        
        ans = [[1],[1,1]]
        last_temp = [1,1]
        
        if numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1],[1,1]]
        
        for i in range(numRows-2):
            temp = [1]
            for j in range(len(last_temp)-1):
                temp.append(last_temp[j]+last_temp[j+1])
            temp.append(1)
            ans.append(temp)
            last_temp = temp
        return ans
```

```python
# 12/17/2019
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if not numRows:
            return []
        ans = [[1]]
        for i in range(numRows-1):
            last_row = ans[-1]
            temp = [1]
            for j in range(len(last_row)-1):
                temp.append(last_row[j]+last_row[j+1])
            temp.append(1)
            ans.append(temp)
        
        return ans
```



### 119. Pascal's Triangle II

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        if rowIndex == 0:
            return [1]
        elif rowIndex == 1:
            return [1,1]
        
        temp = [1,1]

        for i in range(rowIndex-1):
            ans = [1]
            for j in range(len(temp)-1):
                ans.append(temp[j]+temp[j+1])
            ans.append(1)
            temp = ans
        
        return ans
```

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        
        ans = [1]
        
        for i in range(1, rowIndex+1):
            temp = [1] * (i+1)
            for j in range(1,i):
                temp[j] = ans[j-1] + ans[j]
            ans = temp
        
        return ans
```



### 121. Best Time to Buy and Sell Stock

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i in range(len(prices)-1):
            ans = max(ans, max(prices[i+1:])-prices[i])
        return ans
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        min_price = sys.maxsize
        for each in prices:
            if each < min_price:
                min_price = each
            elif each - min_price > ans:
                ans = each - min_price
        return ans
```



### 122. Best Time to Buy and Sell Stock II

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = low = high = i = 0
        while i < len(prices) - 1:
            while i < len(prices) - 1 and prices[i] >= prices[i + 1]:
                i += 1
            low = prices[i]
            while i < len(prices) - 1 and prices[i] <= prices[i + 1]:
                i += 1
            high = prices[i]
            ans += (high - low)
        return ans
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                ans += prices[i] - prices[i-1]
        return ans
```



### 146. LRU Cache

```python
from collections import OrderedDict
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.LRU = OrderedDict()
        
    def get(self, key: int) -> int:
        if key not in self.LRU:
            return -1
        self.LRU.move_to_end(key,last = True)
        return self.LRU[key]
            
    def put(self, key: int, value: int) -> None:
        if key in self.LRU:
            self.LRU.move_to_end(key,last = True)
        self.LRU[key] = value
        if len(self.LRU) > self.capacity:
            self.LRU.popitem(last = False)  #Pop first item
```

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.temp_list = []
        self.temp_dict = {}

    def get(self, key: int) -> int:
        if key not in self.temp_dict:
            return -1
        self.temp_list.remove(key)
        self.temp_list.append(key)
        return self.temp_dict[key]

    def put(self, key: int, value: int) -> None:
        if key in self.temp_dict:
            self.temp_dict[key] = value
            self.temp_list.remove(key)
            self.temp_list.append(key)
        else:
            self.temp_list.append(key)
            self.temp_dict[key] = value
            if len(self.temp_list) > self.capacity:
                old_key = self.temp_list.pop(0)
                self.temp_dict.pop(old_key)
```



### 151. Reverse Words in a String

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        if not s:
            return s
        
        i = len(s) - 1
        ans = ""
        temp = ""
        
        while i >= 0:
            if s[i] != " ":
                temp = s[i] + temp
            elif temp:
                ans += temp if not ans else (" " + temp)
                temp = ""
            i -= 1
        if temp:
            ans += temp if not ans else (" " + temp)
            
        return ans
```



### 165. Compare Version Numbers

```python
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        v1_list = version1.split(".")
        v2_list = version2.split(".")
        
        i = j = 0
        while i < len(v1_list) and j < len(v2_list):
            if int(v1_list[i]) > int(v2_list[j]):
                return 1
            elif int(v1_list[i]) < int(v2_list[j]):
                return -1
            i += 1
            j += 1
        
        while i < len(v1_list):
            if int(v1_list[i]) > 0:
                return 1
            i += 1
        while j < len(v2_list):
            if int(v2_list[j]) > 0:
                return -1
            j += 1
        return 0
```
like merge two sorted array



### 167. Two Sum II - Input array is sorted

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        if not numbers:
            return []
        
        start = 0
        end = len(numbers) - 1
        
        while start < end:
            temp = numbers[start] + numbers[end]
            if temp == target:
                return [start+1, end+1]
            elif temp > target:
                end -= 1
            elif temp < target:
                start += 1
        return []
```

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        low = 0
        high = len(numbers) - 1
        while low < high:
            temp = numbers[low] + numbers[high]
            if temp == target:
                return [low+1, high+1]
            elif temp > target:
                high -= 1
            else:
                low += 1
```

总结 two sum 类题型



### 168. Excel Sheet Column Title

```python
class Solution:
    def convertToTitle(self, n: int) -> str:
        capitals = [chr(x) for x in range(ord('A'), ord('Z')+1)]
        result = []
        while n > 0:
            result.append(capitals[(n-1)%26])
            n = (n-1) // 26
        result.reverse()
        return ''.join(result)
```



### 169. Majority Element

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        from collections import Counter
        
        c = Counter(nums)
        for k,v in c.items():
            if v > len(nums) // 2:
                return k
```

```python
class Solution:
    def majorityElement(self, nums):
        nums.sort()
        return nums[len(nums)//2]
```



### 170. Two Sum III - Data structure design

```python
class TwoSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._list = []

    def add(self, number: int) -> None:
        """
        Add the number to an internal data structure..
        """
        self._list.append(number)

    def find(self, value: int) -> bool:
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        """
        self._set = set()
        for num in self._list:
            c = value - num
            if c in self._set:
                return True
            self._set.add(num)
        return False
```

```python
class TwoSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._dict = defaultdict(int)

    def add(self, number: int) -> None:
        """
        Add the number to an internal data structure..
        """
        self._dict[number] += 1

    def find(self, value: int) -> bool:
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        """
        for num in self._dict:
            c = value - num
            if c in self._dict:
                if c == num:
                    if self._dict[c] > 1:
                        return True
                else:
                    return True
        return False
```



### 171. Excel Sheet Column Number

```python
class Solution:
    def titleToNumber(self, s: str) -> int:
        import string
        temp_dict = {val:i+1 for i,val in enumerate(string.ascii_uppercase)}
        print(temp_dict)
        ans = 0
        i = len(s) - 1
        while i >= 0:
            ans += temp_dict[s[i]] * 26**(len(s)-i-1)
            i -= 1
        return ans
```



### 172. Factorial Traniling Zeroes

```c++
#n!
#num = a * 10^k = a*(5^k*2^k)
int trailingZeroes(int n){
    int sum = 0;
    /*
    for (int i=5;i<=n;i+=5)
    {
        int x=i;
        while(x%5==0)
        {
            x/=5;
            sum++;
        }
    }
    */
    while(n>0)
    {
        sum += n/5;
        # n/5 是 n/5 之前5的个数
        # 比如 12! = 12 *..* 10 *..* 5 *..* 2 * 1
        # 12 / 5 = 2 是 2 之前的质因数里有几个5，就是12*11*10*9*8*7*6*5*4*3里面质因数里有个几个5，显然10里面有一个，5里面有一个，一共2个。
        n/=5; 
        # 上面求出了 2 之前有几个5后，问题转化为2后面有几个5，所以 n = n / 5, n = 2.
        
    }
    return sum;
}
```



### 202. Happy Number

```python
def leetcode202(n):  # Accept happy number
    def sum_digit(num):
        sum = 0
        while True:
            digit = num - int(num / 10) * 10  # 4316
            sum += digit * digit  # sum += (num % 10)**2 注意这里遍历各个位直接%10，平方**2
            num = int(num / 10)
            if num == 0:
                break
        return sum

    n_list = [n]
    while True:
        n = sum_digit(n)
        if n == 1:
            return True
        if n in n_list:
            return False
        else:
            n_list.append(n)
```

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        temp = set()
        
        def get_next(num):
            sum = 0
            while num:
                sum += (num % 10) ** 2
                num //= 10
            return sum
        
        slow = n
        fast = get_next(n)
        while fast != 1 and slow != fast:
            slow = get_next(slow)
            fast = get_next(get_next(fast))
        return fast == 1
```



### 205. Isomorphic Strings

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        temp1 = dict()
        temp2 = dict()
        for i in range(len(s)):
            if s[i] not in temp1:
                temp1[s[i]] = t[i]
            elif temp1[s[i]] != t[i]:
                return False
            if t[i] not in temp2:
                temp2[t[i]] = s[i]
            elif temp2[t[i]] != s[i]:
                return False
        return True
```

```java
class Solution {
public:
    bool isIsomorphic(string s, string t) {
        int m1[256] = {0}, m2[256] = {0}, n = s.size();
        for (int i = 0; i < n; ++i) {
            if (m1[s[i]] != m2[t[i]]) return false;
            m1[s[i]] = i + 1;
            m2[t[i]] = i + 1;
        }
        return true;
    }
};
```



### 217. Contains Duplicate

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))
```



### 219. Contains Duplicate II

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        temp = dict()
        for i in range(len(nums)):
            if nums[i] in temp:
                if i - temp[nums[i]] <= k:
                    return True
            temp[nums[i]] = i
        return False
```



### 242. Valid Anagram

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        def to_dict(s):
            s_dict = {}
            for each in s:
                if each not in s_dict:
                    s_dict[each] = 1
                else:
                    s_dict[each] += 1
            return s_dict

        s_dict = to_dict(s)
        t_dict = to_dict(t)
        return s_dict == t_dict
```

```python
# use one dict
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        s_dict = {}
        for each in s:
            if each not in s_dict:
                s_dict[each] = 1
            else:
                s_dict[each] += 1
        for each in t:
            if each not in s_dict or s_dict[each] == 0:
                return False
            else:
                s_dict[each] -= 1
        for k, v in s_dict.items():
            if v != 0:
                return False
        return True
```



### 243. Shortest Word Distance

```python
class Solution:
    def shortestDistance(self, words: List[str], word1: str, word2: str) -> int:
        index1 = index2 = -1
        res = len(words)
        for i in range(len(words)):
            if words[i] == word1:
                index1 = i
            elif words[i] == word2:
                index2 = i
            if index1 != -1 and index2 != -1:
                res = min(res, abs(index1-index2))
        return res
```



### 249. Group Shifted Strings

```python
class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        groups = collections.defaultdict(list)
        for s in strings:
            groups[tuple((ord(c) - ord(s[0])) % 26 for c in s)] += s,
        return groups.values()
```



### 258. Add Digits

```c++
# ab = a*10+b
# ab%9 = (a*9+a+b)%9 = (a+b)%9
# abc = a*100+b*10+c
# abc%9 = (a*99+b*9+a+b+c)%9 = (a+b+c)%9
# 38%9=2; 11%9=2
int addDigits(int num) {
    if(num==0) return 0;
    return( (num-1)%9 + 1 ); 
}
```

```python
class Solution:
    def addDigits(self, num: int) -> int:
        while num > 9:
            temp = 0
            while num:
                temp += num % 10
                num //= 10
            num = temp
        return num
```

First you should understand:

```
10^k % 9 = 1
a*10^k % 9 = a % 9 
```

Then let's use an example to help explain.

Say a number x = 23456

x = 2* 10000 + 3 * 1000 + 4 * 100 + 5 * 10 + 6

2 * 10000 % 9 = 2 % 9

3 * 1000 % 9 = 3 % 9

4 * 100 % 9 = 4 % 9

5 * 10 % 9 = 5 % 9

Then x % 9 = ( 2+ 3 + 4 + 5 + 6) % 9, note that x = 2* 10000 + 3 * 1000 + 4 * 100 + 5 * 10 + 6

So we have 23456 % 9 = (2 + 3 + 4 + 5 + 6) % 9

因此，此处蕴含着递归，23456%9 = (2+3+4+5+6)%9 = 20%9 = (2+0)%9 = 2%9 = 2



### 263. Ugly Number

浮点转int

2.999 -> 2

3.000 -> 3

Int (f+0.000001)

---

(a-x)%base = (a+base-x)%base

---

base=9

n%base=[0...8]

(n-1)%base + 1 =[1...9]



### 283. Move Zeroes

```python
# two pointers 一个快一个慢，把所有非零数放到前面后，后面全部置零。
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nonzero_num = 0
        for each in nums:
            if each:
                nonzero_num += 1
        
        slow = 0
        fast = 0
        
        while slow <= nonzero_num-1:
            if nums[fast]:
                nums[slow] = nums[fast]
                slow += 1
                fast += 1
            else:
                fast += 1
        for i in range(slow, len(nums)):
            nums[i] = 0
```

```c++
// 更漂亮的写法
void moveZeroes(vector<int>& nums) {
    int lastNonZeroFoundAt = 0;
    // If the current element is not 0, then we need to
    // append it just in front of last non 0 element we found. 
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != 0) {
            nums[lastNonZeroFoundAt++] = nums[i];
        }
    }
 	// After we have finished processing new elements,
 	// all the non-zero elements are already at beginning of array.
 	// We just need to fill remaining array with 0's.
    for (int i = lastNonZeroFoundAt; i < nums.size(); i++) {
        nums[i] = 0;
    }
}
```

```c++
// 优化，交换slow和fast指针，这样不用最后置零。
void moveZeroes(vector<int>& nums) {
    for (int lastNonZeroFoundAt = 0, cur = 0; cur < nums.size(); cur++) {
        if (nums[cur] != 0) {
            swap(nums[lastNonZeroFoundAt++], nums[cur]);
        }
    }
}
```

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        index = 0 
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[index] = nums[i]
                index += 1
        
        for i in range(index, len(nums)):
            nums[i] = 0
```



### 284. Peeking Iterator

```python
# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator:
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.peek_value = None
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if self.peek_value is not None:
            return self.peek_value
        elif not self.iterator.hasNext():
            raise StopIteration()
        else:
            self.peek_value = self.iterator.next()
            return self.peek_value

    def next(self):
        """
        :rtype: int
        """
        if self.peek_value is not None:
            temp = self.peek_value
            self.peek_value = None
            return temp
        else:
            return self.iterator.next()
        

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.peek_value is not None:
            return True
        else:
            return self.iterator.hasNext()
        

# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].
```

```python
class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self._next = iterator.next()
        self._iterator = iterator
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self._next
        

    def next(self):
        """
        :rtype: int
        """
        if self._next is None:
            raise StopIteration()
        temp = self._next
        self._next = None
        if self._iterator.hasNext():
            self._next = self._iterator.next()
        return temp

    def hasNext(self):
        """
        :rtype: bool
        """
        return self._next is not None
```



### 287. Find the Duplicate Number

```python
# 自认为很漂亮的方法，原理是因为有n+1个数且只有一个数字重复，并且每个数字的取值范围在[1, n], 所以只要把每个数字方法它该放的位置，也就是说把1放到0, 2放到1, 3放到2, 4放到3...最后一个数字便是重复的数字。当然在遍历过程中如果遇到在该位置上已经有了正确的数字直接返回该数字
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        if not nums:
            return
        i = 0
        while i <= len(nums)-2:
            if nums[i] == i+1:
                i += 1
                continue
            if nums[nums[i]-1] == nums[i]:
                return nums[i]
            else:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        return nums[len(nums)-1]
```

```python
# Floyd's Tortoise and Hare (Cycle Detection)
# same as Linked List Cycle II 
class Solution:
    def findDuplicate(self, nums):
        # Find the intersection point of the two runners.
        tortoise = nums[0]
        hare = nums[0]
        while True:
            tortoise = nums[tortoise]
            hare = nums[nums[hare]]
            if tortoise == hare:
                break
        
        # Find the "entrance" to the cycle.
        ptr1 = nums[0]
        ptr2 = tortoise
        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]
        
        return ptr1
```



### 288. Unique Word Abbreviation

```python
class ValidWordAbbr:
    from collections import defaultdict
    def __init__(self, dictionary: List[str]):
        self._dict = defaultdict(set)
        for each in dictionary:
            self._dict[self.get_abbr(each)].add(each)
        
    def get_abbr(self, word: str) -> str:
        if len(word) <= 2:
            return word
        return word[0] + str(len(word)-2) + word[-1]

    def isUnique(self, word: str) -> bool:
        addr = self.get_abbr(word)
        words = self._dict.get(addr)
        return words is None or len(words) == 1 and word in words
```



### 290. Word Pattern

```python
class Solution:
    def wordPattern(self, pattern: str, str: str) -> bool:
        str_list = str.split(" ")
        if len(pattern) != len(str_list):
            return False
        temp = dict()
        temp_r = dict()
        for i in range(len(str_list)):
            if pattern[i] in temp and str_list[i] in temp_r:
                if temp[pattern[i]] != str_list[i] or temp_r[str_list[i]] != pattern[i]:
                    return False
            elif pattern[i] in temp or str_list[i] in temp_r:
                return False
            else:
                temp[pattern[i]] = str_list[i]
                temp_r[str_list[i]] = pattern[i]
        return True
```



### 292. Nim Game

```c++
bool canWinNim(int n) {
    return (n%4!=0);
}
```

```c++
bool canWinNim(int n) {
    vector<bool> f(n+1, false);
    for(int i=0;i<n;i++)
    {
        if(!f[i])
        {
            for(int j=1;j<=3;j++)
            {
                f[i+j]=true;
            }
        }
    }
    return f[n];
}
```



如果n存在一种方式一步到达必输状态

存在x，f[n-x]==必输 其中1<=x<=3 那么f[n]则是必胜

不存在x就是必输状态，就是怎么走别人都是必胜

![image-20180619182030574](/var/folders/8g/vw4sgnn93vb5whntp_n610g80000gn/T/abnerworks.Typora/image-20180619182030574.png)

问题2:

两堆石子一堆a个，一堆b个，甲乙两个人取石子

每次2种取法：

1.从任意一堆中取走任意个

2.从两堆中取走相同个

取走最后一个的获胜



### 326. Power of Three

```c++
bool is PowerOfThree(int n){
    //3 9 27 81
    //big3 % n==0; big3 是大于n的3的幂，如果等式成立则n也是3的幂。则问题转化为求big3:int范围内最大的3的幂，此方法只对素数有用
    //big3 = 3^k;
    //k=log3(maxint);
    const int maxint = 0x7fffffff;
    int k = log(maxint)/log(3);
    int big3 = pow(3,k);
    return (big3 % n == 0);
}
```



### 342. Power of Four

```c++
bool isPowerOfFour(int n){
    //2 8 false 2 : 10 8 : 1000
    //4 16 true
    //5 101
    //0101 & 8 1000 = 0000
    //0101 & 4 0100 = 0100
    if( n<=0 ) return false;
    return ((n&(n-1))==0 && (n&0x55555555) );
}
```



### 344. Reverse String

```java
class Solution {
    public void reverseString(char[] s) {
        if (s.length == 0) return;
        int low = 0;
        int high = s.length-1;
        while (low < high){
            char temp = s[low];
            s[low] = s[high];
            s[high] = temp;
            low++;
            high--;
        }
    }
}
```

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def reverseString_helper(i1, i2):
            if i1 < i2:
                s[i1], s[i2] = s[i2], s[i1]
                reverseString_helper(i1+1, i2-1)
        
        reverseString_helper(0, len(s)-1)
```



### 346. Moving Average from Data Stream

```python
class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.size = size
        self.q = []

    def next(self, val: int) -> float:
        self.q.append(val)
        if len(self.q) <= self.size:
            return sum(self.q) / len(self.q)
        else:
            return sum(self.q[-self.size:]) / self.size
```



### 347. Top K Frequent Elements

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        from collections import Counter
        counter = Counter(nums)
        counter = sorted([(key, value) for key, value in counter.items()], key=lambda t:t[1], reverse=True)
        return [each[0] for each in counter[:k]]
```



### 359. Logger Rate Limiter

```python
class Logger:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.recorder = dict()
        

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """
        if message not in self.recorder:
            self.recorder[message] = timestamp
            return True
        else:
            if timestamp - self.recorder[message] < 10:
                return False
            else:
                self.recorder[message] = timestamp
                return True
```



### 380. Insert Delete GetRandom O(1)

```python
class RandomizedSet:
    import random
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._list = []
        self._dict = {}
        

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self._dict:
            return False
        self._dict[val] = len(self._list)
        self._list.append(val)
        return True
        

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self._dict:
            return False
        index_delete, last_element = self._dict[val], self._list[-1]
        self._list[index_delete], self._dict[last_element] = last_element, index_delete
        self._list.pop()
        del self._dict[val]
        return True
        
        

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return random.choice(self._list)
```



 ### 387. First Unique Character in a String

```python
# two years ago since 3/28/2020
class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        is_have_repeat = False
        for i in range(len(s)):
            for j in range(len(s)):
                if i == j:
                    continue
                if s[i] == s[j]:
                    is_have_repeat = True
                    break
                else:
                    is_have_repeat = False
            if is_have_repeat == False:
                return i
        return -1
```

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        from collections import defaultdict
        temp = defaultdict(int)
        for each in s:
            temp[each] += 1
        for i, val in enumerate(s):
            if temp[val] == 1:
                return i
        return -1
```

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        dict_ = dict()
        for i, val in enumerate(s):
            if val not in dict_:
                dict_[val] = [1, i]
            else:
                dict_[val][0] += 1
        temp = [v[1] for k, v in dict_.items() if v[0] == 1]
        return min(temp) if temp else -1
```



### 392. Is Subsequence

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        index_s = index_t = 0
        while index_s < len(s) and index_t < len(t):
            if t[index_t] == s[index_s]:
                index_t += 1
                index_s += 1
            else:
                index_t += 1
        return index_s == len(s)
```



###  406. Queue Reconstruction by Height

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people_sort = sorted(people, key=lambda k: (-k[0], k[1]))
        ans = []
        for each in people_sort:
            ans.insert(each[1], each)
        return ans
```



### 409. Longest Palindrome

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        from collections import Counter
        count = Counter(s)
        if_odd = False
        ans = 0
        for v in count.values():
            ans += (v // 2) * 2
            if v % 2 != 0:
                if_odd = True
        return ans if not if_odd else ans+1
```



### 410. Split Array Largest Sum

```java
public boolean guess(long mid, int[] nums, int m) {
    long sum = 0;
    long mm = 0;
    for (int i=0;i<nums.length;++i) {
        if (sum + nums[i] > mid) {
            ++mm;
            sum = nums[i];
            if (nums[i] > mid) {
                return false;
            }
        }
            else {
                sum += nums[i]
            }
    }
    return mm < m;
}

public int splitArray(int[] nums, int m) {
    long n = nums.length;
    long R = 1; //m >= 1
    for (int i=0;i<n;++1) {
        R += nums[i];
    }
    long L = 0;
    long ans = 0;
    while (L < R) {
        long mid = (L + R) / 2;
        if (guess(mid, nums, m)) {
            ans = mid;
            R = mid;
        } else {
            L = mid + 1
        }
    }
    return (int) ans;
}
```



### 447. Number of Boomerangs

```python
class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        from collections import defaultdict
        temp = defaultdict(int)
        
        def get_instance(x, y):
            a = x[0] - y[0]
            b = x[1] - y[1]
            return a*a + b*b
        
        ans = 0
        
        for i in range(len(points)):
            for j in range(len(points)):
                if i == j:
                    continue
                d = get_instance(points[i], points[j])
                temp[d] += 1
            for val in temp.values():
                ans += val*(val-1)
            temp = defaultdict(int)
        return ans
```



### 454. 4Sum II

```python
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        AB = collections.Counter(a+b for a in A for b in B)
        return sum([AB[-(c+d)] for c in C for d in D])
```

We add up a and b and store the result-frequency pair into AB.
E.g.
AB[5] = 2 means a+b=5 appears 2 times in the a+b scenario.
Then we are looking for how many times does c+d = -5 appear so that it could be paired with AB[5] and form a 0.
That's why we then look for AB[-c-d] (or AB[-(c+d)] )



### 459. Repeated Substring Pattern

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return s in (s + s)[1:-1]
```



### 470. Implement Rand10() Using Rand7()

```python
class Solution:
    def rand10(self):
        """
        :rtype: int
        """
        rand40 = 40
        while rand40 >= 40:
            rand40 = (rand7() - 1) * 7 + rand7() - 1
        return rand40 % 10 + 1
```



### 509. Fibonacci Number

```python
class Solution:
    def fib(self, N: int) -> int:
        if N < 2:
            return N
        return self.fib(N-1) + self.fib(N-2)
```

```python
class Solution:
    def fib(self, N: int) -> int:
        
        cache = dict()
        
        def fib_helper(N):
            if N < 2:
                return N
            if N in cache:
                return cache.get(N)
            else:
                temp = fib_helper(N-1) + fib_helper(N-2)
                cache[N] = temp
                return temp
        return fib_helper(N)
```



### 556. Next Greater Element III

```python
class Solution:
    def nextGreaterElement(self, n: int) -> int:
        if n <= 10: return -1
        nList = [int(each) for each in list(str(n))]
        i = len(nList) - 2
        while i >= 0:
            if nList[i] < nList[i+1]:
                break
            i -= 1
        if i == -1: return -1
        smallestIndex = i + 1
        for j in range(i+2, len(nList)):
            if nList[j] > nList[i] and nList[j] < nList[smallestIndex]:
                smallestIndex = j
        nList[i], nList[smallestIndex] = nList[smallestIndex], nList[i] 
        res = int("".join([str(each) for each in nList[:i+1] + sorted(nList[i+1:])]))
        return res if res <= 0x7fffffff else -1
```

https://leetcode.com/problems/next-greater-element-iii/discuss/101824/Simple-Java-solution-(4ms)-with-explanation.

n1 = 0x80000000; *// 最大负数, -2147483648* 

n2 = 0x7fffffff; *// 最大正数, 2147483647*



### 557. Reverse Words in a String III

```python
class Solution(object):
    def reverseWords(self, s):
        fArray = []
        strArray = s.split(" ")
        for eachstr in strArray:
            a = list(eachstr)
            a.reverse()
            a = "".join(a)
            fArray.append(a)
        return " ".join(fArray)
```

```python
# 10/12/2019
# two pointers 
class Solution:
    def reverseWords(self, s: str) -> str:
        if not s:
            return s
        
        s = list(s)
        
        def reverseOneWord(s, head, tail):
            if not s:
                return
            
            while head < tail:
                temp = s[head]
                s[head] = s[tail]
                s[tail] = temp
                head += 1
                tail -= 1
            
        head = 0
        for i in range(len(s)):
            if s[i] == ' ':
                reverseOneWord(s, head, i-1)
                head = i+1
        reverseOneWord(s, head, len(s)-1)
        
        return "".join(s)
```

```python
# 10/12/2019
# 反向遍历，直接添加
class Solution:
    def reverseWords(self, s: str) -> str:
        if not s:
            return s
        
        ans = ""
        temp = ""
        
        i = len(s)-1
        while i >= 0:
            if s[i] != " ":
                temp += s[i]
            else:
                temp = " " + temp
                ans = temp + ans
                temp = ""
            i -= 1
        return temp + ans
```



### 593. Valid Square

```python
class Solution:
    def validSquare(self, p1: List[int], p2: List[int], p3: List[int], p4: List[int]) -> bool:
        
        def distance(a, b):
            return (a[0] - b[0])**2 + (a[1] - b[1])**2
        
        p1p2 = distance(p1, p2)
        p1p3 = distance(p1, p3)
        p1p4 = distance(p1, p4)
        temp = set([p1p2, p1p3, p1p4])
        
        if len(temp) != 2:
            return False
        edge, diagonal = min(temp), max(temp)
        if not edge or not diagonal: return False
        
        if p1p2 == diagonal:
            if distance(p2, p3) != edge or distance(p2, p4) != edge or distance(p3, p4) != diagonal:
                return False
        elif p1p3 == diagonal:
            if distance(p2, p3) != edge or distance(p3, p4) != edge or distance(p2, p4) != diagonal:
                return False
        elif p1p4 == diagonal:
            if distance(p4, p3) != edge or distance(p2, p4) != edge or distance(p2, p3) != diagonal:
                return False
        return True
```



### 599. Minimum Index Sum of Two Lists

```python
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        dict1 = {list1[i]:i for i in range(len(list1))}
        dict2 = {list2[i]:i for i in range(len(list2))}
        res = []
        for each in dict1:
            if each in dict2:
                res.append((each, dict1[each]+dict2[each]))
        
        res.sort(key=lambda x:x[1])
        
        for i in range(1, len(res)):
            if res[i][1] != res[0][1]:
                return [each[0] for each in res[:i]]
        return [each[0] for each in res]
```

```python
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        temp = dict()
        ans = []
        import sys
        min_index = sys.maxsize
        for i, val in enumerate(list1):
            temp[val] = i
        for i, val in enumerate(list2):
            if val in temp:
                if i + temp[val] < min_index:
                    ans = [val]
                    min_index = i + temp[val]
                elif i + temp[val] == min_index:
                    ans.append(val)
        return ans
```



### 621. Task Scheduler

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        from collections import Counter
        counter = Counter(tasks)
        maxCount = max(counter.values())
        sumMaxCount = 0
        for k, v in counter.items():
            if v == maxCount:
                sumMaxCount += 1
        
        parts = maxCount - 1
        free_for_each_part = n - (sumMaxCount - 1)
        all_free_part = parts * free_for_each_part
        not_max_tasks = len(tasks) - maxCount * sumMaxCount
        remain_free_part = all_free_part - not_max_tasks
        free_part = max(0, remain_free_part)
        
        return len(tasks) + free_part
```

https://leetcode.com/problems/task-scheduler/discuss/104500/Java-O(n)-time-O(1)-space-1-pass-no-sorting-solution-with-detailed-explanation



### 628.Maximum Product of Three Numbers

```python
        '''
        if len(nums) == 3:
            return nums[-1]*nums[-2]*nums[-3]
        
        nums.sort()
        
        if nums[0] >= 0 or nums[-1] <= 0:
            return nums[-1]*nums[-2]*nums[-3]
        
        if len(nums) >= 5:
            if nums[1] <0 and nums[-3] > 0:
                return max(nums[0]*nums[1]*nums[-1],nums[-1]*nums[-2]*nums[-3])
        
        if len(nums) >= 4:
            if (nums[0] < 0 and nums[1] <= 0 and nums[2] >= 0 and nums[3] > 0) or nums[-2] <= 0:
                return nums[0]*nums[1]*nums[-1]
            if nums[0] < 0 and nums[1] >= 0:
                return nums[-1]*nums[-2]*nums[-3]
        '''
        nums.sort()
        return max(nums[0]*nums[1]*nums[-1],nums[-1]*nums[-2]*nums[-3])
```

对数组进行排序非常耗时



### 665. Non-decreasing Array

```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        temp = None
        
        for i in range(len(nums)-1):
            if nums[i] > nums[i+1]:
                if temp is not None:
                    return False
                temp = i
        
        if temp is None or temp == 0 or temp == len(nums) - 2:
            return True
        if nums[temp-1] <= nums[temp+1] or nums[temp] <= nums[temp+2]:
            return True
        return False
```

查找不符合降序排列的index，如果此index个数大于1直接返回false

如果此index为None或0或为倒数第二个元素可直接返回true

否则判断此index前后两个组合是否符合条件：比如[2,3,3,2,4]. 不符合的index==2. 那么判断其前后两个组合[3,3,2] 和 [3,2,4]其中一个可以即可，即nums[index-1] <= nums[index+1] or nums[index] <= nums[index+2]



### 680. Valid Palindrome II

```python
# 07/14/2019
class Solution:
    def validPalindrome(self, s: str) -> bool:
        # eabbade
        # edabbae

        pos = 0
        neg = 0

        for i in range(len(s)):
            if i > len(s)//2-1:
                return True
            if s[i+pos] != s[len(s)-1-i-neg]:
                if pos+neg !=0:
                    return False
                if s[i+1] == s[len(s)-1-i]:
                    if i+2 < len(s):
                        if s[i+2] == s[len(s)-1-i-1]:
                            pos = 1
                    else:
                        pos = 1
                elif s[i] == s[len(s)-1-i-1] and s[i+1] == s[len(s)-1-i-2]:
                    neg = 1
                else:
                    return False
```

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        
        def helper(left, right):
            while left < right:
                if s[left] != s[right]: return False
                left += 1
                right -= 1
            return True
        
        left, right = 0, len(s) - 1
        while left < right :
            if s[left] != s[right]:
                return helper(left+1, right) or helper(left, right-1)
            left += 1
            right -= 1
        return True
```



### 705. Design HashSet

```python
class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keyRange = 769
        self.bucketArray = [Bucket() for i in range(self.keyRange)]
        
    def _hash(self, key):
        return key % self.keyRange

    def add(self, key: int) -> None:
        index = self._hash(key)
        self.bucketArray[index].insert(key)

    def remove(self, key: int) -> None:
        index = self._hash(key)
        self.bucketArray[index].delete(key)
        

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        index = self._hash(key)
        return self.bucketArray[index].exist(key)

class Node(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class Bucket(object):
    def __init__(self):
        self.head = Node(0)
    
    def insert(self, val):
        if not self.exist(val):
            temp = self.head.next
            new_node = Node(val, temp)
            self.head.next = new_node
    
    def exist(self, val):
        cur = self.head.next
        while cur:
            if cur.val == val:
                return True
            cur = cur.next
        return False
    
    def delete(self, val):
        prev, curr = self.head, self.head.next
        while curr:
            if curr.val == val:
                prev.next = curr.next
                return
            prev, curr = curr, curr.next

# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
```



### 706. Design HashMap

```python
class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 2069
        self.hash_table = [Bucket() for i in range(self.size)]
        
    def _hash(self, key):
        return key % self.size
    
        
    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        index = self._hash(key)
        self.hash_table[index].update(key, value)
        

    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        index = self._hash(key)
        return self.hash_table[index].get(key)
        

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        index = self._hash(key)
        self.hash_table[index].remove(key)
        
class Bucket(object):
    def __init__(self):
        self.bucketList = []
    
    def get(self, key):
        for (k, v) in self.bucketList:
            if k == key:
                return v
        return -1
    
    def update(self, key, value):
        for i, kv in enumerate(self.bucketList):
            if kv[0] == key:
                self.bucketList[i] = (key, value)
                return
        self.bucketList.append((key, value))
    
    def remove(self, key):
        for i, kv in enumerate(self.bucketList):
            if kv[0] == key:
                del self.bucketList[i]
        


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```



### 752. Open the Lock

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        deadends = set(deadends)
        if target in deadends or '0000' in deadends:
            return -1
        
        ans = 0
        
        Q = ['0000']
        seen = set()
        
        while Q:
            size = len(Q)
            for i in range(size):
                curr = Q.pop(0)
                if curr == target:
                    return ans
                for j in range(4):
                    curr_digit = curr[j]
                    if curr_digit == '0':
                        temp1 = '1'
                        temp2 = '9'
                    elif curr_digit == '9':
                        temp1 = '0'
                        temp2 = '8'
                    else:
                        temp1 = chr(ord(curr_digit)+1)
                        temp2 = chr(ord(curr_digit)-1)
                    temp1 = curr[:j] + temp1 + curr[j + 1:]
                    if temp1 not in seen:
                        seen.add(temp1)
                        if temp1 in deadends:
                            deadends.remove(temp1)
                        else:
                            Q.append(temp1)
                    temp2 = curr[:j] + temp2 + curr[j + 1:]
                    if temp2 not in seen:
                        seen.add(temp2)
                        if temp2 in deadends:
                            deadends.remove(temp2)
                        else:
                            Q.append(temp2)
            ans += 1
        return -1
```

列表的in操作比较耗时，可以remove该元素如果之后不再需要的话，remove后可以优化时间，但也比较有限。将 deadends 转为 set 后时间有明显缩短！因为 `Sets are significantly faster when it comes to determining if an object is present in the set (as in x in s)` set 操作是将列表转为字典，字典in操作平均为O(1)



### 779. K-th Symbol in Grammar

```python
class Solution:
    def kthGrammar(self, N: int, K: int) -> int:
        if N == 1 or K == 1:
            return 0
        if K % 2 == 0:
            return 1 - self.kthGrammar(N-1, K//2)
        else:
            return self.kthGrammar(N-1, K//2+1)
```

https://leetcode.com/problems/k-th-symbol-in-grammar/discuss/438528/Explanation-Python



### 811. Subdomain Visit Count

```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        from collections import Counter
        count = Counter()
        
        for each in cpdomains:
            num, domain = each.split(" ")
            domain_list = domain.split(".")
            for i in range(len(domain_list)-1, -1, -1):
                count[domain_list[i]] += int(num)
                if i > 0:
                    domain_list[i-1] += "." + domain_list[i]
        return ["{} {}".format(num, domain) for domain, num in count.items()]
```



### 819. Most Common Word

```python
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        ban = set(banned)
        par_list = re.findall(r'\w+', paragraph.lower())
        
        from collections import Counter
        par_dict = Counter(par_list)
        par_order_list = sorted([(key, value) for key, value in par_dict.items()], key=lambda t:t[1], reverse=True)
        for each in par_order_list:
            if each[0] not in ban:
                return each[0]
```



### 841. Keys and Rooms

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        Q = rooms[0]
        room_num_set = set(rooms[0])
        room_num_set.add(0)
        while Q:
            room_num = Q.pop(0)
            for each in rooms[room_num]:
                if each not in room_num_set:
                    Q.append(each)
                    room_num_set.add(each)
            
        if len(room_num_set) == len(rooms):
            return True
        else:
            return False
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



### 925. Long Pressed Name

```python
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        if len(name) > len(typed) or name[0] != typed[0]:
            return False
        i = j = 0
        while i < len(name) and j < len(typed):
            if name[i] == typed[j]:
                i += 1
                j += 1
            else:
                if typed[j] == typed[j-1]:
                    j += 1
                else:
                    return False
        if i < len(name):
            return False
        
        while j < len(typed):
            if typed[j] == typed[j-1]:
                j += 1
            else:
                return False
        return True
```



### 933. Number of Recent Calls

```python
class RecentCounter:

    def __init__(self):
        self.count = 0
        self._list = []

    def ping(self, t: int) -> int:
        self.count += 1
        self._list.append(t)
        if len(self._list) > 1 and t - self._list[0] > 3000:
            for i, val in enumerate(self._list):
                if t - val > 3000:
                    self.count -= 1
                else:
                    self._list = self._list[i:]
                    break
        return self.count
```



### 937. Reorder Data in Log Files

```python
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def f(log):
            id_, rest = log.split(" ", 1)
            return (0, rest, id_) if rest[0].isalpha() else (10,)

        return sorted(logs, key = f)
```



### 939. Minimum Area Rectangle

```python
class Solution:
    def minAreaRect(self, points: List[List[int]]) -> int:
        if not points:
            return
        import sys
        ans = sys.maxsize
        S = set(map(tuple, points))
        for i in range(len(points)):
            for j in range(i):
                p1 = points[i]
                p2 = points[j]
                if p1[0] != p2[0] and p1[1] != p2[1] and (p1[0], p2[1]) in S and (p2[0], p1[1]) in S:
                    ans = min(ans, abs(p2[1]-p1[1]) * abs(p2[0] - p1[0]))
        return ans if ans < sys.maxsize else 0
```



### 948. Bag of Tokens

```python
class Solution:
    def bagOfTokensScore(self, tokens: List[int], P: int) -> int:
        tokens.sort()
        queue = collections.deque(tokens)
        ans = cur = 0
        while queue and (cur or P >= queue[0]):
            while queue and P >= queue[0]:
                P -= queue.popleft()
                cur += 1
            
            ans = max(ans, cur)
            if queue:
                P += queue.pop()
                cur -= 1
        return ans
```



### 967. Numbers With Same Consecutive Differences

```python
class Solution:
    def numsSameConsecDiff(self, N: int, K: int) -> List[int]:
        if N == 1:
            return [i for i in range(10)]
        
        ans = []
        
        def dfs(N, num):
            if N == 0:
                ans.append(num)
                return
            
            last_digit = num % 10
            last_digit_list = set([last_digit + K, last_digit - K])
            for digit in last_digit_list:
                if digit >= 0 and digit <= 9:
                    new_num = num * 10 + digit
                    dfs(N-1, new_num)
        
        for i in range(1, 10):
            dfs(N-1, i)
        return ans
```

https://leetcode.com/problems/numbers-with-same-consecutive-differences/solution/



### 973. K Closest Points to Origin

```python
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        dist = lambda i: points[i][0]**2 + points[i][1]**2

        def sort(i, j, K):
            # Partially sorts A[i:j+1] so the first K elements are
            # the smallest K elements.
            if i >= j: return

            # Put random element as A[i] - this is the pivot
            k = random.randint(i, j)
            points[i], points[k] = points[k], points[i]

            mid = partition(i, j)
            if K < mid - i + 1:
                sort(i, mid - 1, K)
            elif K > mid - i + 1:
                sort(mid + 1, j, K - (mid - i + 1))

        def partition(i, j):
            # Partition by pivot A[i], returning an index mid
            # such that A[i] <= A[mid] <= A[j] for i < mid < j.
            oi = i
            pivot = dist(i)
            i += 1

            while True:
                while i < j and dist(i) < pivot:
                    i += 1
                while i <= j and dist(j) >= pivot:
                    j -= 1
                if i >= j: break
                points[i], points[j] = points[j], points[i]

            points[oi], points[j] = points[j], points[oi]
            return j

        sort(0, len(points) - 1, K)
        return points[:K]
```



### 994. Rotting Oranges

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        
        from collections import deque
        q = deque()
        
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == 2:
                    q.append((row, col, 0))
        
        def get_n(row, col):
            for row, col in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
                    yield row, col
        ans = 0
        while q:
            row, col, depth = q.popleft()
            for r, c in get_n(row, col):
                if grid[r][c] == 1:
                    grid[r][c] = 2
                    ans = depth+1
                    q.append((r, c, depth+1))
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == 1:
                    return -1
        return ans
```



### 997. Find the Town Judge

```python
class Solution:
    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        from collections import defaultdict
        
        trust_dict = defaultdict(set)
        
        for a, b in trust:
            trust_dict[a].add(b)
        
        if len(trust_dict) == N-1:
            for i in range(1, N+1):
                if i not in trust_dict:
                    judge_ = i
                    break
            for val in trust_dict.values():
                if judge_ not in val:
                    return -1
            return judge_
        return -1
```



### 1002. Find Common Characters

```python
class Solution:
    def commonChars(self, A: List[str]) -> List[str]:
        ans = []
        temp = collections.defaultdict(dict)
        for index, string in enumerate(A):
            for c in string:
                if index+1 not in temp[c]:
                    temp[c].update({index+1: 1})
                else:
                    temp[c][index+1] += 1
        for c, v_dict in temp.items():
            if len(v_dict) == len(A):
                for i in range(min(v_dict.values())):
                    ans.append(c)
        return ans
```



### 1029. Two City Scheduling

```python
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        costs.sort(key=lambda k: k[1]-k[0], reverse=True)
        ans = 0
        print(costs)
        for i in range(len(costs)//2):
            ans += costs[i][0]
        for i in range(len(costs)//2, len(costs)):
            ans += costs[i][1]
        return ans
```
https://leetcode.com/problems/two-city-scheduling/discuss/278716/C%2B%2B-O(n-log-n)-sort-by-savings



### 1041. Robot Bounded In Circle

```python
# https://leetcode.com/problems/robot-bounded-in-circle/discuss/456726/Java-Solution-Easy-to-Understand-and-Efficient
# 最后形成圈的条件有二：1. 完成一遍instructions后仍回到原点。2. 完成一遍后没有回到原点但是方向变了
# index 代表方向，0: north, 1: west, 2: south, 3: east 相对应的移动方向为[[0, 1], [-1, 0], [0, -1], [1, 0]]
# 向左转与上面list顺序相同所以+1
# 向右转与上面list顺序相反所以+3
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        x = y = index = 0
    
        direction = [[0, 1], [-1, 0], [0, -1], [1, 0]]
        
        for each in instructions:
            if each == 'L':
                index = (index + 1) % 4
            elif each == 'R':
                index = (index + 3) % 4
            else:
                x += direction[index][0]
                y += direction[index][1]
        return (x == 0 and y == 0) or index != 0
```



### 1046. Last Stone Weight

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        h = [-each for each in stones]
        heapq.heapify(h)
        while len(h) > 1:
            temp = heapq.heappop(h) - heapq.heappop(h)
            if temp != 0:
                heapq.heappush(h, temp)
        return -h[0] if h else 0
```



### 1094. Car Pooling

 ```python
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        
        temp = []
        for p_num, start, end in trips:
            temp += [[start, p_num], [end, -p_num]]
        
        count = 0
        temp.sort()
        
        
        for _, p_num in temp:
            count += p_num
            if count > capacity:
                return False
        return True
 ```



### 1099. Two Sum Less Than K

```python
class Solution:
    def twoSumLessThanK(self, A: List[int], K: int) -> int:
        sort_A = sorted(A)
        i, j = 0, len(A) - 1
        ans = -1
        while i < j:
            if sort_A[i] + sort_A[j] < K:
                ans = max(ans, sort_A[i]+sort_A[j])
                i += 1
            else:
                j -= 1
        return ans
```



### 1103. Distribute Candies to People

```python
# not finished
class Solution:
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        temp = 0
        for i in range(1, 10**9+1):
            if temp + i > candies:
                break
            temp += i
        count, remain = i-1, candies - temp
        print(count)
        print(remain)
        rows = count // num_people
        remain_rows = count % num_people
        print(rows)
        print(remain_rows)
        ans = [0] * num_people
        for i in range(num_people):
            ans[i] = ((i+1) + (i+1)+(rows-1)*num_people) * rows // 2
        print(ans)
        for i in range(remain_rows):
            ans[i] += (i+1)+(rows-1)*num_people + num_people
        ans[remain_rows-1] += remain
        return ans
```

```python
class Solution:
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        ans = [0] * num_people
        given = 0
        while candies > 0:
            ans[given%num_people] += min(candies, given+1)
            given += 1
            candies -= given
        return ans
```



### 1119. Remove Vowels from a String

```python
class Solution:
    def removeVowels(self, S: str) -> str:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return ("").join([each for each in S if each not in vowels])
```



### 1160. Find Words That Can Be Formed by Characters

```python
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        from collections import Counter
        ans = 0
        chars_counter = Counter(chars)
        for word in words:
            word_counter = Counter(word)
            flag = True
            for k, v in word_counter.items():
                if k not in chars_counter or v > chars_counter[k]:
                    flag = False
                    break
            if flag:
                ans += len(word)
        return ans
```



### 1207. Unique Number of Occurrences

```python
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        from collections import Counter
        count = Counter(Counter(arr).values())
        for each in count.values():
            if each != 1:
                return False
        return True
```



### 1232. Check If It Is a Straight Line

```python
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        if not coordinates:
            return False
        if len(coordinates) <= 2:
            return True
        
        slope = None
        for i in range(1, len(coordinates)):
            point1 = coordinates[i-1]
            point2 = coordinates[i]
            new_slope = (point2[1]-point1[1])/(point2[0]-point1[0]) if point2[0] != point1[0] else sys.maxsize
            if slope is None:
                slope = new_slope
            elif slope != new_slope:
                return False
        return True
```



### 1306. Jump Game III

```python
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        if start < len(arr) and start >= 0 and arr[start] >= 0:
            arr[start] = -arr[start]
            return arr[start] == 0 or self.canReach(arr, start + arr[start]) or self.canReach(arr, start - arr[start])
        return False
```



### 1344. Angle Between Hands of a Clock

```python
class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        hour %= 12
        minutes_degree = minutes / 60 * 360
        hour_degree = hour / 12 * 360 + 30 * minutes / 60
        ans = abs(minutes_degree - hour_degree)
        return ans if ans < 180 else 360 - ans
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



### 1365. How Many Numbers Are Smaller Than the Current Number

```python
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        from collections import Counter
        
        count = Counter(nums)
        ans = []
        for each in nums:
            temp = 0
            for _ in count:
                if _ < each:
                    temp += count[_]
            ans.append(temp)
        return ans
```

```python
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        array = [0] * 102
        for num in nums:
            array[num+1] += 1
        for i in range(1, len(array)):
            array[i] += array[i-1]
        return [array[each] for each in nums]
```



### 1387. Sort Integers by The Power Value

```python
class Solution:
    def getKth(self, lo: int, hi: int, k: int) -> int:
        temp = {1:0}
        
        def helper(num):
            if num in temp:
                return temp.get(num)
            
            if num % 2 == 0:
                temp[num] = helper(num // 2) + 1
            else:
                temp[num] = helper(3 * num + 1) + 1
            return temp[num]
        
        
        for i in range(lo, hi+1):
            helper(i)
        
        return sorted(range(lo, hi+1), key=temp.__getitem__)[k-1]
```

```python
    def getKth(self, lo, hi, k):
        return sorted(range(lo, hi + 1), key='寒寒寓寙寔寗寚寢寕寥寘寠寛寛寣寣寖寞實實寙寙寡寡寜審寜屁寤寤寤尼寗寬察察寧寧寧寴寚尿寚寯寢寢寢尺寝寪寪寪寝寝层层寥寲寥寲寥寥尽尽寘寭寭寭寠寠寠尸寨居寨寠寨寨寵寵寛寨局局寛寛寰寰寣寰寣尮寣寣尻尻寞屈寫寫寫寫寫尩寞寸寞尶屃屃屃尗實寞寳寳實實寳寳實就實尀尾尾尾尀寙屋寮寮寮寮寮寻寡尬寡寻寡寡尹尹審屆屆屆審審寡寡審寶審尧寶寶寶專寜尴審審屁屁屁尕寜尃寜屎寱寱寱尢寤寱寱寱寤寤尯尯寤対寤対尼尼尼対察屉屉屉寬寬寬屉寬寤寬对寬寬尪尪察对对对察察尷尷屄寬屄将屄屄尘尘寧将察察寴寴寴屑寧尥寧屑寴寴寴将寧寧尲尲寧寧封封尿封尿尓尿尿封封寚屌屌屌寯寯寯尠寯屌寯寧寯寯导导寢寯尭尭寢寢导导寢导寢導尺尺尺导寪寯屇屇屇屇屇尉寪尛寪屇寢寢寢导寪寷寷寷寪寪尨尨寷屔寷寷寷寷尉尉寝寪尵尵寪寪寪屡层射层寪层层尖尖寝层射射寝寝屏屏寲屏寲屏寲寲尣尣寥屏寲寲寲寲寲射寥寿寥寿尰尰尰寿寥寥寿寿寥寥寿寿尽少尽尌尽尽寿寿寠寲届届届届届届寭尌寭尞寭寭届届寭寥寥寥寭寭寺寺寭寺寭屗尫尫尫屗寠屗寺寺寺寺寺寲寠尌寠將尸尸尸寺居寭寭寭居居將將居寭居將尙尙尙尳寨居將將寠寠寠寺寵屒寵屒寵寵屒屒寨寵尦尦寨寨屒屒寵寵寵寭寵寵將將寨専寨寨尳尳尳屟寨専寨屟専専専尳局寨専専局局尔尔局小局寵専専専小寛寵屍屍屍屍屍小寰屍寰屍寰寰尡尡寰寰屍屍寰寰寨寨寰寨寰専寽寽寽屚寣寽寰寰尮尮尮寽寣屚寣寰寽寽寽尩寣寽寽寽寣寣小小尻尊尻寰尻尻寽寽寫寰寰寰屈屈屈寰屈尊屈屈屈屈尊尊寫尜尜尜寫寫屈屈寣尊寣尗寣寣寽寽寫展寸寸寸寸寸尗寫展寫展尩尩尩展寸寫展展寸寸寸寸寸寰寸寰尊尊尊展寞尅寫寫尶尶尶寸寫屢寫尶寫寫屢屢屃尅尅尅屃屃寫寫屃尅屃屢尗尗尗就寞尒屃屃尅尅尅尒寞尒寞寸屐屐屐寸寳屐屐屐寳寳屐屐寳屐寳尒尤尤尤屼實寳屐屐寳寳寳尒寳寫寳寫寳寳尅尅實尀尀尀實實尀尀就寳就屝就就尀尀實屝實實尀尀尀就實尬實尀尀尀尀屝尾實尒尒尾尾對對尾寳尾屪尀尀尀對寡寳寳寳屋屋屋屪屋寳屋對屋屋屋屋寮屋對對寮寮尟尟寮尟寮尹屋屋屋尚寮對實實實實實尚寮尀寮屘寻寻寻屘寮寻寻寻寮寮屘屘尬屘尬寻尬尬屘屘寡寮屘屘寻寻寻尧寻寻寻寻寻寻寳寳寡對對對寡寡專專尹寮尹履尹尹寻寻屆履寮寮寮寮寮岄屆履屆寮專專專履屆屆寮寮屆屆專專尚履尚尀尚尚尴尴審尕屆屆專專專屆寡尕寡專寡寡寻寻寶屓屓屓寶寶屓屓寶屓寶尕屓屓屓屆審屓寶寶尧尧尧屓審屿審尧屓屓屓寶寶寶寶寶寶寶寮寮寶寮寶寮專專專屓審尃尃尃審審審屠尴尃尴寶尴尴屠屠審尴尃尃審審屠屠尃審尃寶尃尃尴尴屁尯審審尃尃尃尃屁'.__getitem__)[k - 1]
```

https://leetcode.com/problems/sort-integers-by-the-power-value/discuss/546573/Just-for-fun

https://stackoverflow.com/questions/3417760/how-to-sort-a-python-dicts-keys-by-value



### 1431. Kids With the Greatest Number of Candies

```python
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_ = max(candies)
        return [each+extraCandies >= max_ for each in candies]
```



### 1512. Number of Good Pairs

```python
class Solution(object):
    def numIdenticalPairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        from collections import defaultdict
        ans = 0
        temp = defaultdict(int)
        for num in nums:
            ans += temp.get(num, 0)
            temp[num] += 1
        return ans
```

```python
class Solution(object):
    def numIdenticalPairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum([val * (val-1) // 2 for val in collections.Counter(nums).values()])
```

val * (val - 1) // 2 == A(n, 2) // 2



### 1640. Check Array Formation Through Concatenation

```python
class Solution:
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        temp = {each[0]: each for each in pieces}
        i = 0
        while i < len(arr):
            if arr[i] not in temp: return False
            for each in temp[arr[i]]:
                if arr[i] != each: return False
                i += 1
        return True
```





### [Amazon | OA 2019 | Favorite Genres](https://leetcode.com/discuss/interview-question/373006)

```python
def Favorite_Genres(userSongs, songGenres):
  	ans = dict()
  	if not songGenres:
      	for each_user in userSongs.keys():
          	ans[each_user] = []
        return ans

    re_songGenres = dict()
    for k,v in songGenres.items():
        for each in v:
            re_songGenres[each] = k

    for v in userSongs.values():
        for i in range(len(v)):
            v[i] = re_songGenres[v[i]]



    for user,genres_list in userSongs.items():
        temp_dict = dict()
        for each in genres_list:
            if each not in temp_dict.keys():
                temp_dict[each] = 1
            else:
                temp_dict[each] += 1
        max_num = max(temp_dict.values())
        for genre,genre_num in temp_dict.items():
            if genre_num == max_num:
                if user not in ans.keys():
                    ans[user] = [genre]
                else:
                    ans[user].append(genre)
		return ans
```



### Naveego OA - [Look and say numbers](https://www.codewars.com/kata/53ea07c9247bc3fcaa00084d)

```python
def look_and_say(data='1', maxlen=5):
    def helper(temp_str):
        seen = temp_str[0]
        count = 1
        res = ''
        for i in range(1, len(temp_str)):
            if temp_str[i] == seen:
                count += 1
            else:
                res += str(count) + seen
                seen = temp_str[i]
                count = 1
        res += str(count) + seen
        return res

    ans = []
    for i in range(maxlen):
        data = helper(data)
        ans.append(data)
    return ans
```



### 斐波纳切

```python
def fac(num):
    a = b = 1
    for i in range(num):
        c = a + b
        if i == 0 or i == 1:
            yield 1
            continue
        yield c
        a, b = b, c
```



### Perform String Shifts

You are given a string `s` containing lowercase English letters, and a matrix `shift`, where `shift[i] = [direction, amount]`:

- `direction` can be `0` (for left shift) or `1` (for right shift). 
- `amount` is the amount by which string `s` is to be shifted.
- A left shift by 1 means remove the first character of `s` and append it to the end.
- Similarly, a right shift by 1 means remove the last character of `s` and add it to the beginning.

Return the final string after all operations.

**Example 1:**

```
Input: s = "abc", shift = [[0,1],[1,2]]
Output: "cab"
Explanation: 
[0,1] means shift to left by 1. "abc" -> "bca"
[1,2] means shift to right by 2. "bca" -> "cab"
```

**Example 2:**

```
Input: s = "abcdefg", shift = [[1,1],[1,1],[0,2],[1,3]]
Output: "efgabcd"
Explanation:  
[1,1] means shift to right by 1. "abcdefg" -> "gabcdef"
[1,1] means shift to right by 1. "gabcdef" -> "fgabcde"
[0,2] means shift to left by 2. "fgabcde" -> "abcdefg"
[1,3] means shift to right by 3. "abcdefg" -> "efgabcd"
```

 

**Constraints:**

- `1 <= s.length <= 100`
- `s` only contains lower case English letters.
- `1 <= shift.length <= 100`
- `shift[i].length == 2`
- `0 <= shift[i][0] <= 1`
- `0 <= shift[i][1] <= 100`

```python
class Solution:
    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        if not shift:
            return s
        
        ans = s
        
        if len(shift) > 1:
            for i in range(1, len(shift)):
                if shift[0][0] == shift[i][0]:
                    shift[0][1] += shift[i][1]
                else:
                    if shift[0][1] > shift[i][1]:
                        shift[0][1] -= shift[i][1]
                    elif shift[0][1] < shift[i][1]:
                        temp = shift[i][1] - shift[0][1]
                        shift[0] = [shift[i][0], temp]
                    else:
                        shift[0][1] = 0
        
        shift[0][1] %= len(s)
        if shift[0][1] == 0:
            return s
            
        if shift[0][0] == 0:
            ans += ans[:shift[0][1]]
            return ans[shift[0][1]:]
        else:
            ans = ans[-shift[0][1]:] + ans
            return ans[:-shift[0][1]]
```


### Leftmost Column with at Least a One

```python
class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        row_num, col_num = binaryMatrix.dimensions()
        def find_leftmost_1(row_index):
            if binaryMatrix.get(row_index, col_num-1) == 0:
                return col_num
            left = 0
            right = col_num-1
            while left < right:
                mid = left + (right - left) // 2
                if binaryMatrix.get(row_index, mid) == 1:
                    if mid == 0:
                        return mid
                    if binaryMatrix.get(row_index, mid-1) == 0:
                        return mid
                    else:
                        right = mid - 1
                else:
                    if binaryMatrix.get(row_index, mid+1) == 1:
                        return mid+1
                    else:
                        left = mid + 1
            if binaryMatrix.get(row_index, left) == 1:
                return left
            else:
                return col_num
        
        leftmost_1_list = [find_leftmost_1(i) for i in range(row_num)]
        for each in leftmost_1_list:
            if each != col_num:
                return min(leftmost_1_list)
        return -1
```



### 华为机试HJ37 统计每个月兔子总数

有一只兔子，从出生后第3个月起每个月都生一只兔子，小兔子长到第三个月后每个月又生一只兔子，假如兔子都不死，问每个月的兔子总数为多少？

```python
def helper(month):
    res = 1
    tempList = [1, 0, 0]
    i = 1
    while i < month:
        tempList[2] += tempList[1]
        res += tempList[2]
        tempList[1] = tempList[0]
        tempList[0] = tempList[2]
        i += 1
    return res

try:
    while 1:
        month = int(input())
        print(helper(month))
except:
    pass
```

