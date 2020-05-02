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



### 4. Median of Two Sorted Arrays

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        res_index = (len(nums1) + len(nums2)) // 2
        if not nums1 or not nums2:
            new_nums = nums1 or nums2
        else:
            new_nums = []
            if nums2[0] >= nums1[len(nums1)-1]:
                new_nums = nums1 + nums2
            elif nums1[0] >= nums2[len(nums2)-1]:
                new_nums = nums2 + nums1
        if new_nums:
            if (len(nums1) + len(nums2)) % 2 == 0:
                return (new_nums[res_index]+new_nums[res_index-1])/2
            else:
                return new_nums[res_index]

        nums1_index = 0
        nums2_index = 0
        
        for i in range(len(nums1)+len(nums2)):
            if nums1_index <= len(nums1)-1 and nums2_index <= len(nums2)-1:
                if nums1[nums1_index] <= nums2[nums2_index]:
                    new_nums.append(nums1[nums1_index])
                    nums1_index += 1
                else:
                    new_nums.append(nums2[nums2_index])
                    nums2_index += 1
            else:
                if nums1_index <= len(nums1)-1:
                    new_nums.append(nums1[nums1_index])
                    nums1_index += 1
                elif nums2_index <= len(nums2)-1:
                    new_nums.append(nums2[nums2_index])
                    nums2_index += 1
            if i == res_index:
                if (len(nums1) + len(nums2)) % 2 == 0:
                    return (new_nums[res_index]+new_nums[res_index-1])/2
                else:
                    return new_nums[res_index]
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



### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        res = []
        cur_ele = intervals[0]
        nxt_index = 1
        while nxt_index <= len(intervals) - 1:
            if cur_ele[1] < intervals[nxt_index][0]:
                res.append(cur_ele)
                cur_ele = intervals[nxt_index]
            else:
                cur_ele = [min(cur_ele[0], intervals[nxt_index][0]), max(cur_ele[1], intervals[nxt_index][1])]
            nxt_index += 1
        res.append(cur_ele)
        return res
```

try to use divide and conquer



### 61. Rotate List

```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head or k == 0:
            return head
        
        temp_move = temp_l = head
        
        # get length of linked list
        length = 0
        while temp_l:
            length += 1
            temp_l = temp_l.next
        
        # 计算倒数第几个
        num = k % length
        if num == 0:
            return head
        
        # 将指针挪到上面那个数的前一个, 并取出后面那个数作为新头且断开与后面那个数的链接
        for i in range(length-num-1):
            temp_move = temp_move.next
        new_head = temp3 = temp_move.next
        temp_move.next = None
        
        # 将结尾与原来的头链接
        while temp3 and temp3.next:
            temp3 = temp3.next
        temp3.next = head
        
        return new_head
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



### 70. Climbing Stairs

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        cache = dict()
        
        def climbStairs_helper(i, n):
            if i in cache:
                return cache[i]
            
            if i > n:
                return 0
            if i == n:
                return 1
            temp = climbStairs_helper(i+1, n) + climbStairs_helper(i+2, n)
            cache[i] = temp
            return temp
        
        return climbStairs_helper(0, n)
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



### 136. Single Number

```python
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for num in nums:
            res ^= num
        return res
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



### 152. Maximum Product Subarray

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        min_, max_, ans = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            max_temp = max(max(max_*nums[i], min_*nums[i]), nums[i])
            min_temp = min(min(max_*nums[i], min_*nums[i]), nums[i])
            ans = max(ans, max_temp)
            max_ = max_temp
            min_ = min_temp
        return ans
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



### 189. Rotate Array

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums:
            return
        k = k % len(nums);
        count = 0
        for i in range(len(nums)):
            last_starter = i
            index = i
            temp1 = nums[index]
            temp2 = None
            while 1:
                wait_index = (index + k) % len(nums)
                temp2 = nums[wait_index]
                nums[wait_index] = temp1
                index = wait_index
                temp1 = temp2
                count += 1
                if count == len(nums):
                    return
                if wait_index == last_starter:
                    break
```



### 190. Reverse Bits

```python
    def reverseBits(self, n):
        n_bin = bin(n)
        is_positive = False
        if '-' in n_bin:
            n_bin = n_bin[3:]
            is_positive = True
        else:
            n_bin = n_bin[2:]

        for i in range(32 - len(n_bin)):
            n_bin = '0' + n_bin

        re_n_bin = n_bin[::-1]
        if is_positive == True:
            re_n_bin = '-' + re_n_bin
        int_re_n_bin = int(re_n_bin, 2)
        return int_re_n_bin
    

'''
        reverse = 0
        count = 0
        while count < 32:
            reverse = reverse << 1
            bit = n & 1
            reverse = reverse + bit
            n = n >> 1
            count += 1
        return reverse
'''
```

n & 1 一个数&1是取到这个数二进制的最后一位k



### 191.Number of 1 Bits

```c++
int bammingWeight(uint32_t n){
    int ans = 0;
    while(n>0){
        n=n&(n-1);//最低位1=0
        ans++;
    }
    return ans;
}
```

这种方法速度比较快，其运算次数与输入n的大小无关，只与n中1的个数有关。如果n的二进制表示中有k个1，那么这个方法只需要循环k次即可。其原理是不断清除n的二进制表示中最右边的1，同时累加计数器，直至n为0。

为什么n &= (n – 1)能清除最右边的1呢？因为从二进制的角度讲，n相当于在n - 1的最低位加上1。举个例子，8（1000）= 7（0111）+ 1（0001），所以8 & 7 = （1000）&（0111）= 0（0000），清除了8最右边的1（其实就是最高位的1，因为8的二进制中只有一个1）。再比如7（0111）= 6（0110）+ 1（0001），所以7 & 6 = （0111）&（0110）= 6（0110），清除了7的二进制表示中最右边的1（也就是最低位的1）。



### 201. Bitwise AND of Numbers Range

```python
class Solution:
    def rangeBitwiseAnd(self, m: int, n: int) -> int:
        count = 0
        while m != n:
            count += 1
            m >>= 1
            n >>= 1
        return m << count
```
区间内所有数字的与，实则是寻找从左边起有多少位全部是1，所以将mn右移并计算此数，如果mn相等则右移次数为右边有多少位0，此时在m右边补相应位数的0并返回。



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



### 231. Power of Two

```
1 1
2 10
4 100
8 1000
```

```c++
bool isPowerOfTwo(int n){
    if( n<=0 ) return false;
    return ((n&(n-1))==0)
}
```



### 238. Product of Array Except Self

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return
        
        left_to_right, right_to_left, ans = [0]*len(nums), [0]*len(nums), [0]*len(nums)
        
        left_to_right[0] = 1
        for i in range(1, len(nums)):
            left_to_right[i] = left_to_right[i-1] * nums[i-1]
            
        right_to_left[-1] = 1
        for i in range(len(nums)-2, -1, -1):
            right_to_left[i] = right_to_left[i+1] * nums[i+1]
        
        for i in range(len(nums)):
            ans[i] = left_to_right[i] * right_to_left[i]
        return ans
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



### 349. Intersection of Two Arrays

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1_set = set(nums1)
        nums2_set = set(nums2)
        res = []
        for each in nums1_set:
            if each in nums2_set:
                res.append(each)
        return res
```

```js
This is a Facebook interview question.
They ask for the intersection, which has a trivial solution using a hash or a set.

Then they ask you to solve it under these constraints:
O(n) time and O(1) space (the resulting array of intersections is not taken into consideration).
You are told the lists are sorted.

Cases to take into consideration include:
duplicates, negative values, single value lists, 0's, and empty list arguments.
Other considerations might include
sparse arrays.

function intersections(l1, l2) {
    l1.sort((a, b) => a - b) // assume sorted
    l2.sort((a, b) => a - b) // assume sorted
    const intersections = []
    let l = 0, r = 0;
    while ((l2[l] && l1[r]) !== undefined) {
       const left = l1[r], right = l2[l];
        if (right === left) {
            intersections.push(right)
            while (left === l1[r]) r++;
            while (right === l2[l]) l++;
            continue;
        }
        if (right > left) while (left === l1[r]) r++;
         else while (right === l2[l]) l++;
        
    }
    return intersections;
}
```



### 350. Intersection of Two Arrays II

```python
# 2 years ago since 3/22/2020
class Solution(object):
    def intersect(self, nums1, nums2):
        finArray = []
        for each in nums1:
            if each in nums2:
                finArray.append(each)
                nums2.remove(each)
        return finArray
        
```

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        from collections import Counter
        
        nums1Dict = Counter(nums1)
        res = []
        for each in nums2:
            if each in nums1Dict and nums1Dict[each] != 0:
                res.append(each)
                nums1Dict[each] -= 1
        return res
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



### 476. Number Complement 

```python
ans = ''
for i in range(len(bin(num)) - 2):
	if num & 1 == 1:
		ans += '0'
	else:
		ans += '1'
	num >>= 1
return int(ans[::-1], 2)
'''
if num < 1:
	return 1
i = 1
while i <= num:
    i <<= 1
return (i-1) ^ num
'''
```

求二进制数每位取反，用相同位个1与原数进行异或运算：1111 ^ 1011 = 0100



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



### 525. Contiguous Array

```python
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        temp_dict = dict()
        count = ans = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                count -= 1
            else:
                count += 1
            if count == 0:
                ans = max(ans, i+1)
            if count not in temp_dict:
                temp_dict[count] = i
            else:
                ans = max(ans, i - temp_dict[count])
        return ans
```
遇0减一遇1加一，并将此数作为key，index作为值存入字典，如果遇到相同的数组说明有相等的0和1出现，用当前index值键字典中相同数字的index得到长度。
如果遇到数字为0，说明从0到目前index为止的0和1数量相同



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



### 697. Degree of an Array

```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        count = collections.Counter(nums)
        degrees = []
        max_degree = max(count.values())
        for key, val in count.items():
            if val == max_degree:
                degrees.append(key)
        ans = sys.maxsize
        start = end = None
        for each in degrees:
            i = 0
            while True:
                if nums[i] == each:
                    start = i
                    break
                i += 1
            i = len(nums)-1
            while True:
                if nums[i] == each:
                    end = i
                    break
                i -= 1
            ans = min(ans, end-start+1)
        return ans
```

```python
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        left, right = {}, {}
        from collections import defaultdict
        count = defaultdict(int)
        for i in range(len(nums)):
            if nums[i] not in left:
                left[nums[i]] = i
            right[nums[i]] = i
            count[nums[i]] += 1
        
        max_degree = max(count.values())
        ans = sys.maxsize
        for key, value in count.items():
            if value == max_degree:
                ans = min(ans, right[key]-left[key]+1)
        return ans
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



### 836. Rectangle Overlap

```python
        r1x1,r1y1,r1x2,r1y2 = rec1
        r2x1,r2y1,r2x2,r2y2 = rec2
        
        finx1 = max(r1x1,r2x1)
        finy1 = max(r1y1,r2y1)
        finx2 = min(r1x2,r2x2)
        finy2 = min(r1y2,r2y2)
        
        if (finx2 > finx1) and (finy1 < finy2):
            return True
        else:
            return False
```

求相交矩阵坐标，并判断该坐标是否可构成矩阵

[假定矩形是用一对点表达的(minx, miny) (maxx, maxy)，那么两个矩形 rect1{(minx1, miny1)(maxx1, maxy1)} rect2{(minx2, miny2)(maxx2, maxy2)}  ](https://www.cnblogs.com/0001/archive/2010/05/04/1726905.html)

相交的结果一定是个矩形，构成这个相交矩形rect{(minx, miny) (maxx, maxy)}的点对坐标是：  

minx=max(minx1, minx2)  

miny=max(miny1, miny2)  

maxx=min(maxx1, maxx2)  

maxy=min(maxy1, maxy2)  

如果两个矩形不相交，那么计算得到的点对坐标必然满足：  

（ minx > maxx ） 或者 （ miny > maxy ） 

 判定是否相交，以及相交矩形是什么都可以用这个方法一体计算完成。

从这个算法的结果上，我们还可以简单的生成出下面的两个内容：

㈠ 相交矩形：  (minx, miny) (maxx, maxy)

㈡ 面积： 面积的计算可以和判定一起进行
        if ( minx>maxx ) return 0;
        if ( miny>maxy ) return 0;
        return (maxx-minx)*(maxy-miny)

第二种方法

两个矩形相交的条件:两个矩形的重心距离在X和Y轴上都小于两个矩形长或宽的一半之和.这样,分两次判断一下就行了.

bool CrossLine(Rect r1,RECT r2)
{
if(abs((r1.x1+r1.x2)/2-(r2.x1+r2.x2)/2)<((r1.x2+r2.x2-r1.x1-r2.x1)/2) && abs((r1.y1+r1.y2)/2-(r2.y1+r2.y2)/2)<((r1.y2+r2.y2-r1.y1-r2.y1)/2))
return true;
return false;
}



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



### 912. Sort an Array

```python
# QuickSort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        
        def sort_method(i, j):
            oi = i
            pivot = nums[i]
            i += 1
            while True:
                while i < j and nums[i] < pivot:
                    i += 1
                while i <= j and nums[j] >= pivot:
                    j -= 1
                if i >= j: break
                nums[i], nums[j] = nums[j], nums[i]
            nums[oi], nums[j] = nums[j], nums[oi]
            return j
        
        def quick_sort(i, j):
            if i >= j: return
            
            k = random.randint(i, j)
            nums[i], nums[k] = nums[k], nums[i]
            mid = sort_method(i, j)
            quick_sort(i, mid)
            quick_sort(mid+1, j)
        
        quick_sort(0, len(nums)-1)
        return nums
```

```python
# QuickSort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def sort(i, j):
            pivot = nums[i]
            slow = i
            for fast in range(i+1, j+1):
                if nums[fast] < pivot:
                    slow += 1
                    nums[slow], nums[fast] = nums[fast], nums[slow]
            nums[i], nums[slow] = nums[slow], nums[i]
            return slow
        
        def partition(i, j):
            if i >= j: return
            mid = sort(i, j)
            partition(i, mid-1)
            partition(mid+1, j)
            
        
        partition(0, len(nums)-1)
        return nums
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



### 1119. Remove Vowels from a String

```python
class Solution:
    def removeVowels(self, S: str) -> str:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return ("").join([each for each in S if each not in vowels])
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



### 1431. Kids With the Greatest Number of Candies

```python
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_ = max(candies)
        return [each+extraCandies >= max_ for each in candies]
```



### Topo sort 

```java
# DFS + TOPO Sort 并判断是否有环，如果有环打印构成环的边，只能判断有一个环的图。
# 要点：
# 1. DFS 访问完成的顺序即为拓扑排序
# 2. 若访问到一个点发现该点已经访问过但没有访问完成，也就是没有在访问完成栈里，则说明有环。
# 3. DFS 遍历时，访问一个点将一个点入栈，访问完成出栈，如果到某个点发现有环，则栈顶到该点的所有点构成环。

Stack topoStack = new Stack(); # 用来构成拓扑排序
Stack st = new Stack(); # 用来打印环

void DFS_recursive(int n, boolean visited[]) {
    visited[n] = true;
    st.push(n);
    System.out.print(n + ","); # 此处打印出的顺序为 DFS 遍历顺序
    Iterator<Integer> i = link[n].listIterator();
    while (i.hasNext()) {
        int temp = i.next();
        if (visited[temp] == true && topoStack.search(temp) == -1) { # 此处判断该点已访问过但没有在topo栈里，说明没有完成，则说明有环
            System.out.println("There is a cycle:");
            printCycle(temp); # temp 为环关闭的点
            return;
        }
        if (visited[temp] == false) {
            DFS_recursive(temp, visited);
        }
    }
    //add point which has already finished
    topoStack.push(n); # 此处n点说明已经访问完成，加入topo栈，出栈顺序即为topo排序
    if (st.empty() == false) {
        st.pop();
    } else {
        topoStack.push(-1); # 若有环push一个-1进行标记，之后不进行打印topo排序
        return;
    }
}

void DFS(int n) { //visit start n
    boolean visited[] = new boolean[Node_number];
    DFS_recursive(n, visited);
}

void printCycle(int temp) {
    int top = (Integer) st.peek(); # 取栈顶元素但不删除，留之后打印边用
    while (st.empty() == false) { # 从栈顶到 temp 点，这几个点构成环儿，打印之
        int a = (Integer) st.pop();
        int b = (Integer) st.peek();
        System.out.println(a + "<-" + b);
        if (b == temp) {
            System.out.println(b + "<-" + top);
            break;
        }
    }
}

void getTOPO() {
    if ((Integer) topoStack.peek() == -1) {
        return;
    }
    System.out.println("\nTOPO Sort:");
    while (topoStack.empty() == false) {
        System.out.print(topoStack.pop());
        System.out.print(',');
    }
}
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



### OA

```python
class Solution(object):
    def generate_graph(self, sources, destinations, weights):
        G = dict()
        cost_dict = dict()
        for i in range(len(sources)):
            if sources[i] not in G:
                G[sources[i]] = [destinations[i]]
            else:
                G[sources[i]].append(destinations[i])
            cost_dict[(sources[i], destinations[i])] = weights[i]
            if destinations[i] not in G:
                G[destinations[i]] = [sources[i]]
            else:
                G[destinations[i]].append(sources[i])
            cost_dict[(destinations[i], sources[i])] = weights[i]
        return G, cost_dict

    def find_min(self, dist_dict):
        nodes = []
        min_value = min(dist_dict.values())
        for each in dist_dict:
            if dist_dict[each] == min_value:
                nodes.append(each)
        return nodes

    def get_route(self, prev, current_node, res):
        if current_node == 1:
            return
        for each in prev.get(current_node):
            res.append({current_node, each})
            self.get_route(prev, each, res)

    def shortest_path(self, G, cost_dict):
        dist_dict = {i: 2 ** 32 - 1 for i in range(1, len(G) + 1)}
        dist_dict[1] = 0
        prev = {i: [] for i in range(1, len(G) + 1)}
        S = set()

        while len(S) != len(G):
            start_nodes = self.find_min(dist_dict)
            for start_node in start_nodes:
                S.add(start_node)
                link_node_list = G[start_node]
                for each_link_node in link_node_list:
                    if each_link_node not in S:
                        if dist_dict[each_link_node] > dist_dict[start_node] + cost_dict[(start_node, each_link_node)]:
                            dist_dict[each_link_node] = dist_dict[start_node] + cost_dict[(start_node, each_link_node)]
                            prev[each_link_node] = [start_node]
                        elif dist_dict[each_link_node] == dist_dict[start_node] + cost_dict[
                            (start_node, each_link_node)]:
                            prev[each_link_node].append(start_node)
                dist_dict.pop(start_node)
        return prev

    def checkYourRoute(self, nodes, sources, destinations, weights, end):
        G, cost_dict = self.generate_graph(sources, destinations, weights)
        prev = self.shortest_path(G, cost_dict)
        route_res = []
        ans = []
        self.get_route(prev, end, route_res)
        for i in range(len(sources)):
            if {sources[i], destinations[i]} in route_res:
                ans.append('YES')
            else:
                ans.append('NO')
        return ans


solution = Solution()
nodes = 4
sources = [1, 1, 1, 2, 2]
destinations = [2, 3, 4, 3, 4]
weights = [1, 1, 1, 1, 1]
end = 4
print(solution.checkYourRoute(nodes, sources, destinations, weights, end))
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



### Naveego OA - [Directions Reduction](https://www.codewars.com/kata/550f22f4d758534c1100025a)

```python
def dirReduc(arr):
    stack = []
    
    for each in arr:
        if stack and ({stack[-1], each} == {'NORTH', 'SOUTH'} or {stack[-1], each} == {'EAST', 'WEST'}):
            stack.pop()
        else:
            stack.append(each)
    return stack
```



```python
def ShortestPath(strArr):
  import collections
  # code goes here
  def generate_graph(sou_des_list):
    G = collections.defaultdict(list)
    for each in sou_des_list:
      source, destination = each.split("-")
      G[source].append(destination)
      G[destination].append(source)
    return G
  
  def find_path(source, destination, f_node_dict):
    res = destination
    curr_node = destination
    while curr_node != source:
      curr_node = f_node_dict.get(curr_node)
      res = curr_node + '-' + res
    return res
  
  node_num = int(strArr[0])
  source = strArr[1]
  destination = strArr[node_num]
  G = generate_graph(strArr[node_num+1:])
  f_node_dict = {source:None}
  Q = [source]
  while Q:
    node = Q.pop(0)
    for each in G.get(node):
      if each in f_node_dict:
        continue
      f_node_dict[each] = node
      if each == destination:
        return find_path(source, destination, f_node_dict)
      Q.append(each)
  return -1

# keep this function call here 
print(ShortestPath(input()))
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
