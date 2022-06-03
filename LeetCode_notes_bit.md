## Leetcode - Bits

[toc]

https://leetcode.com/problems/sum-of-two-integers/discuss/84278/A-summary%3A-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently



### 29. Divide Two Integers

```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        if dividend == -2**31 and divisor == -1:
            return 2**31 - 1
        x, y = abs(dividend), abs(divisor)
        ans = 0
        while x >= y:
            temp, m = y, 1
            while temp << 1 <= x:
                temp <<= 1
                m <<= 1
            
            x -= temp
            ans += m
        
        return ans if dividend > 0 and divisor > 0 or dividend < 0 and divisor < 0 else -ans
```

https://leetcode.com/problems/divide-two-integers/discuss/13407/C%2B%2B-bit-manipulations

The key observation is that the quotient of a division is just the number of times that we can subtract the `divisor` from the `dividend` without making it negative.

Suppose `dividend = 15` and `divisor = 3`, `15 - 3 > 0`. We now try to subtract more by *shifting* `3` to the left by `1` bit (`6`). Since `15 - 6 > 0`, shift `6` again to `12`. Now `15 - 12 > 0`, shift `12` again to `24`, which is larger than `15`. So we can at most subtract `12` from `15`. Since `12` is obtained by shifting `3` to left twice, it is `1 << 2 = 4` times of `3`. We add `4` to an answer variable (initialized to be `0`). The above process is like `15 = 3 * 4 + 3`. We now get part of the quotient (`4`), with a remaining dividend `3`.

Then we repeat the above process by subtracting `divisor = 3` from the remaining `dividend = 3` and obtain `0`. We are done. In this case, no shift happens. We simply add `1 << 0 = 1` to the answer variable.

This is the full algorithm to perform division using bit manipulations. The sign also needs to be taken into consideration. And we still need to handle one overflow case: `dividend = INT_MIN` and `divisor = -1`.



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



### 260. Single Number III

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        temp = 0
        for num in nums:
            temp ^= num
        
        mask = temp & -temp
        
        ans1, ans2 = 0, 0
        
        for num in nums:
            if num & mask == 0:
                ans1 ^= num
            else:
                ans2 ^= num
        return [ans1, ans2]
```

先对所有数字进行异或计算，由于有两个数的个数是1，其余都是2，那么结果便是这两个数异或。两个数异或的结果意味着这两个数的二进制表示中哪些位是不同的，我们取任意一个不同位，按此对原数组进行分组，之后分别对两组进行异或计算便得到结果。

https://leetcode.com/problems/single-number-iii/discuss/68901/Sharing-explanation-of-the-solution

其中取任意不同位 `mask = temp & -temp` 意思是求最右侧不同位是哪一位：

```
 10: 0000 0000 0000 0000 0000 0000 0000 1010
-10: 1111 1111 1111 1111 1111 1111 1111 0110
10 & -10 = 10(2)
```

取负数方法：https://blog.csdn.net/u013790019/article/details/44627719



### 304. Range Sum Query 2D - Immutable

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        rows, cols = len(matrix), len(matrix[0])
        self.dp = [[0] * (cols+1) for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                self.dp[row][col+1] = self.dp[row][col] + matrix[row][col]
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = 0
        for row in range(row1, row2+1):
            res += self.dp[row][col2+1] - self.dp[row][col1]
        return res
        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```



### 338. Counting Bits

```python
# O(n*sizeof(integer))
class Solution:
    def countBits(self, num: int) -> List[int]:
        ans = []
        
        def num_bit1(num):
            ans = 0
            while num > 0:
                num &= (num-1)
                ans += 1
            return ans
        
        for i in range(num+1):
            if i == 0:
                ans.append(0)
                continue
            if i == 1:
                ans.append(1)
                continue
            ans.append(num_bit1(i))
        return ans
```

```python
# dp
class Solution:
    def countBits(self, num: int) -> List[int]:
        if num == 0:
            return [0]
        if num == 1:
            return [0, 1]
        
        ans = [0, 1]
        
        for i in range(2, num+1):
            if i & 1:
                ans.append(ans[i >> 1] + 1)
            else:
                ans.append(ans[i >> 1])
        return ans
```



### 461. Hamming Distance

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        res = 0
        while x and y:
            if x & 1 != y & 1:
                res += 1
            x >>= 1
            y >>= 1
        while x:
            if x & 1:
                res += 1
            x >>= 1
        while y:
            if y & 1:
                res += 1
            y >>= 1
        
        return res
```



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

same as 1009. 求二进制数每位取反，用相同位个1与原数进行异或运算：1111 ^ 1011 = 0100



### 1009. Complement of Base 10 Integer

```python
class Solution:
    def bitwiseComplement(self, N: int) -> int:

        if N == 0: return 1
        res = ''
        while N > 0:
            res = str(1-N&1) + res
            N >>= 1
        return int(res, 2)
'''
if num < 1:
	return 1
i = 1
while i <= num:
    i <<= 1
return (i-1) ^ num
'''
```

same as 476.

```python
class Solution:
    def bitwiseComplement(self, n: int) -> int:
        if n == 0: return 1
        temp1 = n
        temp2 = 1
        while temp1:
            temp2 <<= 1
            temp1 >>= 1
        return temp2-1 ^ n
```



### 1178. Number of Valid Words for Each Puzzle

```python
# TLE
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        res = []
        word_set_list = [set(word) for word in words]
        for puzzle in puzzles:
            puzzle_set = set(puzzle)
            temp = 0
            for word_set in word_set_list:
                if puzzle[0] in word_set and word_set.issubset(puzzle_set):
                    temp += 1
            res.append(temp)
        
        return res
```

```python
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        def bitmask(word: str) -> int: # 注意此处将字符串转成bitmask的方法
            mask = 0
            for letter in word:
                mask |= 1 << (ord(letter) - ord('a'))
            return mask

        word_count = Counter(bitmask(word) for word in words)

        result = []
        for puzzle in puzzles:
            mask = bitmask(puzzle)
            first = 1 << (ord(puzzle[0]) - ord('a'))
            count = 0
            submask = mask
            while submask:
                if submask & first: # 包含第一个字母的子集
                    count += word_count[submask]
                submask = (submask - 1) & mask # 注意此处使用bit求所有subset的方法
            result.append(count)
        return result
```

https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/solution/

https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/discuss/1567324/C%2B%2BPython-Clean-Solutions-w-Detailed-Explanation-or-Bit-masking-and-Trie-Approaches



### 1404. Number of Steps to Reduce a Number in Binary Representation to One

```python
class Solution:
    def numSteps(self, s: str) -> int:
        
        res = carry = 0
        
        for each in s[:0:-1]:
            res += 1
            if int(each) + carry == 1:
                carry = 1
                res += 1
        
        return res + carry
```

从最后一个元素开始向前遍历到第二个元素。每个元素判断该元素加进位数carry的值，该值有三种可能（0，1，2）如果为0，直接除以2也就是右移一位，动作总数加一。如果为1需要进行两步操作：加一，除以二。之后把进位数设置为1。如果是2，说明此时元素为1且有1个进位，相加后为10，有一个进位且当前位置为0，所以除以二即可。最后遍历结束后剩下第一个元素1，如果此时有进位carry=1，那么1+1=10，需要再进行一次右移，如果此时carry=0，那么不需要再进行任何操作，所以最后直接返回res+carry即可。



### 1461. Check If a String Contains All Binary Codes of Size K

```python
class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        need = 1 << k
        got = set()

        for i in range(k, len(s)+1):
            tmp = s[i-k:i]
            if tmp not in got:
                got.add(tmp)
                need -= 1
                # return True when found all occurrences
                if need == 0:
                    return True
        return False
```

Maybe you think the approach above is not fast enough. Let's write the hash function ourselves to improve the speed.

Note that we will have at most 2^k2*k* string, can we map each string to a number in [00, 2^k-12*k*−1]?

We can. Recall the binary number, we can treat the string as a binary number, and take its decimal form as the hash value. In this case, each binary number has a unique hash value. Moreover, the minimum is all `0`, which is zero, while the maximum is all `1`, which is exactly 2^k-12*k*−1.

Because we can directly apply bitwise operations to decimal numbers, it is not even necessary to convert the binary number to a decimal number explicitly.

What's more, we can get the current hash from the last one. This method is called [Rolling Hash](https://en.wikipedia.org/wiki/Rolling_hash). All we need to do is to remove the most significant digit and to add a new least significant digit with bitwise operations.

> For example, say `s="11010110"`, and `k=3`, and we just finish calculating the hash of the first substring: `"110"` (`hash` is 4+2=6, or `110`). Now we want to know the next hash, which is the hash of `"101"`.
>
> We can start from the binary form of our hash, which is `110`. First, we shift left, resulting `1100`. We do not need the first digit, so it is a good idea to do `1100 & 111 = 100`. The all-one `111` helps us to align the digits. Now we need to apply the lowest digit of `"101"`, which is `1`, to our hash, and by using `|`, we get `100 | last_digit = 100 | 1 = 101`.

Write them together, we have: `new_hash = ((old_hash << 1) & all_one) | last_digit_of_new_hash`.

With rolling hash method, we only need \mathcal{O}(1)O(1) to calculate the next hash, because bitwise operations (`&`, `<<`, `|`, etc.) are only cost \mathcal{O}(1)O(1).

This time, we can use a simple list to store our hashs, and we will not have hash collision. Those advantages make this approach faster.

```python
class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        need = 1 << k
        got = [False] * need
        all_one = need - 1
        hash_val = 0
        
        
        for i in range(len(s)):
            hash_val = (hash_val << 1) & all_one | int(s[i])
            if i >= k - 1 and got[hash_val] is False:
                got[hash_val] = True
                need -= 1
                if need == 0:
                    return True
        return False
```



### 1680. Concatenation of Consecutive Binary Numbers

```python
# TLE
class Solution:
    def concatenatedBinary(self, n: int) -> int:
        res = 0
        i = n
        temp = 0
        while i > 0:
            res += i<<temp
            temp += len(bin(i))-2
            i -= 1
        return res % (10**9+7)
```

```python
# TLE
class Solution:
    def concatenatedBinary(self, n: int) -> int:
        res = 0
        for i in range(1, n+1):
            res = res * (1 << (len(bin(i)) - 2)) + i
        return res % (10**9 + 7) # 此步骤需要在for中进行
```

```python
class Solution:
    def concatenatedBinary(self, n: int) -> int:
        res = 0
        for i in range(1, n+1):
            res = (res * (1 << (len(bin(i)) - 2)) + i) % (10**9+7)
        return res
```

