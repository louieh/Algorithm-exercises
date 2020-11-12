## Leetcode - Bits

[toc]

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

