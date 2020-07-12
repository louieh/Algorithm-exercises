## Leetcode - Bits

[toc]

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
# | 按位并
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

