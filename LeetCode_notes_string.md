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



### 209. Minimum Size Subarray Sum

```python
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        for i in range(1, len(nums)+1): # 遍历长度 (1-6)
            for j in range(len(nums)):  # 遍历起始点 (0-5)
                if j + i - 1 >= len(nums):
                    break
                sum_temp = 0
                for k in range(i): # 从 j 开始遍历 i 长度
                    sum_temp += nums[k+j]
                if sum_temp >= s:
                    return i
        return 0
```

$O(n^3)$ 超时。将数组中各个位的累积和存在另一个数组中，这样时间复杂度缩小到 $O(n^2)$

```java
class Solution {
    public int minSubArrayLen(int s, int[] nums) {
        if (nums.length == 0)
            return 0;
        
        for(int i=0; i<nums.length; i++){
            if (nums[i] >= s)
                return 1;
        }
        
        int[] dest = nums.clone();
        
        for(int i=2; i<=nums.length; i++){ // 遍历长度
            for(int j=0; j<nums.length; j++){ // 遍历起始点
                if(j+i-1 >= nums.length)
                    break;
                if (dest[j] + nums[j+i-1] >= s)
                    return i;
                else{
                    dest[j] += nums[j+i-1];
                }
            }
        }
        return 0;
    }
}
```

Java $O(n^2)$ accepted

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        
        for each in nums:
            if each >= s:
                return 1

        nums_sum_temp = nums.copy()
        
        for i in range(2, len(nums)+1): # 遍历长度
            for j in range(len(nums)):  # 遍历起始点
                if j + i - 1 >= len(nums):
                    break
                if nums_sum_temp[j] + nums[j+i-1] >= s:
                    return i
                else:
                    nums_sum_temp[j] += nums[j+i-1]
        return 0
```

Python $O(n^2)$ 超时

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        sums = 0
        left = 0
        ans = len(nums) + 1
        for i in range(len(nums)):
            sums += nums[i]
            while (sums >= s):
                ans = min(ans, i + 1 - left)
                sums -= nums[left]
                left += 1
        if ans < len(nums) + 1:
            return ans
        else:
            return 0
```

$O(n)$ 先一直向前加，加到和>=s后再从左边开始向右减，然后再从刚刚加到的位置开始。



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

