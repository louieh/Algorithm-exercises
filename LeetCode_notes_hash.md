## LeetCode - Tree

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



### 1679. Max Number of K-Sum Pairs

```python
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        record = defaultdict(int)
        res = 0
        for num in nums:
            if record[k-num] > 0:
                res += 1
                record[k-num] -= 1
            else:
                record[num] += 1
        return res
```

和 two sum 一样？？？



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



### 535. Encode and Decode TinyURL

```python
class Codec:
    def __init__(self):
        import string
        import random
        self.str_list = list(string.ascii_letters + string.digits)
        self.l = 6
        self.ans_dict = dict()
    
    def get_random_str(self, l):
        temp = ''
        for i in range(l):
            temp += random.choice(self.str_list)
        return temp

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        while 1:
            temp = self.get_random_str(self.l)
            if temp not in self.ans_dict.keys():
                self.ans_dict[temp] = longUrl
                return temp
        

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        return self.ans_dict[shortUrl]
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))
```

```python
class Codec:
    def __init__(self):
        import hashlib
        self._m = hashlib.md5()
        self._dict1 = {}
        self._dict2 = {}

    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        self._m.update(longUrl.encode("utf8"))
        md5 = self._m.hexdigest()
        self._dict1[longUrl] = md5
        self._dict2[md5] = longUrl
        return md5

    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        return self._dict2.get(shortUrl)
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))
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

```python
class MyHashMap:

    def __init__(self):
        self.length = 769
        self.hashmap = [Bucket() for _ in range(self.length)]
    
    def _hash(self, key):
        return key % self.length

    def put(self, key: int, value: int) -> None:
        index = self._hash(key)
        self.hashmap[index].insert(key, value)

    def get(self, key: int) -> int:
        index = self._hash(key)
        return self.hashmap[index].get(key)
        
    def remove(self, key: int) -> None:
        index = self._hash(key)
        return self.hashmap[index].remove(key)

class Node(object):
    def __init__(self, key, val, next=None):
        self.key = key
        self.val = val
        self.next = next

class Bucket(object):
    def __init__(self):
        self.head = Node(-1, -1)
    
    def get(self, key):
        head = self.head
        while head:
            if head.key == key:
                return head.val
            head = head.next
        return -1
    
    def insert(self, key, val):
        head = self.head
        while head:
            if head.key == key:
                head.val = val
                return
            head = head.next
        node = Node(key, val)
        node.next = self.head.next
        self.head.next = node
    
    def remove(self, key):
        head = self.head
        while head.next:
            if head.next.key == key:
                head.next = head.next.next
                return
            head = head.next


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```



### 1396. Design Underground System

```python
class UndergroundSystem:

    def __init__(self):
        self.checkin_dict = dict()
        self.alltravel_dict = defaultdict(lambda: defaultdict(int))

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.checkin_dict[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        start_station, checkin_time = self.checkin_dict.pop(id)
        self.alltravel_dict[(start_station, stationName)]["times"] += 1
        self.alltravel_dict[(start_station, stationName)]["summation"] += (t - checkin_time)

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        _dict = self.alltravel_dict[(startStation, endStation)]
        return _dict["summation"] / _dict["times"]


# Your UndergroundSystem object will be instantiated and called as such:
# obj = UndergroundSystem()
# obj.checkIn(id,stationName,t)
# obj.checkOut(id,stationName,t)
# param_3 = obj.getAverageTime(startStation,endStation)
```
