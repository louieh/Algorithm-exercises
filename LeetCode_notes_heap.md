## LeetCode - Heap

[toc]

### 767. Reorganize String

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        counter = Counter(s)
        res = ""
        hq = [(-v, k) for k, v in counter.items()]
        heapq.heapify(hq)
        prev_val, prev_key = 0, ""
        while hq:
            val, key = heapq.heappop(hq)
            res += key
            if prev_val < 0:
                heapq.heappush(hq, (prev_val, prev_key))
            val += 1
            prev_val, prev_key = val, key
        if len(res) != len(s): return ""
        return res
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

