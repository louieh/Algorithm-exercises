## LeetCode - Heap

[toc]

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

