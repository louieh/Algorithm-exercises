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

初始化一个优先队列，先取出一个字母和对应的count，之后把这个字母和count作为一个中间变量存起来，之后再从队列里取字母，然后将中间变量中字母放入队列，将刚刚取出的字母设置为中间变量，也就是将每次赋值的字母取出队列以保证下一次不会从队列里取到。

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        counter = Counter(s)
        counter_list = [(c, count) for c, count in counter.items()]
        counter_list.sort(key=lambda x: x[1], reverse=True)

        if counter_list[0][1] > math.ceil(len(s) / 2): return ""

        res = [None] * len(s)

        index = 0
        for c, count in counter_list:
            while count > 0:
                if index >= len(s):
                    index = 1
                res[index] = c
                count -= 1
                index += 2
        
        return "".join(res)
```

这个方法是按偶数位置和奇数位置按序赋值，现将字母按频率排序，从最多的字母开始从偶数位置赋值，最开始判断一下最多的字母大小是否大于多大容量。不过有点没太想清楚这个方法是如何保证正确性的。



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



### 1834. Single-Threaded CPU

```python
class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        # heapq.heappush(heap, (processTime, i, enqueueTime))
        # heapq.heappop(heap)

        for i in range(len(tasks)):
            tasks[i].append(i)
        tasks.sort()

        heap, time = [], 0
        res = []

        for enq, pro, i in tasks:
            while heap and time < enq: # 还没到下个任务的时间
                pro_time, index, enq_time = heapq.heappop(heap)
                res.append(index)
                time = max(time, enq_time) + pro_time
            heapq.heappush(heap, (pro, i, enq))
        
        while heap:
            res.append(heapq.heappop(heap)[1])
        
        return res
```

现将任务按enqueueTime, processTime, index排序，之后遍历每个任务，遍历的过程中将任务插入heap中，插入的排序规则是processTime, index, enqueueTime，因为题目要求当有多个任务要执行的时候先执行processTime最短的，如果processTime相同先执行index小的，可以把heap当作当前要执行的任务队列。

遍历过程中在还没到下个任务执行时间时（`time < enq`）进入`while`语句，也就是开始执行队列中任务。先把任务从堆中pop出来，把index插入结果列表中，执行完当前队列中所有任务的时间是所有任务最大的enqueueTime + 所有的processTime，在任务执行过程中，仍有可能会插入新的任务，当时间到达下个任务的执行时间时。最后将堆中剩余的任务加入结果列表中。
