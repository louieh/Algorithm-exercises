## LeetCode - Graph

[toc]

### 207. Course Schedule

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        from collections import defaultdict
        graph = defaultdict(list)
        for i, j in prerequisites:
            graph[j].append(i)
            
        ans = []
        ans_set = set()
        seen = set()
        self.possible = True
        def dfs(node):
            seen.add(node)
            if node in graph:
                for each in graph.get(node):
                    if each in seen and each not in ans_set:
                        self.possible = False
                    if each not in seen:
                        dfs(each)
            ans.append(node)
            ans_set.add(node)
        
        for i in range(numCourses):
            if i not in seen:
                dfs(i)
        
        return self.possible
```

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        graph = defaultdict(list)
        for i, j in prerequisites:
            graph[j].append(i)
        
        seen = set()
        topo = set()

        def dfs(node):
            seen.add(node)
            for each in graph[node]:
                if each in seen and each not in topo:
                    return False
                if each not in seen:
                    if not dfs(each): # 这里直接return可能不会走下面topo.add(node)
                        return False
            topo.add(node)
            return True
        
        for node in range(numCourses):
            if node not in seen:
                res = dfs(node)
                if not res:
                    return False
        return True
```

DFS + Topological Sort 判断是否有环，参考下面 Topo sort ：

1. DFS 访问完成的顺序即为拓扑排序

2. 若访问到一个点发现该点已经访问过但没有访问完成，也就是没有在访问完成栈里，则说明有环。

3. DFS 遍历时，访问一个点将一个点入栈，访问完成出栈，如果到某个点发现有环，则栈顶到该点的所有点构成环。

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(set)
        for a, b in prerequisites:
            graph[a].add(b)

        finish = set()
        course_set = set(range(numCourses))

        while True:
            courses = course_set - graph.keys()
            courses.difference_update(finish)
            if not courses or len(finish) == numCourses: break
            finish = finish.union(courses)
            wait_to_delete = []
            for k, v in graph.items():
                v.difference_update(courses)
                if not v:
                    wait_to_delete.append(k)
            for each in wait_to_delete:
                graph.pop(each)
        
        return len(finish) == numCourses
```





### 210. Course Schedule II

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        from collections import defaultdict
        graph = defaultdict(list)
        for i, j in prerequisites:
            graph[j].append(i)
            
        ans = []
        ans_set = set()
        seen = set()
        self.possible = True
        def dfs(node):
            seen.add(node)
            if graph.get(node):
                for each in graph.get(node):
                    if each in seen and each not in ans_set:
                        self.possible = False
                    if each not in seen:
                        dfs(each)
            ans.append(node)
            ans_set.add(node)
            
        for i in range(numCourses):
            if i not in seen:
                dfs(i)
        return ans[::-1] if self.possible else []
```

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        for a, b in prerequisites:
            graph[a].append(b)

        seen = set()
        fini = []

        def dfs(node):
            seen.add(node)
            for depend_node in graph[node]:
                if depend_node in seen and depend_node not in fini:
                    return False
                if depend_node not in seen:
                    if not dfs(depend_node):
                        return False
            fini.append(node)
            return True

        for i in range(numCourses):
            if i not in seen:
                if not dfs(i):
                    return []
        return fini
```

Topo order，dfs判断有环

最后的结果是否倒序取决于在构造图时候的方向，如果构造成 a->b && a depended on b，也就是先完成b才能完成a，边的方法是指向依赖项的话，那么最后结果不需要反转，因为 topo order 添加的顺序是最先完成子节点递归的节点到最后完成递归的节点，那么如果这样构造图的话，最先完成的节点就是没有依赖的课程先完成，所以最后不需要反转。

最后循环那里，上面是遍历所有节点，理论上是可以从没有入度节点开始，也就是不作为任何课程的依赖课程的节点，但是当图有环的时候这样遍历会有问题。

```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(set)
        for a, b in prerequisites:
            graph[a].add(b)

        finish = []
        course_set = set(range(numCourses))

        while True:
            courses = course_set - graph.keys()
            courses.difference_update(set(finish))
            if not courses or len(finish) == numCourses: break
            finish += list(courses)
            wait_to_delete = []
            for k, v in graph.items():
                v.difference_update(courses)
                if not v:
                    wait_to_delete.append(k)
            for each in wait_to_delete:
                graph.pop(each)
        
        return finish if len(finish) == numCourses else []
```

如果我们将图中边的方法定义为「任务 --> 先前任务」该方法是直接找到没有先前任务的任务开始执行，也就是找到没有出度为零的点，执行完成后，将该任务从所有任务的依赖列表中删除，也就是 release 边，之后再找到图中没有先前任务的任务，直到执行完所有任务，可以理解为直接从叶节点开始，边执行边删除边，直到根节点。



### 310. Minimum Height Trees

```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]
        from collections import defaultdict
        graph = defaultdict(list)
        for i, j in edges:
            graph[i].append(j)
            graph[j].append(i)
        
        leaves = [k for k,v in graph.items() if len(v) == 1]
        while n > 2:
            n -= len(leaves)
            new_leaves = []
            for leaf in leaves:
                graph.get(graph.get(leaf)[0]).remove(leaf)
                if len(graph.get(graph.get(leaf)[0])) == 1:
                    new_leaves.append(graph.get(leaf)[0])
                graph.get(leaf).pop()
            leaves = new_leaves
        return leaves
```

https://leetcode.com/problems/minimum-height-trees/discuss/76055/Share-some-thoughts



### 332. Reconstruct Itinerary

Eulerian Path and Eulerian Circuit 经历每条边一次，对应哈密尔顿问题

|            | Eulerian Circuit                               | Eulerian Path                                                |
| ---------- | ---------------------------------------------- | ------------------------------------------------------------ |
| Undirected | Every vertex has an even degree                | Either every vertex has even degree or exactly two vertices have odd degree |
| Directed   | Every vertex has dqual indegree and out degree | At most one vertex has (outdegree) - (indegree) = 1 and at most one vertex has (indegree) - (outdegree) = 1 and all other vertices have equal in and out degrees |

 ```python
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        from collections import defaultdict
        dict_helper = defaultdict(list)
        ans = []
        for i, j in tickets:
            dict_helper[i].append(j)
        
        for each in dict_helper.values():
            each.sort()
        
            
        def dfs(start):
            while dict_helper[start]:
                node = dict_helper[start][0]
                dict_helper[start] = dict_helper[start][1:]
                dfs(node)
            ans.append(start)
        
        dfs("JFK")

        return ans[::-1]
 ```



### 399. Evaluate Division

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
#         a / b = x
#         b / c = y
#         c / d = z
        
#         a   b   a
#         - * - = - = x * y 
#         b   c   c
        
#         a   b   c   a
#         - * - * - = - = x * y * z
#         b   c   d   d
        graph = defaultdict(dict)
        res = []
        nodes = set()
        for i, equa in enumerate(equations):
            a, b = equa
            nodes.add(a)
            nodes.add(b)
            graph[a].update({b: values[i]})
            graph[b].update({a: 1/values[i]})
        
        def dfs(node, dest, visited, res):
            if node not in visited:
                visited.add(node)
                for nei, val in graph[node].items():
                    if nei not in visited:
                        if nei == dest:
                            return res * val
                        else:
                            temp = dfs(nei, dest, visited, res*val)
                            if temp:
                                return temp

        for a, b in queries:
            if a not in nodes or b not in nodes:
                res.append(-1)
                continue
            if a == b:
                res.append(1)
                continue
            val = graph[a].get(b)
            if val is not None:
                res.append(val)
                continue
            visited = set()
            print("a: ", a)
            print("b: ", b)
            temp = dfs(a, b, visited, 1)
            if temp is None: temp = -1
            res.append(temp)
        return res
```



### 547. Number of Provinces

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        from collections import defaultdict
        graph = defaultdict(set)
        seen = set()
        res = 0
        rows = len(isConnected)
        cols = len(isConnected[0])
        for row in range(rows):
            for col in range(cols):
                if isConnected[row][col] == 1:
                    graph[row].add(col)
                    graph[col].add(row)
        
        for each in graph:
            if each in seen: continue
            stack = [each]
            while stack:
                node = stack.pop()
                if node not in seen:
                    seen.add(node)
                    for each_node in graph[node]:
                        stack.append(each_node)
            res += 1
        return res
```



### 684. Redundant Connection

```python
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        from collections import defaultdict
        seen = set()
        def dfs(source, target):
            for each in graph[source]:
                if each not in seen:
                    if each == target:
                        return True
                    seen.add(each)
                    if dfs(each, target):
                        return True
        
        graph = defaultdict(list)
        for u, v in edges:
            seen = {u}
            if u in graph and v in graph and dfs(u, v):
                return [u, v]
            graph[u].append(v)
            graph[v].append(u)
```

It's basically doing Cycle **Prevention** and not Detection, which was the key piece I missed.

You're building the graph one edge at a time. However, before adding an edge between **u** and **v**, you first check if there already is a path between them, avoiding a cycle.



### 721. Accounts Merge

```python
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        from collections import defaultdict
        graph = defaultdict(set)
        email_to_name = dict()
        for acc in accounts:
            for email in acc[1:]:
                graph[acc[1]].add(email)
                graph[email].add(acc[1])
                email_to_name[email] = acc[0]
        
        ans, seen = [], set()
        for email in graph:
            if email not in seen:
                seen.add(email)
                component = [email]
                stack = [email]
                while stack:
                    temp = stack.pop()
                    for each in graph[temp]:
                        if each not in seen:
                            seen.add(each)
                            stack.append(each)
                            component.append(each)
                ans.append([email_to_name[email]] + sorted(component))
        return ans
```

https://leetcode.com/articles/accounts-merge/

把每个account的第一个邮件和剩下邮件相连，形成一个完全图，再从每个邮件开始dfs查找连通分量，每个连通分量即答案。



### 743. Network Delay Time

```python
# DFS
# TLE
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        
        graph = defaultdict(list)
        for source, target, time in times:
            graph[source].append((target, time))
        
        for each in graph:
            graph[each].sort()
        
        recev_time = [sys.maxsize] * (n + 1)
        
        def dfs(node, curr_time):
            nonlocal recev_time
            if curr_time >= recev_time[node]: return
            recev_time[node] = curr_time
            for nei in graph[node]:
                target, time = nei
                dfs(target, curr_time + time)
        
        dfs(k, 0)
        max_time = max(recev_time[1:])
        return max_time if max_time < sys.maxsize else -1
```

```python
# BFS
# Accept
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        
        import queue
        
        graph = defaultdict(list)
        for source, target, time in times:
            graph[source].append((target, time))
        
        for each in graph:
            graph[each].sort()
        
        recev_time = [sys.maxsize] * (n + 1)
        
        def bfs(node):
            q = queue.Queue()
            q.put(node)
            recev_time[node] = 0
            
            while not q.empty():
                node = q.get()
                for nei in graph[node]:
                    target, time = nei
                    new_recev_time = recev_time[node] + time
                    if new_recev_time < recev_time[target]:
                        recev_time[target] = new_recev_time
                        q.put(target)
        
        bfs(k)
        max_time = max(recev_time[1:])
        return max_time if max_time < sys.maxsize else -1
```



### 785. Is Graph Bipartite?

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        colored = {}
        
        def dfs(node):
            for each in graph[node]:
                if each not in colored:
                    colored[each] = -colored[node]
                    if not dfs(each):
                        return False
                else:
                    if colored[each] == colored[node]:
                        return False
            return True
        
        for i in range(len(graph)):
            if i not in colored:
                colored[i] = 1
                if not dfs(i):
                    return False
        return True
```

To be able to split the node set `{0, 1, 2, ..., (n-1)}` into sets A and B, we will try to color nodes in set A with color A (i.e., value 1) and nodes in set B with color B (i.e., value -1), respectively.

If so, ***the graph is bipartite if and only if the two ends of each edge must have opposite colors***. Therefore, we could just start with standard BFS to traverse the entire graph and

- color neighbors with opposite color if not colored, yet;
- ignore neighbors already colored with oppsite color;
- annouce the graph can't be bipartite if any neighbor is already colored with the same color.

**NOTE:** The given graph might not be connected, so we will need to loop over all nodes before BFS.

题目要求图中点是否可分为两组，使得每条边连接的两个点分在不同组。

算法：我们对点进行着色，所以只需要判断所有边的两个点是否可以为不同颜色

对未着色的点上色为1，dfs或bfs遍历相邻节点，未着色的话上相反颜色，已经有颜色的话判断颜色是否一致，一致则return False，不一致跳过。



### 787. Cheapest Flights Within K Stops

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        cost = [sys.maxsize for i in range(n)]
        cost[src] = 0
        for o in range(K+1):
            temp = cost.copy()
            for i, j, price in flights:
                if temp[i] == sys.maxsize:
                    continue
                elif cost[i] + price < temp[j]:
                    temp[j] = cost[i] + price
            cost = temp
        return cost[dst] if cost[dst] != sys.maxsize else -1
```

Bellman Ford算法，执行K轮，因为题目要求at most K stops 注意每轮当中要复制一遍数组，用原数组的开始点+price与现在数组中点进行比较。

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        cost = [sys.maxsize for i in range(n)]
        cost[src] = 0
        for _ in range(k+1):
            temp_cost = cost.copy()
            for fr, to, price in flights:
                if temp_cost[fr] != sys.maxsize:
                    temp_cost[to] = min(temp_cost[to], cost[fr]+price)
            cost = temp_cost
        
        return cost[dst] if cost[dst] != sys.maxsize else -1
```



### 797. All Paths From Source to Target

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        graph_dict = {index:v for index, v in enumerate(graph)}
        N = len(graph)
        ans = []
        
        def helper(node, path):
            for each in graph_dict[node]:
                temp_path = path.copy()
                temp_path.append(each)
                if each == N-1:
                    ans.append(temp_path)
                    continue
                helper(each, temp_path)
        
        helper(0, [0])
        
        return ans
```

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        
        g = collections.defaultdict(list)

        for i, vs in enumerate(graph):
            for v in vs:
                g[i].append(v)
        
        res = []

        def helper(node, path): # dfs
            path.append(node)
            if node == len(graph) - 1:
                res.append(path)
                return
            for each in g[node]:
                helper(each, path.copy())
        
        helper(0, [])
        return res
```



### 802. Find Eventual Safe States

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        G = {}
        q = collections.deque()
        ans = [False] * len(graph)
        for i in range(len(graph)):
            if not graph[i]:
                q.append(i)
            for j in graph[i]:
                if j in G:
                    G[j].add(i)
                else:
                    G[j] = {i}
        while q:
            node = q.popleft()
            ans[node] = True
            if node in G:
                for i in G[node]:
                    graph[i].remove(node)
                    if not graph[i]:
                        q.append(i)
        return [index for index, val in enumerate(ans) if val]
```

```python
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        
        ans = [False] * len(graph)
        G = collections.defaultdict(set)
        q = []

        for i, each in enumerate(graph):
            if not each:
                q.append(i)
            for _each in each:
                G[_each].add(i)
        
        while q:
            node = q.pop()
            ans[node] = True
            for each in G[node]:
                graph[each].remove(node)
                if not graph[each]:
                    q.append(each)
        
        return [i for i, each in enumerate(ans) if each]
```

给定的图为有向图，构造一个反向的图，也就是入节点指向出节点，构造过程中将出度为0的点（terminal node）插入队列中。

While 循环队列，pop出一个节点，将其在ans列表中设置为True，之后遍历该节点的入度节点，也就是G[node]，将入度节点到该节点的边去掉，判断是否还有其他边，如果没有则说明此入度节点为safe node，再插入到队列中，因为题目定义safe node：A node is a **safe node** if every possible path starting from that node leads to a **terminal node** (or another safe node).

graph[each].remove(node) 这一步可以优化



### 886. Possible Bipartition

```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        
        ans = True

        graph = collections.defaultdict(list)
        for f, t in dislikes:
            graph[f].append(t)
            graph[t].append(f)
        
        def dfs(node, color):
            nonlocal ans
            color_dict[node] = color
            for nei in graph[node]:
                if color_dict.get(nei) == color:
                    ans = False
                elif nei not in color_dict:
                    dfs(nei, 1 - color)

         
        color_dict = dict()

        for i in range(1, n + 1):
            if i not in color_dict:
                dfs(i, 0)
                if not ans:
                    return False
        return True
```

```python
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
    
        graph = collections.defaultdict(list)
        for f, t in dislikes:
            graph[f].append(t)
            graph[t].append(f)
        
        def dfs(node, color):
            color_dict[node] = color
            for nei in graph[node]:
                if color_dict.get(nei) == color:
                    return False
                elif nei not in color_dict:
                    if not dfs(nei, 1 - color):
                        return False
            return True

        color_dict = dict()

        for i in range(1, n + 1):
            if i not in color_dict:
                if not dfs(i, 0):
                    return False
        return True
```

 [bipartite graph](https://en.wikipedia.org/wiki/Bipartite_graph) 问题，目的是将图中点分成两组，使得每组中点两两间没有连线。

大致思路是给点着色，相邻点着反色，如果在着色过程中发现有相邻点颜色相同那么方法False



### 1192. Critical Connections in a Network

```python
class Solution:
    
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        from collections import defaultdict
        graph = defaultdict(list)
        for v in connections:
            graph[v[0]].append(v[1])
            graph[v[1]].append(v[0])
        
        disc = [None for _ in range(n+1)]
        low = [None for _ in range(n+1)]

        res = []
        self.cur = 0
        
        def dfs(node, parent):
            if disc[node] is None:
                disc[node] = self.cur
                low[node] = self.cur
                self.cur += 1
                for n in graph[node]:
                    if disc[n] is None:
                        dfs(n, node)
                if parent is not None:
                    l = min([low[i] for i in graph[node] if i!=parent]+[low[node]])
                else:
                    l = min(low[i] for i in graph[node]+[low[node]])
                low[node] = l
        
        dfs(0, None)
        
        for v in connections:
            if low[v[0]]>disc[v[1]] or low[v[1]]>disc[v[0]]:
                res.append(v)
        
        return res
```



### 1202. Smallest String With Swaps

```python
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        graph = defaultdict(list)
        for a, d in pairs:
            graph[a].append(d)
            graph[d].append(a)
        s = list(s)
        visited = [False for _ in range(len(s))]
        
        def dfs(s, i, chars, index):
            if visited[i]:
                return
            chars.append(s[i])
            index.append(i)
            visited[i] = True
            for each in graph[i]:
                dfs(s, each, chars, index)
        
        for i in range(len(s)):
            if not visited[i]:
                chars = []
                index = []
                dfs(s, i, chars, index)
                chars.sort()
                index.sort()
                for c, i in zip(chars, index):
                    s[i] = c
        
        return "".join(s)
```

```python
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        graph = defaultdict(list)
        for a, d in pairs:
            graph[a].append(d)
            graph[d].append(a)
        s = list(s)
        visited = [False for _ in range(len(s))]
        
        def dfs(s, i, chars, index):
            if visited[i]:
                return
            chars.append(s[i])
            index.append(i)
            visited[i] = True
            for each in graph[i]:
                dfs(s, each, chars, index)
        res = []
        for i in range(len(s)):
            if not visited[i]:
                chars = []
                index = []
                dfs(s, i, chars, index)
                chars.sort()
                index.sort()
                res.extend(list(zip(chars, index)))
        for c, i in res:
            s[i] = c
        return "".join(s)
```



### [1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/)

```python
class UF(object):
    def __init__(self, n):
        self.uf = list(range(n))
        self.size = [1] * n
        self.maxsize = 1
    
    def find(self, x):
        if self.uf[x] != x:
            self.uf[x] = self.find(self.uf[x])
        return self.uf[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y: return False
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.uf[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.maxsize = max(self.maxsize, self.size[root_x])
        return True

class Solution:
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        critical, pseudo_critical = [], []
        edges_cp = copy.deepcopy(edges)
        for i, edge in enumerate(edges_cp):
            edge.append(i)
        
        edges_cp.sort(key=lambda x: x[2])
        weight_std = 0
        uf = UF(n)
        for i, j, w, index in edges_cp:
            if uf.union(i, j):
                weight_std += w
        
        for i, j, w, index in edges_cp:
            
            # find critical
            weight_critical = 0
            uf_critical = UF(n)
            for _i, _j, _w, _index in edges_cp:
                if index == _index: continue
                if uf_critical.union(_i, _j):
                    weight_critical += _w
                
            if uf_critical.maxsize < n or weight_critical > weight_std:
                critical.append(index)
                continue
            
            # find pseudo-critical
            weight_pse_critical = w
            uf_pse_critical = UF(n)
            uf_pse_critical.union(i, j)
            for _i, _j, _w, _index in edges_cp:
                if index == _index: continue
                if uf_pse_critical.union(_i, _j):
                    weight_pse_critical += _w
            if weight_pse_critical == weight_std:
                pseudo_critical.append(index)
        
        return [critical, pseudo_critical]
```

https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/editorial/

这道题有几个点是：

1. 明确最小生成树定义与在本题中的意义：

   A graph has exactly one minimum spanning tree (MST) weight, but there could be multiple MSTs with this weight.

2. 明确 *critical* edge 和 *pseudo-critical* edge 的定义与查找方法：

   A *critical* edge is an edge that, if removed from the graph, would increase the MST weight. It means that the edge appears in every MST.

   On the other hand, a *pseudo-critical* edge is an edge that can appear in some MSTs but not all. It means that the edge isn't necessary to maintain the MST weight, but we can include it without increasing the MST weight.

3. 用并查集使用 Kruskal's 算法计算最小生成树

其中并查集单独使用一个类构造，类中存放父子关系的是一个长度为 n 的列表，因为我们已知图中节点个数且省去了 find 方法中设置自己是自己父节点的步骤。size 属性记录以该节点为根的集合大小，注意这里设置 size 的原因是为了计算最大集合数量，所以设置 size 时候只会把小集合加到大集合里。maxsize 属性便是最大集合数量。



### 1514. Path with Maximum Probability

```python
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:

        graph = defaultdict(list)
        prob = dict()

        for i, (a, b) in enumerate(edges):
            graph[a].append(b)
            graph[b].append(a)
            prob[(a, b)] = succProb[i]
            prob[(b, a)] = succProb[i]

        seen = set()
        records = [(-1, start)]

        while records:
            dis, node = heappop(records)
            if node == end: return -dis # 这里之所以是负数是因为 start 初始值是 -1，而之所以初始值设置为 -1 是因为要求最大概率且默认为小根堆
            seen.add(node)
            for each in graph[node]:
                if each in seen: continue
                heappush(records, (dis * prob.get((node, each), 0), each))
        
        return 0
```

Dijkstra 算法 https://www.freecodecamp.org/chinese/news/dijkstras-shortest-path-algorithm-visual-introduction/



### 1557. Minimum Number of Vertices to Reach All Nodes

```python
class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        hasIndegree = set([b for a, b in edges])
        res = set()
        for a, b in edges:
            if a not in hasIndegree:
                res.add(a)
        return list(res)
        
        def dfs(node):
            seen = {node}
            stack = [node]
            while stack:
                temp = stack[-1]
                allAdjSeen = True
                for each in graph[temp]:
                    if each not in seen:
                        seen.add(each)
                        stack.append(each)
                        addAdjSeen = False
                        break
                if addAdjSeen:
                    stack.pop()
```

返回所有入度为0的点。dfs非递归算法。



### 1584. Min Cost to Connect All Points

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        heap = [(0, 0)]
        used = [False] * len(points)
        cost = 0
        edges_used = 0
        
        while edges_used < len(points):
            weight, curr_node = heapq.heappop(heap)
            
            if used[curr_node]: continue
            
            used[curr_node] = True
            
            cost += weight
            edges_used += 1
            
            for nxt_node in range(len(points)):
                if used[nxt_node] is True: continue
                nxt_weight = abs(points[curr_node][0] - points[nxt_node][0]) + abs(points[curr_node][1] - points[nxt_node][1])
                
                heapq.heappush(heap, (nxt_weight, nxt_node))
        return cost
```

Prim 算法，找最小生成树

https://zhuanlan.zhihu.com/p/136387766

https://leetcode.com/problems/min-cost-to-connect-all-points/solution/



### [1615. Maximal Network Rank](https://leetcode.com/problems/maximal-network-rank/)

```python
class Solution:
    def maximalNetworkRank(self, n: int, roads: List[List[int]]) -> int:

        if not roads: return 0

        graph = defaultdict(set)

        for fr, to in roads:
            graph[fr].add(to)
            graph[to].add(fr)
        
        graph_list = [(k, v) for k, v in graph.items()]
        # graph_list.sort(key=lambda x: len(x[1]), reverse=True)

        # print(f"graph_list: {graph_list}")
        
        res = -sys.maxsize
        for i in range(len(graph_list)):
            for j in range(i+1, len(graph_list)):
                _res = len(graph_list[i][1]) + len(graph_list[j][1])
                if graph_list[i][0] in graph_list[j][1]:
                    _res -= 1
                res = max(res, _res)
        
        return res
```

本质就是找两个出度和最大值，且如果这两个点是直接相连的，那么直接相连的边只能算一次，也就是需要减 1。

因为节点数量最大就只有100个所以嵌套循环是可以接受的。

最开始想的是把所有节点按出度排序，然后用最大出度点与其他点比较计算结果，但是也可能有很多最大出度点，不同最大出度点之间可能不相连，所以直接暴力循环判断不用排序了。

下面是不用嵌套循环的方法：

https://leetcode.com/problems/maximal-network-rank/solutions/3924675/beat-100-o-v-e-most-efficient-solution-greedy-no-hash-no-double-loop/

**Intuition**

The so called **network rank** of two cities is simply the sum of their degrees except when they are adjacent we minus that by 1.

> We consider this as a graph problem: a city is a *vertex* and a road is an *edge*. The *degree* is the concept in the graph theory.

Obviously we only care about cities with the largest degrees.

- If there are more than one cities with the largest degree we call them candidates and:
  - If there are a pair of `candidates` that are not adjacent, then the answer is `max_degree * 2`.
  - Otherwise the answer is `max_degree * 2 - 1`.
- If there is a single city with the largest degree, we call it king and the cities with the second largest degrees candidates.
  - If any one from the `candidates` is **not** connected to the `king` then the answer is `max_degree + second_max_degree`.
  - Otherwise the answer is `max_degree + second_max_degree - 1`

> **Combinatorics Knowledge**:
> The number of pairs in nn*n* items is simply:
> (n2)=n(n−1)/2{n \choose 2} = n(n-1) / 2(2*n*​)=*n*(*n*−1)/2
> If the total count of directly-connnected-pairs among `candidates` is less than that then we are guaranteed to have at least one pair of not-directly-connected candidates.

**Approach**

For max performance we avoided hash containers and double loop. We first go through all edges to find the degree info and then candidates info. Then we go through the edges again to check the connection relation between candidates (and king).

**Complexity**

- Time complexity:

Θ(*V*+*E*)
*V* is the *vertex* count which is the number of cities here.
*E* is the *edge* count which is the number of roads here.

- Space complexity:

Θ(*V*)

**code**

```java
class Solution {
public:
  int maximalNetworkRank(int n, vector<vector<int>>& roads) {
    vector<int> degrees(n);
    for (const vector<int>& road : roads) {
      int a = road[0];
      int b = road[1];
      ++degrees[a];
      ++degrees[b];
    }

    int max_degree = 0;
    int second_max_degree = 0;
    for (int degree : degrees) {
      if (degree < second_max_degree) {
        continue;
      }
      second_max_degree = degree;
      if (second_max_degree > max_degree) {
        swap(second_max_degree, max_degree);
      }
    }

    vector<bool> is_candidate(n);
    int candidate_count = 0;
    int king = -1;
    for (int i = 0; i < n; ++i) {
      if (degrees[i] == second_max_degree) {
        is_candidate[i] = true;
        ++candidate_count;
      }
      if (max_degree > second_max_degree && degrees[i] == max_degree) {
        king = i;
      }
    }

    if (max_degree == second_max_degree) {
      // Case 1: We have multiple candidates with the same max degrees.
      if (candidate_count > max_degree + 1) {
        return max_degree * 2;
      }
      int connection_count = 0;
      for (const vector<int>& road : roads) {
        int a = road[0];
        int b = road[1];
        if (is_candidate[a] && is_candidate[b]) {
          ++connection_count;
        }
      }
      if (connection_count < candidate_count * (candidate_count - 1) / 2) {
        return max_degree * 2;
      }
      return max_degree * 2 - 1;
    }

    // Case 2: We have a single max degree (king) and multiple second max degree candidates.
    int connection_count = 0;
    for (const vector<int>& road : roads) {
      int a = road[0];
      int b = road[1];
      if (a != king && b != king) {
        continue;
      }
      if (is_candidate[a] || is_candidate[b]) {
        ++connection_count;
      }
    }
    if (connection_count < candidate_count) {
      return max_degree + second_max_degree;
    }
    return max_degree + second_max_degree - 1;
  }
};
```



### 1631. Path With Minimum Effort

```python
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        rows, cols = len(heights), len(heights[0])
        dist = [[math.inf] * cols for i in range(rows)]
        dist[0][0] = 0
        minHeap = [(0, 0, 0)]
        
        while minHeap:
            history_dist, row, col = heappop(minHeap)
            if history_dist > dist[row][col]: continue
            if row == rows - 1 and col == cols - 1:
                return history_dist
            for a, b in [[row+1, col], [row, col+1], [row-1, col], [row, col-1]]:
                if 0 <= a < rows and 0 <= b < cols:
                    newDist = max(history_dist, abs(heights[a][b] - heights[row][col]))
                    if dist[a][b] > newDist:
                        dist[a][b] = newDist
                        heappush(minHeap, (dist[a][b], a, b))
```

Dijikstra 算法

https://leetcode.com/problems/path-with-minimum-effort/discuss/909017/JavaPython-Dijikstra-Binary-search-Clean-and-Concise



### 1971. Find if Path Exists in Graph

```python
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        graph = collections.defaultdict(list)

        if not edges and source == destination: return True
        
        for f, t in edges:
            graph[f].append(t)
            graph[t].append(f)
        
        visited = set()

        q = [source]

        while q:
            node = q.pop()
            if node not in visited:
                visited.add(node)
                for nei in graph[node]:
                    if nei == destination:
                        return True
                    q.append(nei)
        
        return False
```

简单的dfs即可



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



### Topo sort 

```java
# DFS + Topological Sort 并判断是否有环，如果有环打印构成环的边，只能判断有一个环的图。
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

