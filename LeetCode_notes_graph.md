## LeetCode - Graph

[toc]

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

Topo order

dfs判断有环



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

##### Dijikstra 算法

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

