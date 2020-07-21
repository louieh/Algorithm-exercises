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

