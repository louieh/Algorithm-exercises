## LeetCode - Union-find

[toc]

### 1061. Lexicographically Smallest Equivalent String

```python
class Solution:
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        
        U = {}

        def find(x):
            U.setdefault(x, x)
            if U[x] != x:
                U[x] = find(U[x])
            return U[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)

            if root_x > root_y:
                U[root_x] = root_y
            else:
                U[root_y] = root_x
        
        for i in range(len(s1)):
            union(s1[i], s2[i])
        
        res = [find(each) for each in baseStr]
        return "".join(res)
```

Union-find 方法，构造一个邻接关系字典，key 是子节点，value 是父节点，child:parent，find 方法是返回最终根节点，并且在找根节点的过程中，把中间节点的 value 也会置成根节点，个人理解这一步是为了减少递归深度。union 方法是将两个点的根节点连接在一起，把他们放到一组。

https://leetcode.com/problems/lexicographically-smallest-equivalent-string/solutions/3047517/python3-union-find-template-explanations/

@md2030

> **Intuition**
>
> - The rules for the "equivalent characters" introduced in the problem simply means **the two characters are belong to the same group**.
> - And our job is that for each character in `baseStr`, we need to find its belonging group and find the smallest character from that group.
> - So we have two tasks:
>   - Create all the groups with equivalent characters from `s1` and `s2`.
>   - Find the group each character in `baseStr`, and find the smallest character in that group
>
> Grouping connected elements can be done using DFS/BFS/Union-find, I personally like Union-find because it makes the most sense, and very easy to implement using a template. If you don't know about union-find or don't have a good template, I put the best one I know below which I used a lot.
>
> **Basic Union-find template**
>
> ```python
> # UF is a hash map where you can find the root of a group of elements giving an element.
> # A key in UF is a element, UF[x] is x's parent.
> # If UF[x] == x meaning x is the root of its group.
> UF = {}
> 
> # Given an element, find the root of the group to which this element belongs.
> def find(x):
>     # this may be the first time we see x or y, so set itself as the root.
>     if x not in UF:
>         UF[x] = x
>     # If x == UF[x], meaning x is the root of this group.
>     # If x != UF[x], we use the find function again on x's parent UF[x] 
>     # until we find the root and set it as the parent (value) of x in UF.
>     if x != UF[x]:
>         UF[x] = find(UF[x])
>     return UF[x]
> 
> # Given two elements x and y, we know that x and y should be in the same group, 
> # this means the group that contains x and the group that contains y 
> # should be merged together if they are currently separate groups.
> # So we first find the root of x and the root of y using the find function.
> # We then set the root of y (rootY) as the root of the root of x (rootX).
> def union(x, y):
> 
>     rootX = find(x)
>     rootY = find(y)
>     # set the root of y (rootY) as the root of the root of x (rootX)
>     UF[rootX] = rootY
> ```
>
> The tricky part in this problem using Union-find template is to set the smallest element in a group as root of that group. Please see the changes below.
>
> **Union-find Solution**
>
> ```python
> class Solution:
>     def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
>         
>         UF = {}
>         def find(x):
>             UF.setdefault(x,x)
>             if x != UF[x]:
>                 UF[x] = find(UF[x])
>             return UF[x]
>         
>         def union(x,y):
>             rootX = find(x)
>             rootY = find(y)
>             # The main issue we need to take care of in this problem is
>             # that we want the root of a group to be 
>             # the smallest element in the group
>             # So every time we add an element in a group, we check if it is the smallest one,
>             # If it is, we set it as the root.
>             if rootX>rootY:
>                 UF[rootX] = rootY
>             else:
>                 UF[rootY] = rootX
>         
>         # Union the two equivalent characters
>         # at the same position from s1 and s2 into the same group.
>         for i in range(len(s1)):
>             union(s1[i],s2[i])
>         
>         # Simply find the root of the group a character belongs to
>         # Note that if c is not in any group, 
>         # we have UF.setdefault(x,x) in def find(x) to take care of it
>         res = []
>         for c in baseStr:
>             res.append(find(c))
>             
>         return ''.join(res)
> ```
