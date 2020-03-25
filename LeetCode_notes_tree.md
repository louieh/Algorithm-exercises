## LeetCode - Tree

[toc]

### 98. Validate Binary Search Tree

```python
def isValidBST(self, root: TreeNode) -> bool:
    if not root:
        return True
    if not root.left and not root.right:
        return True

    if root.left:
        if not root.left.val < root.val:
            self.isValidBST(root.left)
        else:
            return False

    if root.right:
        if not root.right.val > root.val:
            self.isValidBST(root.right)
        else:
            return False
    return True
```

第一遍：单纯判断树的每个节点是否左节点小于根、右节点大于根。问题在于 `self.isValidBST(root.left)`处递归时没有加 `return` 这会导致无论返回值是对或错都直接跳过不做反应(导致[2,1,3,2,null,null,null]返回True)。所以此处应该加上 `return` .

```python
def isValidBST(self, root: TreeNode) -> bool:
    if not root:
        return True
    if not root.left and not root.right:
        return True

    if root.left:
        if not root.left.val < root.val:
            return self.isValidBST(root.left)
        else:
            return False

    if root.right:
        if not root.right.val > root.val:
            return self.isValidBST(root.right)
        else:
            return False
    return True
```

第二遍：加上 `return` 后不会导致[2,1,3,2,null,null,null]返回True了，但会导致[3,2,1,1,null,null,null]返回True，因为当判断左子树为True时，由于此处添加了 `return` 所以函数直接返回True，而不会继续判断右子树。

```python
def isValidBST(self, root: TreeNode) -> bool:
    if not root:
        return True
    if not root.left and not root.right:
        return True

    if root.left:
        if not root.left.val < root.val or not self.isValidBST(root.left):
            return False

    if root.right:
        if not root.right.val > root.val or not self.isValidBST(root.right):
            return False
    return True
```

第三遍：修改了第二遍提前回返回True的问题：左右子树判断为真时不返回任何，只有假时返回False，全部判断完成后返回True。此时达到了判断每个节点的目的，但是会漏过不同层间错误导致误判成BST. 比如判断[10,5,15,null,null,6,20]为True. 因为当插入6时应该会插入到左节点而不是根的右子树。

```python
def isValidBST_tool(self, root, lower, upper):
    if not root:
        return True
    if not (root.val < upper and root.val > lower):
        return False
    return self.isValidBST_tool(root.left, lower, root.val) and self.isValidBST_tool(root.right, root.val, upper)

def isValidBST(self, root: TreeNode) -> bool:
    return self.isValidBST_tool(root, -2**32, 2**32)
```

添加递归函数，参数为根、下限、上限。根值须在上下限之间，在递归过程中设根值为其右子树下限，上限为upper，设根为左子树上限，下限为lower，上下限会维持，所以不会产生下层左节点小于上层根节点的情况。

```python
#9/21/2019 再做
class Solution:
    def __init__(self):
        self.ans = True
        
    def isValidBST_tool(self, root, low, high):
        if not root:
            return
        
        if low is not None and high is not None:
            if not (root.val < high and root.val > low):
                self.ans = False
                return
        elif low is not None or high is not None:
            if low is not None:
                if root.val <= low:
                    self.ans = False
                    return
            if high is not None:
                if root.val >= high:
                    self.ans = False
                    return

        self.isValidBST_tool(root.left, low, root.val)
        self.isValidBST_tool(root.right, root.val, high)
    
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        self.isValidBST_tool(root, None, None)
        
        return self.ans
```

```python
# 3/18/2020
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        def helper(root, low, high):
            if root.left:
                if root.left.val < root.val and root.left.val > low:
                    if not helper(root.left, low, root.val):
                        return False
                else:
                    return False
            
            if root.right:
                if root.right.val > root.val and root.right.val < high:
                    if not helper(root.right, root.val, high):
                        return False
                else:
                    return False
            return True
        import sys
        return helper(root, -sys.maxsize, sys.maxsize)
```



### 104. Maximum Depth of Binary Tree

```python
"""
def __init__(self):
    self.stack_list = []
    self.depth = 0

def maxDepth(self, root):
    if not root:
        return 0
    else:
        self.stack_list.append([root])

    while len(self.stack_list):
        temp_list = []
        temp_node = self.stack_list.pop()
        for each_node in temp_node:
            if each_node.left:
                temp_list.append(each_node.left)
            if each_node.right:
                temp_list.append(each_node.right)
        if temp_list:
            self.stack_list.append(temp_list)

        self.depth += 1
    return self.depth
"""

def maxDepth_tool(self, root):
    if not root:
        return 0
    if not root.left and not root.right:
        return 0

    lval, rval = 0, 0
    if root.left:
        lval = self.maxDepth_tool(root.left) + 1
    if root.right:
        rval = self.maxDepth_tool(root.right) + 1
    return max(lval, rval)

def maxDepth_tool1(self, root):
    if not root:
        return 0

    lval = 0
    rval = 0
    if root.left:
        lval = self.maxDepth_tool1(root.left)
    if root.right:
        rval = self.maxDepth_tool1(root.right)
    return max(lval, rval) + 1

def maxDepth(self, root):
    if not root:
        return 0
    return self.maxDepth_tool(root)+1
```

求树深度两种方法，一个是队列层序遍历，一个递归，递归代码比较简洁，上面maxDepth_tool和maxDepth_tool1是递归方法，两个函数加1的位置不同而已。还有一种求深度递归方法是增加当前深度参数到递归函数中，相对好理解，参见543题。 

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root: return 0
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return max(l, r) + 1
```

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        def maxDepth_helper(root, h):
            if not root: return h-1
            l = maxDepth_helper(root.left, h+1)
            r = maxDepth_helper(root.right, h+1)
            return max(l,r)
        
        return maxDepth_helper(root, 1)
```



### 108. Convert Sorted Array to Binary Search Tree

```python
class Solution:
    def insertBST_tool(self, root, val):
        if not root:
            return TreeNode(val)
        if val > root.val:
            root.right = self.insertBST_tool(root.right, val)
        else:
            root.left = self.insertBST_tool(root.left, val)
        return root
    
    def splitNum(self, root, nums, low, high):
        if low > high:
            return root
        mid = int((high-low)/2+low)
        root = self.insertBST_tool(root, nums[mid])
        root = self.splitNum(root, nums, low, mid-1)
        root = self.splitNum(root, nums, mid+1, high)
        return root
        
    
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return
        root = self.splitNum(None, nums, 0, len(nums)-1)
        return root
```

因为数组是有序的，主要思想和 MergeSort 类似，取中点插入树中，递归进行左右两侧数组。速度较慢。

改进将两个递归合并成一个递归

```python
class Solution:
    def sortedArrayToBST_tool(self, nums, low, high):
        if low > high:
            return
        mid = (high-low)//2+low
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST_tool(nums, low, mid-1)
        root.right = self.sortedArrayToBST_tool(nums, mid+1, high)
        return root
        
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return
        root = self.sortedArrayToBST_tool(nums, 0, len(nums)-1)
        return root
```



### 110. Balanced Binary Tree

```python
def levelorder_(self, node):
    levelcount = 0
    temp_stack = [[node]]
    while temp_stack:
        temp_list = []
        temp_node = temp_stack.pop()
        for each_node in temp_node:
            if each_node.left:
                temp_list.append(each_node.left)
            if each_node.right:
                temp_list.append(each_node.right)
        if temp_list:
            temp_stack.append(temp_list)
        levelcount += 1
    return levelcount

def balance_value(self, node):
    left_value = 0
    right_value = 0
    if node.left:
        left_value = self.levelorder_(node.left)
    if node.right:
        right_value = self.levelorder_(node.right)
    return left_value - right_value

def isBalanced(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    ans = True
    if not root or (not root.left and not root.right):
        return True
    temp_stack = [[root]]
    while temp_stack:
        temp_list = []
        temp_node = temp_stack.pop()
        for each_node in temp_node:
            ans_value = self.balance_value(each_node)
            if ans_value not in [-1,0,1]:
                ans = False
                break
            if each_node.left:
                temp_list.append(each_node.left)
            if each_node.right:
                temp_list.append(each_node.right)
        if temp_list:
            temp_stack.append(temp_list)
    return ans
```

判断是否是平衡二叉树，很蛋疼的方法，不断的进行层序遍历。

```python
def isBalanced_tool(self, root):
    l = 0
    r = 0
    if root.left:
        l = self.isBalanced_tool(root.left)
    if root.right:
        r = self.isBalanced_tool(root.right)

    if l - r not in [-1,0,1]:
        self.ans = False
    return max(l,r) + 1

def isBalanced(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """
    if not root:
        return True
    if not root.left and not root.right:
        return True
    self.ans = True
    self.isBalanced_tool(root)
    return self.ans
```

递归方法。

```python
# 9/24/2019
class Solution:
    def __init__(self):
        self.ans = True
        
    def isBalanced_tool(self, root):
        if not root:
            return 0
        
        left = self.isBalanced_tool(root.left)
        right = self.isBalanced_tool(root.right)
        
        if left - right not in [-1, 0, 1]:
            self.ans = False
            
        return max(left, right) + 1
    
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        self.isBalanced_tool(root)
        
        return self.ans
```



### 235. Lowest Common Ancestor of a Binary Search Tree

```python
class Solution:
    def __init__(self):
        self.ans = None
        
    def lowestCommonAncestor_tool(self, root, p, q):
        if not root:
            return False
        

        left = self.lowestCommonAncestor_tool(root.left, p, q)
        right = self.lowestCommonAncestor_tool(root.right, p, q)
        
        if root.val == p.val or root.val == q.val:
            temp = True
        else:
            temp = False
        

        if temp + left + right >= 2:
            self.ans = root
            
        return left or right or temp
        
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return
        
        if p.val > root.val and q.val > root.val:
            self.lowestCommonAncestor_tool(root.right, p, q)
        elif p.val < root.val and q.val < root.val:
            self.lowestCommonAncestor_tool(root.left, p, q)
        else:
            self.lowestCommonAncestor_tool(root, p, q)
        return self.ans
```

不是最优解，递归过程中没有利用到二分查找树

```python
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return 
        
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
```



### 236. Lowest Common Ancestor of a Binary Tree

```python
class Solution:

    def __init__(self):
        # Variable to store LCA node.
        self.ans = None

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def recurse_tree(current_node):

            # If reached the end of a branch, return False.
            if not current_node:
                return False

            # Left Recursion
            left = recurse_tree(current_node.left)

            # Right Recursion
            right = recurse_tree(current_node.right)

            # If the current node is one of p or q
            mid = current_node == p or current_node == q

            # If any two of the three flags left, right or mid become True.
            if mid + left + right >= 2:
                self.ans = current_node

            # Return True if either of the three bool values is True.
            return mid or left or right

        # Traverse the tree
        recurse_tree(root)
        return self.ans
```

DFS 回溯，应熟练掌握此模式。



### 250. Count Univalue Subtrees

```python
class Solution:
    def countUnivalSubtrees(self, root: TreeNode) -> int:
        if not root:
            return 0

        def helper(root):
            if not root:
                return 0, None
            l_score, l_value = helper(root.left)
            r_score, r_value = helper(root.right)
            if l_value is None and r_value is None:
                return 1, root.val
            elif (l_value == r_value and l_value == root.val) or (l_value is None and r_value == root.val) or (r_value is None and l_value == root.val):
                return l_score+r_score+1, root.val
            else:
                return l_score+r_score, 'None'
        
        score, value = helper(root)
        return score
```



### 404. Sum of Left Leaves

```python
def __init__(self):
    self.ans = 0

def sumOfLeftLeaves(self, root: TreeNode) -> int:
    if not root:
        return 0

    if root.left:
        if not root.left.left and not root.left.right:
            self.ans += root.left.val
        else:
            self.sumOfLeftLeaves(root.left)

    self.sumOfLeftLeaves(root.right)

    return self.ans
```



### 429. N-ary Tree Level Order Traversal

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, children):
        self.val = val
        self.children = children
"""
class Solution:    
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []
        
        ans = [[root.val]]
        q = [[root]]
        
        while q:
            valList = []
            nodeList = []
            tempNodeList = q.pop()
            for each in tempNodeList:
                if each.children:
                    nodeList += each.children
            if nodeList:
                for each in nodeList:
                    valList.append(each.val)
            if valList:
                ans.append(valList)
                q.append(nodeList)
        
        return ans
```

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []
        
        ans = [[root.val]]
        temp = [root]
        
        while temp:
            temp_ans = []
            temp_node = []
            for each in temp:
                temp_ans += [i.val for i in each.children]
                temp_node += each.children
            if temp_ans:
                ans.append(temp_ans)
            temp = temp_node
        
        return ans
```



### 450. Delete Node in a BST

```python
# 错误
class Solution:
    def deleteNode_tool(self, root, key):
        if not root:
            return
        if root.left and root.left.val == key:
            if not root.left.left and not root.left.right:
                root.left = None
            elif root.left.left and not root.left.right:
                root.left = root.left.left
            elif root.left.right and not root.left.left:
                root.left = root.left.right
            else:
                if root.left.right.left:
                    root.left.val = root.left.right.left.val
                    root.left.right.left = None
                else:
                    root.left.val = root.left.right.val
                    root.left.right = root.left.right.right
        elif root.right and root.right.val == key:
            if not root.right.left and not root.right.right:
                root.right = None
            elif root.right.left and not root.right.right:
                root.right = root.right.left
            elif root.right.right and not root.right.left:
                root.right = root.right.right
            else:
                if root.right.right.left:
                    root.right.val = root.right.right.left.val
                    root.right.right.left = None
                else:
                    root.right.val = root.right.right.val
                    root.right.right = root.right.right.right
        elif key > root.val:
            self.deleteNode_tool(root.right, key)
        else:
            self.deleteNode_tool(root.left, key)
    
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return
        
        if root.val == key:
            if not root.left and not root.right:
                return
            elif root.left and not root.right:
                return root.left
            elif root.right and not root.left:
                return root.right
            else:
                if root.right.left:
                    root.val = root.right.left.val
                    root.right.left = root.right.left.left or root.right.left.right
                else:
                    root.val = root.right.val
                    root.right = root.right.right
            return root
        
        ans = root
        self.deleteNode_tool(root, key)
        return ans
```

```python
class Solution:
    
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return
        
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            else:
                mini = self.get_mini(root.right)
                root.val = mini
                root.right = self.deleteNode(root.right, mini)
        return root

    def get_mini(self, root):
        while root.left:
            root = root.left
        return root.val
```



### 543. Diameter of Binary Tree

```python
def __init__(self):
    self.max_diameter = 0

def diameterOfBinaryTree_tool(self, root, temp, temp_list):
    if not root.left and not root.right:
        if temp > temp_list[0]:
            temp_list[0] = temp

    if root.left:
        self.diameterOfBinaryTree_tool(root.left, temp+1, temp_list)

    if root.right:
        self.diameterOfBinaryTree_tool(root.right, temp+1, temp_list)

def tool(self, root):
    if not root:
        return 0
    if not root.left and not root.right:
        return 0
    max_l = 0
    max_r = 0
    temp_list = [0]

    if root.left:
        self.diameterOfBinaryTree_tool(root.left, 1, temp_list)
        max_l = temp_list[0]

    if root.right:
        temp_list = [0]
        self.diameterOfBinaryTree_tool(root.right, 1, temp_list)
        max_r = temp_list[0]

    return max_l + max_r

def diameterOfBinaryTree(self, root: TreeNode) -> int:
    if not root:
        return 0
    a = self.tool(root)
    if a > self.max_diameter:
        self.max_diameter = a
    if root.left:
        self.diameterOfBinaryTree(root.left)
    if root.right:
        self.diameterOfBinaryTree(root.right)
    return self.max_diameter

```

diameterOfBinaryTree_tool函数负责计算节点深度，tool函数负责相加每个节点左右子树深度和。

```python
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans = 1
        def helper(root):
            if not root:
                return 0
            left = helper(root.left)
            right = helper(root.right)
            self.ans = max(self.ans, left+right+1)
            return max(left, right) + 1
        
        helper(root)
        return self.ans-1
```





### 559. Maximum Depth of N-ary Tree

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, children):
        self.val = val
        self.children = children
"""
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        
        level = 0
        q = [[root]]
        
        while q:
            temp = []
            NodeList = q.pop()
            level += 1
            for each in NodeList:
                temp += each.children
            if temp:
                q.append(temp)
        return level
                
```

```python
# top-down
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution:       
    def __init__(self):
        self.ans = 0
        
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        
        def maxDepth_helper(root, depth):
            if not root.children:
                self.ans = max(self.ans, depth)
            else:
                for each in root.children:
                    maxDepth_helper(each, depth+1)
        
        maxDepth_helper(root, 1)
        
        return self.ans
```

```python
class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root: return 0
        if not root.children: return 1
        return max([self.maxDepth(each) for each in root.children]) + 1
```



### 563. Binary Tree Tilt

```python
    def findTilt_tool(self, root):
        if not root:
            return 0
        
        l = self.findTilt_tool(root.left)
        r = self.findTilt_tool(root.right)
        self.ans += abs(l-r)
        return root.val+l+r
        
    def findTilt(self, root: TreeNode) -> int:
        if not root:
            return 0
        if not root.left and not root.right:
            return 0
        
        self.ans = 0
        self.findTilt_tool(root)
        
        return self.ans
```

熟记此递归方法。自底向上回溯？



### 572. Subtree of Another Tree

```python
class Solution:
    def isSametree(self, root1, root2):
        if not root1 and not root2:
            return True
        if not root1 and root2 or not root2 and root1:
            return False
        
        if root1.val != root2.val:
            return False
        
        return self.isSametree(root1.left, root2.left) and self.isSametree(root1.right, root2.right)
    
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if not s and not t:
            return True
        if not s and t or not t and s:
            return False
        if s.val == t.val:
            if self.isSametree(s, t):
                return True
        if self.isSubtree(s.left, t):
            return True
        if self.isSubtree(s.right,t):
            return True
        
        return False
```



### 589. N-ary Tree Preorder Traversal

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution:
    def __init__(self):
        self.ans = []
        
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        self.ans.append(root.val)
        for each in root.children:
            self.preorder(each)
        return self.ans
```



### 590. N-ary Tree Postorder Traversal

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution:
    def __init__(self):
        self.ans = []
        
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return
        
        for each in root.children:
            self.postorder(each)
        self.ans.append(root.val)
        
        return self.ans
```



### 654. Maximum Binary Tree

```python
class Solution:  
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        if not nums:
            return
        root = TreeNode(max(nums))
        root.left = self.constructMaximumBinaryTree(nums[:nums.index(root.val)])
        root.right = self.constructMaximumBinaryTree(nums[nums.index(root.val)+1:])
        return root
```

注意总结此类分别向左右子树赋值的题目。



### 701. Insert into a Binary Search Tree

```python
# 熟记此种遍历方式，可直接返回结果
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)
        
        if val > root.val:
            root.right = self.insertIntoBST(root.right, val)
        else:
            root.left = self.insertIntoBST(root.left, val)
        return root
```

```python
# 9/22/2019
class Solution:
    def  insertIntoBST_tool(self, root, val):
        if val > root.val:
            if root.right:
                self.insertIntoBST_tool(root.right, val)
            else:
                root.right = TreeNode(val)
        else:
            if root.left:
                self.insertIntoBST_tool(root.left, val)
            else:
                root.left = TreeNode(val)
    
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)
        
        ans = root
        self.insertIntoBST_tool(root, val)
        return ans
```



### 703. Kth Largest Element in a Stream

```java
class KthLargest {
    TreeNode root;
    int k;
    public KthLargest(int k, int[] nums) {
        this.k = k;
        for (int num: nums) root = add(root, num);
    }

    public int add(int val) {
        root = add(root, val);
        return findKthLargest();
    }

    private TreeNode add(TreeNode root, int val) {
        if (root == null) return new TreeNode(val);
        root.count++;
        if (val < root.val) root.left = add(root.left, val);
        else root.right = add(root.right, val);

        return root;
    }

    public int findKthLargest() {
        int count = k;
        TreeNode walker = root;

        while (count > 0) {
            int pos = 1 + (walker.right != null ? walker.right.count : 0);
            if (count == pos) break;
            if (count > pos) {
                count -= pos;
                walker = walker.left;
            } else if (count < pos)
                walker = walker.right;
        }
        return walker.val;
    }

    class TreeNode {
        int val, count = 1;
        TreeNode left, right;
        TreeNode(int v) { val = v; }
    }
}

/**
 * Your KthLargest object will be instantiated and called as such:
 * KthLargest obj = new KthLargest(k, nums);
 * int param_1 = obj.add(val);
 */
```

```python
# 超时
class KthLargest:
    class TreeNode(object):
        def __init__(self, val):
            self.val = val
            self.count = 1
            self.left = None
            self.right = None

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.root = None
        for num in nums:
            self.root = self.add_init(self.root, num)
            
    def add_init(self, root, num):
        if not root:
            return self.TreeNode(num)
        root.count += 1
        if num > root.val:
            root.right = self.add_init(root.right, num)
        else:
            root.left = self.add_init(root.left, num)
        return root
        

    def add(self, val: int) -> int:
        self.root = self.add_init(self.root, val)
        return self.findKthLargest()
    
    def findKthLargest(self):
        temp_root = self.root
        k = self.k
        while k > 0:
            if temp_root.right:
                pos = temp_root.right.count + 1
            else:
                pos = 1

            if k == pos:
                break
            elif k > pos:
                k -= pos
                temp_root = temp_root.left
            else: #k < pos:
                temp_root = temp_root.right
        return temp_root.val
```

>I reimplement this solution in Python but I got a case that exceeded time limitation. This is a case whose initialization is an empty list. Its number of `add` operations is large and the added element is in the ascending order. If we use the simple step to add nodes in the tree, it will degrade into a linked-list. In such a case, will the complexity of `findKthLargest()` get worse? That makes me confused.



另一个版本的 java 代码要更快一些，因为非递归。

```java
class KthLargest {
    class Node {
        int val;
        int cnt;
        Node left, right;

        Node(int val) {
            this.val = val;
            cnt = 1;
        }
    }

    private int k;
    private Node root;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        if (nums.length == 0) {
            root = null;
            return;
        } else {
            root = new Node(nums[0]);
            Node cur;
            for (int i = 1; i < nums.length; i++) {
                cur = root;
                Node pre = null;
                while (cur != null) {
                    cur.cnt++;
                    pre = cur;
                    if (cur.val > nums[i]) cur = cur.left;
                    else if (cur.val < nums[i]) cur = cur.right;
                    else if (cur.val == nums[i]) break;
                }
                if (pre.val > nums[i]) pre.left = new Node(nums[i]);
                else if (pre.val < nums[i]) pre.right = new Node(nums[i]);
                else {
                    Node curLeft = cur.left;
                    cur.left = new Node(nums[i]);
                    cur.left.left = curLeft;
                }
            }
        }

    }

    public int add(int val) {
        Node cur;
        Node pre;
        if (root == null) {
            root = new Node(val);
        } else {
            cur = root;
            pre = null;
            while (cur != null) {
                cur.cnt++;
                pre = cur;
                if (cur.val > val) cur = cur.left;
                else if (cur.val < val) cur = cur.right;
                else if (cur.val == val) break;
            }
            if (pre.val > val) pre.left = new Node(val);
            else if (pre.val < val) pre.right = new Node(val);
            else {
                Node curLeft = cur.left;
                cur.left = new Node(val);
                cur.left.left = curLeft;
            }
        }

        int temp = k;
        cur = root;
        int rightnum = cur.right == null ? 0 : cur.right.cnt;
        int leftnum = cur.left == null ? 0 : cur.left.cnt;
        while (temp - rightnum != 1) {
            if (rightnum >= temp) {
                cur = cur.right;
            } else {
                cur = cur.left;
                temp = temp - rightnum - 1;
            }
            rightnum = cur.right == null ? 0 : cur.right.cnt;
            leftnum = cur.left == null ? 0 : cur.left.cnt;
        }
        return cur.val;
    }
}
```

将上面非递归改写为python

```python
class KthLargest(object):
    class TreeNode(object):
        def __init__(self, val):
            self.count = 1
            self.val = val
            self.left = None
            self.right = None

    def __init__(self, k, nums):
        self.k = k

        if not nums:
            return
        self.root = self.TreeNode(nums[0])
        for i in range(1, len(nums)):
            self.add_without(nums[i])

    def add_without(self, num):
        cur = self.root
        pre = None
        while cur:
            cur.count += 1
            pre = cur
            if num < cur.val:
                cur = cur.left
            elif num > cur.val:
                cur = cur.right
            else:
                break
        if pre.val > num:
            pre.left = self.TreeNode(num)
        elif pre.val < num:
            pre.right = self.TreeNode(num)
        else:
            curLeft = cur.left
            cur.left = self.TreeNode(num)
            cur.left.left = curLeft

    def add(self, num):
        if not self.root:
            self.root = self.TreeNode(num)
        else:
            self.add_without(num)
        return self.findKthLargest()

    def findKthLargest(self):
        k = self.k
        root = self.root
        while k > 0:
            if root.right:
                pos = root.right.count + 1
            else:
                pos = 1

            if k == pos:
                break
            elif k > pos:
                k -= pos
                root = root.left
            else:
                root = root.right
        return root.val

    def inorder(self, root):
        if not root:
            return

        self.inorder(root.left)
        print("root.val: {0} root.count: {1}".format(root.val, root.count))
        self.inorder(root.right)
```



### 814. Binary Tree Pruning

```python
class Solution:
    def pruneTree_tool(self, root):
        if not root:
            return False
        l = self.pruneTree_tool(root.left)
        r = self.pruneTree_tool(root.right)
        if not l:
            root.left = None
        if not r:
            root.right = None
        return root.val == 1 or l or r
    
    def pruneTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return
        self.pruneTree_tool(root)
        return root
```



### 993. Cousins in Binary Tree

```python
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        if not root:
            return False
        
        q = [root]
        while q:
            next_q = []
            existx = False
            existy = False
            for node in q:
                if node.val == x:
                    existx = True
                if node.val == y:
                    existy = True
                if node.left and node.right:
                    if (node.left.val == x and node.right.val == y) or (node.left.val == y and node.right.val == x):
                        return False
                if node.left:
                    next_q.append(node.left)
                if node.right:
                    next_q.append(node.right)
            if existx and existy:
                return True
            elif existx or existy:
                return False
            q = next_q
        return False
```



### 998. Maximum Binary Tree II

```python
class Solution:
    def insertIntoMaxTree(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)
        
        if val > root.val:
            temp = TreeNode(val)
            temp.left = root
            return temp
        else:
            root.right = self.insertIntoMaxTree(root.right, val)
            return root
```



### 1022. Sum of Root To Leaf Binary Numbers

```python
class Solution:
    def __init__(self):
        self.ans = []
        
    def sumRootToLeaf_tool(self, root, str_now):
        str_now += str(root.val)
        if root.left:
            self.sumRootToLeaf_tool(root.left, str_now)
        if root.right:
            self.sumRootToLeaf_tool(root.right, str_now)
        if not root.left and not root.right:
            self.ans.append(str_now)
    
    def sumRootToLeaf(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        self.sumRootToLeaf_tool(root, '')
        sum = 0
        for each in self.ans:
            sum += int(each,2)
        return sum
```

熟记：获取树每条路径。



### 1104. Path In Zigzag Labelled Binary Tree

```python
class Solution:
    def get_oppo(self, num_now, f):
        return 2*(2**(f-1))-1-num_now+2**(f-1)
    
    def pathInZigZagTree(self, label: int) -> List[int]:
        if label == 1:
            return [1]
        ans = [label]
        import math
        f = math.ceil(math.log(label+1, 2))
        while 1:
            if f == 2:
                ans.append(1)
                return ans[::-1]
            if f % 2 == 0: # 倒序
                label = self.get_oppo(label, f) // 2
                ans.append(label)
            else: # 正序
                label = self.get_oppo(label // 2, f-1)
                ans.append(label)
            f -= 1
```

满二叉树第n个节点所在层数（层数从1开始）: `math.ceil(math.log(n+1, 2))` 

设完全二叉树根节点层数为1，设有n个节点的完全二叉树层数为k，则其满足：k-1层满二叉树<=n<=k层满二叉树。

$2^{k-1}-1 \leq n \leq 2^k-1$ 则 k 为 $\left \lceil \ log_2n+1\right \rceil$ 以2为底n+1的指数，向上取整。



### 1110. Delete Nodes And Return Forest

```python
class Solution:
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        if not root:
            return []
        
        res = []
        
        def helper(root, is_root):
            if not root:
                return False
            
            deleted = root.val in to_delete
            if is_root and not deleted:
                res.append(root)
            
            if helper(root.left, deleted):
                root.left = None
            if helper(root.right, deleted):
                root.right = None
            return deleted
        
        helper(root, True)
        return res
```





**二叉树**

第n层的节点个数最多为: 2^n-1^ 

深度为n的二叉树最多节点个数：2^n^-1

**满二叉树**

节点个数：2^n^-1

节点层次：(log~2~i)+1

父节点：i=1 根无父节点 i!=1 父节点=i/2

leetcode 236 求2个节点的lca

**leetcode 112 path sum 树的深度遍历**

Leetcode 105/106 已知前序中序遍历还原树

leetcode 145 后序 非递归



Leetcode230 找第k大的数

leetcode 108

左儿子右兄弟表示法

python 构造树和链表



**617: 合并两个二叉树**

**965: 判断一个二叉树所有元素是否都相同**

**700: 二叉搜索树，查找，返回节点，没有返回空**

**589: n叉树前续遍历**

**590: n叉树后续遍历**

**559: n叉树最大深度**

897: Increasing Order Search Tree 

**872: 判断两二叉树叶子结点顺序是否相同**

669: Trim a Binary Search Tree

**104: 二叉树最大深度**

**429: n叉树层序遍历**

**637: 返回二叉树每层平均值**

**226: 反转二叉树**

993: 判断二叉树两节点是否是cousins

653: 给定二叉搜索树，判断树中是否有两个元素和等于给定值

606: Construct String from Binary Tree

538: Convert BST to Greater Tree

**100: 判断两棵树是否相同**

108: Convert Sorted Array to Binary Search Tree

**404: 求二叉树左叶子节点和**

**563: Binary Tree Tilt**

**543: 返回二叉树任意两节点最大路径**

**107: 层序遍历，从底向上返回每层数组**

**257: 返回二叉树层根到叶节点每条路径**

235: 二叉搜索树，给定两点，返回最小公共祖先

671: Second Minimum Node In a Binary Tree

**101: 判断二叉树是否对称**

437: 给定二叉树，返回路径和等于给定值，路径不一定经过根或叶

572: 判断一棵二叉树是否是另一棵的子树

**110: 判断二叉树是否平衡**

501: Find Mode in Binary Search Tree

**112: 给定二叉树，找从根到叶路径和等给定值**

**111: 给定二叉树，找最小深度**

687: 给定二叉树，找最大路径，路径上所有元素值相同ma