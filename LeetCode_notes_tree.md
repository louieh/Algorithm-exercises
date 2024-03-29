## LeetCode - Tree

[toc]

### 94. Binary Tree Inorder Traversal

```python
class Solution:
    def __init__(self):
        self.ans=[]
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return

        self.inorderTraversal(root.left)
        self.ans.append(root.val)
        self.inorderTraversal(root.right)
        return self.ans
```

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = []
        res = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            node = stack.pop()
            res.append(node.val)
            root = node.right
        return res
```



### 95. Unique Binary Search Trees II

```python
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if not n:
            return []
        
        def helper(start, end):
            if start > end:
                return [None, ]
            
            res = []
            for i in range(start, end+1):
                left = helper(start, i-1)
                right = helper(i+1, end)
                
                for j in left:
                    for k in right:
                        curr = TreeNode(i)
                        curr.left = j
                        curr.right = k
                        res.append(curr)
            return res
        
        return helper(1, n)
```

https://leetcode.com/articles/unique-binary-search-trees-ii/



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



### 99. Recover Binary Search Tree

```python
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        nums = []
        
        def helper1(root):
            if not root: return
            helper1(root.left)
            nums.append(root.val)
            helper1(root.right)
        
        def helper2(root):
            if not root: return
            helper2(root.left)
            root.val = nums.pop()
            helper2(root.right)
        
        helper1(root)
        nums.sort(reverse=True)
        helper2(root)
```

```python
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.fir = self.sec = None
        self.prev = TreeNode(-2147483649)
        
        def helper(root):
            if not root: return
            helper(root.left)
            if self.fir is None and self.prev.val > root.val:
                self.fir = self.prev
            if self.fir is not None and self.prev.val > root.val:
                self.sec = root
            self.prev = root
            helper(root.right)
        
        helper(root)
        
        self.fir.val, self.sec.val = self.sec.val, self.fir.val
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

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```



### 106. Construct Binary Tree from Inorder and Postorder Traversal

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not postorder or not inorder: return
        root = TreeNode(postorder.pop())
        inorder_root_index = inorder.index(root.val)
        root.right = self.buildTree(inorder[inorder_root_index+1:], postorder)
        root.left = self.buildTree(inorder[:inorder_root_index], postorder)
        return root
```

the time complexity of line 5 is O(N). We can improve it to O(1) by making a dict a advance.

and line 6 and line 7 also need to take O(K). 

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        inorder_dict = {}
        for i, v in enumerate(inorder):
            inorder_dict[v] = i
        
        def helper(low, high):
            if low > high: return 
            root = TreeNode(postorder.pop())
            inorder_root_index = inorder_dict[root.val]
            root.right = helper(inorder_root_index+1, high)
            root.left = helper(low, inorder_root_index-1)
            return root
        
        return helper(0, len(inorder)-1)
```

 先构造inorder字典，之后递归的时候不做切片操作，而是在helper函数中给定inorder列表范围



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



### 109. Convert Sorted List to Binary Search Tree

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def findMid(self, head):
        slow = fast = head
        mid_prev = None
        while fast and fast.next:
            mid_prev = slow
            slow = slow.next
            fast = fast.next.next
        if mid_prev:
            mid_prev.next = None
        return slow
    
    def sortedListToBST(self, head: ListNode) -> TreeNode:
    
        if not head:
            return
        mid = self.findMid(head)
        root = TreeNode(mid.val)
        if mid == head: return root
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(mid.next)
        return root
```

https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/solution/



### 100. Same Tree

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q: return True
        if p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False
```

```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q: return True
        if p and not q or not p and q or q.val != p.val: return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

递归前写清所有base case



### 101. Symmetric Tree

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root: return True

        def helper(root1, root2): # 与 same tree 一样，只是下面递归传入的左右子树不同
            if not root1 and not root2: return True
            if root1 and root2 and root1.val == root2.val:
                return helper(root1.left, root2.right) and helper(root1.right, root2.left)
            else:
                return False
        
        return helper(root.left, root.right)
```

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
      	# helper与isSameTree类似，base case一样只是递归传参数
        def helper(root1, root2):
            if root1 and root2 and root1.val != root2.val:
                return False
            if not root1 and not root2:
                return True
            if not root1 and root2 or root1 and not root2:
                return False
            return helper(root1.left, root2.right) and helper(root1.right, root2.left)
        return helper(root.left, root.right)
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



### 111. Minimum Depth of Binary Tree

```python
class Solution(object):
    def __init__(self):
        self.stack_list = []
        self.depth = 1
        
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        else:
            self.stack_list.append([root])
        
        while len(self.stack_list):
            temp_list = []
            temp_node = self.stack_list.pop()
            for each_node in temp_node:
                if not each_node.left and not each_node.right:
                    return self.depth
                if each_node.left:
                    temp_list.append(each_node.left)
                if each_node.right:
                    temp_list.append(each_node.right)
            self.stack_list.append(temp_list)
            self.depth += 1
```

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root: return 0
        res = sys.maxsize
        def helper(root, temp):
            nonlocal res
            if not root.left and not root.right:
                res = min(res, temp)
                return
            if root.left:
                helper(root.left, temp+1)
            if root.right:
                helper(root.right, temp+1)
        helper(root, 1)
        return res
```

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0

        depth = 1

        Q = [root]

        while Q:
            temp = []
            for node in Q:
                if not node.left and not node.right:
                    return depth
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            Q = temp
            depth += 1
        
        return depth
```



### 112. Path Sum

```python
class Solution:
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            return False
        else:
            return self.DFS(root, sum, 0)
    
    def DFS(self, root, sum, sum_now):
        if not root:
            return False
        sum_now += root.val
        if not root.left and not root.right:
            if sum_now != sum:
                return False
            else:
                return True
        return self.DFS(root.left, sum, sum_now) or self.DFS(root.right, sum, sum_now)
```

```python
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        
        def helper(root, res):
            if not root.left and not root.right and res + root.val == sum:
                return True
            
            if root.left:
                if helper(root.left, res+root.val):
                    return True
            if root.right:
                if helper(root.right, res+root.val):
                    return True
        ans = helper(root, 0)
        return True if ans is True else False
```



### 113. Path Sum II

```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:
            return []
        
        ans = []
        def helper(root, temp_sum, temp_str):
            if not root:
                return
            
            if not root.left and not root.right and temp_sum + root.val == sum:
                temp_str += str(root.val)
                ans.append(temp_str.split("*"))
            helper(root.left, temp_sum+root.val, temp_str+str(root.val)+"*")
            helper(root.right, temp_sum+root.val, temp_str+str(root.val)+"*")
        helper(root, 0, "")
        return ans
```



### 114. Flatten Binary Tree to Linked List

```python
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        
        if not root:
            return 
        def helper(root):
            if not root.left and not root.right:
                return
            
            if root.left:
                helper(root.left)
            if root.right:
                helper(root.right)
            temp = root.right
            root.right = root.left
            root.left = None
            while root.right:
                root = root.right
            root.right = temp
        helper(root)
        return root
```



### 116. Populating Next Right Pointers in Each Node

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left, right, next):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return
        
        Q = [root]
        
        while Q:
            temp = []
            for i in range(len(Q)):
                if Q[i].left:
                    temp.append(Q[i].left)
                if Q[i].right:
                    temp.append(Q[i].right)
                if i == len(Q)-1:
                    Q[i].next = None
                    break
                Q[i].next = Q[i+1]
            Q = temp
        return root
```

```python
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root: return
        
        q  = [root]
        while q:
            next_q = []
            for i in range(len(q)):
                if i == len(q) - 1:
                    q[i].next = None
                else:
                    q[i].next = q[i+1]
                if q[i].left:
                    next_q.append(q[i].left)
                if q[i].right:
                    next_q.append(q[i].right)
            q = next_q
        return root
```



### 117. Populating Next Right Pointers in Each Node II

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left, right, next):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return 
        
        Q = [root]
        
        while Q:
            temp = []
            for i in range(len(Q)): 
                if Q[i].left:
                    temp.append(Q[i].left)
                if Q[i].right:
                    temp.append(Q[i].right)
                if i == len(Q)-1:
                    break
                Q[i].next = Q[i+1]
                
            Q = temp
        return root
```



### 124. Binary Tree Maximum Path Sum

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.ans = -sys.maxsize
        def helper(root):
            if not root:
                return 0
            
            left = max(helper(root.left), 0)
            right = max(helper(root.right), 0)
            
            path_now = root.val + left + right
            self.ans = max(self.ans, path_now)
            
            return root.val + max(left, right)
        
        helper(root)
        return self.ans
```



### 129. Sum Root to Leaf Numbers

```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        def helper(root, num_str):
            nonlocal ans
            if root:
                if not root.left and not root.right:
                    ans += int(num_str+str(root.val))
                    return
                helper(root.left, num_str+str(root.val))
                helper(root.right, num_str+str(root.val))
        ans = 0
        helper(root, "")
        return ans
```

```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        stack = [(root, 0)]
        ans = 0
        while stack:
            root, temp_sum = stack.pop()
            temp_sum = temp_sum * 10 + root.val
            if not root.left and not root.right:
                ans += temp_sum
            else:
                if root.left:
                    stack.append((root.left, temp_sum))
                if root.right:
                    stack.append((root.right, temp_sum))
        return ans
```

```python
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        
        self.res = 0
        
        def helper(root, cur):
            if not root:
                return
            cur = cur * 10 + root.val
            if root.left:
                helper(root.left, cur)
            if root.right:
                helper(root.right, cur)
            if not root.left and not root.right:
                self.res += cur
        helper(root, 0)
        return self.res
```



### 173. Binary Search Tree Iterator

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.temp = []
        self.helper(root)
        self.index = 0
        
    def helper(self, root):
        if not root: return
        self.helper(root.left)
        self.temp.append(root.val)
        self.helper(root.right)

    def next(self) -> int:
        res = self.temp[self.index]
        self.index += 1
        return res

    def hasNext(self) -> bool:
        return self.index <= len(self.temp) - 1


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
```

```python
# 非递归
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        self.helper(root)
    
    def helper(self, root):
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        res = self.stack.pop()
        self.helper(res.right)
        return res.val

    def hasNext(self) -> bool:
        return bool(self.stack)
```



### 222. Count Complete Tree Nodes

```python
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```

```python
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        def get_depth(root):
            if not root:
                return 0
            return 1 + get_depth(root.left)
        
        depth = get_depth(root) - 1
        
        if depth == 0:
            return 1
        
        def exists(root, depth, index):
            left = 0
            right = 2**depth-1
            for i in range(depth):
                mid = left + (right - left) // 2
                if index <= mid:
                    root = root.left
                    right = mid
                else:
                    root = root.right
                    left = mid + 1
            return root is not None
        
        left, right = 0, 2**depth-1
        while left <= right:
            mid = left + (right - left) // 2
            if exists(root, depth, mid):
                left = mid + 1
            else:
                right = mid - 1
        
        return 2**depth-1 + left
```



### 226. Invert Binary Tree

```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        right = self.invertTree(root.right)
        left = self.invertTree(root.left)
        root.left = right
        root.right = left
        return root
```

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return

        temp = [root]

        while temp:
            nxt = []
            for node in temp:
                node.left, node.right = node.right, node.left
                if node.left:
                    nxt.append(node.left)
                if node.right:
                    nxt.append(node.right)
            temp = nxt
        
        return root
```



### 230. Kth Smallest Element in a BST

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        
        self.index = 0
        self.ans = None
        
        def helper(root):
            if root.left:
                helper(root.left)
            self.index += 1
            if self.index == k:
                self.ans = root.val
            if root.right:
                helper(root.right)
        helper(root)
        return self.ans
```

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        def inorder(r):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
    
        return inorder(root)[k - 1]
```

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right
```
非递归中序遍历

```python
# inorder
stack = []
while stack or root:
    while root:
        stack.append(root)
        root = root.left
    root = stack.pop()
    print(root.val)
    root = root.right
# preorder
stack = []
while stack or root:
    while root:
      	print(root.val)
        stack.append(root)
        root = root.left
    root = stack.pop()
    root = root.right
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



### 255. Verify Preorder Sequence in Binary Search Tree

```python
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        if not preorder:
            return True
        
        def helper(start, end):
            if start == end:
                return True
            root = preorder[start]
            index1 = index2 = None
            for i in range(start+1, end+1):
                if preorder[i] < root and index1 is None:
                    index1 = i
                elif preorder[i] > root and index2 is None:
                    index2 = i
                if index1 is not None and index2 is not None:
                    break
            if index1 is not None and index2 is not None:
                if index1 > index2:
                    return False
                for i in range(index2, end+1):
                    if preorder[i] < root:
                        return False
                return helper(index1, index2-1) and helper(index2, end)
            else:
                return helper((index1 or index2), end)
        
        return helper(0, len(preorder)-1)
```
Time Limit Exceeded

```python
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        if not preorder:
            return True
        
        stack = []
        low = -sys.maxsize
        for each in preorder:
            if each < low:
                return False
            while stack and each > stack[-1]:
                low = stack.pop()
            stack.append(each)
        return True
```



### 257. Binary Tree Paths

```python
class Solution:
    def __init__(self):
        self.ans = []
        
    def binaryTreePaths_tool(self, root, temp):
        if not root.left and not root.right:
            self.ans.append(temp+str(root.val))
        
        if root.left:
            self.binaryTreePaths_tool(root.left, temp+str(root.val)+"->")
        
        if root.right:
            self.binaryTreePaths_tool(root.right, temp+str(root.val)+"->")
        
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return []
        
        self.binaryTreePaths_tool(root, "")
        
        return self.ans
```

```python
class Solution:
    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        if not root:
            return []
        
        ans = []
        def helper(root, temp):
            if not root.left and not root.right:
                ans.append(temp + str(root.val))
            
            if root.left:
                helper(root.left, temp+str(root.val)+"->")
            if root.right:
                helper(root.right, temp+str(root.val)+"->")
        helper(root, "")
        return ans
```



### 285. Inorder Successor in BST

```python
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        if not root:
            return
        self.if_found = False
        self.ans = None
        def helper(root):
            if root.left:
                helper(root.left)
            if self.ans is None:
                if self.if_found:
                    self.ans = root
            if root.val == p.val:
                self.if_found = True
            if root.right:
                helper(root.right)
        
        helper(root)
        return self.ans
```

```python
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        if not root:
            return
        
        if p.right:
            p = p.right
            while p.left:
                p = p.left
            return p
        
        stack = []
        temp = None
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            if temp == p.val:
                return root
            temp = root.val
            
            root = root.right
        return None
```

非递归inorder

```python
class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        ans = None
        
        while root:
            if p.val >= root.val:
                root = root.right
            else:
                ans = root
                root = root.left
        return ans
```

Here is a much simpler solution to the problem. The idea is pretty straight forward.
We start from the root, utilizing the property of BST:

- If current node's value is less than or equal to p's value, we know our answer must be in the right subtree.
- If current node's value is greater than p's value, current node is a candidate. Go to its left subtree to see if we can find a smaller one.
- If we reach `null`, our search is over, just return the candidate.



### 298. Binary Tree Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        self.ans = 0
        def helper(root, num):
            self.ans = max(self.ans, num)
            if root.left:
                if root.left.val == root.val + 1:
                    helper(root.left, num+1)
                else:
                    helper(root.left, 1)
            if root.right:
                if root.right.val == root.val + 1:
                    helper(root.right, num+1)
                else:
                    helper(root.right, 1)
        
        helper(root, 1)
        return self.ans
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

```python
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        
        self.res = 0
        
        def helper(root):
            if not root: return
            if root.left and not root.left.left and not root.left.right:
                self.res += root.left.val
            helper(root.left)
            helper(root.right)
        
        helper(root)
        return self.res
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



### 437. Path Sum III

```python
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        if not root:
            return 0
        
        self.ans = 0
        def helper(root, sum_list):
            if not root:
                return
            sum_list.append(0)
            temp = [each+root.val for each in sum_list]
            for each in temp:
                if each == sum:
                    self.ans += 1
            helper(root.left, temp.copy())
            helper(root.right, temp.copy())
        
        helper(root, [])
        return self.ans
```



### 449. Serialize and Deserialize BST

```
class Codec:

    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.
        """
        
        if not root: return ""
        import json
        serialize_dict = dict()
        temp = [root]
        d = 0
        while temp:
            child_list = []
            for node in temp:
                child_val = [None, None]
                if node.left:
                    child_val[0] = node.left.val
                    child_list.append(node.left)
                if node.right:
                    child_val[1] = node.right.val
                    child_list.append(node.right)
                serialize_dict["{},{}".format(node.val, d)] = child_val
            temp = child_list
            d += 1
        return json.dumps(serialize_dict)
        
    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        if not data: return None
        serialize_dict = json.loads(data)
        root = None
        for each in serialize_dict:
            val, d = each.split(",")
            if d == '0': 
                root = TreeNode(int(val))
                break
        temp, d = [root], 0
        while temp:
            child_list = []
            for node in temp:
                left_val, right_val = serialize_dict.get("{},{}".format(node.val, d))
                if left_val is not None:
                    left_node = TreeNode(left_val)
                    node.left = left_node
                    child_list.append(left_node)
                if right_val is not None:
                    right_node = TreeNode(right_val)
                    node.right = right_node
                    child_list.append(right_node)
            d += 1
            temp = child_list
        return root
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



### 501. Find Mode in Binary Search Tree

```python
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        from collections import Counter
        self.temp_dict = Counter()
        def helper(root):
            if not root:
                return
            self.temp_dict[root.val] += 1
            helper(root.left)
            helper(root.right)
        helper(root)
        max_val = max([v for k,v in self.temp_dict.items()])
        return [k for k,v in self.temp_dict.items() if v == max_val]
```



### 513. Find Bottom Left Tree Value

```python
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        if not root:
            return
        
        stack = [root]
        
        while stack:
            temp_stack = []
            for node in stack:
                if node.left:
                    temp_stack.append(node.left)
                if node.right:
                    temp_stack.append(node.right)
            if not temp_stack:
                return stack[0].val
            else:
                stack = temp_stack
```



### 515. Find Largest Value in Each Tree Row

```python
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        if not root:
            return
        
        list_ = [root]
        ans = [root.val]
        while list_:
            temp_list = []
            for node in list_:
                if node.left:
                    temp_list.append(node.left)
                if node.right:
                    temp_list.append(node.right)
            if not temp_list:
                break
            ans.append(max([each.val for each in temp_list]))
            list_ = temp_list
        return ans
```



### 530. Minimum Absolute Difference in BST

```python
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        
        stack = []
        
        prev = None
        ans = sys.maxsize
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if prev is not None:
                ans = min(ans, root.val - prev)
            prev = root.val
            root = root.right
        return ans
```

the question is the same as 783

```python
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:

        prev = res = None
        
        def inorder(root):
            if not root: return
            inorder(root.left)
            nonlocal prev
            nonlocal res
            if prev is not None:
                res = min(res, abs(root.val - prev)) if res is not None else root.val - prev
            prev = root.val
            inorder(root.right)
        
        inorder(root)
        return res
```

中序遍历



### 538. Convert BST to Greater Tree

```python
class Solution:
    def __init__(self):
        self.temp = 0
        
    def convertBST(self, root: TreeNode) -> TreeNode:
        if not root:
            return
        self.convertBST(root.right)
        self.temp += root.val
        root.val = self.temp
        self.convertBST(root.left)
        return root
```

```python
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        total = 0
        
        node = root
        stack = []
        
        while stack or node:
            while node:
                stack.append(node)
                node = node.right
            node = stack.pop()
            total += node.val
            node.val = total
            node = node.left
        return root
```

is the same as 1038



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
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        res = 0
        def helper(root):
            nonlocal res
            if not root: return 0
            left = helper(root.left)
            right = helper(root.right)
            res = max(res, left+right+1)
            return max(left, right) + 1
        helper(root)
        return res-1
```



### 549. Binary Tree Longest Consecutive Sequence II

```python
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        self.ans = 0
        def helper(root):
            inc, dec = 1, 1
            if root.left:
                inc_temp, dec_temp = helper(root.left)
                if root.val == root.left.val + 1:
                    dec = dec_temp + 1
                elif root.val == root.left.val - 1:
                    inc = inc_temp + 1
            if root.right:
                inc_temp, dec_temp = helper(root.right)
                if root.val == root.right.val + 1:
                    dec = max(dec, dec_temp + 1)
                elif root.val == root.right.val - 1:
                    inc = max(inc, inc_temp + 1)
            self.ans = max(self.ans, inc + dec - 1)
            return inc, dec
        helper(root)
        return self.ans
```

No.687



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



### 606. Construct String from Binary Tree

```python
class Solution:
    def tree2str(self, t: TreeNode) -> str:
        if not t:
            return ""
        self.ans = ""
        
        def helper(t):
            self.ans += str(t.val)
            if not t.left and not t.right:
                return
            self.ans += "("
            if t.left:
                helper(t.left)
            self.ans += ")"
            if t.right:
                self.ans += "("
                helper(t.right)
                self.ans += ")"
        helper(t)
        return self.ans
```



### 652. Find Duplicate Subtrees

```python
class Solution(object):
    def findDuplicateSubtrees(self, root):
        """
        :type root: TreeNode
        :rtype: List[TreeNode]
        """
        from collections import defaultdict
        temp = defaultdict(int)
        ans = []
        def helper(root):
            if not root:
                return '#'
            left = helper(root.left)
            right = helper(root.right)
            id_str = str(root.val) + left + right
            temp[id_str] += 1
            if temp[id_str] == 2:
                ans.append(root)
            return id_str
        helper(root)
        return ans
```

使用一个id去表示每个子树并存放在set中



### 653. Two Sum IV - Input is a BST

```python
class Solution:
    def findTarget(self, root: TreeNode, k: int) -> bool:
        if not root:
            return False
        
        self._set = set()
        def helper(root):
            if not root:
                return
            c = k - root.val
            if c in self._set:
                return True
            self._set.add(root.val)
            return helper(root.left) or helper(root.right)
        ans = helper(root)
        return True if ans is True else False
```

```python
class Solution:
    def findTarget(self, root: TreeNode, k: int) -> bool:
        if not root:
            return False
        
        self._list = []
        def get_inorder_list(root):
            if not root:
                return
            get_inorder_list(root.left)
            self._list.append(root.val)
            get_inorder_list(root.right)
        get_inorder_list(root)
        start, end = 0, len(self._list) - 1
        while start < end:
            temp = self._list[start] + self._list[end]
            if temp > k:
                end -= 1
            elif temp < k:
                start += 1
            else:
                return True
        return False
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



### 662. Maximum Width of Binary Tree

```python
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        if not root:
            return 0
        ans = 0
        stack = [root]
        while any(stack):
            next_stack = []
            most_left_index = sys.maxsize
            most_right_index = -sys.maxsize
            for i in range(len(stack)):
                if stack[i] is not None:
                    most_left_index = min(i, most_left_index)
                    most_right_index = max(i, most_right_index)
                    next_stack.append(stack[i].left)
                    next_stack.append(stack[i].right)
                else:
                    next_stack.extend([None, None])
            ans = max(ans, most_right_index-most_left_index+1)
            left, right = sys.maxsize, -sys.maxsize
            for i in range(len(next_stack)):
                if next_stack[i] is not None:
                    left = min(left, i)
                    right = max(right, i)
            stack = next_stack[left: right+1]
        return ans
```

```python
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        if not root:
            return 0
        temp = dict()
        
        def dfs(node, node_id, level, mapping):
            if not node:
                return 0
            if level not in mapping:
                mapping[level] = node_id
            current = node_id - mapping[level] + 1
            left = dfs(node.left, node_id*2, level+1, mapping)
            right = dfs(node.right, node_id*2+1, level+1, mapping)
            return max(current, left, right)
        
        return dfs(root, 1, 0, temp)
```



### 669. Trim a Binary Search Tree

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def trimBST_tool(self, root, L, R):
        if not root:
            return None
        
        if root.val < L:
            return self.trimBST_tool(root.right, L, R)
        
        elif root.val > R:
            return self.trimBST_tool(root.left, L, R)
        
        else:
            root.left = self.trimBST_tool(root.left, L, R)
            root.right = self.trimBST_tool(root.right, L, R)
            return root
    
    def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
        if not root:
            return

        ans = self.trimBST_tool(root, L, R)
        return ans
```

```python
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if not root: return
        if root.val < low:
            return self.trimBST(root.right, low, high)
        elif root.val > high:
            return self.trimBST(root.left, low, high)
        else:
            root.left = self.trimBST(root.left, low, high)
            root.right = self.trimBST(root.right, low, high)
            return root
```



### 671. Second Minimum Node In a Binary Tree

```python
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        if not root or not root.left:
            return -1
        mini = root.val
        ans = sys.maxsize
        def helper(root):
            nonlocal ans
            if not root:
                return
            if root.val > mini and root.val < ans:
                ans = root.val
            helper(root.left)
            helper(root.right)
        helper(root)
        return ans if ans != sys.maxsize else -1
```

```python
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        if not root or not root.left:
            return -1
        mini = root.val
        ans = sys.maxsize
        def helper(root):
            nonlocal ans
            if not root:
                return
            if root.val > mini and root.val < ans:
                ans = root.val
            elif root.val == mini:
                helper(root.left)
                helper(root.right)
        helper(root)
        return ans if ans != sys.maxsize else -1
```



### 687. Longest Univalue Path

```python
    def longestUnivaluePath(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        self.ans = 0
        def helper(root):
            if not root:
                return 0
            left = helper(root.left)
            right = helper(root.right)
            left_temp = right_temp = 0
            if root.left and root.left.val == root.val:
                left_temp = left + 1
            if root.right and root.right.val == root.val:
                right_temp = right + 1
            self.ans = max(self.ans, left_temp+right_temp)
            return max(left_temp, right_temp)
        helper(root)
        return self.ans
```

No.549



### 700. Search in a Binary Search Tree

```python
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root: return None
        if root.val == val: return root
        return self.searchBST(root.right, val) if val > root.val else self.searchBST(root.left, val)
```



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

```python
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.pool = heapq.nlargest(k, nums)
        heapq.heapify(self.pool)

    def add(self, val: int) -> int:
        if len(self.pool) < self.k:
            heapq.heappush(self.pool, val)
        else:
            heapq.heappushpop(self.pool, val)
        return self.pool[0]
        


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```



### 771. Jewels and Stones

```python
# one year ago since 3/29/2020
class Solution:
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        sum = 0
        for eachJ in J:
            for eachS in S:
                if eachJ == eachS:
                    sum += 1
        return sum
```

```python
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        from collections import Counter
        S_c = Counter(S)
        ans = 0
        for l in J:
            ans += S_c[l]
        return ans
```



### 783. Minimum Distance Between BST Nodes

the question is the same as 530



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



### 863. All Nodes Distance K in Binary Tree

```python
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
        distance_dict = dict()
        ans = []
        def get_distance_dict(root):
            if not root:
                return -1
            if root.val == target.val:
                distance_dict[root] = 0
                return 0
            left = get_distance_dict(root.left)
            if left >= 0:
                distance_dict[root] = left + 1
                return left + 1
            right = get_distance_dict(root.right)
            if right >= 0:
                distance_dict[root] = right + 1
                return right + 1
            return -1
        
        def dfs(root, distance):
            if not root:
                return
            if root in distance_dict:
                distance = distance_dict.get(root)
            if distance == K:
                ans.append(root.val)
            dfs(root.left, distance + 1)
            dfs(root.right, distance + 1)
        
        get_distance_dict(root)
        dfs(root, 0)
        return ans
```

重点

```python
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:

        graph = collections.defaultdict(list)

        def helper(node):
            if not node: return
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val].append(node.val)
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val].append(node.val)
            helper(node.left)
            helper(node.right)
        
        helper(root)

        ans, seen = [], set()
        def search(node, dis):
            if node not in seen:
                seen.add(node)
                if dis == k:
                    ans.append(node)
                    return
                for each in graph.get(node, []):
                    search(each, dis+1)
        search(target.val, 0)
        return ans
```

先将树转为图字典，然后求与target距离为k的点

也可以给每个节点增加parent指针，思路类似。



### 865. Smallest Subtree with all the Deepest Nodes

```python
class Solution:
    def subtreeWithAllDeepest(self, root: TreeNode) -> TreeNode:
        if not root:
            return
        if not root.left and not root.right:
            return root
        dict_ = dict()
        
        def dfs(root, depth):
            if not root:
                return
            dict_[root.val] = depth
            dfs(root.left, depth+1)
            dfs(root.right, depth+1)
        dfs(root, 0)
        max_depth = max(dict_.values())
        def helper(root):
            if not root or dict_.get(root.val) == max_depth:
                return root
            left, right = helper(root.left), helper(root.right)
            return root if left and right else left or right
        return helper(root)
```

两遍dfs，第一遍记录每个点的深度，第二遍返回答案。



### 872. Leaf-Similar Trees

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def __init__(self):
        self.tempList1 = []
        self.tempList2 = []
    
    def get_leaf_list(self, root, tempList):
        if root.left:
            self.get_leaf_list(root.left, tempList)
        if not root.left and not root.right:
            tempList.append(root.val)
        if root.right:
            self.get_leaf_list(root.right, tempList)
        return tempList
            
    def leafSimilar(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """
        if not root1 and not root2:
            return True
        
        list1 = self.get_leaf_list(root1, self.tempList1)
        list2 = self.get_leaf_list(root2, self.tempList2)
        
        if list1 == list2:
            return True
        else:
            return False
```

```python
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:

        def get_leaf_list(root: TreeNode, res: List) -> None:
            if not root.left and not root.right:
                res.append(root.val)
                return
            if root.left:
                get_leaf_list(root.left, res)
            if root.right:
                get_leaf_list(root.right, res)
        
        root1_leaf_list = []
        root2_leaf_list = []

        get_leaf_list(root1, root1_leaf_list)
        get_leaf_list(root2, root2_leaf_list)

        return root1_leaf_list == root2_leaf_list
```



### 889. Construct Binary Tree from Preorder and Postorder Traversal

```python
class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        if not pre:
            return
        
        root = TreeNode(pre[0])
        if len(pre) == 1:
            return root
        
        num_left_branch = post.index(pre[1]) + 1
        root.left = self.constructFromPrePost(pre[1:num_left_branch+1], post[:num_left_branch])
        root.right = self.constructFromPrePost(pre[num_left_branch+1:], post[num_left_branch:-1])
        return root
```

左子树的节点数是前序遍历的第二个点在后续遍历中的index+1，因为后续遍历是访问完左右子树再访问根节点。



### 894. All Possible Full Binary Trees

```python
class Solution:
    
    memo = {0: [], 1: [TreeNode(0)]}
    
    def allPossibleFBT(self, N: int) -> List[TreeNode]:
        if N in self.memo:
            return self.memo[N]
        ans = []
        for x in range(N):
            y = N - 1 - x
            for left in self.allPossibleFBT(x):
                for right in self.allPossibleFBT(y):
                    temp = TreeNode(0)
                    temp.left = left
                    temp.right = right
                    ans.append(temp)
        self.memo[N] = ans
        return ans
```

https://leetcode.com/articles/all-possible-full-binary-trees/



### 897. Increasing Order Search Tree

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def __init__(self):
        self.ans = TreeNode(0)
        self.ans_ = self.ans
        
    def temp(self, root):
        if root.left:
            self.temp(root.left)
        self.ans_.right = TreeNode(root.val)
        self.ans_ = self.ans_.right
        if root.right:
            self.temp(root.right)
    
    def increasingBST(self, root: TreeNode) -> TreeNode:
        if not root:
            return
        self.temp(root)
        
        return self.ans.right
```

```python
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        if not root: return
        
        self.dummy = self.res = TreeNode()
        
        def helper(root):
            if not root: return
            helper(root.left)
            self.dummy.right = TreeNode(root.val)
            self.dummy = self.dummy.right
            helper(root.right)
        
        helper(root)
        
        return self.res.right
```



### 919. Complete Binary Tree Inserter

```python
class CBTInserter:

    def __init__(self, root: TreeNode):
        self.q = collections.deque()
        self.root = root
        temp_q = collections.deque([root])
        while temp_q:
            node = temp_q.popleft()
            if not node.left or not node.right:
                self.q.append(node)
            if node.left:
                temp_q.append(node.left)
            if node.right:
                temp_q.append(node.right)

    def insert(self, v: int) -> int:
        node = self.q[0]
        self.q.append(TreeNode(v))
        if not node.left:
            node.left = self.q[-1]
        else:
            node.right = self.q[-1]
            self.q.popleft()
        return node.val
        

    def get_root(self) -> TreeNode:
        return self.root
```

树的bfs



### 938. Range Sum of BST

```python
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        res = 0
        
        def helper(root):
            nonlocal res
            if not root: return
            if low <= root.val <= high:
                res += root.val
            helper(root.left)
            helper(root.right)
        
        helper(root)
        return res
```

```python
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root: return 0
        elif root.val < low:
            return self.rangeSumBST(root.right, low, high)
        elif root.val > high:
            return self.rangeSumBST(root.left, low, high)
        return root.val + self.rangeSumBST(root.left, low, high) + self.rangeSumBST(root.right, low, high)
```

```python
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root: return 0

        def helper(root):
            if not root: return
            nonlocal res
            if root.val >= low and root.val <= high:
                res += root.val
                helper(root.left)
                helper(root.right)
            # 不能加 root.right.val >= low 这个条件，因为 root.right.right 可能会符合要求的，比如下面图片中的树，上下限是[7,15]，5和6是小于7的，但6的右节点是符合条件的，如果加上下面的if语句就会把7过滤掉
            # elif root.val < low and root.right and root.right.val >= low:
            elif root.val < low:
                helper(root.right)
            # elif root.val > high and root.left and root.left.val <= high:
            elif root.val > high:
                helper(root.left)
        
        res = 0
        helper(root)

        return res
```

<img src="./截屏2022-12-28 22.51.50.png" alt="截屏2022-12-28 22.51.50" style="zoom:50%;" />

### 951. Flip Equivalent Binary Trees

```python
class Solution:
    def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool:
        if root1 is root2:
            return True
        if not root1 or not root2 or root1.val != root2.val:
            return False
        return self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right) or self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left)
```



### 958. Check Completeness of a Binary Tree

```python
# BFS
class Solution:
    def isCompleteTree(self, root: TreeNode) -> bool:
        q = collections.deque([root])
        while q[0]:
            temp = q.popleft()
            q.append(temp.left)
            q.append(temp.right)
        while q and q[0] is None:
            q.popleft()
        return len(q) == 0
```

```python
# level
class Solution:
    def isCompleteTree(self, root: TreeNode) -> bool:
        q = collections.deque([root])
        
        aboveNone = False
        while q:
            nowNone = False
            for i in range(len(q)):
                temp = q.popleft()
                if temp:
                    if nowNone or aboveNone:
                        return False
                    q.append(temp.left)
                    q.append(temp.right)
                else:
                    nowNone = True
            aboveNone = nowNone
        return True
```



### 988. Smallest String Starting From Leaf

```python
class Solution:
    def smallestFromLeaf(self, root: TreeNode) -> str:
        temp_dict = {i: chr(i+97) for i in range(26)}
        if not root.left and not root.right:
            return temp_dict[root.val]
        
        temp_dict = {i: chr(i+97) for i in range(26)}
        
        ans = []
        
        def dfs(root, cur):
            if not root.left and not root.right:
                ans.append(temp_dict[root.val]+cur)
                return
            if root.left:
                dfs(root.left, temp_dict[root.val]+cur)
            if root.right:
                dfs(root.right, temp_dict[root.val]+cur)
        
        dfs(root, "")
        ans.sort()
        return ans[0]
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



### Construct Binary Search Tree from Preorder Traversal

```python
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        
        self.index = 0
        def helper(upper):
            if self.index >= len(preorder) or preorder[self.index] > upper:
                return None
            root = TreeNode(preorder[self.index])
            self.index += 1
            root.left = helper(root.val)
            root.right = helper(upper)
            return root
        return helper(sys.maxsize)
```
给定一个上限，判断下一个node是否小于当前上限，小于的话作为当前节点。



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



### 1026. Maximum Difference Between Node and Ancestor

```python
class Solution:
    def maxAncestorDiff(self, root: TreeNode) -> int:
        if not root:
            return
        
        def dfs(root, mx, mn):
            if not root:
                return mx - mn
            mx = max(root.val, mx)
            mn = min(root.val, mn)
            return max(dfs(root.left, mx, mn), dfs(root.right, mx, mn))
        return dfs(root, root.val, root.val)
```

```python
class Solution:
    def maxAncestorDiff(self, root: TreeNode) -> int:
        if not root:
            return
        self.ans = 0
        def dfs(root, mx, mn):
            mx = max(root.val, mx)
            mn = min(root.val, mn)
            if root.left:
                dfs(root.left, mx, mn)
            if root.right:
                dfs(root.right, mx, mn)
            if not root.left and not root.right:
                self.ans = max(self.ans, mx-mn)
        dfs(root, root.val, root.val)
        return self.ans
```

```python
# 上面两个解法类似，都是从顶至下遍历树的时候计算最大最小值，然后在叶子结点处计算最大差值
# 此方法则是，计算最大最小值与最大差值均是在遍历树回溯的时候进行，也就是从低到上
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:

        res = 0

        def helper(root):
            if not root: return 0, 10**5
            l_max, l_min = helper(root.left)
            r_max, r_min = helper(root.right)
            temp_max = max(root.val, l_max, r_max)
            temp_min = min(root.val, l_min, r_min)
            nonlocal res
            res = max(res, abs(root.val - temp_max), abs(root.val - temp_min))
            return temp_max, temp_min

        _max, _min = helper(root)

        return res
```

```python
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        
        res = 0

        def helper(root):
            if not root: return sys.maxsize, -sys.maxsize
            l_min, l_max = helper(root.left)
            r_min, r_max = helper(root.right)

            nonlocal res
            if root.left:
                res = max(res, abs(root.val - l_min), abs(root.val - l_max))
            if root.right:
                res = max(res, abs(root.val - r_min), abs(root.val - r_max))

            cur_min = min(root.val, l_min, r_min)
            cur_max = max(root.val, l_max, r_max)

            return cur_min, cur_max
        
        helper(root)
        return res
```



### 1038. Binary Search Tree to Greater Sum Tree

```python
class Solution:
    def __init__(self):
        self.temp = 0
        
    def bstToGst(self, root: TreeNode) -> TreeNode:
        if not root:
            return
        self.bstToGst(root.right)
        self.temp += root.val
        root.val = self.temp
        self.bstToGst(root.left)
        return root
```

is the same as 538



### 1080. Insufficient Nodes in Root to Leaf Paths

```python
class Solution:
    def sufficientSubset(self, root: TreeNode, limit: int) -> TreeNode:
        if not root:
            return None
        if not root.left and not root.right:
            return None if root.val < limit else root
        
        if root.left:
            root.left = self.sufficientSubset(root.left, limit - root.val)
        if root.right:
            root.right = self.sufficientSubset(root.right, limit - root.val)
        
        return root if root.left or root.right else None
```



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



### 1123. Lowest Common Ancestor of Deepest Leaves

```python
class Solution:
    def lcaDeepestLeaves(self, root: TreeNode) -> TreeNode:
        
        def helper(root):
            if not root:
                return 0, None
            l_height, l_lca = helper(root.left)
            r_height, r_lca = helper(root.right)
            if l_height > r_height: return l_height+1, l_lca
            elif l_height < r_height: return r_height+1, r_lca
            else:
                return l_height+1, root
        return helper(root)[1]
```



### 1130. Minimum Cost Tree From Leaf Values

```python
class Solution:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        ans = 0
        while len(arr) > 1:
            i = arr.index(min(arr))
            ans += min(arr[i-1:i] + arr[i+1:i+2]) * arr.pop(i)
        return ans
```

```python
class Solution:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        stack = [sys.maxsize]
        ans = 0
        for each in arr:
            while stack[-1] <= each:
                mid = stack.pop()
                ans += min(stack[-1], each) * mid
            stack.append(each)

        while len(stack) > 2:
            ans += stack.pop() * stack[-1]
        return ans
```
https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/discuss/339959/One-Pass-O(N)-Time-and-Space



### 1161. Maximum Level Sum of a Binary Tree

```python
class Solution:
    def maxLevelSum(self, root: TreeNode) -> int:
        sum_list = [root.val]
        stack = [root]
        
        while stack:
            temp_sum = 0
            next_node_list = []
            for node in stack:
                if node.left:
                    temp_sum += node.left.val
                    next_node_list.append(node.left)
                if node.right:
                    temp_sum += node.right.val
                    next_node_list.append(node.right)
            if next_node_list:
                stack = next_node_list
                sum_list.append(temp_sum)
            else:
                break
        for i in range(len(sum_list)):
            if sum_list[i] == max(sum_list):
                return i+1
```

```python
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:

        level_nodes = [root]
        level_val = [root.val]

        while level_nodes:
            temp = []
            for each in level_nodes:
                if each.left:
                    temp.append(each.left)
                if each.right:
                    temp.append(each.right)
            level_nodes = temp
            if temp: # 这里用reduce求和会有问题
                val = 0
                for each in temp:
                    val += each.val
                level_val.append(val)
        
        max_val = max(level_val)
        for i, val in enumerate(level_val):
            if val == max_val:
                return i + 1
```



### 1261. Find Elements in a Contaminated Binary Tree

```python
class FindElements:

    def __init__(self, root: TreeNode):
        self.val = {0}
        root.val = 0
        self.recover(root)
    
    def recover(self, root: TreeNode):
        if not root:
            return
        self.val.add(root.val)
        if root.left:
            root.left.val = 2 * root.val + 1
            self.recover(root.left)
        if root.right:
            root.right.val = 2 * root.val + 2
            self.recover(root.right)
    def find(self, target: int) -> bool:
        return target in self.val
```



### 1302. Deepest Leaves Sum

```python
class Solution:
    def deepestLeavesSum(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        stack = [root]
        ans = 0
        while stack:
            temp_list = []
            temp_sum = 0
            for each in stack:
                temp_sum += each.val
                if each.left:
                    temp_list.append(each.left)
                if each.right:
                    temp_list.append(each.right)
            stack = temp_list
            ans = temp_sum
        return ans
```



### 1305. All Elements in Two Binary Search Trees

```python
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        
        stack1 = []
        stack2 = []
        ans = []
        while stack1 or stack2 or root1 or root2:
            while root1:
                stack1.append(root1)
                root1 = root1.left
            while root2:
                stack2.append(root2)
                root2 = root2.left
                
            if not stack2 or (stack1 and stack1[-1].val <= stack2[-1].val):
                ans.append(stack1[-1].val)
                root1 = stack1.pop()
                root1 = root1.right
            else:
                ans.append(stack2[-1].val)
                root2 = stack2.pop()
                root2 = root2.right
        return ans
```



### 1315. Sum of Nodes with Even-Valued Grandparent

```python
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        if not root:
            return 0
        self.ans = 0
        
        def helper(root):
            if root.left:
                if root.val % 2 == 0:
                    self.ans += root.left.left.val if root.left.left else 0
                    self.ans += root.left.right.val if root.left.right else 0
                helper(root.left)
            if root.right:
                if root.val % 2 == 0:
                    self.ans += root.right.left.val if root.right.left else 0
                    self.ans += root.right.right.val if root.right.right else 0
                helper(root.right)
        helper(root)
        return self.ans
```

```python
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        def helper(root, parent_val, gradparent_val):
            if not root:
                return 0 
            temp = root.val if gradparent_val % 2 == 0 else 0
            return temp + helper(root.left, root.val, parent_val) + helper(root.right, root.val, parent_val)
        return helper(root, 1, 1)
```


### 1325. Delete Leaves With a Given Value

```python
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        if root.left:
            root.left = self.removeLeafNodes(root.left, target)
        if root.right:
            root.right = self.removeLeafNodes(root.right, target)
        return None if not root.left and not root.right and root.val == target else root
```

熟记，回溯删除节点



### 1361. Validate Binary Tree Nodes

```py
class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        '''
        CHECK CONDITION 1 : There must be EXACTLY (n - 1) edges 
        '''
        # COUNT THE TOTAL EDGES - O(N)
        leftEdges = len([l for l in leftChild if l != -1])
        rightEdges = len([r for r in rightChild if r != -1])

        # CHECK FOR VIOLATION OF CONDITION 1 - O(1)
        if leftEdges + rightEdges != n - 1:
            return False
        
        '''
        CHECK CONDITION 2 : Each node, except the root, has EXACTLY 1 parent
        '''
        # GET THE PARENT OF EACH NODE - O(N)
        parent = [[] for _ in range(n)]
        
        # FILL THE PARENT - O(N)
        for i in range(n):
            if leftChild[i] != -1: parent[leftChild[i]].append(i)                
            if rightChild[i] != -1: parent[rightChild[i]].append(i)  
        
        # 4
        # [1,0,3,-1]
        # [-1,-1,-1,-1]
        for i in range(n):
            if parent[i] and parent[parent[i][0]]==[i]:
                return False
                
        # FIND ALL ROOT NODES (IE. THOSE WITHOUT PARENT) - O(N)
        roots = [i for i in range(len(parent)) if not parent[i]]

        # CHECK IF THERE'S EXACTLY 1 ROOT NODE  - O(1)
        if len(roots) != 1:
            return False
        
        # ENSURE ROOT HAS > 1 CHILD, IF N > 1 - O(N)
        root = roots[0]
        return max(leftChild[root], rightChild[root]) != -1 or n == 1
```



### 1367. Linked List in Binary Tree

```python
class Solution:
    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        if not head or not root:
            return False
        
        def isSame(root, head):
            if not root and head:
                return False
            if not head:
                return True
            if root.val != head.val:
                return False
            
            left = isSame(root.left, head.next)
            right = isSame(root.right, head.next)
            return left or right
        
        def helper(root):
            if not root:
                return
            if root.val == head.val:
                if isSame(root, head):
                    return True
            return helper(root.left) or helper(root.right)
        
        return True if helper(root) else False
```

和 572 Subtree of Another Tree 类似

https://leetcode.com/problems/linked-list-in-binary-tree/discuss/684678/Java-DFS



### 1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree

```python
class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        
        self.res = None
        
        def helper(ori, clo):
            if ori == target:
                self.res = clo
                return
            if ori.left:
                helper(ori.left, clo.left)
            if ori.right:
                helper(ori.right, clo.right)
        helper(original, cloned)
        return self.res
```

```python
class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        
        def pre_order(o, c):
            if not o: return
            if o is target:
                self.ans = c
            pre_order(o.left, c.left)
            pre_order(o.right, c.right)
        
        pre_order(original, cloned)
        return self.ans
```



### 1443. Minimum Time to Collect All Apples in a Tree

```python
class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        from collections import defaultdict
        graph = defaultdict(list)
        for i, j in edges:
            graph[i].append(j)
            graph[j].append(i)
        
        seen = set()
        def dfs(node):
            if node in seen:
                return 0
            
            seen.add(node)
            temp_ans = 0
            for child in graph[node]:
                temp_ans += dfs(child)
            if temp_ans > 0:
                return temp_ans + 2
            return 2 if hasApple[node] else 0
        
        return max(dfs(0)-2, 0)
```

dfs回溯时计算



### 1448. Count Good Nodes in Binary Tree

```python
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        self.ans = 0
        def helper(root, max_now):
            if not root:
                return 
            if root.val >= max_now:
                self.ans += 1
            helper(root.left, max(max_now, root.val))
            helper(root.right, max(max_now, root.val))
        helper(root, -sys.maxsize)
        return self.ans
```
记录每个节点的父节点和父节点的父节点



### 1457. Pseudo-Palindromic Paths in a Binary Tree

```python
class Solution:
    def pseudoPalindromicPaths (self, root: TreeNode) -> int:
        
        self.ans = 0
        def helper(root, node_set):
            if not root:
                return
            if root.val in node_set:
                node_set.remove(root.val)
            else:
                node_set.add(root.val)
            if not root.left and not root.right:
                if len(node_set) <= 1:
                    self.ans += 1
            helper(root.left, node_set.copy())
            helper(root.right, node_set.copy())
        helper(root, set())
        return self.ans
```



### 1466. Reorder Routes to Make All Paths Lead to the City Zero

```python
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        g = collections.defaultdict(list)
        g_r = collections.defaultdict(list)
        
        for a, b in connections:
            g[a].append(b)
            g_r[b].append(a)
        
        ans = 0
        seen = {0}
        q = collections.deque([0])
        while q:
            node = q.popleft()
            for each in g[node]:
                if each not in seen:
                    seen.add(each)
                    ans += 1
                    q.append(each)
            for each in g_r[node]:
                if each not in seen:
                    seen.add(each)
                    q.append(each)
        return ans
```



### 1519. Number of Nodes in the Sub-Tree With the Same Label

```python
class Solution:
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:

        graph = defaultdict(list)

        for f, t in edges:
            graph[f].append(t)
            graph[t].append(f)
        
        res_list, seen = [None] * n, set()

        def merge(dict1, dict2):
            for k, v in dict2.items():
                if k not in dict1:
                    dict1[k] = v
                else:
                    dict1[k]+= v

        def dfs(node):
            seen.add(node)
            has_not_seen_nei = False
            temp_dict = {}
            for each in graph[node]:
                if each not in seen:
                    has_not_seen_nei = True
                    merge(temp_dict, dfs(each))
            if not has_not_seen_nei:
                res_list[node] = 1
                return {labels[node]: 1}
            else:
                merge(temp_dict, {labels[node]: 1})
                res_list[node] = temp_dict[labels[node]]
                return temp_dict
        
        dfs(0)
        return res_list
```

本质是树的题，不过题目中用图的方式给出，所以只能按图的形式构造。本质思想还是dfs遍历树至叶子结点，在从叶子结点向上回溯的时候填充最后答案。由于我们是用图的结果存储的，所以不好直接判断是否是叶子结点，所以只能在遍历相邻结点之前增加标签（has_not_seen_nei），如果相邻结点均遇见过（has_not_seen_nei is False）那么该结点就是叶子结点。

如果是叶子结点那么直接填写结果数组 res_list[node] = 1，以字典形式返回相应label数量 {labels[node]: 1}。

从叶子结点网上走，非叶子结点的话，叶子结点的结果会在地26行返回，返回之后会将所有子节点的结果merge在临时字典（temp_dict）中，之后将当前节点的label数量merge进临时字典中，之后填充当前节点到结果数组中并返回临时字典作为当前节点的结果，以上步骤对应31-33行。



### 2359. Find Closest Node to Given Two Nodes

```python
class Solution:
    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
        
        graph = defaultdict(list)
        for i, val in enumerate(edges):
             if val != -1:
                graph[i].append(val)
        
        dis_from_node1 = {}
        dis_from_node2 = {}

        def dfs(node, dist, record):
            if node not in seen:
                seen.add(node)
                record[node] = dist
                if node in graph:
                    for each in graph[node]:
                        dfs(each, dist+1, record)
                    
        seen = set()
        dfs(node1, 0, dis_from_node1)
        seen = set()
        dfs(node2, 0, dis_from_node2)

        res, max_dis = -1, sys.maxsize
        for i in range(len(edges)):
						# check if the end node is reachable from both starting nodes
            if i in dis_from_node1 and i in dis_from_node2:
                temp_max_dis = max(dis_from_node1[i], dis_from_node2[i])
								# update the distance and the final answer if relevant
                if temp_max_dis < max_dis:
                    max_dis = temp_max_dis
                    res = i
        return res
```

描述很迷惑，其实是求到达node1和node2的最大距离的最小值的那个点，min(max(dis_to_node1, dis_to_node2))

### 判断二叉查找树

```python
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def maxDepth_tool(self, root):
        if not root:
            return 0
        if not root.left and not root.right:
            return 0

        lval = 0
        rval = 0
        if root.left:
            lval = self.maxDepth_tool(root.left) + 1
        if root.right:
            rval = self.maxDepth_tool(root.right) + 1
        return max(lval, rval)

    def maxDepth(self, root):
        print("###")
        a = self.maxDepth_tool(root)
        return a+1


def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root


def intToString(input):
    if input is None:
        input = 0
    return str(input)


def main():
   # line = lines.next()
    line = '[3,9,20,null,null,15,7]'
    root = stringToTreeNode(line)

    ret = Solution().maxDepth(root)

    out = intToString(ret)
    print(out)


if __name__ == '__main__':
    main()
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

687: 给定二叉树，找最大路径，路径上所有元素值相同
