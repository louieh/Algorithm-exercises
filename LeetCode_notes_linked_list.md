## LeetCode - Linked List

[toc]

### 21. Merge Two Sorted Lists

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeFunction(self, l1, l2, temp):
        if l1 and not l2:
            temp.next = l1
            return
        elif l2 and not l1:
            temp.next = l2
            return
        else:
            if l1.val <= l2.val:
                temp.next = l1
                self.mergeFunction(l1.next, l2, temp.next)
            else:
                temp.next = l2
                self.mergeFunction(l1, l2.next, temp.next)
    
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 and not l2:
            return
        
        ans = ListNode(0)
        temp = ans
        self.mergeFunction(l1, l2, temp)
        
        return ans.next
```

```python
# 10/19/2019
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 and not l2:
            return
        if not l1 or not l2:
            return l1 or l2
        
        dummy = head = ListNode(0)
        
        while l1 and l2:
            if l1.val <= l2.val:
                dummy.next = ListNode(l1.val)
                l1 = l1.next
            else:
                dummy.next = ListNode(l2.val)
                l2 = l2.next
            dummy = dummy.next

        dummy.next = l1 or l2
        return head.next
```

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        def mergeTwoLists_helper(head, l1, l2):
            if l1 and l2:
                if l1.val <= l2.val:
                    head.next = ListNode(l1.val)
                    mergeTwoLists_helper(head.next, l1.next, l2)
                else:
                    head.next = ListNode(l2.val)
                    mergeTwoLists_helper(head.next, l1, l2.next)
            else:
                head.next = l1 or l2
                return
        head = ListNode(0)
        temp = head
        mergeTwoLists_helper(temp, l1, l2)
        return head.next
```

```python
# 02/22/2020
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return l1 or l2
        
        head = ListNode(0)
        res = head
        while l1 and l2:
            if l1.val <= l2.val:
                head.next = l1
                l1 = l1.next
            else:
                head.next = l2
                l2 = l2.next
            head = head.next
        head.next = l1 or l2
        return res.next
```



### 23. Merge k Sorted Lists

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    import sys
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return
        
        dummy = ListNode(0)
        ans = dummy
        
        while 1:
            mark = 0
            min_index = None
            min_value = sys.maxsize
            
            for i in range(len(lists)):
                if lists[i]:
                    mark += 1
                    if lists[i].val < min_value:
                        min_value = lists[i].val
                        min_index = i
            if mark == 0:
                break
            dummy.next = ListNode(min_value)
            dummy = dummy.next
            lists[min_index] = lists[min_index].next
            
        return ans.next
```



### 24. Swap Nodes in Pairs

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head: return
        
        def swapPaires_helper(node):
            if not node or not node.next: return
            node.val, node.next.val = node.next.val, node.val
            swapPaires_helper(node.next.next)
        
        node = head
        swapPaires_helper(node)
        return head
```

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        
        def helper(node):
            ress = node.next
            temp = node.next.next
            node.next.next = node
            node.next = temp
            return ress
        
        res = old_head = ListNode(0)
        while head and head.next:
            old = helper(head)
            old_head.next = old
            old_head = head
            head = head.next
        return res.next
```



### 141.Linked List Cycle

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return False
        
        slow = head
        fast = head.next
        
        while slow != fast:
            if not fast or not fast.next:
                return False
            
            slow = slow.next
            fast = fast.next.next
            
        return True
```



### 142. Linked List Cycle II

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                start = head
                while slow != start:
                    slow = slow.next
                    start = start.next
                return slow
```



### 138. Copy List with Random Pointer

```python
# 不太懂
"""
# Definition for a Node.
class Node:
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return
        
        head_dict = dict()
        
        def copyRandomList_tool(head):
            if not head:
                return
            
            if head in head_dict.keys():
                return head_dict[head]
            
            cp_head = Node(head.val, None, None)
            head_dict[head] = cp_head
            
            cp_head.random = copyRandomList_tool(head.random)
            cp_head.next = copyRandomList_tool(head.next)
            
            return cp_head
            
        return copyRandomList_tool(head)
```



### 203. Remove Linked List Elements

```python
class Solution:
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        if not head:
            return
        while head.val == val:
            head = head.next
            if not head:
                return head
        tempNode = head
        while tempNode.next:
            if tempNode.next.val == val:
                if tempNode.next.next:
                    tempNode.next = tempNode.next.next
                    continue
                else:
                    tempNode.next = None
            tempNode = tempNode.next
            if not tempNode:
                return head
        return head
```

```python
# 02/15/2020
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        if not head:
            return
        
        temp_head = head
        while head and head.val == val:
            head = head.next
            temp_head = head
        
        while head:
            if head.next and head.next.val == val:
                head.next = head.next.next
            else:
                head = head.next
        return temp_head
```



### 206. Reverse Linked List

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null){
            return null;
        }
      // 1->2->3->4->5->NULL
      // 一共三个标记，node, head, nxt
      // 第一次循环node为空
      // 之后三个标记变量依次往后挪，最后 node 为反转后的链表头
        ListNode node = null;
        while (head != null){
            ListNode nxt = head.next;
            head.next = node;
            node = head;
            head = nxt;
        }
        return node;
    }
}
```

```python
# 10/27/2019
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        
        old_head = None
        while head:
            new_head = ListNode(head.val)
            new_head.next = old_head
            old_head = new_head
            head = head.next
        
        return new_head
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        
        ans_head = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return ans_head
```

```python
# 02/15/2020
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head:
            return 
        
        temp_head = head
        while head.next:
            temp = head.next
            head.next = head.next.next
            temp.next = temp_head
            temp_head = temp
        return temp_head
```



### 234. Palindrome Linked List

```python
def isPalindrome(self, head: ListNode) -> bool:
    if not head:    # 0 node
        return True
    if not head.next: # 1 node
        return True
		# 找中点，slow 为中点
    fast = slow = head 
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
		
    # 反转从 slow 开始后面的链表
    node = None
    while slow:
        nxt = slow.next
        slow.next = node
        node = slow
        slow = nxt
		
    # 用 slow 前的部分和 slow 后已经反转的部分比较，相等则是回文
    while head and node:
        if head.val != node.val:
            return False
        head = head.next
        node = node.next
    return True
```

```python
# 02/15/2020
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if not head or not head.next:
            return True
        
        # find the middle node
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # reverse the linked list from slow node
        temp = None
        while slow:
            nxt = slow.next
            slow.next = temp
            temp = slow
            slow = nxt
        
        # compare
        while head and temp:
            if head.val != temp.val:
                return False
            head = head.next
            temp = temp.next
        return True
```



### 328. Odd Even Linked List

```python
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next or not head.next.next:
            return head
        
        wait_to_add = None
        temp_head = head
        while head.next:
            if not wait_to_add:
                wait_to_add = head.next
                temp_wait_to_add = wait_to_add
            else:
                wait_to_add.next = head.next
                wait_to_add = wait_to_add.next
            head.next = head.next.next
            if head.next:
                head = head.next
            
        wait_to_add.next = None
        head.next = temp_wait_to_add
        return temp_head
```

```python
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return
        
        odd = head
        even = head.next
        even_head = even
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = even_head
        return head
```



### 430. Flatten a Multilevel Doubly Linked List

```python
class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        if not head:
            return
        
        res = temp = Node(0)
        
        def helper(head):
            nonlocal temp
            new_node = Node(head.val)
            new_node.prev = temp
            temp.next = new_node
            temp = temp.next
            if head.child:
                helper(head.child)
            if head.next:
                helper(head.next)
        
        helper(head)
        res = res.next
        res.prev = None // 把 prve 置空，否则会报is not a valid doubly linked list
        return res
```

```python
class Solution:
    def flatten(self, head: 'Node') -> 'Node':
        if not head:
            return
        
        res = temp = Node(0)
        
        def helper(head):
            if not head:
                return 
            nonlocal temp
            new_node = Node(head.val)
            new_node.prev = temp
            temp.next = new_node
            temp = temp.next
            helper(head.child)
            helper(head.next) // 少用 if 会比较快
        
        helper(head)
        res = res.next
        res.prev = None
        return res
```

```python
class Solution(object):
    def flatten(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        def flattenR(head, parent):
            
            
            if not head:
                return None
            
            head.prev = parent
            curr = head
            prev = None
            
            while curr:
                if curr.child:
                    end = flattenR(curr.child, curr)
                
                    temp = curr.next
                    curr.next = curr.child
                    curr.child = None
                    end.next = temp
                    if temp:
                        temp.prev = end
                    
                    prev = end
                    curr = temp
                else:
                    prev = curr
                    curr = curr.next
                    
            return prev
        
        
        flattenR(head, None)
        return head
```



### 707. Design Linked List

```python
class MyLinkedList:

    class Node(object):
        def __init__(self, value):
            self.val = value
            self.next = None
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.head = None
        

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if not self.head:
            return -1
        if index == 0:
            return self.head.val
        
        temp = self.head
        for i in range(index):
            if temp.next:
                temp = temp.next
            else:
                return -1
        return temp.val
        

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        newNode = self.Node(val)
        if not self.head:
            self.head = newNode
        else:
            newNode.next = self.head
            self.head = newNode
        # print("addAtHead: ")
        # self.printLinkedList()
        

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        if not self.head:
            self.head = self.Node(val)
        else:
            temp = self.head
            while temp.next:
                temp = temp.next
            newNode = self.Node(val)
            temp.next = newNode
        # print("addAtTail: ")
        # self.printLinkedList()

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index == 0:
            self.addAtHead(val)
        else:
            temp = self.head
            for i in range(index-1):
                if temp.next:
                    temp = temp.next
                else:
                    return
            newNode = self.Node(val)
            newNode.next = temp.next
            temp.next = newNode
        # print("addAtIndex: ")
        # self.printLinkedList()
    
    def printLinkedList(self):
        if self.head:
            temp = self.head
            while temp:
                print("{0}->".format(temp.val), end="")
                temp = temp.next
            print()
        else:
            print("None")
        

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if not self.head:
            return
        if index == 0:
            self.head = self.head.next
        else:
            temp = self.head
            for i in range(index-1):
                if temp.next:
                    temp = temp.next
                else:
                    return
            if temp.next:
                temp.next = temp.next.next
        # print("deleteAtIndex: ")
        # self.printLinkedList()


# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)
```

