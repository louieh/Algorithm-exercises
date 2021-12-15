## LeetCode - Linked List

[toc]

### 19. Remove Nth Node From End of List

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        slow = fast = head
        for i in range(n):
            fast = fast.next
        if not fast: return head.next # 删除第一个点
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head
```

设置两个点，第一个点先向前走n步，此时两个点步数间隔n，之后两个点同时先前走



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



### 25. Reverse Nodes in k-Group

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        
        def reverse_helper(head):
            count = k - 1
            tempHead = head
            while count > 0:
                nextHead = head.next
                head.next = head.next.next
                nextHead.next = tempHead
                tempHead = nextHead
                count -= 1
            return tempHead, head, head.next
        
        def helper(head):
            count = 0
            dmy = head
            while dmy and count != k:
                dmy = dmy.next
                count += 1
            if count != k: return head
            tempHead, prev, curHead = reverse_helper(head)
            prev.next = helper(curHead)
            return tempHead
        
        return helper(head)
```



### 82. Remove Duplicates from Sorted List II

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head: return
        dummy = ListNode(0, head)
        prev = dummy
        while head:
            if head.next and head.val == head.next.val:
                while head.next and head.val == head.next.val:
                    head = head.next
                prev.next = head.next
            else:
                prev = prev.next
            head = head.next
        return dummy.next
```



### 83. Remove Duplicates from Sorted List

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head: return
        temp = head
        while temp.next:
            if temp.val == temp.next.val:
                temp.next = temp.next.next
                continue
            temp = temp.next
        return head
```



### 86. Partition List

```python
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        if not head:
            return 
        
        before = before_head = ListNode(0)
        after  = after_head = ListNode(0)
        
        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            
            head = head.next
        after.next = None
        before.next = after_head.next
        return before_head.next
```



### 92. Reverse Linked List II

```python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if not head or m == n:
            return head
        
        ans = head
        for i in range(m-1):
            final_temp_head = head
            head = head.next
        temp_head = head
        for i in range(n-m):
            temp_next = head.next
            head.next = head.next.next
            temp_next.next = temp_head
            temp_head = temp_next
        if m != 1:
            final_temp_head.next = temp_head
            return ans
        else:
            return temp_head
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



### 143. Reorder List

```python
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return
        
        p1, p2 = head, head
        
        while p2.next and p2.next.next:
            p1 = p1.next
            p2 = p2.next.next
        
        preMiddle = p1
        preCurrent = p1.next
        temp_head = p1.next
        while preCurrent.next:
            temp_next = preCurrent.next
            preCurrent.next = preCurrent.next.next
            temp_next.next = temp_head
            temp_head = temp_next
        preMiddle.next = temp_head
        
        p1, p2 = head, preMiddle.next
        while p1 != preMiddle:
            preMiddle.next = p2.next
            p2.next = p1.next
            p1.next = p2
            p1 = p2.next
            p2 = preMiddle.next
```

https://leetcode.com/problems/reorder-list/discuss/44992/Java-solution-with-3-steps

![IMG_3359](/Users/hanluyi/Downloads/other_Python_ex/leetcode/IMG_3359.jpeg)



### 147. Insertion Sort List

```python
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        res = ListNode()
        
        while head:
            prev_node = res
            next_node = prev_node.next
            while next_node:
                if head.val < next_node.val:
                    break
                prev_node = prev_node.next
                next_node = prev_node.next

            new_head = head.next
            prev_node.next = head
            head.next = next_node

            head = new_head
        return res.next
```

```python
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        res = ListNode()
        
        while head:
            res_cpy = res
            while res_cpy.next:
                if head.val < res_cpy.next.val:
                    break
                res_cpy = res_cpy.next

            new_head = head.next
            
            head.next = res_cpy.next
            res_cpy.next = head

            head = new_head
        return res.next
```

```python
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head: return

        dummyHead = ListNode()
        dummyHead.next = head
        lastSorted = head
        curr = head.next

        while curr:
            if lastSorted.val <= curr.val:
                curr = curr.next
                lastSorted = lastSorted.next
            else:
                prev = dummyHead
                while prev.next.val < curr.val:
                    prev = prev.next
                lastSorted.next = curr.next
                curr.next = prev.next
                prev.next = curr
                curr = lastSorted.next
        
        return dummyHead.next
```

插入排序的基本思想是，维护一个有序序列，初始时有序序列只有一个元素，每次将一个新的元素插入到有序序列中，将有序序列的长度增加 11，直到全部元素都加入到有序序列中。

如果是数组的插入排序，则数组的前面部分是有序序列，每次找到有序序列后面的第一个元素（待插入元素）的插入位置，将有序序列中的插入位置后面的元素都往后移动一位，然后将待插入元素置于插入位置。

对于链表而言，插入元素时只要更新相邻节点的指针即可，不需要像数组一样将插入位置后面的元素往后移动，因此插入操作的时间复杂度是 O(1)O(1)，但是找到插入位置需要遍历链表中的节点，时间复杂度是 O(n)O(n)，因此链表插入排序的总时间复杂度仍然是 O(n^2)O(n2)，其中 nn 是链表的长度。

对于单向链表而言，只有指向后一个节点的指针，因此需要从链表的头节点开始往后遍历链表中的节点，寻找插入位置。

对链表进行插入排序的具体过程如下。

1. 首先判断给定的链表是否为空，若为空，则不需要进行排序，直接返回。

2. 创建哑节点 dummyHead，令 dummyHead.next = head。引入哑节点是为了便于在 head 节点之前插入节点。

3. 维护 lastSorted 为链表的已排序部分的最后一个节点，初始时 lastSorted = head。

4. 维护 curr 为待插入的元素，初始时 curr = head.next。

5. 比较 lastSorted 和 curr 的节点值。

   * 若 lastSorted.val <= curr.val，说明 curr 应该位于 lastSorted 之后，将 lastSorted 后移一位，curr 变成新的 lastSorted。

   * 否则，从链表的头节点开始往后遍历链表中的节点，寻找插入 curr 的位置。令 prev 为插入 curr 的位置的前一个节点，进行如下操作，完成对 curr 的插入：

     ```python
     lastSorted.next = curr.next
     curr.next = prev.next
     prev.next = curr
     ```

6. 令 curr = lastSorted.next，此时 curr 为下一个待插入的元素。

7. 重复第 5 步和第 6 步，直到 curr 变成空，排序结束。

8. 返回 dummyHead.next，为排序后的链表的头节点。

   

### 148. Sort List

 ```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        def mergeSort(head):
            if not head or not head.next: return head
            mid = findMid(head)
            left = mergeSort(head)
            right = mergeSort(mid)
            return merge(left, right)
        
        def findMid(head):
            slow = fast = head
            mid = None
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            mid = slow.next
            slow.next = None
            return mid
        
        def merge(l1, l2):
            dummyhead = ListNode()
            tail = dummyhead
            while l1 and l2:
                if l1.val <= l2.val:
                    tail.next = l1
                    l1 = l1.next
                else:
                    tail.next = l2
                    l2 = l2.next
                tail = tail.next
            tail.next = l1 or l2
            return dummyhead.next
        
        return mergeSort(head)
 ```

merge sort

![mergeSort](/Users/hanluyi/Downloads/other_Python_ex/leetcode/mergeSort.png)



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



### 382. Linked List Random Node

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {

    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    int LinkLength;
    HashMap<Integer, Integer> dict;
    Random rand;
    public Solution(ListNode head) {
        int i = 0;
        this.dict = new HashMap<>();
        this.rand = new Random();
        while (head != null) {
            i += 1;
            dict.put(i, head.val);
            head = head.next;
        }
        this.LinkLength = i;
    }
    
    /** Returns a random node's value. */
    public int getRandom() {
        int random = this.rand.nextInt(this.LinkLength) + 1;
        return this.dict.get(random);
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(head);
 * int param_1 = obj.getRandom();
 */
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



### 445. Add Two Numbers II

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return l1 or l2
        
        def com_length(l):
            length = 0
            while l:
                length += 1
                l = l.next
            return length
        
        def add_length(l, num):
            new_header = temp_header = ListNode(0)
            for i in range(num-1):
                temp_node = ListNode(0)
                temp_header.next = temp_node
                temp_header = temp_header.next
            temp_header.next = l
            return new_header
        
        def add(l1, l2):
            if not l1 and not l2:
                return 0
            temp = add(l1.next, l2.next)
            old_l1 = l1.val
            l1.val = (l1.val + l2.val + temp) % 10
            return (old_l1 + l2.val + temp) // 10
        
        l1_length = com_length(l1)
        l2_length = com_length(l2)
        
        if l1_length > l2_length:
            l2 = add_length(l2, l1_length-l2_length)
        elif l1_length < l2_length:
            l1 = add_length(l1, l2_length-l1_length)
        res = add(l1, l2)
        if res != 0:
            ans = ListNode(res)
            ans.next = l1
            return ans
        else:
            return l1
```

```python
# discuss 中的代码，what do you think now?
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
		# post-order generator
        def it(node):
            if node:
                yield from it(node.next)
                yield node.val
        
        ans = ListNode(None)
        carry = 0
        for v1, v2 in itertools.zip_longest(it(l1), it(l2), fillvalue=0):
            v = v1 + v2 + carry
            digit, carry = ListNode(v % 10), v // 10
            digit.next, ans.next = ans.next, digit
        
        if carry > 0:
            digit = ListNode(carry)
            digit.next, ans.next = ans.next, digit
        
        return ans.next
```

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        stack1, stack2 = [], []
        
        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        while l2:
            stack2.append(l2.val)
            l2 = l2.next
        
        head = ListNode(0)
        tempSum = 0
        while stack1 or stack2:
            if stack1:
                tempSum += stack1.pop()
            if stack2:
                tempSum += stack2.pop()
            head.val = tempSum % 10
            newHead = ListNode(tempSum // 10, head)
            head = newHead
            tempSum //= 10
        return head if head.val != 0 else head.next
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



### 708. Insert into a Sorted Circular Linked List

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        if head == None:
            newNode = Node(insertVal, None)
            newNode.next = newNode
            return newNode
 
        prev, curr = head, head.next
        toInsert = False

        while True:
            
            if prev.val <= insertVal <= curr.val:
                # Case #1.
                toInsert = True
            elif prev.val > curr.val:
                # Case #2. where we locate the tail element
                # 'prev' points to the tail, i.e. the largest element!
                if insertVal >= prev.val or insertVal <= curr.val:
                    toInsert = True

            if toInsert:
                prev.next = Node(insertVal, curr)
                # mission accomplished
                return head

            prev, curr = curr, curr.next
            # loop condition
            if prev == head:
                break
        # Case #3.
        # did not insert the node in the loop
        prev.next = Node(insertVal, curr)
        return head
```



### 844. Backspace String Compare

```python
class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        def helper(string):
            ans = []
            for each in string:
                if each != "#":
                    ans.append(each)
                elif ans:
                    ans.pop()
            return ans
        return helper(S) == helper(T)
```

```python
class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        def helper(string):
            skip = 0
            for each in reversed(string):
                if each == "#":
                    skip += 1
                elif skip:
                    skip -= 1
                else:
                    yield each
        return all(x == y for x, y in itertools.zip_longest(helper(S), helper(T)))
```



### 876. Middle of the Linked List

```python
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        if not head:
            return None
        
        slow = fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```



### 1019. Next Greater Node In Linked List

```python
class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        stack = []
        res = []
        def helper(head):
            if head.next:
                helper(head.next)
            while stack and head.val >= stack[-1]:
                stack.pop()
            temp = stack[-1] if stack else 0
            res.insert(0, temp)
            stack.append(head.val)
        helper(head)
        return res
```

Similar as stack 739. Daily Temperatures



### 1290. Convert Binary Number in a Linked List to Integer

```python
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        if not head:
            return 0
        
        index_list = []
        index = 0
        value = False
        while head:
            if head.val == 1:
                value = True
                index_list.append(index)
            head = head.next
            index += 1
        if not value:
            return 0

        ans = 0
        for each in index_list:
            ans += 2 ** (index-1-each)
        return ans
```

