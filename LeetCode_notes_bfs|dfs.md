## Leetcode - BFS | DFS

[toc]

### 1298. Maximum Candies You Can Get from Boxes
```python
class Solution:
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        opened, closed, res, not_used_keys = collections.deque(), set(), 0, set()
        for each in initialBoxes:
            if status[each]:
                opened.append(each)
            else:
                closed.add(each)
        key = [False] * len(status)

        while opened:
            cur_box = opened.popleft()
            
            # 1. add candies
            res += candies[cur_box]

            # 2. deal with keys
            for contain_key in keys[cur_box]:
                key[contain_key] = True

            # 3. deal with contained boxes
            for contain_box in containedBoxes[cur_box]:
                if status[contain_box] or key[contain_box]:
                    opened.append(contain_box)
                else:
                    closed.add(contain_box)
            
            # 4. deal with closed box
            delete_from_close = set()
            for close_box in closed:
                if key[close_box]:
                    delete_from_close.add(close_box)
                    opened.append(close_box)
            closed -= delete_from_close

        return res        

```
BFS
