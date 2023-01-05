## LeetCode - Interval

[toc]

### 56. Merge Intervals

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        res = []
        cur_ele = intervals[0]
        nxt_index = 1
        while nxt_index <= len(intervals) - 1:
            if cur_ele[1] < intervals[nxt_index][0]:
                res.append(cur_ele)
                cur_ele = intervals[nxt_index]
            else:
                cur_ele = [min(cur_ele[0], intervals[nxt_index][0]), max(cur_ele[1], intervals[nxt_index][1])]
            nxt_index += 1
        res.append(cur_ele)
        return res
```

try to use divide and conquer

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        
        def merge_fun(intervals, low, high):
            mid = (low + high) // 2
            if low < high:
                merge_fun(intervals, low, mid)
                merge_fun(intervals, mid+1, high)
                merge_method(intervals, low, mid, high)
        
        
        def merge_method(inter1, inter2):
            temp1 = inter1[-1]
            temp2 = inter2[0]
            if temp1[1] < temp2[0]:
                return inter1+inter2
            elif temp1[1] >= temp2[0] and temp1[1] <= temp2[1]:
                return inter1[:-1] + [min(temp1[0], temp2[0]), max(temp1[1], temp2[1])] + inter2[1:]
            elif temp1[0] >= temp2[0] and temp1[1] <= temp2[1]:
                return merge_method(inter1[:-1], inter2)
```

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        intervals.sort(key=lambda x: x[0])
        
        prev = intervals[0]
        
        for i in range(1, len(intervals)):
            cur = intervals[i]
            if prev[1] < cur[0]:
                res.append(prev)
                prev = cur
            else:
                prev = [prev[0], max(prev[1], cur[1])]
        res.append(prev)
        return res
```



### 57. Insert Interval

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        ans = []
        intervals.append(newInterval)
        intervals = sorted(intervals, key=lambda k:k[0])
        for i in range(len(intervals)):
            if ans and ans[-1][1] >= intervals[i][0]:
                ans[-1][1] = max(ans[-1][1], intervals[i][1])
            else:
                ans.append(intervals[i])
        return ans
```

Similar to 56



### 435. Non-overlapping Intervals

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) <= 1:
            return 0
        
        temp = sorted(intervals, key=lambda k: k[1])
        ans = 1
        end = temp[0][1]
        for i in range(1, len(temp)):
            if temp[i][0] >= end:
                end = temp[i][1]
                ans += 1
        return len(temp)-ans
```



### 452. Minimum Number of Arrows to Burst Balloons

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points: return 0
        points = sorted(points, key=lambda x:x[0])
        end_cur = points[0][1]
        arrows = 1
        for i in range(1, len(points)):
            start, end = points[i]
            if start > end_cur:
                arrows += 1
                end_cur = end
            else:
                end_cur = min(end_cur, end)
        return arrows
```

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort()
        res = 0
        cur = points[0]
        
        for left, right in points[1:]:
            cur_left, cur_right = cur[0], cur[1]
            if cur_right < left:
                res += 1
                cur = [left, right]
            else:
                cur = [max(cur_left,left), min(cur_right, right)]
        return res + 1
```

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:

        res = 1

        points.sort()

        cur_x, cur_y = points[0]

        for i in range(1, len(points)):
            nxt_x, nxt_y = points[i]
            if cur_y < nxt_x:
                res += 1
                cur_x, cur_y = nxt_x, nxt_y
            else:
                cur_x, cur_y = max(cur_x, nxt_x), min(cur_y, nxt_y)
        
        return res
```



### 836. Rectangle Overlap

```python
        r1x1,r1y1,r1x2,r1y2 = rec1
        r2x1,r2y1,r2x2,r2y2 = rec2
        
        finx1 = max(r1x1,r2x1)
        finy1 = max(r1y1,r2y1)
        finx2 = min(r1x2,r2x2)
        finy2 = min(r1y2,r2y2)
        
        if (finx2 > finx1) and (finy1 < finy2):
            return True
        else:
            return False
```

求相交矩阵坐标，并判断该坐标是否可构成矩阵

[假定矩形是用一对点表达的(minx, miny) (maxx, maxy)，那么两个矩形 rect1{(minx1, miny1)(maxx1, maxy1)} rect2{(minx2, miny2)(maxx2, maxy2)}  ](https://www.cnblogs.com/0001/archive/2010/05/04/1726905.html)

相交的结果一定是个矩形，构成这个相交矩形rect{(minx, miny) (maxx, maxy)}的点对坐标是：  

minx=max(minx1, minx2)  

miny=max(miny1, miny2)  

maxx=min(maxx1, maxx2)  

maxy=min(maxy1, maxy2)  

如果两个矩形不相交，那么计算得到的点对坐标必然满足：  

（ minx > maxx ） 或者 （ miny > maxy ） 

 判定是否相交，以及相交矩形是什么都可以用这个方法一体计算完成。

从这个算法的结果上，我们还可以简单的生成出下面的两个内容：

㈠ 相交矩形：  (minx, miny) (maxx, maxy)

㈡ 面积： 面积的计算可以和判定一起进行
        if ( minx>maxx ) return 0;
        if ( miny>maxy ) return 0;
        return (maxx-minx)*(maxy-miny)

第二种方法

两个矩形相交的条件:两个矩形的重心距离在X和Y轴上都小于两个矩形长或宽的一半之和.这样,分两次判断一下就行了.

bool CrossLine(Rect r1,RECT r2)
{
if(abs((r1.x1+r1.x2)/2-(r2.x1+r2.x2)/2)<((r1.x2+r2.x2-r1.x1-r2.x1)/2) && abs((r1.y1+r1.y2)/2-(r2.y1+r2.y2)/2)<((r1.y2+r2.y2-r1.y1-r2.y1)/2))
return true;
return false;
}

### 986. Interval List Intersections

```python
class Solution:
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        i = j = 0
        ans = []
        while i < len(A) and j < len(B):
            if A[i][0] <= B[j][1] and A[i][0] >= B[j][0] or B[j][0] <= A[i][1] and B[j][0] >= A[i][0]:
                ans.append([max(A[i][0], B[j][0]), min(A[i][1], B[j][1])])
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return ans
```

```python
class Solution:
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        i = j = 0
        ans = []
        while i < len(A) and j < len(B):
            if A[i][0] <= B[j][1] and A[i][0] >= B[j][0] or B[j][0] <= A[i][1] and B[j][0] >= A[i][0]:
                ans.append([max(A[i][0], B[j][0]), min(A[i][1], B[j][1])])
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return ans
```



### 1288. Remove Covered Intervals

```python
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 1:
            return 1
        intervals.sort()
        
        cur = intervals[0]
        res = len(intervals)
        for i in range(1, len(intervals)):
            start, end = intervals[i]
            if start == cur[0] and end >= cur[1]:
                cur = intervals[i]
                res -= 1
            elif start > cur[0] and end <= cur[1]:
                res -= 1
            else:
                cur = intervals[i]
        return res
```

```python
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        res = 0
        cur = intervals[0]
        for interval in intervals[1:]:
            cur_left, cur_right = cur
            nex_left, nex_right = interval
            if nex_left > cur_right:
                cur = interval
            elif nex_left >= cur_left and nex_right <= cur_right:
                res += 1
            elif nex_left <= cur_left and nex_right >= cur_right:
                cur = interval
                res += 1
            else:
                cur = interval
        return len(intervals) - res
```

