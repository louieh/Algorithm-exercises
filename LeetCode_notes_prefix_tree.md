## LeetCode - Prefix Tree

[toc]

### 6301 sp10

```java
// starter code for Tries
/**
 * Course : CS 6301.001 - Special Topics in Computer Science - F19
 * SP10 : Tries
 *
 * @Authors : Luyi Han (lxh180006)
 */

import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;


public class Trie<V> {
    /**
     * Entry class
     */
    private class Entry {
        V value;
        HashMap<Character, Entry> child;
        int depth;

        /**
         * Constructor for building Entry
         *
         * @param value
         * @param depth
         */
        Entry(V value, int depth) {
            this.value = value;
            child = new HashMap<>();
            this.depth = depth;
        }
    }

    private Entry root;
    private int size;

    /**
     * Constructor for building trie
     */
    public Trie() {
        root = new Entry(null, 0);
        size = 0;
    }

    /**
     * private method: insert a String to prefix tree
     *
     * @param iter
     * @param value
     * @return
     */
    private V put(Iterator<Character> iter, V value) {
        Entry temp_root = root;
        while (iter.hasNext()) {
            Character temp_char = iter.next();
            if (!temp_root.child.containsKey(temp_char))
                temp_root.child.put(temp_char, new Entry(null, temp_root.depth + 1));
            temp_root = temp_root.child.get(temp_char);
        }
        temp_root.value = value;
        return value;
    }

    /**
     * private method: get the value of String
     *
     * @param iter
     * @return
     */
    private V get(Iterator<Character> iter) {
        Entry temp_root = root;
        while (iter.hasNext()) {
            Character temp_char = iter.next();
            if (!temp_root.child.containsKey(temp_char))
                return null;
            temp_root = temp_root.child.get(temp_char);
        }
        return temp_root.value;
    }

    private V remove(Iterator<Character> iter) {
        return null;
    }

    /**
     * private helper method of remove: remove a String
     *
     * @param current
     * @param s
     * @param index
     * @return
     */
    private boolean remove(Entry current, String s, int index) {
        if (index == s.length()) {
            if (current.value == null) {
                return false;
            }
            current.value = null;
            return current.child.isEmpty();
        }
        Character ch = s.charAt(index);
        Entry node = current.child.get(ch);
        if (node == null)
            return false;
        boolean shouldDelCurrent = remove(node, s, index + 1) && node.value == null;
        if (shouldDelCurrent) {
            current.child.remove(ch);
            return current.child.isEmpty();
        }
        return false;
    }

    /**
     * private helper method: dfs
     *
     * @param current
     * @param count
     */
    private void dfs(Entry current, int[] count) {
        if (current.value != null)
            count[0]++;
        for (Character key : current.child.keySet()) {
            dfs(current.child.get(key), count);
        }
    }


    // public methods

    /**
     * public method: insert String
     *
     * @param s
     * @param value
     * @return
     */
    public V put(String s, V value) {
        if (get(s) != null)
            return null;
        size++;
        return put(new StringIterator(s), value);
    }

    /**
     * public method: get String
     *
     * @param s
     * @return
     */
    public V get(String s) {
        return get(new StringIterator(s));
    }

    /**
     * public method: remove String
     *
     * @param s
     * @return
     */
    public V remove(String s) {
        V res = get(s);
        if (res != null) {
            size--;
            remove(root, s, 0);
        }
        return res;
    }

    // How many words in the dictionary start with this prefix?

    /**
     * public method: get the number of words in the prefix tree start with this String
     *
     * @param s
     * @return
     */
    public int prefixCount(String s) {
        Entry temp_node = root;
        int[] count = new int[1];
        Iterator<Character> iter = new StringIterator(s);
        while (iter.hasNext()) {
            Character ch = iter.next();
            if (temp_node.child.containsKey(ch)) {
                temp_node = temp_node.child.get(ch);
            } else {
                return 0;
            }
        }

        dfs(temp_node, count);
        return count[0];
    }

    public int size() {
        return size;
    }

    /**
     * String Iterator
     */
    public static class StringIterator implements Iterator<Character> {
        char[] arr;
        int index;

        public StringIterator(String s) {
            arr = s.toCharArray();
            index = 0;
        }

        public boolean hasNext() {
            return index < arr.length;
        }

        public Character next() {
            return arr[index++];
        }

        public void remove() {
            throw new java.lang.UnsupportedOperationException();
        }
    }

    public static void main(String[] args) {
        Trie<Integer> trie = new Trie<>();
        trie.put("ant", 1);
        trie.put("ante", 2);
        trie.put("anteater", 3);
        trie.put("antelope", 4);
        trie.put("antique", 5);
        System.out.println(trie.get("anteater"));
        trie.remove("anteater");
        System.out.println(trie.get("anteater"));
        System.out.println(trie.get("antique"));
        System.out.println(trie.prefixCount("anted"));
        System.out.println(trie.size());
//        int wordno = 0;
//        Scanner in = new Scanner(System.in);
//        while (in.hasNext()) {
//            String s = in.next();
//            if (s.equals("End")) {
//                break;
//            }
//            wordno++;
//            trie.put(s, wordno);
//        }
//
//        while (in.hasNext()) {
//            String s = in.next();
//            Integer val = trie.get(s);
//            System.out.println(s + "\t" + val);
//        }
    }
}

```

* 尤其注意 `remove` 函数的尾递归
* 还有 dfs 函数。



### 208. Implement Trie (Prefix Tree)

```python
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = dict()
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        # if not self.search(word)
        
        index, temp_dict = self.tool(word)
        for i in range(index, len(word)):
            if word[i] not in temp_dict:
                temp_dict[word[i]] = dict()
            temp_dict = temp_dict[word[i]]
        
        if '$' not in temp_dict:
            temp_dict['$'] = None
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        if not self.trie:
            return False
        index, temp_dict = self.tool(word)
        if index == len(word) and '$' in temp_dict:
            return True
        else:
            return False
    
    def tool(self, word):
        temp_dict = self.trie
        for i in range(len(word)):
            if word[i] in temp_dict:
                temp_dict = temp_dict.get(word[i])
            else:
                return i, temp_dict
        return len(word), temp_dict
        

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        if not self.trie:
            return False
        index, temp_dict = self.tool(prefix)
        if index == len(prefix):
            return True
        return False
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```



### 211. Add and Search Word - Data structure design

```python
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.prefix_tree = dict()
        

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        temp_dict = self.prefix_tree
        for each in word:
            if each not in temp_dict:
                temp_dict.update({each: {}})
            temp_dict = temp_dict.get(each)
        temp_dict['$'] = None
    
    def search_helper(self, current, word, index):
        if index == len(word):
            if '$' in current:
                return True
        else:
            if word[index] != '.':
                if word[index] in current:
                    if self.search_helper(current.get(word[index]), word, index+1):
                        return True
            else:
                for each_key in current:
                    if each_key != '$':
                        if self.search_helper(current.get(each_key), word, index+1):
                            return True
        return False
            

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        temp_dict = self.prefix_tree
        return self.search_helper(temp_dict, word, 0)
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

重点再看 search_helper 函数，几个错误点：

* search_helper 函数中第一个 `if` 后加 `else` 否则会继续执行第二个`if`.

* search_helper 函数中 `for` 循环遍历 `current` 中所有键时注意排除 `'$'`.

* Search_helper 递归函数有一个 `True` 则返回 `True` 否则返回 `False`, 这一点需要再考虑。。。递归返回值问题。最后循环处这样处理：循环中有返回 `True` 的话立刻返回，循环结束后加 `return False`, 这样这个循环是 `or` 类型。

  ```python
      def search_helper(self, current, word, index):
          if index == len(word):
              if '$' in current:
                  return True
              else:
                  return False
          else:
              if word[index] != '.':
                  if word[index] in current:
                      if self.search_helper(current.get(word[index]), word, index+1):
                          return True
                  else:
                      return False
              else:
                  for each_key in current:
                      if each_key != '$':
                          if self.search_helper(current.get(each_key), word, index+1):
                              return True
                  return False
  ```



### 212. Word Search II

```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        prefix_tree = dict()
        
        def insert(string):
            temp_dict = prefix_tree
            for each in string:
                if each not in temp_dict:
                    temp_dict.update({each:{}})
                temp_dict = temp_dict.get(each)
            temp_dict['$'] = None
        
        for each in words:
            insert(each)
        
        def dfs(row, col, temp_dict, str_now, res):
            if '$' in temp_dict:
                res.add(str_now)
            temp = board[row][col]
            board[row][col] = None
            if row > 0 and board[row-1][col] in temp_dict:
                dfs(row-1, col, temp_dict.get(board[row-1][col]), str_now+board[row-1][col], res)
            if row < len(board)-1 and board[row+1][col] in temp_dict:
                dfs(row+1, col, temp_dict.get(board[row+1][col]), str_now+board[row+1][col], res)
            if col > 0 and board[row][col-1] in temp_dict:
                dfs(row, col-1, temp_dict.get(board[row][col-1]), str_now+board[row][col-1], res)
            if col < len(board[0])-1 and board[row][col+1] in temp_dict:
                dfs(row, col+1, temp_dict.get(board[row][col+1]), str_now+board[row][col+1], res)
            board[row][col] = temp
        
        rows = len(board)
        cols = len(board[0])
        res = set()
        
        for row in range(rows):
            for col in range(cols):
                if board[row][col] in prefix_tree:    
                    dfs(row, col, prefix_tree.get(board[row][col]), board[row][col], res)
        
        return list(res)
```

先将列表中单词插入到prefix_tree中，之后dfs遍历字母矩阵，每走一个字母将其设为None以防之后再回来，因为每次都是上下左右四个方向。



### 336. Palindrome Pairs

```python
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        word2idx = {w:i for i, w in enumerate(words)}
        def is_palindrome(arr, start, end):
            while start < end:
                if arr[start] != arr[end]: return False
                start += 1
                end -= 1
            return True

        result = []
        for i, w in enumerate(words):
            for j in range(len(w) + 1):
                if is_palindrome(w, 0, j - 1):
                    candidate = ''.join(reversed(w[j:]))
                    if candidate not in word2idx: continue
                    idx = word2idx[candidate]
                    if idx == i: continue
                    result.append([idx, i])
            for j in range(len(w) - 1, -1, -1):
                if is_palindrome(w, j, len(w) - 1):
                    candidate = ''.join(reversed(w[:j]))
                    if candidate not in word2idx: continue
                    idx = word2idx[candidate]
                    if idx == i: continue
                    result.append([i, idx])
        return result

```

这题貌似和prefix_tree没什么关系，首先生成单词与index对应的字典。然后遍历列表中单词，

从左至右，若整个单词是回文，反转单词看之前字典中有没有对应key，若有则可拼接成回文；向右走一个字母，反转剩余字母，判断是否在字典中；向右走一个字母，判断截止到这是否是回文，若是则反转剩余字母。。。

从右至左同上。




### 421. Maximum XOR of Two Numbers in an Array

```python
class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        max_num_length = len(bin(max(nums))) - 2
        bits_list = [[(num >> i) & 1 for i in reversed(range(max_num_length))] for num in nums]
        
        prefix_tree = dict()
        for num, bits in zip(nums, bits_list):
            temp_dict = prefix_tree
            for bit in bits:
                if bit not in temp_dict:
                    temp_dict.update({bit: {}})
                temp_dict = temp_dict.get(bit)
            temp_dict['$'] = num
        
        max_res = 0
        
        for num, bits in zip(nums, bits_list):
            temp_dict = prefix_tree
            for bit in bits:
                toggled_bit = 1 - bit
                if toggled_bit in temp_dict:
                    temp_dict = temp_dict.get(toggled_bit)
                else:
                    temp_dict = temp_dict.get(bit)
            max_res = max(max_res, temp_dict['$'] ^ num)
        return max_res
```

https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/discuss/404504/Python-O(N)-Trie-Solution-wcomments-and-explanations

https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/discuss/407424/Using-a-Trie...-(accepted)



### 648. Replace Words

```python
class Solution:
    def replaceWords(self, dict: List[str], sentence: str) -> str:
        prefix_tree = {}
        
        def insert(string, index):
            temp_dict = prefix_tree
            for each in string:
                if each not in temp_dict:
                    temp_dict.update({each:{}})
                temp_dict = temp_dict.get(each)
            temp_dict['$'] = index

        for i in range(len(dict)):
            insert(dict[i], i)
        
            
        def get(string):
            temp_dict = prefix_tree
            for each in string:
                if each in temp_dict:
                    temp_dict = temp_dict.get(each)
                    if '$' in temp_dict:
                        return temp_dict.get('$')
                else:
                    return -1
            return -1
        
        sentence_list = sentence.split(" ")
        for i in range(len(sentence_list)):
            index = get(sentence_list[i])
            if index >= 0:
                sentence_list[i] = dict[index]
        
        return " ".join(sentence_list)
```



### 677. Map Sum Pairs

```python
class MapSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = dict()

    def insert(self, key: str, val: int) -> None:
        temp_dict = self.root
        for each in key:
            if each not in temp_dict:
                temp_dict.update({each:{}})
            temp_dict = temp_dict.get(each)
        temp_dict['$'] = val
    
    def dfs(self, current, count):
        if '$' in current:
            count += current.get('$')
        for each in current:
            if each != '$':
                count = self.dfs(current.get(each), count)
        return count
        
    def sum(self, prefix: str) -> int:
        temp_dict = self.root
        for each in prefix:
            if each not in temp_dict:
                return 0
            temp_dict = temp_dict.get(each)
        return self.dfs(temp_dict, 0)


# Your MapSum object will be instantiated and called as such:
# obj = MapSum()
# obj.insert(key,val)
# param_2 = obj.sum(prefix)
```



### 1032. Stream of Characters

```python
class StreamChecker:

    def __init__(self, words: List[str]):
        self.trie = self.initTrie(words)
        self.letters = []
    
    def initTrie(self, words: List[str]):
        res = dict()
        for word in words:
            temp = res
            for i, c in enumerate(word[::-1]):
                if c not in temp:
                    temp[c] = {}
                temp = temp[c]
                if i == len(word)-1:
                    temp.update({"#": "#"})
        return res

    def query(self, letter: str) -> bool:
        self.letters.append(letter)
        i = len(self.letters)-1
        trie = self.trie
        while i >= 0:
            if self.letters[i] not in trie:
                return False
            trie = trie.get(self.letters[i])
            if '#' in trie: return True
            i -= 1
        return '#' in trie


# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)
```



### 1268. Search Suggestions System

```python
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        trie = dict()
        ans = []
        
        def insert(word, trie):
            temp_dict = trie
            for c in word:
                if c not in temp_dict:
                    temp_dict.update({c:{}})
                temp_dict = temp_dict.get(c)
            temp_dict.update({'$': None})
        
        def get_word_list_by_prefix(trie, prefix):
            word_list = []
            def tool(temp_dict):
                for each in prefix:
                    if each in temp_dict:
                        temp_dict = temp_dict.get(each)
                    else:
                        return None
                return temp_dict
                
            def dfs(temp_dict, result):
                for each in temp_dict:
                    if each == "$":
                        word_list.append(result)
                    else:
                        dfs(temp_dict[each], result+each)
            temp_dict = tool(trie)
            if temp_dict is None:
                return []
            else:
                dfs(temp_dict, prefix)
                return word_list
        
        for word in products:
            insert(word, trie)
        
        cur_pre = ''
        for i in range(len(searchWord)):
            cur_pre += searchWord[i]
            word_list = sorted(get_word_list_by_prefix(trie, cur_pre))
            if len(word_list) <= 3:
                ans.append(word_list)
            else:
                ans.append(word_list[:3])
        return ans
```

