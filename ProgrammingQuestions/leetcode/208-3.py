class Node:
    def __init__(self):
        self.is_leaf = False
        self.next = {}

# 逻辑和上一版本没有太大区别
class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.next = {}

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        cur = self
        for c in word:
            if c not in cur.next.keys():
                new_node = Node()
                cur.next[c] = new_node
            cur = cur.next[c]
        cur.is_leaf = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        cur = self
        for c in word:
            if c in cur.next.keys():
                cur = cur.next[c]
            else:
                return False
        return cur.is_leaf

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        cur = self.next
        for c in prefix:
            if c in cur.keys():
                cur = cur[c].next
            else:
                return False
        return True


        # Your Trie object will be instantiated and called as such:
        # obj = Trie()
        # obj.insert(word)
        # param_2 = obj.search(word)
        # param_3 = obj.startsWith(prefix)
