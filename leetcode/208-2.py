# 比上一版快了不少，使用字典来存储子节点

class Node:
    def __init__(self, val=None):
        self.children = {}  # 使用字典来存储加快查找速度
        self.is_leaf = False


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root_node = Node()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        root_node = self.root_node
        for c in word:
            if c not in root_node.children:
                root_node.children[c] = Node()
            root_node = root_node.children[c]

        root_node.is_leaf = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        root_node = self.root_node
        for c in word:
            if c not in root_node.children:
                return False
            else:
                root_node = root_node.children[c]
        return root_node.is_leaf

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        root_node = self.root_node
        for c in prefix:
            if c not in root_node.children:
                return False
            root_node = root_node.children[c]
        return True


# Your Trie object will be instantiated and called as such:
obj = Trie()
word = 'asd'
obj.insert(word)
param_2 = obj.search(word)
param_3 = obj.startsWith(word)
print(param_2)
print(param_3)
