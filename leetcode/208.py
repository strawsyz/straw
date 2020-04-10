class Node:
    def __init__(self, val=None):
        self.val = val
        self.next_node_values = []
        self.next_node = []
        self.is_leaf = False

    def add_next(self, val):
        if val not in self.next_node_values:
            self.next_node_values.append(val)
            node = Node(val)
            self.next_node.append(node)
            # node.pre_node
        else:
            index = self.next_node_values.index(val)
            node = self.next_node[index]
        return node

    def find_next(self, val):
        if val in self.next_node_values:
            index = self.next_node_values.index(val)
            node = self.next_node[index]
            return node
        else:
            return False


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
            root_node = root_node.add_next(c)
            # root_node = Node(c, pre_node=root_node)
        root_node.is_leaf = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        root_node = self.root_node
        for c in word:
            res = root_node.find_next(c)
            if res:
                root_node = res
            else:
                return False
        return root_node.is_leaf

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        root_node = self.root_node
        for c in prefix:
            res = root_node.find_next(c)
            if res:
                root_node = res
            else:
                return False
        return True


# Your Trie object will be instantiated and called as such:
obj = Trie()
word = 'asd'
obj.insert(word)
param_2 = obj.search(word)
param_3 = obj.startsWith(word)
print(param_2)
print(param_3)
