def preorder(root):
    # 先序遍历
    if root:
        print(root.val)
        preorder(root.left)
        preorder(root.right)


def preorder_non_recursion(root):
    # 不递归的先序遍历
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            print(node.val)
            stack.append(node.left)
            stack.append(node.right)


def inorder(root):
    # 中序遍历
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)


def inorder_non_recursion(root):
    # 不递归的中序遍历
    stack = []
    node = root

    while stack or node:
        if node:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            print(node.val)


def postorder(root):
    # 后序遍历
    if root:
        postorder(root.left)
        postorder(root.right)
        print(root.val)


def postorder_non_recursion(root):
    # 不使用递归的后序遍历
    stack_1 = [root]
    stack_2 = []
    while stack_1:
        node = stack_1.pop()
        stack_2.append(node)
        if node.left:
            stack_1.append(node.left)
        if node.right:
            stack_1.append(node.right)
    while stack_2:
        print(stack_2.pop().val)


# 层次遍历
def levelorder(root):
    queue = [root]
    while queue:
        node = queue.pop(0)
        print(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

# 二插查找树（Binary Search Tree）
# 左《 中 《 右

# 平衡搜索树定义
# 首先是个二查搜索树
# 每个节点的左右子树的高度之差最多为1
