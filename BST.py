from tools import Tools

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.size = len(value[0])

    def __repr__(self):
        return str(self.value)

    def getU(self):
        return self.value[0]

    def getSize(self):
        return self.size

    def getVal(self):
        return self.value

class Leaf:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __repr__(self):
        return str(self.value)

    def getVal(self):
        return self.value

    def getSize(self):
        return len(self.value)

class BST:
    def __init__(self, root):
        self.root = root

    def getRoot(self):
        return self.root

    def getDepth(self, root):
        node = root
        count = 0
        while node is not None:
            count += 1
            node = node.left
        return count

    def put(self, parent, child, side):
        self.root = self._put(parent, child, side)

    def _put(self, parent, child, side):
        if side == "left":
            parent.left = child
        else:
            parent.right = child

    def printTree(self, root):
        if root is None:
            return

        self.printTree(root.left)
        print('-------')

        if type(root) is Node:
            print('upperright')
            print('U: ' + str(root.value[0].round(2)))
            print('E: ' + str(root.value[1].round(2)))
            print('Vt: ' + str(root.value[2].round(2)))
            print('lowerleft')
            print('U: ' + str(root.value[3].round(2)))
            print('E: ' + str(root.value[4].round(2)))
            print('Vt: ' + str(root.value[5].round(2)))

            '''
            A = root.value
            U1, S1, VT1, U2, S2, VT2 = A[0], A[1], A[2], A[3], A[4], A[5]
            print('upperright')
            print(str((U1.dot(S1)).dot(VT1).round(2)))
            print('lowerleft')
            print(str((U2.dot(S2)).dot(VT2).round(2)))
            '''
        if type(root) is Leaf:
            print(root.value)

        # print(root.value)
        print('-------')
        self.printTree(root.right)
        print('-------')


    def printTreeDims(self, root):
        t = Tools()
        if root is None:
            return

        self.printTreeDims(root.left)

        if type(root) is Node:
            print('SVD: ' + t.printDim(root.value[0].round(2)))
        if type(root) is Leaf:
            print(t.printDim(root.value))

        # print(root.value)
        self.printTreeDims(root.right)


    def compareTrees(self, root, root2):
        if root is None:
            return

        self.compareTrees(root.left, root2.left)
        print('-------')

        if type(root) is Node:
            print('U: ' + str((root.value[0] - root2.value[0]).round(2)))
            print('E: ' + str((root.value[1] - root2.value[1]).round(2)))
            print('Vt: ' + str((root.value[2] - root2.value[2]).round(2)))
            print('U: ' + str((root.value[3] - root2.value[3]).round(2)))
            print('E: ' + str((root.value[4] - root2.value[4]).round(2)))
            print('Vt: ' + str((root.value[5] - root2.value[5]).round(2)))
        if type(root) is Leaf:
            print(root.value - root2.value)

        # print(root.value)
        print('-------')
        self.compareTrees(root.right, root2.right)
        print('-------')