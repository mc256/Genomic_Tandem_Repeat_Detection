class Node:
    leaves = []
    nodes = []
    value = ""

    def __init__(self):
        self.leaves = []
        self.nodes = []
        self.value = ""
        pass

    def checkNodes(self, st):
        for idx in range(0, len(self.nodes)):
            if self.nodes[idx].value == st[0]:
                self.nodes[idx].addSuffix(st[1:])
                return True
        return False

    def checkLeaves(self, st):
        for idx in range(0, len(self.leaves)):
            leaf = self.leaves[idx]
            if leaf[0] == st[0]:
                node = Node()
                node.value = leaf[0]
                node.addSuffix(st[1:])
                node.addSuffix(leaf[1:])
                self.nodes.append(node)
                del self.leaves[idx]
                return
        self.leaves.append(st)

    def addSuffix(self, st):
        if len(st) == 0 or st == "":
            return
        else:
            if not self.checkNodes(st):
                self.checkLeaves(st)

    def getLongestRepeatedSubString(self):
        str = ""
        for idx in range(0, len(self.nodes)):
            temp = self.nodes[idx].getLongestRepeatedSubString()
            if len(temp) > len(str):
                str = temp
        return self.value + str


class LongestRepeatSubSequence:
    def __init__(self, sequence=""):
        self.sequence = sequence
        self.root = Node()
        for idx in range(0, len(sequence)):
            self.root.addSuffix(sequence[idx:])

    def get_LRS(self):
        return self.root.getLongestRepeatedSubString()



#test = [1,2,3,4]
#del test[1]
#print(test)
c = LongestRepeatSubSequence("GTTTTGATGAAGAAGATGGGGATGAAGAATCC")
print(c.get_LRS())
