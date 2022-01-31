from graphs.BayesNet import *
import itertools

class ImaginedBayesNet(BayesNet):

    def __init__(self, in_bnet):
        BayesNet.__init__(self, in_bnet.nodes)
        self.in_bnet = in_bnet
        self.build_self()

    def build_self(self):
        nodeX = self.in_bnet.get_node_named("X")
        nodeY = self.get_node_named("Y")
        assert nodeX.has_child(nodeY),\
            "Node Y is not a child on node Y"
        parentsY = self.in_bnet.parents

        id_nums = [nd.id_num for nd in self.in_bnet.nodes]
        max_id = max(id_nums)
        nodeY0 = BayesNode(max_id + 1, "Y0")
        nodeY1 = BayesNode(max_id + 2, "Y1")

        nodeY0.add_parents(parentsY + nodeX)
        nodeY1.add_parents(parentsY + nodeX)

        nodeY0.add_child(nodeY)
        nodeY1.add_child(nodeY)

        for nd in parentsY:
            nodeY.remove_parent(nd)
        nodeY.add_parents([nodeX, nodeY0, nodeY1])

        potY = DiscreteCondPot(False, [nodeX, nodeY0, nodeY1, nodeY])
        potY.pot_arr = np.zeros(shape=(2,2,2,2),dtype=np.float64)
        arr = potY.pot_arr

        for prod in itertools.product([0,1], repeat=4):
            x, y0, y1, y = prod
            if y == y0*(1-x) + y1*x:
                arr[x, y0, y1, y] = 1



