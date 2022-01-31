import numpy as np

from nodes.BayesNode import *
from graphs.BayesNet import *
from potentials.DiscreteUniPot import *
from potentials.DiscreteCondPot import *

class WetGrass:

    @staticmethod
    def build_bnet():
        """
        Builds CBnet called WetGrass with diamond shape
                Cloudy
                /    \
             Rain    Sprinkler
               \      /
               WetGrass
        All arrows pointing down
        """

        cl = BayesNode(0, name="Cloudy")
        sp = BayesNode(1, name="Sprinkler")
        ra = BayesNode(2, name="Rain")
        we = BayesNode(3, name="WetGrass")

        we.add_parent(sp)
        we.add_parent(ra)
        sp.add_parent(cl)
        ra.add_parent(cl)

        nodes = {cl, ra, sp, we}

        cl.potential = DiscreteUniPot(False, cl)  # P(a)
        sp.potential = DiscreteCondPot(False, [cl, sp])  # P(b| a)
        ra.potential = DiscreteCondPot(False, [cl, ra])
        we.potential = DiscreteCondPot(False, [sp, ra, we])

        # in general
        # DiscreteCondPot(False, [y1, y2, y3, x]) refers to P(x| y1, y2, y3)
        # off = 0
        # on = 1

        cl.potential.pot_arr[:] = [.5, .5]

        ra.potential.pot_arr[1, :] = [.5, .5]
        ra.potential.pot_arr[0, :] = [.4, .6]

        sp.potential.pot_arr[1, :] = [.7, .3]
        sp.potential.pot_arr[0, :] = [.2, .8]

        we.potential.pot_arr[1, 1, :] = [.01, .99]
        we.potential.pot_arr[1, 0, :] = [.01, .99]
        we.potential.pot_arr[0, 1, :] = [.01, .99]
        we.potential.pot_arr[0, 0, :] = [.99, .01]

        return BayesNet(nodes)


if __name__ == "__main__":
    def main():
        bnet = WetGrass.build_bnet()
        # introduce some evidence
        bnet.get_node_named("WetGrass").active_states = [1]

        bnet.write_bif('../examples_cbnets/WetGrass.bif', False)
        bnet.write_dot('../examples_cbnets/WetGrass.dot')
    main()
