import numpy as np
import pymc3 as pm
import theano

# Installing pymc3 correctly proved to be a bit tricky.
# See
# https://github.com/pymc-devs/pymc/wiki/Installation-Guide-%28Linux%29
# I set up a conda virtual environment with Python 3.9. Then used:
# conda install -c conda-forge pymc3 theano-pymc mkl mkl-service
# conda install graphviz
# conda install pydotplus
# conda install python-graphviz
# conda install pydot

# ran into trouble installing jupyter lab in conda virtual env. This solved it
# https://stackoverflow.com/questions/68261254/conda-error-sysconfigdata-x86-64-conda-linux-gnu


class PyMC3_model_builder:
    """
    Note: This class uses code in directories: nodes, graphs, potentials,
    that was borrowed from my app Quantum Fog
    https://github.com/artiste-qb-net/quantum-fog

    This class also benifitted greatly from the following thread in the
    PyMC3 Discourse
    https://discourse.pymc.io/t/bayes-nets-belief-networks-and-pymc/5150

    Theano is complementary to our Quantum Fog classes. They both use DAGs
    and potentials, but QFog uses Bayesian networks whereas Theano uses
    block diagrams. (rv=random variable, TPM=Transition Probability matrix)

    Here is how I translate a block diagram to a bnet
    block diagram -> Bayesian network
    rvs live in arrows, TPMs live in nodes-> both rvs and TPMs live in nodes
    arrow=rv -> arrow just a mapping thing, nothing lives there
    node= theano function or shared array -> TPM
    shared array node -> general TPM
    function node -> determinstic TPM

    Attributes
    ----------
    controlled_bnet :  ImaginedBayesNet, DoX_BayesNet
        an object of a child class of BayesNet that possesses the attribute
        trol_list
    pm_model : pm.Model
    topo_index_to_nd : dict[int, BayesNode]
        dictionary mapping topological index to node
    trol_list : list[BayesNode]
        list of control nodes
    trol_to_pos : dict[BayesNode, int]
        dictionary mapping control node to its position in the trol_list
    """

    def __init__(self, controlled_bnet):
        """
        Constructor

        Parameters
        ----------
        controlled_bnet : ImaginedBayesNet, DoX_BayesNet
        """
        self.controlled_bnet = controlled_bnet
        self.pm_model = None

        self.topo_index_to_nd = {}
        for nd in self.controlled_bnet.nodes:
            self.topo_index_to_nd[nd.topo_index] = nd

        # num_nds = len(self.controlled_bnet.nodes)
        # for ind in range(num_nds):
        #     nd = self.topo_index_to_nd[ind]
        #     print("mmmkk", ind, nd.name,
        #           "parents=", [x.name for x in nd.parents],
        #           "children=", [x.name for x in nd.children], '\n',
        #           nd.potential.pot_arr)

        self.trol_list = self.controlled_bnet.trol_list
        self.trol_to_pos = {trol: pos for pos, trol in enumerate(
            self.trol_list)}
    
    def refresh_pm_model(self, trol_coords):
        """
        Refreshes the pm (PyMC3) model for the control coordinates
        trol_coords.

        Parameters
        ----------
        trol_coords : tuple[int]
            control coordinates is a tuple giving the state of each
            control node in the order self.trol_list.

        Returns
        -------
        None

        """
        if self.pm_model is not None:
            del self.pm_model
        self.pm_model = pm.Model()

        with self.pm_model:

            # nd_to_rv = node to random variable. This dictionary maps each
            # BayesNode object in controlled_bnet to a pymc3 random variable
            nd_to_rv = {}
            num_nds = len(self.controlled_bnet.nodes)
            for topo_index in range(num_nds):
                nd = self.topo_index_to_nd[topo_index]
                # print("llhhhf", topo_index, nd.name)
                nd_pa_list = nd.potential.ord_nodes[:-1]
                num_parents = len(nd_pa_list)

                if nd in self.trol_list:
                    active_state = trol_coords[self.trol_to_pos[nd]]

                    nd_arr = np.zeros(shape=nd.potential.pot_arr.shape)
                    # slicex = tuple([slice(None), slice(None), ...,
                    # slice(None), active_state])
                    slicex = tuple([slice(None)] * num_parents +
                                   [active_state])
                    # print('aaasss', num_parents, active_state, slicex)
                    nd_arr[slicex] = 1.0
                else:
                    nd_arr = nd.potential.pot_arr
                # print("vvvvv", nd.name, nd_arr)

                if num_parents == 0:
                    nd_to_rv[nd] = pm.Categorical(nd.name, nd_arr)
                    # print("dxxxx", nd.name, nd_arr)
                else:
                    lookup_table = theano.shared(np.asarray(nd_arr))

                    def fun(*rv_pa_list):
                        return lookup_table[tuple(rv_pa_list)]
                    # print("lllkk", nd.name, fun(*([0]*num_parents)))

                    rv_pa_list = [nd_to_rv[nd1] for nd1 in nd_pa_list]
                    nd_to_rv[nd] = pm.Categorical(nd.name, fun(*rv_pa_list))


if __name__ == "__main__":

    def main():
        lookup_table = theano.shared(np.asarray([
            [[.99, .01], [.1, .9]],
            [[.9, .1], [.1, .9]]]))
        print(lookup_table.get_value()[0, 0])

        def fun1(*pa_list):
            return lookup_table[tuple(pa_list)]
        print("fun1", fun1(1, 0))

        def fun2(p1, p2):
            return lookup_table[p1, p2]
        print("fun2", fun2(1, 0))

    main()