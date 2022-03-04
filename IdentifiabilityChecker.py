from DoX_BayesNet import *
from PyMC3_model_builder import *
from utils_JudeasRx import next_mu_sigma

import numpy as np
import itertools
from pprint import pprint
from collections import OrderedDict

np.set_printoptions(precision=3, floatmode="fixed")

class IdentifiabilityChecker:
    """
    This class has a doX bnet (an object of DoX_BayesNet) as input. The main
    goal of this class is to calculate bounds for the query Q = P(y=1|do(
    X)=x, z) for each stratum z. Here a stratum is one possible assignment of
    states to the control nodes of the doX bnet.

    One particular random assignment to TPMs of the unobserved nodes of the
    doX bnet is called a "world". This class loops over num_worlds number of
    worlds.

    For each world, this class loops over a pm model for each element of the
    cartesian product of the set of states of every control node. We refer
    to the elements of that cartesian product as the trol coords ( i.e.,
    control coordinates). The control nodes that are the axes of a trol
    coord are stored in self.trol_list.

    Attributes
    ----------
    doX_bnet : DoX_BayesNet
    num_1world_samples : int
        number of samples taken by pm for a single world.
    num_worlds : int
        number of worlds. Each world has a different random assignment to the
        TPMs of the unobserved nodes.
    pm_model_builder : PyMC3_model_builder
    trol_coords_to_query_bds : OrderedDict[tuple[int], np.array]
        dictionary mapping control coordinates to Q bounds.
        np.array of is [low Q, high Q].
    trol_coords_to_query_stats : OrderedDict[tuple[int], np.array]
        dictionary mapping control coordinates to Q statistics
        np.array of is [mean of Q, sigma of Q].
        mu=mean, sigma=standard deviation.
    trol_list : list[BayesNode]
        list of control nodes
    """

    def __init__(self, doX_bnet, num_1world_samples, num_worlds):
        """
        Constructor.

        Parameters
        ----------
        doX_bnet : DoX_BayesNet
        num_1world_samples : int
        num_worlds : int
        """
        self.doX_bnet = doX_bnet
        self.num_1world_samples = num_1world_samples
        self.num_worlds = num_worlds
        self.pm_model_builder = PyMC3_model_builder(doX_bnet)

        self.trol_coords_to_query_bds = OrderedDict()
        self.trol_coords_to_query_stats = OrderedDict()
        self.init_query_bds_and_stats()
        self.trol_list = self.doX_bnet.trol_list

    def init_query_bds_and_stats(self):
        """
        Initializes query bounds and statistics.

        Returns
        -------
        None

        """
        trol_range_list = [range(nd.size) for nd in
                           self.doX_bnet.trol_list]

        for trol_coords in itertools.product(*trol_range_list):
            self.trol_coords_to_query_bds[trol_coords] = np.array([1., 0])
            self.trol_coords_to_query_stats[trol_coords] = np.array([0.0, 0])


    def estimate_query_for_these_trol_coords(self, trol_coords):
        """
        Estimates query for a single tuple of control coordinates.

        Parameters
        ----------
        trol_coords : tuple[int]

        Returns
        -------
        float

        """
        self.pm_model_builder.refresh_pm_model(trol_coords)
        pm_model = self.pm_model_builder.pm_model
        with pm_model:
            trace = pm.sample_prior_predictive(self.num_1world_samples)
            prob_Y_is_1 = trace['Y'].mean()
            return prob_Y_is_1

    def estimate_query_for_all_trol_coords(self):
        """
        Estimates the query bounds for each of all the possible control
        coordinates.

        Returns
        -------
        dict[tuple, float]

        """
        trol_range_list = [list(range(nd.size)) for nd in
                           self.doX_bnet.trol_list]
        trol_coords_to_query = {}
        for trol_coords in itertools.product(*trol_range_list):
            Q = self.estimate_query_for_these_trol_coords(trol_coords)
            # print("xxccd, trol_coords, Q", trol_coords, Q)
            trol_coords_to_query[trol_coords] = Q

        return trol_coords_to_query

    def get_query_bds(self):
        """
        Gets class attribute trol_coords_to_query_bds.

        Returns
        -------
        OrderedDict[tuple, np.array]
            np.array is [low Q, high Q]

        """
        return self.trol_coords_to_query_bds

    def get_query_stats(self):
        """
        Gets class attribute trol_coords_to_query_stats.

        Returns
        -------
        OrderedDict[tuple, np.array]
            np.array is [mu of Q, sigma of Q]
            mu=mean, sigma=standard deviation

        """
        return self.trol_coords_to_query_stats


    def set_query_bds_and_stats(self):
        """
        Sets class attributes trol_coords_to_query_bds and
        trol_coords_to_query_stats.

        Returns
        -------
        None

        """
        # this is necessary when reuse self for several x_val
        self.init_query_bds_and_stats()

        print("world:")
        for world in range(self.num_worlds):
            if (world+1) % 10 == 0 or world == self.num_worlds-1:
                print(world, end="\n")
            else:
                print(world, end=", ")
            self.doX_bnet.refresh_unobs_nodes()
            trol_coords_to_query = self.estimate_query_for_all_trol_coords()
            for trol_coords, query in trol_coords_to_query.items():
                q_bds = self.trol_coords_to_query_bds[trol_coords]
                low, high = q_bds
                if query <= low:
                    q_bds[0] = query
                if query > high:
                    q_bds[1] = query
                q_stats = self.trol_coords_to_query_stats[trol_coords]
                next_mu, next_sigma = next_mu_sigma(world, query,
                                                    q_stats[0], q_stats[1])
                q_stats[0] = next_mu
                q_stats[1] = next_sigma

    def print_query_bds_and_stats(self):
        """
        Prints class attributes trol_coords_to_query_bds and
        trol_coords_to_query_stats.

        Returns
        -------
        None

        """
        print("x_value=", self.doX_bnet.x_val)
        print("control nodes:",
              [nd.name for nd in self.trol_list])
        print("control coords to query bounds (low, high):")
        pprint(dict(self.get_query_bds()))
        print("control coords to query statistics (mu, sigma):")
        pprint(dict(self.get_query_stats()))

if __name__ == "__main__":
    from Plotter_nz import *
    def main():
        doX_bnet = DoX_BayesNet.build_test_doX_bnet(draw=False)
        # print("kkkll", doX_bnet.nodes)
        checker = IdentifiabilityChecker(doX_bnet,
                          num_1world_samples=100,
                          num_worlds=5)
        checker.set_query_bds_and_stats()
        checker.print_query_bds_and_stats()
        Plotter_nz.plot_query_bds(doX_bnet.x_val, checker.get_query_bds(),
            zname_to_query_stats=checker.get_query_stats(),
            horizontal=True)
    main()
