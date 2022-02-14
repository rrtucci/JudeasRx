from ImaginedBayesNet import *
from PyMC3_model_builder import *

import numpy as np
import itertools
from pprint import pprint

np.set_printoptions(precision=3, floatmode="fixed")

class MultiBounder_MC:
    """
    We call this class MultiBounder_MC= MultiBounder Monte Carlo,
    to distinguish it from MultiBounder_ana= MultiBounder analytic.

    This class has an imagined bnet (an object of ImaginedBayesNet) as
    input. The main goal of this class is to calculate bounds for the 3
    quantities PNS3=(PNS, PN, PS), for each strata z. Here a strata is one
    possible assignment of states to the control nodes of the imagined bnet.

    One particular assignment to the TPMs of the random nodes of the
    imagined bnet is called a "world". This class loops over num_worlds
    number of worlds.

    For each world, this class loops over a pm model for each element of the
    cartesian product of the set of states of every control node. We refer
    to the elements of that cartesian product as the trol coords ( i.e.,
    control coordinates). The control nodes that are the axes of a trol
    coord are stored in self.trol_list.

    Attributes
    ----------
    imagined_bnet : ImaginedBayesNet
    num_1world_samples : int
        number of samples taken by pm for a single world.
    num_worlds : int
        number of worlds. Each world has a different random assignment to the
        TPMs of the random nodes.
    pm_model_builder : PyMC3_model_builder
    trol_coords_to_PNS3_bds : dict[tuple[int], np.array]
        np.array of shape=(3, 2). dictionary mapping control coordinates to
        PNS3 bounds
    trol_list : list[BayesNode]
        list of control nodes
    """

    def __init__(self, imagined_bnet, num_1world_samples, num_worlds):
        """
        Constructor.

        Parameters
        ----------
        imagined_bnet : ImaginedBayesNet
        num_1world_samples : int
        num_worlds : int
        """
        self.imagined_bnet = imagined_bnet
        self.num_1world_samples = num_1world_samples
        self.num_worlds = num_worlds
        self.pm_model_builder = PyMC3_model_builder(imagined_bnet)

        self.trol_coords_to_PNS3_bds = {}
        trol_range_list = [range(nd.size) for nd in
                           self.imagined_bnet.trol_list]

        for trol_coords in itertools.product(*trol_range_list):
            self.trol_coords_to_PNS3_bds[trol_coords] = np.array([[1., 0],
                                                                  [1., 0],
                                                                  [1., 0]])
        self.trol_list  = self.imagined_bnet.trol_list

    def estimate_PNS3_for_these_trol_coords(self, trol_coords):
        """
        Estimates PNS3 = (PNS, PN, PS) for a single tuple of control
        coordinates.

        Parameters
        ----------
        trol_coords : tuple[int]

        Returns
        -------
        np.array
            shape=(3, ), [PNS, PN, PS]

        """
        self.pm_model_builder.refresh_pm_model(trol_coords)
        pm_model = self.pm_model_builder.pm_model
        with pm_model:
            trace = pm.sample_prior_predictive(self.num_1world_samples)

            prob_Y0_is_1 =\
                trace['Y0'][(trace['X'] == 1) & (trace['Y'] == 1)].mean()
            PN = 1 - prob_Y0_is_1
            prob_Y1_is_1 = \
                trace['Y1'][(trace['X'] == 0) & (trace['Y'] == 0)].mean()
            PS = prob_Y1_is_1

            prob_Y0_is_0 = 1 - trace['Y0'].mean()
            prob_Y1_is_1_if_Y0_is_0 = trace['Y1'][(trace['Y0'] == 0)].mean()
            PNS = prob_Y1_is_1_if_Y0_is_0 * prob_Y0_is_0

            return np.array([PNS, PN, PS])

    def estimate_PNS3_for_all_trol_coords(self):
        """
        Estimates the PNS3 bounds for each of all the possible control
        coordinates.

        Returns
        -------
        dict[tuple, np.array]
            np.array of shape=(3, )

        """
        trol_range_list = [list(range(nd.size)) for nd in
                           self.imagined_bnet.trol_list]
        trol_coords_to_PNS3 = {}
        for trol_coords in itertools.product(*trol_range_list):
            PNS, PN, PS = self.estimate_PNS3_for_these_trol_coords(trol_coords)
            # print("xxccd, trol_coords, PNS, PN, PS", trol_coords, PNS, PN,
            # PS)
            trol_coords_to_PNS3[trol_coords] = np.array([PNS, PN, PS])

        return trol_coords_to_PNS3

    def get_PNS3_bds(self):
        """
        Gets class attribute trol_coords_to_PNS3_bds, where PNS3=(PNS, PN,
        PS).

        Returns
        -------
        dict[tuple, np.array]
            np.array shape=(3, 2)
            [[PNS low, PNS high],
            [PN low, PN high],
            [PS low, PS high]]

        """
        return self.trol_coords_to_PNS3_bds

    def set_PNS3_bds(self):
        """
        Sets class attribute trol_coords_to_PNS3_bds, where PNS3=(PNS, PN,
        PS).

        Returns
        -------
        None

        """
        for world in range(self.num_worlds):
            print("world", world)
            self.imagined_bnet.refresh_random_nodes()
            trol_coords_to_PNS3 = self.estimate_PNS3_for_all_trol_coords()
            for trol_coords, PNS3 in trol_coords_to_PNS3.items():
                PNS3_bounds = self.trol_coords_to_PNS3_bds[trol_coords]
                for k in range(3):
                    low, high = PNS3_bounds[k]
                    if PNS3[k] <= low:
                        PNS3_bounds[k][0] = PNS3[k]
                    if PNS3[k] > high:
                        PNS3_bounds[k][1] = PNS3[k]
                    # print("kkhhh", f"k={k}, low={low}, high={high}, "
                    #                f"PNS[k]={PNS3[k]}, "
                    #                f"new low={PNS3_bounds[k][0]},"
                    #                f"new high={PNS3_bounds[k][1]}")


if __name__ == "__main__":
    def main():
        imagined_bnet = ImaginedBayesNet.build_test_imagined_bnet(
            draw=False, use_Y0Y1=True, only_obs=False)
        # print("kkkll", imagined_bnet.nodes)
        bder = MultiBounder_MC(imagined_bnet,
                          num_1world_samples=100,
                          num_worlds=5)
        bder.set_PNS3_bds()
        print("control nodes:",
              [nd.name for nd in bder.trol_list])
        pprint(bder.get_PNS3_bds())

    main()
