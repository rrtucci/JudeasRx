import numpy as np
np.set_printoptions(precision=3, floatmode="fixed")


class Bounder:
    def __init__(self, o_y_bar_x, px, e_y_bar_x=None):
        """
        The theory and notation in this class is explained in the 2 chapters
        entitled "Personalized Treatment Effects" and "Personalized Expected
        Utility" of my free open source book Bayesuvius. Those 2 chapters
        are totally based on work by Pearl, Tian and later Pearl, Li, Mueller.

        https://qbnets.wordpress.com/2020/11/30/my-free-book-bayesuvius-on-bayesian-networks/

        The main 2 goals of this class are: (1) to calculate the bounds for
        PNS, PN and PS, given the probability matrices O_{y|x}, P(x) and E_{
        y|x}. (2) If also a utility function alpha_{y0,y1} is given,
        to calculate the bounds for the expected utility EU.

        We will use PNS3 to stand for the trio (PNS, PN, PS).

        O_{y|x} and P(x) are called the Observational Probabilities (these
        come from a survey), whereas E_{y|x} are called the Experimental
        Probabilities (these come from a RCT).

        Each column of  O_{y|x}, P(x) and E_{y|x} sums to 1, so we chose as
        independent dofs (degrees of freedom) the probabilties O_{1|0},
        O_{1|1}, P(x=1), E_{1|0} and E_{1|1}.

        A given O_{y|x} imposes bounds on E_{y|x} that must be obeyed for
        consistency. This class also calculates those bounds.

        In this app, we consider a Bounder object bounder_m for males,
        and a Bounder object bounder_f for females.

        In this app, we consider two cases:

        1. Only Observational data

        2. Both Observational and Experimental data.

        We also allow the user to impose the additional constraints of
        exogeneity, strong exogeneity, monotonicity and some specific DAG
        families.

        The more constraints, the tighter the bounds on PNS3 and EU.

        Attributes
        ----------
        alp_y0_y1 : np.array[shape=(2, 2)]
            \alpha_{y0, y1}, utility function
        e0b0 : float
            E_{0|0}
        e0b1 : float
            E_{0|1}
        e1b0 : float
            E_{1|0}
        e1b1 : float
            E_{1|1}
        e_y_bar_x : np.array[shape=(2, 2)]
            E_{y|x}
        eu_bds : np.array[shape=(2, )]
            bounds for EU (expected utility)
        exogeneity : bool
        left_bds_e_y_bar_x : np.array[shape=(2, 2)]
            left (low) bounds for each element of E_{y|x}
        monotonicity : bool
        o00 : float
            O_{0,0}
        o01 : float
            O_{0,1}
        o0b0 : float
            O_{0|0}
        o0b1 : float
            O_{0|1}
        o10 : float
            O_{1,0}
        o11 : float
            O_{1,1}
        o1b0 : float
            O_{1|0}
        o1b1 : float
            O_{1|1}
        o_y_bar_x : np.array[shape=(2, 2)]
            O_{y|x}
        pns3_bds : np.array[shape =(3, 2)]
            [[PNS_low, PNS_high],
            [PN_low, PN_high],
            [PS_low, PS_high]]
        px : float
            P(x)
        px0 : float
            P(x=0)
        px1 : float
            P(x=1)
        right_bds_e_y_bar_x : np.array[shape=(2, 2)]
            right (high) bounds for each element of E_{y|x}
        strong_exo : bool

        Parameters
        ----------
        o_y_bar_x : np.array[shape=(2, 2)]
            O_{y|x}
        px : np.array[shape=(2, )]
            P(x)
        e_y_bar_x : np.array[shape=(2, )], None
            E_{y|x}
        """

        Bounder.check_prob_vec(px)
        self.px = px
        self.px0 = px[0]
        self.px1 = px[1]

        Bounder.check_2d_trans_matrix(o_y_bar_x)
        self.o_y_bar_x = o_y_bar_x
        self.o0b0 = o_y_bar_x[0, 0]
        self.o0b1 = o_y_bar_x[0, 1] 
        self.o1b0 = o_y_bar_x[1, 0]
        self.o1b1 = o_y_bar_x[1, 1]
        self.o00 = o_y_bar_x[0, 0]*px[0]
        self.o01 = o_y_bar_x[1, 0]*px[0]   # oxy and oybx  so flip x,y
        self.o10 = o_y_bar_x[0, 1]*px[1]   # oxy and oybx  so flip x,y
        self.o11 = o_y_bar_x[1, 1]*px[1]

        self.e_y_bar_x = None
        self.e0b0 = None
        self.e0b1 = None
        self.e1b0 = None
        self.e1b1 = None
        if e_y_bar_x is not None:
            self.set_exp_probs(e_y_bar_x)
        self.left_bds_e_y_bar_x = np.zeros(shape=(2, 2))
        self.right_bds_e_y_bar_x = np.zeros(shape=(2, 2))
        self.pns3_bds = np.zeros(shape=(3, 2))

        self.exogeneity = False
        self.monotonicity = False
        self.strong_exo = False

        self.alp_y0_y1 = np.zeros(shape=(2,2))
        self.eu_bds = np.array([-1, 1])

    def set_obs_probs(self, o_y_bar_x, px):
        """
        This method refreshes the class attributes with new observational
        probabilities. It checks the consistency of the input.

        Parameters
        ----------
        o_y_bar_x : np.array[shape=(2, 2)]
            O_{y|x}
        px : np.array[shape=(2, )]
            P(x)

        Returns
        -------
        None

        """
        Bounder.check_prob_vec(px)
        self.px = px
        self.px0 = px[0]
        self.px1 = px[1]

        Bounder.check_2d_trans_matrix(o_y_bar_x)
        self.o_y_bar_x = o_y_bar_x
        self.o0b0 = o_y_bar_x[0, 0]
        self.o0b1 = o_y_bar_x[0, 1]
        self.o1b0 = o_y_bar_x[1, 0]
        self.o1b1 = o_y_bar_x[1, 1]
        self.o00 = o_y_bar_x[0, 0]*px[0]
        self.o01 = o_y_bar_x[1, 0]*px[0]
        self.o10 = o_y_bar_x[0, 1]*px[1]
        self.o11 = o_y_bar_x[1, 1]*px[1]

    def set_exp_probs(self, e_y_bar_x):
        """
        This method refreshes the class attributes with new experimental
        probabilities. It checks the consistency of the input.

        Parameters
        ----------
        e_y_bar_x : np.array[shape=(2, 2)]
            E_{y|x}

        Returns
        -------
        None

        """
        Bounder.check_2d_trans_matrix(e_y_bar_x)
        self.e_y_bar_x = e_y_bar_x
        self.e0b0 = e_y_bar_x[0, 0]
        self.e0b1 = e_y_bar_x[0, 1]
        self.e1b0 = e_y_bar_x[1, 0]
        self.e1b1 = e_y_bar_x[1, 1]

    def set_utility_fun(self, alp_y0_y1):
        """
        Sets the utility function \alpha_{y_0, y_1}

        Parameters
        ----------
        alp_y0_y1 : np.array[shape=(2, 2)]
            \alpha_{y_0, y_1}, utility function

        Returns
        -------
        None

        """
        self.alp_y0_y1 = alp_y0_y1
        assert alp_y0_y1.shape == (2,2)

    @staticmethod
    def check_2d_trans_matrix(mat):
        """
        Checks that the 2x2 transition probability matrix mat is well defined.

        Parameters
        ----------
        mat : np.array[shape=(2, 2)]

        Returns
        -------
        None

        """
        assert mat.shape == (2, 2)
        assert (0 <= mat).all()
        assert (mat <= 1).all()  # can't check 0<=mat<=1 at once
        assert np.abs(sum(mat[:, 0]) - 1) < 1e-5
        assert np.abs(sum(mat[:, 1]) - 1) < 1e-5

    @staticmethod
    def check_prob_vec(vec):
        """
        Checks that the probability vector vec is well defined.

        Parameters
        ----------
        vec : np.array[shape=(2, )]

        Returns
        -------
        None

        """
        assert vec.shape == (2, )
        assert (0 <= vec).all()
        assert (vec <= 1).all()
        assert np.abs(sum(vec) - 1) < 1e-5

    def get_ate(self):
        """
        Returns ATE = E_{1|1} - E_{1|0} or None

        Returns
        -------
        float, None

        """
        if self.e_y_bar_x is not None:
            return self.e1b1 - self.e1b0
        else:
            return None

    def get_py(self):
        """
        Returns P(y) for y=0,1.

        Returns
        -------
        float, float
            P(y=0), P(y=1)

        """
        py0 = self.o00 + self.o10
        py1 = self.o01 + self.o11
        return py0, py1

    def get_e_star_bar_star(self):
        """
        Returns E_{*|*} = E_{0|0} + E_{1|1}.

        Returns
        -------
        float
            E_{*|*} = E_{0|0} + E_{1|1}

        """
        return self.e0b0 + self.e1b1

    def get_o_star_bar_star(self):
        """
        Returns O_{*|*} = O_{0|0} + O_{1|1}.

        Returns
        -------
        float
            O_{*|*} = O_{0|0} + O_{1|1}

        """
        return self.o0b0 + self.o1b1

    def get_o_star_star(self):
        """
        Returns O_{*,*} = O_{0,0} + O_{1,1}.

        Returns
        -------
        float
            O_{*,*} = O_{0,0} + O_{1,1}

        """
        return self.o00 + self.o11

    def get_sigma(self):
        """
        Returns sigma = a01 + a10 - (a00 + a11) where
        aij= self.alp_y0_y1[i,j]

        Returns
        -------
        float

        """
        return self.alp_y0_y1[0,1] + self.alp_y0_y1[1,0]- (
            self.alp_y0_y1[1,1] + self.alp_y0_y1[0,0])

    def set_exp_probs_bds(self):
        """
        This method sets the class attributes for the elementwise bounds on
        the transition probability matrix E_{ y|x}.

        Returns
        -------
        None

        """

        right = self.right_bds_e_y_bar_x
        left = self.left_bds_e_y_bar_x
        if not self.monotonicity:
            left[1, 1] = self.o11
            right[1, 1] = 1 - self.o10
            left[1, 0] = self.o01
            right[1, 0] = 1 - self.o00

            # use if a <= x <= b then 1-b <= 1-x <= 1-a
            left[0, 1] = self.o10
            right[0, 1] = 1 - self.o11
            left[0, 0] = self.o00
            right[0, 0] = 1 - self.o01
        else:
            py0, py1 = self.get_py()

            left[1, 1] = py1
            right[1, 1] = 1 - self.o10
            left[1, 0] = self.o01
            right[1, 0] = py1

            # use if a <= x <= b then 1-b <= 1-x <= 1-a
            left[0, 1] = self.o10
            right[0, 1] = py0
            left[0, 0] = py0
            right[0, 0] = 1 - self.o01

    def get_exp_probs_bds(self):
        """
        Returns left (low) and right (high) bounds of e_y_bar_x

        Returns
        -------
        np.array[shape=(2, 2)], np.array[shape=(2, 2)]
            self.left_bds_e_y_bar_x,  self.right_bds_e_y_bar_x

        """
        return self.left_bds_e_y_bar_x, self.right_bds_e_y_bar_x

    def set_pns3_bds(self):
        """
        This method sets the class attribute for the bounds for PNS3 = (PNS,
        PN, PS).

        Returns
        -------
        None

        """
        any_exo = self.strong_exo or self.exogeneity
        if self.e_y_bar_x is None:         # no experimental data
            pns_bds = [0, self.get_o_star_star()]
            pn_bds = [0, 1]
            ps_bds = [0, 1]
        else:
            py0, py1 = self.get_py()
            e_star_bar_star = self.get_e_star_bar_star()
            o_star_bar_star = self.get_o_star_bar_star()
            o_star_star = self.get_o_star_star()

            if not any_exo and not self.monotonicity:
                # pns bounds
                pns_left = max(
                    0,
                    e_star_bar_star - 1,
                    self.e0b0 - py0,
                    self.e1b1 - py1)
                pns_right = min(
                    self.e1b1,
                    self.e0b0,
                    o_star_star,
                    e_star_bar_star - o_star_star)

                # pn bounds
                if self.o11 <= 0:
                    pn_left = 0
                    pn_right = 1
                else:
                    pn_left = max(
                        0,
                        (self.e0b0 - py0)/self.o11)
                    pn_right = min(
                        1,
                        (self.e0b0 - self.o00)/self.o11)

                # ps bounds
                if self.o00 <= 0:
                    ps_left = 0
                    ps_right = 1
                else:
                    ps_left = max(
                        0,
                        (self.e1b1 - py1)/self.o00)
                    ps_right = min(
                        1,
                        (self.e1b1 - self.o11)/self.o00)

            elif any_exo and not self.monotonicity:
                # pns bounds
                pns_left = max(
                    0,
                    o_star_bar_star - 1)
                pns_right = min(
                    self.o1b1,
                    self.o0b0)
                # pn bounds
                if self.o1b1 <= 0:
                    pn_left = 0
                    pn_right = 1
                else:
                    err = (self.o1b1 - self.o1b0)/self.o1b1
                    pn_left = max(0, err)
                    pn_right = min(1, self.o0b0/self.o1b1)

                # ps bounds
                if self.o0b0 <= 0:
                    ps_left = 0
                    ps_right = 1
                else:
                    err_tilde = (self.o0b0 - self.o0b1)/self.o0b0
                    ps_left = max(0, err_tilde)
                    ps_right = min(1, self.o1b1/self.o0b0)
            elif not any_exo and self.monotonicity:
                # pns bounds
                pns_left = e_star_bar_star - 1
                pns_right = pns_left
                # pn bounds
                if self.o11 <= 0:
                    pn_left = 1
                else:
                    pn_left = (self.e0b0 - py0)/self.o11
                pn_right = pn_left
                # ps bounds
                if self.o00 <= 0:
                    ps_left = 1
                else:
                    ps_left = (self.e1b1 - py1)/self.o00
                ps_right = ps_left
            elif any_exo and self.monotonicity:
                # pns bounds
                pns_left = o_star_bar_star - 1
                pns_right = pns_left
                # pn bounds
                if self.o11 <= 0:
                    pn_left = 1
                else:
                    pn_left = (self.o0b0 - py0)/self.o11
                pn_right = pn_left
                # ps bounds
                if self.o00 <= 0:
                    ps_left = 1
                else:
                    ps_left = (self.o1b1 - py1)/self.o00
                ps_right = ps_left
            else:
                assert False
            if self.strong_exo:
                # when strong exo holds, pns = pn*o1b1 = ps*o0b0
                left = max(pns_left, pn_left*self.o1b1, ps_left*self.o0b0)
                right = min(pns_right, pn_right*self.o1b1, ps_right*self.o0b0)
                pns_left = left
                pns_right = right
                if self.o1b1 > 0:
                    pn_left = left/self.o1b1
                    pn_right = right/self.o1b1
                if self.o0b0 > 0:
                    ps_left = left/self.o0b0
                    ps_right = right/self.o0b0
            pns_bds = [pns_left, pns_right]
            pn_bds = [pn_left, pn_right]
            ps_bds = [ps_left, ps_right]

        self.pns3_bds = np.array([pns_bds, pn_bds, ps_bds])

    def get_pns3_bds(self):
        """
        Returns PNS3 bounds.

        Returns
        -------
         np.array[shape=(3, 2)]
            [[PNS_low, PNS_high],
            [PN_low, PN_high],
            [PS_low, PS_high]]

        """
        return self.pns3_bds

    def set_eu_bds(self):
        """
        Sets the class attribute self.eu_bds to the bounds for the expected
        utility EU.

        Returns
        -------
        None

        """
        if self.e_y_bar_x is None:
            self.eu_bds = np.array([-1, 1])
            return

        a00 = self.alp_y0_y1[0, 0]
        a01 = self.alp_y0_y1[0, 1]
        a10 = self.alp_y0_y1[1, 0]
        a11 = self.alp_y0_y1[1, 1]

        if self.monotonicity:
            eu = a00*self.e0b1 + a11*self.e1b0 + a01*self.get_ate()
            self.eu_bds = np.array([eu, eu])
            return

        sigma = a01 + a10 - (a00 + a11)

        p5 = (a11 - a10)*self.e1b1 + a00*self.e0b0 + a10*self.e1b0
        py0, py1 = self.get_py()

        p1 = (a01 - a00)*self.e1b1 + a00*self.e0b0 + a10*self.e1b0
        p2 = (a01 - a11)*self.e0b0 + a10*self.e1b0 + a11*self.e1b1
        p3 = p5 + sigma*self.get_o_star_star()
        p4 = p1 + sigma*(-self.e1b0 + 1 - self.get_o_star_star())
        p6 = p1 - sigma*self.e1b0
        p7 = p5 + sigma*(-self.e1b0 + py1)
        p8 = p1 - sigma*py1

        low = max(p1, p2, p3, p4)
        high = min(p5, p6, p7, p8)
        if low > high:
            low = -1
            high = 1

        self.eu_bds = np.array([low, high])

    def get_eu_bds(self):
        """
        Returns the bounds for EU (expected utility).

        Returns
        -------
        np.array[shape=(2,)]

        """
        return self.eu_bds

    def print_exp_probs(self,  st="", array_format=False):
        """
        Prints the Experimental probabilities E_{y|x}.

        Parameters
        ----------
        st : str
            st is used for more explicit labeling of the male and female
            strata.
        array_format : bool
            True iff print Experimental Probabilities as numpy arrays

        Returns
        -------
        None

        """
        if self.e_y_bar_x is not None:
            if array_format:
                print("E_{y|x" + st + "}=\n", self.e_y_bar_x)
            else:
                print("E_{1|0" + st + "}=", "%.3f" % self.e1b0)
                print("E_{1|1" + st + "}=", "%.3f" % self.e1b1)
        else:
            print("E_{y|x" + st + "}= None")

    def print_obs_probs(self,  st="", array_format=False):
        """
        Prints the Observational Probabilities O_{y|x} and P(x).

        Parameters
        ----------
        st : str
            st is used for more explicit labeling of the male and female
            strata.
        array_format : bool
            True iff print Observational Probabilities as numpy arrays

        Returns
        -------
        None

        """
        if array_format:
            print("O_{y|x" + st + "}=\n", self.o_y_bar_x)
            print("P_{x" + st + "}=", self.px)
        else:
            print("O_{1|0" + st + "}=", "%.3f" % self.o_y_bar_x[1, 0])
            print("O_{1|1" + st + "}=", "%.3f" % self.o_y_bar_x[1, 1])
            print("P_{1" + st + "}=", self.px[1])

    def print_all_probs(self,  st=""):
        """
        Calls print_obs_probs() and print_exp_probs()

        Parameters
        ----------
        st : str
            st is used for more explicit labeling of the male and female
            strata.

        Returns
        -------
        None

        """
        self.print_obs_probs(st)
        self.print_exp_probs(st)

    def print_utility_fun(self, st=""):
        """
        Prints utility function \alpha_{y0, y1}.

        Parameters
        ----------
        st : str
            st is used for more explicit labeling of the male and female
            strata.

        Returns
        -------
        None

        """
        print("alpha_{y0,y1}" + st + "=\n", self.alp_y0_y1)

    def print_exp_probs_bds(self,  st=""):
        """
        Prints left (low) and right (high) bounds for each element of E_{y|x}.

        Parameters
        ----------
        st : str
            st is used for more explicit labeling of the male and female
            strata.

        Returns
        -------
        None

        """
        left = self.left_bds_e_y_bar_x
        right = self.right_bds_e_y_bar_x
        mid = self.e_y_bar_x
        for x in range(2):
            for y in range(2):
                print("E_{" + str(y) + "|" + str(x) + st + "}: " +
                        "%.3f <= %.3f <= %.3f"
                        % (left[y, x],
                        mid[y, x],
                        right[y, x]))

    def print_pns3_bds(self,  st=""):
        """
        Prints bounds on PNS3 = (PNS, PN, PS).

        Parameters
        ----------
        st : str
            st is used for more explicit labeling of the male and female
            strata.

        Returns
        -------
        None

        """
        for i,  st1 in zip([0, 1, 2], ['PNS', ' PN', ' PS']):
            print("%.3f" % self.pns3_bds[i, 0]
                   + " <= " + st1 + st + " <= "
                   + "%.3f" % self.pns3_bds[i, 1])

    def print_eu_bds(self, st=""):
        """
        Prints bounds for Expected Utility EU

        Parameters
        ----------
        st : str
            st is used for more explicit labeling of the male and female
            strata.

        Returns
        -------
        None

        """
        print("%.3f" % self.eu_bds[0]
              + " <= " + "EU" + st + " <= "
              + "%.3f" % self.eu_bds[1])

if __name__ == "__main__":
    def main():
        print("FEMALE-----------------------")
        print("input probabilities obtained from exp. and obs. data:")
        e_y_bar_x_f = np.array([[.79, .52],
                                [.21, .48]])
        o_y_bar_x_f = np.array([[.3, .73],
                               [.7, .27]])
        px_f = np.array([.3, .7])
        alp_y0_y1_f = np.array([[2, 4],
                              [-3, 7]])
        f = Bounder(o_y_bar_x_f, px_f, e_y_bar_x=e_y_bar_x_f)
        f.set_utility_fun(alp_y0_y1_f)
        f.print_all_probs(",f")
        f.print_utility_fun("_f")
        print("---------------------------")
        print("Check exp. data is within bds imposed by obs. data:")
        f.set_exp_probs_bds()
        f.print_exp_probs_bds(",f")
        print("---------------------------")
        f.set_pns3_bds()
        f.print_pns3_bds("_f")
        print("---------------------------")
        f.set_eu_bds()
        f.print_eu_bds("_f")

        print("MALE--------------------------")
        print("input probabilities obtained from exp. and obs. data:")
        e_y_bar_x_m = np.array([[.79, .51],
                                [.21, .49]])
        o_y_bar_x_m = np.array([[.3, .3],
                                [.7, .7]])
        px_m = np.array([.3, .7])
        alp_y0_y1_m = np.array([[2, 4],
                              [-3, 7]])
        m = Bounder(o_y_bar_x_m, px_m, e_y_bar_x=e_y_bar_x_m)
        m.set_utility_fun(alp_y0_y1_m)
        m.print_all_probs(",m")
        m.print_utility_fun("_m")
        print("---------------------------")
        print("Check exp. data is within bds imposed by obs. data:")
        m.set_exp_probs_bds()
        m.print_exp_probs_bds(",m")
        print("---------------------------")
        m.set_pns3_bds()
        m.print_pns3_bds("_m")
        print("---------------------------")
        m.set_eu_bds()
        m.print_eu_bds("_m")

    main()