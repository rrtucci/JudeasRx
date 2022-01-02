import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    """
    This class has no constructor or attributes. It consists of just one
    static method that plots using matplotlib.

    """
    @staticmethod
    def plot_pns3_bds(bds_m, bds_f):
        """
        This method plots as 3 error bars the bounds bds_m for PNS3 = (PNS,
        PN, PS) for male patients. Then it plots side-by-side as 3
        additional error bars the bounds bds_f for female patients.

        Parameters
        ----------
        bds_m : np.array[shape=(3, 2)]
            the ouput of bounder_m.get_pns3_bds()
        bds_f : np.array[shape=(3, 2)]
            the output of bounder_f.get_pns3_bds()


        Returns
        -------
        None

        """
        plt.figure(figsize=(10, 5))
        bar_width = 0.3
        x_labels = ("PNS", "PN", "PS")
        plt.xticks(range(3), x_labels)
        plt.ylim(0, 1)
        y_labels = np.arange(0, 1.1, .1)
        plt.yticks(y_labels)
        plt.grid(linestyle='--', axis='y')
        plt.ylabel('probability')
        plt.bar(np.arange(3) - bar_width/2, bds_m[:, 1]-bds_m[:, 0],
                width=bar_width, bottom=bds_m[:, 0], color='blue')
        for k, x in enumerate(np.arange(3) - bar_width):
            txt = '(%.2f, %.2f)' % (bds_m[k, 0], bds_m[k, 1])
            plt.text(x, bds_m[k, 1]+.02, txt, size='small', color='blue')
        plt.bar(np.arange(3) + bar_width/2, bds_f[:, 1]-bds_f[:, 0],
                width=bar_width, bottom=bds_f[:, 0], color='hotpink')
        for k, x in enumerate(np.arange(3)):
            txt = '(%.2f, %.2f)' % (bds_f[k, 0], bds_f[k, 1])
            plt.text(x, bds_f[k, 1] + .02, txt, size='small', color='hotpink')
        plt.legend(['male', 'female'])
        plt.show()

    # staticmethod
    # def thicken_line(bds_z):
    #     """
    #
    #     Parameters
    #     ----------
    #     bds_z : np.array[shape=(3, 2)]
    #
    #     Returns
    #     -------
    #     np.array[shape=(3, 2)]
    #
    #     """
    #     assert bds_z.shape == (3, 2)
    #     thick_bds = np.array(bds_z)
    #     for row in range(3):
    #         if abs(thick_bds[row, 1] -thick_bds[row, 0])< 1e-6:
    #             if abs(thick_bds[row, 0] -1) <1e-6:
    #                 thick_bds[row, 1] = .99
    #             elif abs(thick_bds[row, 1]) < 1e-6:
    #                 thick_bds[row, 0] = .01
    #             else:
    #                 thick_bds[row, 0] += .01
    #     return thick_bds


if __name__ == "__main__":
    from Bounder import Bounder

    def main1():
        bds_m = np.array([[.3, .3],
                         [.2, .45],
                         [.5, .68]])
        bds_f = np.array([[.35, .45],
                          [.25, .55],
                          [.55, .72]])

        Plotter.plot_pns3_bds(bds_m, bds_f)

    def main2():
        # FEMALE
        e_y_bar_x_f = np.array([[.79, .52],
                                [.21, .48]])
        o_y_bar_x_f = np.array([[.3, .73],
                               [.7, .27]])
        px_f = np.array([.3, .7])
        f = Bounder(o_y_bar_x_f, px_f, e_y_bar_x=e_y_bar_x_f)
        f.set_exp_probs_bds()
        f.print_exp_probs_bds(",f")
        f.set_pns3_bds()
        f.print_pns3_bds("_f")
        # MALE
        e_y_bar_x_m = np.array([[.79, .51],
                                [.21, .49]])
        o_y_bar_x_m = np.array([[.3, .3],
                                [.7, .7]])
        px_m = np.array([.3, .7])
        m = Bounder(o_y_bar_x_m, px_m, e_y_bar_x=e_y_bar_x_m)
        m.set_exp_probs_bds()
        m.print_exp_probs_bds(",m")
        m.set_pns3_bds()
        m.print_pns3_bds("_m")

        pns3_bds_m= m.get_pns3_bds()
        pns3_bds_f = f.get_pns3_bds()
        # call Plotter
        # thick_pns3_bds_m = Plotter.thicken_line(pns3_bds_m)
        # thick_pns3_bds_f = Plotter.thicken_line(pns3_bds_f)
        # print("bds_m\n", thick_pns3_bds_m)
        # print("bds_f\n", thick_pns3_bds_f)
        Plotter.plot_pns3_bds(pns3_bds_m,
                              pns3_bds_f)

    main2()
