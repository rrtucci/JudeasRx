import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Plotter:
    """
    This class has no constructor or attributes. It consists of just static
    methods that plot using matplotlib.

    """
    @staticmethod
    def plot_pns3_bds(ax, bds_m, bds_f):
        """
        This method plots as 3 error bars the bounds bds_m for PNS3 = (PNS,
        PN, PS) for male patients. Then it plots side-by-side as 3
        additional error bars the bounds bds_f for female patients.

        Parameters
        ----------
        ax :
            an axis from matplotlib
        bds_m : np.array[shape=(3, 2)]
            the ouput of bounder_m.get_pns3_bds()
        bds_f : np.array[shape=(3, 2)]
            the output of bounder_f.get_pns3_bds()

        Returns
        -------
        None

        """
        bar_width = 0.3
        x_labels = ("PNS", "PN", "PS")
        plt.sca(ax)
        plt.xticks(range(3), x_labels)
        ax.set_ylim(0, 1)
        y_labels = np.arange(0, 1.1, .1)
        ax.set_yticks(y_labels)
        ax.grid(linestyle='--', axis='y')
        ax.set_ylabel('probability')
        ax.bar(np.arange(3) - bar_width/2, bds_m[:, 1]-bds_m[:, 0],
                width=bar_width, bottom=bds_m[:, 0], color='blue')
        for k, x in enumerate(np.arange(3) - bar_width):
            txt = '(%.2f, %.2f)' % (bds_m[k, 0], bds_m[k, 1])
            ax.text(x, bds_m[k, 1]+.02, txt, size='small', color='blue')
        ax.bar(np.arange(3) + bar_width/2, bds_f[:, 1]-bds_f[:, 0],
                width=bar_width, bottom=bds_f[:, 0], color='hotpink')
        for k, x in enumerate(np.arange(3)):
            txt = '(%.2f, %.2f)' % (bds_f[k, 0], bds_f[k, 1])
            ax.text(x, bds_f[k, 1] + .02, txt, size='small', color='hotpink')
        ax.legend(['male', 'female'])
        
    @staticmethod
    def plot_eu_bds(ax, eu_bds_m, eu_bds_f):
        """
        This method plots as an error bar the bounds eu_bds_m for male
        patients, and then it plots as another error bar the bounds eu_bds_f
        for female patients.

        Parameters
        ----------
        ax :
            an axis from matplotlib
        eu_bds_m : np.array[shape=(2,)]
            the ouput of bounder_m.get_eu_bds()
        eu_bds_f : np.array[shape=(2,)]
            the output of bounder_f.get_eu_bds()


        Returns
        -------
        None

        """
        bar_width = 1
        plt.sca(ax)
        plt.xticks([0], ["EU"])
        ax.set_ylim(-1, 1)
        y_labels = np.arange(-1, 1.1, .2)
        ax.set_yticks(y_labels)
        ax.grid(linestyle='--', axis='y')
        ax.set_ylabel('utility')

        ax.bar(- bar_width/2, eu_bds_m[1]-eu_bds_m[0],
                width=bar_width, bottom=eu_bds_m[0], color='blue')
        txt = '(%.2f, %.2f)' % (eu_bds_m[0], eu_bds_m[1])
        ax.text(-bar_width, eu_bds_m[1]+.02, txt, size='small', color='blue')

        ax.bar(bar_width/2, eu_bds_f[1]-eu_bds_f[0],
                width=bar_width, bottom=eu_bds_f[0], color='hotpink')
        txt = '(%.2f, %.2f)' % (eu_bds_f[0], eu_bds_f[1])
        ax.text(0, eu_bds_f[1] + .02, txt, size='small', color='hotpink')

        ax.legend(['male', 'female'])
        ax.axhline(y=0, color='black')

    @staticmethod
    def plot_all(bds_m, bds_f, eu_bds_m, eu_bds_f):
        """
        This method calls both plot_pns3_bds() and plot_eu_bds().

        Parameters
        ----------
        bds_m : np.array[shape=(3,2)]
            PNS3 bounds for male patients
        bds_f : np.array[shape=(3,2)]
            PNS3 bounds for female patients
        eu_bds_m : np.array[shape=(2,)]
            EU bounds for male patients
        eu_bds_f : np.array[shape=(2,)]
            Eu bounds for female patients

        Returns
        -------
        None
        """

        plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        Plotter.plot_pns3_bds(ax1, bds_m, bds_f)
        Plotter.plot_eu_bds(ax2, eu_bds_m, eu_bds_f)
        plt.tight_layout()
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

    def main():
        bds_m = np.array([[.3, .3],
                         [.2, .45],
                         [.5, .68]])
        bds_f = np.array([[.35, .45],
                          [.25, .55],
                          [.55, .72]])
        eu_bds_m = np.array([-.4, .8])
        eu_bds_f = np.array([0,.5])

        Plotter.plot_all(bds_m, bds_f, eu_bds_m, eu_bds_f)

    main()
