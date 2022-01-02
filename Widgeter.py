from Bounder import Bounder
from Plotter import Plotter
import numpy as np
import ipywidgets as wid
from IPython.display import display, clear_output


class Widgeter:
    def __init__(self):
        """
        The main method of this class and the only one meant for external
        use is run_gui(). This method runs a GUI (Graphical User Interface)
        as a cell in a Jupyter notebook. The controls of the GUI are
        implemented using the library ipywidgets.

        Attributes
        ----------
        bdoor_crit : bool
            True iff backdoor criterion for node G relative to (X,Y) is
            satisfied
        bounder_f : Bounder
            Bounder object for females
        bounder_m : Bounder
            Bounder object for males
        exogeneity : bool
        exp_sliders_to_latex : dict[wid.FloatSlider, str]
            dictionary mapping experimental sliders to a LaTex string
        exp_slider_to_tbox : dict[wid.FloatSlider, wid.BoundedFloatText]
            dictionary mapping experimental sliders to their text boxes
        exp_sliders : List[wid.FloatSlider]
            list of experimental sliders
        monotonicity : bool
        no_x_to_g : bool
            True iff G is not a descendant of X
        obs_sliders_to_latex : dict[wid.FloatSlider, str]
            dictionary mapping observational sliders to a LaTex string
        obs_slider_to_tbox : dict[wid.FloatSlider, wid.BoundedFloatText]
            dictionary mapping observational sliders to their text boxes
        obs_sliders : List[wid.FloatSlider]
            list of observational sliders        
        only_obs : bool
            Only Observational Probabilities, no Experimental ones
        pmale : float
            P(gender=male)
        strong_exogeneity : bool

        """
        self.only_obs = True
        self.exogeneity = False
        self.strong_exogeneity = False
        self.monotonicity = False
        self.no_x_to_g = False
        self.bdoor_crit = False

        o_y_bar_x_m = np.array([[.5, .5], [.5, .5]])
        px_m = np.array([.5, .5])
        self.bounder_m = Bounder(o_y_bar_x_m, px_m)
        self.bounder_m.set_exp_probs_bds()

        o_y_bar_x_f = np.array([[.5, .5], [.5, .5]])
        px_f = np.array([.5, .5])
        self.bounder_f = Bounder(o_y_bar_x_f, px_f)
        self.bounder_f.set_exp_probs_bds()

        self.pmale = .5
        
        self.obs_sliders = []
        self.exp_sliders = []

        self.obs_slider_to_tbox = {}
        self.exp_slider_to_tbox = {}

        self.obs_slider_to_latex = {}
        self.exp_slider_to_latex = {}

    def refresh_bounders_using_slider_vals(
            self,
            o1b0_m, o1b1_m, px1_m,
            o1b0_f, o1b1_f, px1_f,
            e1b0_m, e1b1_m,
            e1b0_f, e1b1_f,
            pmale
            ):
        """
        This method is called by wid.interactive() which requires it. Its
        inputs are all slider values. After doing some housework,
        this method calls Plotter.plot_pns3_bds().

        Parameters
        ----------
        o1b0_m : float
            O_{1|0,m}
        o1b1_m : float
            O_{1|1,m}
        px1_m : float
            P(x=1) for males
        o1b0_f : float
            O_{1|0,f}
        o1b1_f : float
            O_{1|1,f}
        px1_f : float
            P(x=1) for females
        e1b0_m : float
            E_{1|0,m}
        e1b1_m : float
            E_{1|1,m}
        e1b0_f : float
            E_{1|0,f}
        e1b1_f : float
            E_{1|1,f}
        pmale : float
            P(g=male)

        Returns
        -------
        None

        """
        o_y_bar_x_m = np.array([
            [1 - o1b0_m, 1 - o1b1_m],
            [o1b0_m, o1b1_m]])
        px_m = np.array([1 - px1_m, px1_m])
        self.bounder_m.set_obs_probs(o_y_bar_x_m, px_m)
        self.bounder_m.set_exp_probs_bds()

        o_y_bar_x_f = np.array([
            [1 - o1b0_f, 1 - o1b1_f],
            [o1b0_f, o1b1_f]])
        px_f = np.array([1 - px1_f, px1_f])
        self.bounder_f.set_obs_probs(o_y_bar_x_f, px_f)
        self.bounder_f.set_exp_probs_bds()

        if not self.only_obs:
            e_y_bar_x_m = np.array([
                [1 - e1b0_m, 1 - e1b1_m],
                [e1b0_m, e1b1_m]])
            self.bounder_m.set_exp_probs(e_y_bar_x_m)
            e_y_bar_x_f = np.array([
                [1 - e1b0_f, 1 - e1b1_f],
                [e1b0_f, e1b1_f]])
            self.bounder_f.set_exp_probs(e_y_bar_x_f)

        self.bounder_m.set_pns3_bds()
        self.bounder_f.set_pns3_bds()

    def refresh_slider_colors(self, obs_green):
        """
        This method toggles the colors and disabled status (green/red for
        enabled/disabled) of the sliders for Observational Probabilities and
        the sliders for Experimental Probabilities. In addition, it toggles
        the disabled status of the text boxes that are attached to each of
        those sliders.

        Parameters
        ----------
        obs_green : bool
            True iff Observational sliders are green (enabled)

        Returns
        -------
        None

        """
        def color_it(g, latex_str):
            if g:
                color = 'green'
            else:
                color = 'red'
            return '$\color{' + color + '}{' + latex_str + '}$'

        for x in self.obs_sliders:
            x.disabled = not obs_green
            self.obs_slider_to_tbox[x].disabled = not obs_green
            x.description = color_it(obs_green,
                                     self.obs_slider_to_latex[x])
        for x in self.exp_sliders:
            x.disabled = obs_green
            self.exp_slider_to_tbox[x].disabled = obs_green
            x.description = color_it(not obs_green,
                                     self.exp_slider_to_latex[x])

    def refresh_plot(self):
        """
        This method is a clever way of inducing the method wid.interactive()
        to redraw the plot. The method jiggles the pmale slider,
        thus causing wid.interactive() to redraw the plot.

        Returns
        -------
        None

        """
        # just jiggle the pmale slider
        x = self.exp_sliders[4]
        delta = .1
        x.min -= delta
        x.value -= delta
        x.min += delta
        x.value += delta

    def set_exp_sliders_to_valid_values(self):
        """
        This method calculates the experimental bounds for the male and
        female bounders. Using those results, it sets the values of the min,
        value and max parameters of the Experimental Probabilities sliders.

        Returns
        -------
        None

        """
        def change(slider, a, b):
            slider.disabled = True
            slider.min = a
            slider.max = b
            slider.disabled = False
            slider.value = a

        self.bounder_m.set_exp_probs_bds()
        left_bds, right_bds = self.bounder_m.get_exp_probs_bds()
        # set value of E_{1|i,m} for i=0,1
        for i, slider in zip([0, 1], self.exp_sliders[0:2]):
            a, b = left_bds[1, i], right_bds[1, i]
            change(slider, a, b)

        self.bounder_f.set_exp_probs_bds()
        left_bds, right_bds = self.bounder_f.get_exp_probs_bds()
        # set value of E_{1|i,f} for i=0,1
        for i, slider in zip([0, 1], self.exp_sliders[2:4]):
            a, b = left_bds[1, i], right_bds[1, i]
            change(slider, a, b)

    def run_gui(self):
        """
        This is the main method of this class and the only one meant for
        external use. It draws a GUI.

        The GUI has 11 sliders, 6 sliders for Observational Probabilities
        and 5 sliders for Experimental Probabilities. Each slider has a text
        box attached to it which can be used to enter input by typing
        numbers instead of moving the slider. In addition, the GUI has
        several clickable control buttons and check boxes and one disabled
        text box that gives info about the current status of the calculations.

        Returns
        -------
        None

        """
        slider_params = dict(
            min=0,
            max=1,
            value=.5,
            step=.001,
            orientation='vertical')
        o1b0_m_slider = wid.widgets.FloatSlider(**slider_params)
        o1b1_m_slider = wid.widgets.FloatSlider(**slider_params)
        px1_m_slider = wid.widgets.FloatSlider(**slider_params)
        # order important, 1b0 before 1b1,
        # mnemonic 10 < 11
        obs_m_sliders = [o1b0_m_slider,
                        o1b1_m_slider,
                        px1_m_slider]
        o1b0_f_slider = wid.widgets.FloatSlider(**slider_params)
        o1b1_f_slider = wid.widgets.FloatSlider(**slider_params)
        px1_f_slider = wid.widgets.FloatSlider(**slider_params)
        # order important, 1b0 before 1b1,
        # mnemonic 10 < 11
        obs_f_sliders = [o1b0_f_slider,
                        o1b1_f_slider,
                        px1_f_slider]
        self.obs_sliders = [*obs_m_sliders,
                            *obs_f_sliders]

        e1b0_m_slider = wid.widgets.FloatSlider(**slider_params)
        e1b1_m_slider = wid.widgets.FloatSlider(**slider_params)
        # order important, 1b0 before 1b1,
        # mnemonic 10 < 11
        exp_m_sliders = [e1b0_m_slider, e1b1_m_slider]

        e1b0_f_slider = wid.widgets.FloatSlider(**slider_params)
        e1b1_f_slider = wid.widgets.FloatSlider(**slider_params)
        # order important, 1b0 before 1b1,
        # mnemonic 10 < 11
        exp_f_sliders = [e1b0_f_slider, e1b1_f_slider]
        pmale_slider = wid.widgets.FloatSlider(**slider_params)
        self.exp_sliders = [*exp_m_sliders,
                            *exp_f_sliders,
                            pmale_slider]

        self.obs_slider_to_latex = {
            obs_m_sliders[0]: 'O_{1|0,m}',
            obs_m_sliders[1]: 'O_{1|1,m}',
            obs_m_sliders[2]: '\pi_{1,m}',
            #
            obs_f_sliders[0]: 'O_{1|0,f}',
            obs_f_sliders[1]: 'O_{1|1,f}',
            obs_f_sliders[2]: '\pi_{1,f}'}

        self.exp_slider_to_latex = {
            exp_m_sliders[0]: 'E_{1|0,m}',
            exp_m_sliders[1]: 'E_{1|1,m}',
            exp_f_sliders[0]: 'E_{1|0,f}',
            exp_f_sliders[1]: 'E_{1|1,f}',
            pmale_slider: 'P_m'
        }

        header = wid.HTMLMath("Enter Observational Data from a survey." +
            "<br>Then press the 'Add Experimental Data (RCT)' button\
            if you also have Experimental Data."
            "<br>Sliders with green/red labels are\
            enabled/disabled."
            "<br>$g\in\{m,f\}$ stands for gender.$x,y\in \{0,1\}$."
            "<br>$E_{y|x} = \sum_g E_{y|x,g}P_g$ (backdoor adjustment "
            "formula)"
            "<br>$ATE_g = E_{1|1,g} - E_{1|0,g}$"
            "<br>$ATE=E_{1|1} - E_{1|0}$"
            "<br>$PNS = \sum_g PNS_g P_g$")

        add_but = wid.Button(
            description='Add Experimental Data (RCT)',
            button_style='danger',
            layout=wid.Layout(width='200px')
        )
        exp_bds_sign = wid.HTMLMath()

        def add_but_do(btn):
            if self.only_obs:
                self.refresh_slider_colors(obs_green=False)
                self.set_exp_sliders_to_valid_values()
                self.only_obs = False
        add_but.on_click(add_but_do)

        print_but = wid.Button(
            description='Print',
            button_style='warning',
            layout=wid.Layout(width='50px')
        )
        out = wid.Output()
        display(out)

        def print_but_do(btn):
            with out:
                print("###################################")
                print("Male:------------------------------")
                self.bounder_m.print_all_probs(',m')
                self.bounder_m.print_pns3_bds('_m')
                print("Female:----------------------------")
                self.bounder_f.print_all_probs(',f')
                self.bounder_f.print_pns3_bds('_f')
                print("ATE:-------------------------------")
                ate_f = self.bounder_f.get_ate()
                ate_m = self.bounder_m.get_ate()
                if ate_m is not None and ate_f is not None:
                    ate = ate_m * self.pmale + ate_f * (1 - self.pmale)
                    print("ATE_m=", "%.3f" % ate_m)
                    print("ATE_f", "%.3f" % ate_f)
                    print("ATE=", "%.3f" % ate)
        print_but.on_click(print_but_do)

        exo_but = wid.Checkbox(
            value=self.exogeneity,
            description="Exogeneity",
            indent=False)

        def exo_but_do(change):
            new = change['new']
            self.exogeneity = new
            self.bounder_m.exogeneity = new
            self.bounder_f.exogeneity = new
            if not self.only_obs:
                self.refresh_plot()
        exo_but.observe(exo_but_do, names='value')

        strong_exo_but = wid.Checkbox(
            value=self.strong_exogeneity,
            description="Strong Exogeneity",
            indent=False)

        def strong_exo_but_do(change):
            new = change['new']
            self.strong_exogeneity = new
            self.bounder_m.strong_exogeneity = new
            self.bounder_f.strong_exogeneity = new
            if not self.only_obs:
                self.refresh_plot()
        strong_exo_but.observe(strong_exo_but_do, names='value')
        
        mono_but = wid.Checkbox(
            value=self.monotonicity,
            description="Monotonicity",
            indent=False)

        def mono_but_do(change):
            new = change['new']
            self.monotonicity = new
            self.bounder_m.monotonicity = new
            self.bounder_f.monotonicity = new
            if not self.only_obs:
                self.refresh_plot()
        mono_but.observe(mono_but_do, names='value')

        no_x_to_g_but = wid.Checkbox(
            value=self.no_x_to_g,
            description="$G$ is not a descendant of $X$",
            indent=False)

        def no_x_to_g_but_do(change):
            new = change['new']
            self.no_x_to_g = new
            if not self.only_obs:
                self.refresh_plot()
        no_x_to_g_but.observe(no_x_to_g_but_do, names='value')
        
        bdoor_crit_but = wid.Checkbox(
            value=self.bdoor_crit,
            description="Backdoor criterion is satisfied",
            indent=False)

        def bdoor_crit_but_do(change):
            new = change['new']
            self.bdoor_crit = new
            if not self.only_obs:
                self.refresh_plot()
        bdoor_crit_but.observe(bdoor_crit_but_do, names='value')
        ate_m_sign = wid.Label()
        ate_f_sign = wid.Label()
        ate_sign = wid.Label()

        def box_the_sliders(sliders):
            vbox_list = []
            slider_to_tbox = {}
            for x in sliders:
                tbox = wid.BoundedFloatText(
                    step=x.step,
                    min=x.min,
                    max=x.max,
                    layout=wid.Layout(width='60px'))
                slider_to_tbox[x] = tbox
                wid.jslink((x, 'value'), (tbox, 'value'))
                vbox_list.append(wid.VBox([x, tbox]))
            return wid.HBox(vbox_list), slider_to_tbox
        no_dags_box = wid.VBox([exo_but, strong_exo_but, mono_but])
        dags_box = wid.VBox([no_x_to_g_but, bdoor_crit_but])
        constraints_box = wid.HBox([no_dags_box, dags_box])
        ate_box = wid.VBox([ate_m_sign, ate_f_sign, ate_sign])
        cmd_box = wid.HBox([print_but, add_but])
        obs_box, self.obs_slider_to_tbox = box_the_sliders(self.obs_sliders)
        # margin and padding are given as a single string with the values in
        # the order of top, right, bottom & left . margin (spacing to other
        # widgets) and padding (spacing between border and widgets inside
        obs_box = wid.HBox([obs_box],
            layout=wid.Layout(border='solid',
                              margin='5px 5px 5px 5px'))
        obs_box = wid.HBox([obs_box, exp_bds_sign])
        exp_box, self.exp_slider_to_tbox = box_the_sliders(self.exp_sliders)
        exp_box = wid.HBox([exp_box],
            layout=wid.Layout(border='solid'))
        exp_margin = wid.VBox([constraints_box, ate_box])
        all_boxes = wid.VBox([
            header,
            cmd_box,
            obs_box,
            wid.HBox([exp_box, exp_margin])
        ])

        def fun(o1b0_m_slider, o1b1_m_slider, px1_m_slider,
                o1b0_f_slider, o1b1_f_slider, px1_f_slider,
                e1b0_m_slider, e1b1_m_slider,
                e1b0_f_slider, e1b1_f_slider,
                pmale_slider
                ):
            self.refresh_bounders_using_slider_vals(
                o1b0_m_slider, o1b1_m_slider, px1_m_slider,
                o1b0_f_slider, o1b1_f_slider, px1_f_slider,
                e1b0_m_slider, e1b1_m_slider,
                e1b0_f_slider, e1b1_f_slider,
                pmale_slider
            )
            # refresh self.pmale using slider value
            self.pmale = pmale_slider

            bds_m = self.bounder_m.get_pns3_bds()
            bds_f = self.bounder_f.get_pns3_bds()
            Plotter.plot_pns3_bds(bds_m=bds_m, bds_f=bds_f)

            if self.only_obs:
                exp_bds_sign.value = "Good choices for Observational " \
                    "Probabilities! :) They imply<br> the following bounds " \
                    "for the Experimental Probabilities:"
                left_bds, right_bds = self.bounder_m.get_exp_probs_bds()
                exp_bds_sign.value +=\
                '<br>%.2f $\leq E_{1|0,m} \leq$ %.2f'\
                    % (left_bds[1, 0], right_bds[1, 0]) +\
                '<br>%.2f $\leq E_{1|1,m}\leq$ %.2f' \
                    % (left_bds[1, 1], right_bds[1, 1])
                exp_bds_sign.value += ', '
                left_bds, right_bds = self.bounder_f.get_exp_probs_bds()
                exp_bds_sign.value +=\
                '<br>%.2f $\leq E_{1|0,f} \leq$ %.2f'\
                    % (left_bds[1, 0], right_bds[1, 0]) +\
                '<br>%.2f $\leq E_{1|1,f}\leq$ %.2f' \
                    % (left_bds[1, 1], right_bds[1, 1])
            ate_f = self.bounder_f.get_ate()
            ate_m = self.bounder_m.get_ate()
            if ate_m is not None and ate_f is not None:
                ate = ate_m*self.pmale + ate_f*(1-self.pmale)
                ate_m_sign.value = '$ATE_m=$ %.2f' % ate_m
                ate_f_sign.value = '$ATE_f=$ %.2f' % ate_f
                ate_sign.value = '$ATE=$ %.2f' % ate

        slider_dict = {
            'o1b0_m_slider': o1b0_m_slider,
            'o1b1_m_slider': o1b1_m_slider,
            'px1_m_slider': px1_m_slider,
            'o1b0_f_slider': o1b0_f_slider,
            'o1b1_f_slider': o1b1_f_slider,
            'px1_f_slider': px1_f_slider,
            'e1b0_m_slider': e1b0_m_slider,
            'e1b1_m_slider': e1b1_m_slider,
            'e1b0_f_slider': e1b0_f_slider,
            'e1b1_f_slider': e1b1_f_slider,
            'pmale_slider': pmale_slider
        }
        plot = wid.interactive_output(fun, slider_dict)
        # interactive_plot.layout.height = '800px'
        self.refresh_slider_colors(obs_green=True)
        display(all_boxes, plot)
