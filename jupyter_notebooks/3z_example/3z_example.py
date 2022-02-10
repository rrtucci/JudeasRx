from MultiBounder_ana import MultiBounder_ana
import numpy as np
import pandas as pd

df = pd.read_csv('3z_example.csv')
print(df, '\n')

alp_y0_y1 = np.array([[.5, -.4], [.2, .1]])
mba = MultiBounder_ana.create_from_file('3z_example.csv',
                                alp_y0_y1=alp_y0_y1,
                                only_obs=False,
                                exogeneity=False,
                                strong_exo=False,
                                monotonicity=False)
mba.plot_bds()
mba.plot_both_ATE()
