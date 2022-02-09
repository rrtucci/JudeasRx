from Comparer import Comparer
import numpy as np
import pandas as pd

df = pd.read_csv('3z_example.csv')
print(df, '\n')

alp_y0_y1 = np.array([[.5, -.4], [.2, .1]])
cer = Comparer.create_from_file('3z_example.csv',
                                alp_y0_y1=alp_y0_y1,
                                only_obs=False,
                                exogeneity=False,
                                strong_exo=False,
                                monotonicity=False)
cer.plot_bds()
cer.plot_both_ATE()
