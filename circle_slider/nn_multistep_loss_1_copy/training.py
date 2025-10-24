
#!/usr/bin/env python
import helpers.learn_modules as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time as time


if __name__== "__main__":
    N_states = 4
    num_obs = 100

    data = np.load('../data/random/Fx_5_1900038_combined.npy')
    data = data[:, :3,2:]
    print("Data shape:", data.shape)
    lrn = lm.L3(N_x = N_states, N_z = 32, N_e = num_obs, epochs = 300, batch_size = 8192)
    t0 = time.time()
    lrn.learn(data)
    print("Time elapsed:")
    print(time.time()-t0)