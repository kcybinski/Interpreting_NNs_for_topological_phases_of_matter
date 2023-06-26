#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:55:05 2022

@author: k4cp3rskiii
"""

import plotly.io as pio
import os

import numpy as np
import pandas as pd
import plotly.express as px
import pathlib
from Loaders import Importer
pio.renderers.default = 'browser'

def plot_noisy(noise_pow, M=50, loop_ind=None):
    def_batch_size=250
    
    ds_path = pathlib.Path(f"./Datasets/M={M}/5000-1000-500/")
    ds = Importer(ds_path, def_batch_size)
    test_loader = ds.get_test_loader()
    test_X = test_loader.dataset.X.numpy()
    test_V = np.linspace(0.002, 2.002, 500)
    
    test_norms = []
    test_v_tab = []
    test_state_ind = []
    for state_ind in range(0, M):
        for v_ind, v in enumerate(test_V):
            vec_1 = np.sqrt(test_X[1, :, state_ind]) + np.random.normal(0, noise_pow, size=M)
            vec_2 = np.sqrt(test_X[v_ind, :, state_ind]) + np.random.normal(0, noise_pow, size=M)
            test_norms.append(np.dot(vec_1, vec_2))
            test_v_tab.append(v)
            test_state_ind.append(state_ind)
    
    df_test = pd.DataFrame({"norm": test_norms, "v": test_v_tab, "state_ind" : test_state_ind})
    
    fig = px.line(df_test, x='v', y='norm', color='state_ind', markers=True, title=f"Noise_pow = {noise_pow} | M = {M}")
    
    if not os.path.exists("gaussian_noise_testing"):
        os.mkdir("gaussian_noise_testing")
    
    if loop_ind is not None:
        fig.write_image(f"gaussian_noise_testing/M={M}_|_ind={loop_ind}_|_noise={noise_pow}.jpg", width=1920, height=1080, engine='kaleido')
    else:
        fig.write_image(f"gaussian_noise_testing/M={M}_|_noise={noise_pow}.jpg", width=1920, height=1080, engine='kaleido')
    
    # fig.show()
    
if __name__ == "__main__":
    # noise_pow = 0.02
    for ind, noise_pow in enumerate([0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.20]):
        plot_noisy(noise_pow, M=50, loop_ind=ind)
    
    for ind, noise_pow in enumerate([0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.20]):
        plot_noisy(noise_pow, M=80, loop_ind=ind)