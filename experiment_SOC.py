import numpy as np
from single_trial import single_trials_sim
from network import create_network
from input import network_input
#from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from run import run_sim


con_timing = ['foc','soc','test']
con_neuron = ['control','KCall','MBONgamma1','MBONalpha2','PPLgamma1','PPLgamma2alpha1']

#con_timing = ['soc']
#con_neuron = ['MBONalpha2']

test_all_CS2 = []
test_all_ctl = []

for i in range(len(con_timing)):
                for e in range(len(con_neuron)):
                    KC_rate,\
                    DAN_rate,\
                    DAN_alpha_rate,\
                    MBONalpha_rate,\
                    MBONgamma_rate,\
                    MBONgamma5_rate,\
                    KC_MBONalpha_weights,\
                    KC_MBONgamma_weights,\
                    MBONalpha_DAN_weights,test_CS1 ,\
                    test_CS2 ,\
                    test_ctl=run_sim(trials_phase1=3,
                       trials_phase2=3,
                       num_KC=2000,
                       KC_baseline=0,
                       odor_activation=10,
                       # KC spike rate should not exceed 10Hz much
                       odor_phase1='odor1',
                       odor_phase2='odor1_2',
                       w1_init=0.0001,
                       w2_init=0.035,
                       # KC:MBONgamma #reasonable real rate should not exceed 80Hz
                       w3_init=0.2,
                       w5_init=0.095,#0.1,
                       w6_init = 0,#0.035,
                       w7_init = 0,#2,
                       w8_init = 0,#0.6,
                       w1_connection=2000,
                       w2_connection=2000,
                       lr_wKC_gamma=0.0009,
                       lr_wKC_alpha=0.0008,
                       DAN_activation=10,
                       DAN_activation_gamma_alpha=20,
                       con_timing=con_timing[i],
                       con_neuron=con_neuron[e])


                    test_all_CS2.append(test_CS2[1])
                    test_all_ctl.append(test_ctl[1])


np.seterr(invalid='ignore')  

behav_bias = np.subtract(np.array(test_all_CS2),np.array(test_all_ctl)) / np.sum([test_all_CS2,test_all_ctl],axis=0)
behav_bias = np.nan_to_num(behav_bias) #in test KO condition two nans bc 0 divided by 0


