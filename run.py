import numpy as np
from single_trial import single_trials_sim
from network import create_network
from input import network_input
import matplotlib.pyplot as plt




def run_sim(trials_phase1,trials_phase2,num_KC,KC_baseline,odor_activation,odor_phase1,odor_phase2,w1_init,w2_init,w3_init,w5_init,w6_init, w7_init,w8_init, w1_connection,w2_connection,lr_wKC_gamma,lr_wKC_alpha,CO_factor,DAN_activation=False,DAN_activation_gamma_alpha=False,con_timing=False,con_neuron=False):
    '''

    runs the network simulation for any number of trials for N model instance

    :param trials: (int) number of trials
    :param num_KC: (int) number of KC neurons
    :param KC_baseline: (num) KC baseline spike rate
    :param odor_activation: (num) KC odor activation rate
    :param odor: (iterable) odor activation pattern
    :param w1_init: (num) initialization weights wKC_MBONalpha
    :param w2_init: (num) initialization weights wKC_MBONgamma
    :param w3_init: (num) initialization weights wMBONalpha_DAN
    :param w4_init: (num) initialization weights wMBONgamma_MBONalpha
    :param w1_connection: (int) number of random KC_MBONalpha connections
    :param w2_connection: (int) number of random KC_MBONalpha connections
    :param lr_wKC_gamma: (num) learning rate
    :param DAN_activation: (num) DAN spike rate

    :param w6_init: (num) initialization weights wKC_MBONgamma5
    :param w7_init: (num) initialization weights wMBONgamma1_MBONgamma5
    :param w8_init: (num) initialization weights wMBONgamma5_MBONalpha

    :return: DAN_rate, MBONalpha_rate, MBONgamma_rate, KC_MBONalpha_weights, KC_MBONgamma_weights, MBONgamma_MBONalpha_weights, MBONalpha_DAN_weights
            np.arrays with trials + 1 elements/columns, the first element/column contains the initial value
    '''

    trials = trials_phase1 + trials_phase2

    # save DAN , KC, MBON rates and all weights for each trial
    KC_rate = np.zeros((num_KC,trials + 1))
    DAN_rate = np.zeros(trials + 1)
    DAN_gamma_alpha_rate = np.zeros(trials + 1)
    DAN_CO = np.zeros(trials + 1)
    MBONalpha_rate = np.zeros(trials + 1)
    MBONgamma_rate = np.zeros(trials + 1)
    MBONgamma5_rate = np.zeros(trials + 1)

    KC_MBONalpha_weights = np.zeros((num_KC,trials + 1))
    KC_MBONgamma_weights = np.zeros((num_KC, trials + 1))
    MBONalpha_DAN_weights = np.zeros(trials + 1)



    # create and initialize network
    KC, wKC_MBONalpha, wKC_MBONgamma, wKC_MBONgamma5,wMBONg1_MBONg5,wMBONg5_MBONa2,wMBONalpha_DAN, wMBONgamma_DAN, MBONalpha, MBONgamma, MBONgamma5,DAN, DAN_gamma_alpha, CO = create_network(num_KC=num_KC,w1_init=w1_init,w2_init=w2_init,w3_init=w3_init,w5_init=w5_init,w6_init=w6_init,w7_init=w7_init,w8_init=w8_init,w1_connection=w1_connection,w2_connection=w2_connection)

    # save initial values
    KC_rate[:,0] = KC
    DAN_rate[0] = DAN
    DAN_gamma_alpha_rate[0] = DAN_gamma_alpha
    DAN_CO[0] = CO
    MBONalpha_rate[0] = MBONalpha
    MBONgamma_rate[0] = MBONgamma
    MBONgamma5_rate[0] = MBONgamma5
    KC_MBONalpha_weights[:, 0] = wKC_MBONalpha
    KC_MBONgamma_weights[:, 0] = wKC_MBONgamma
    MBONalpha_DAN_weights[0] = wMBONalpha_DAN

    #test model for odors for innate valence towards odors
    odors = network_input(num_KC,KC_baseline,odor_activation)
    KC_CS1 = odors['odor1']
    KC_CS2 = odors['odor2']
    KC_ctl = odors['odor3']

    test_CS1 = np.dot(KC_CS1, wKC_MBONgamma)
    test_CS2 = np.dot(KC_CS2, wKC_MBONgamma)
    test_ctl = np.dot(KC_ctl, wKC_MBONgamma)


    if con_timing=='foc':
        ko_trial=con_neuron
    else:
        ko_trial=False

    for i, trial in enumerate(np.arange(1,trials_phase1+1)):

        KC, DAN, DAN_gamma_alpha, CO, MBONalpha, MBONgamma, MBONgamma5, weights = single_trials_sim(num_KC=num_KC, KC_baseline=KC_baseline,
                                                                   odor_activation=odor_activation, odor=odor_phase1,
                                                                   w1_init=w1_init, w2_init=w2_init, w3_init=w3_init,
                                                                   w1_connection=w1_connection,
                                                                   w2_connection=w2_connection, KC=KC,
                                                                   wKC_MBONalpha=wKC_MBONalpha,
                                                                   wKC_MBONgamma=wKC_MBONgamma,
                                                                   wKC_MBONgamma5 = wKC_MBONgamma5,
                                                                   wMBONg1_MBONg5 = wMBONg1_MBONg5,
                                                                   wMBONg5_MBONa2 = wMBONg5_MBONa2,
                                                                   wMBONalpha_DAN=wMBONalpha_DAN,
                                                                   wMBONgamma_DAN=wMBONgamma_DAN,
                                                                   MBONalpha=MBONalpha, MBONgamma=MBONgamma, DAN=DAN,DAN_gamma_alpha=DAN_gamma_alpha, CO=CO,
                                                                   lr_wKC_gamma=lr_wKC_gamma,lr_wKC_alpha=lr_wKC_alpha, CO_factor=CO_factor,
                                                                   DAN_activation=DAN_activation,DAN_activation_gamma_alpha=DAN_activation_gamma_alpha,
                                                                   ko_trial=ko_trial)

        # save values
        KC_rate[:,trial] = KC
        DAN_rate[trial] = DAN
        DAN_gamma_alpha_rate[trial] = DAN_gamma_alpha
        DAN_CO[trial] = CO
        MBONalpha_rate[trial] = MBONalpha
        MBONgamma_rate[trial] = MBONgamma
        MBONgamma5_rate[trial] = MBONgamma5
        KC_MBONalpha_weights[:, trial] = wKC_MBONalpha
        KC_MBONgamma_weights[:, trial] = wKC_MBONgamma
        MBONalpha_DAN_weights[trial] = wMBONalpha_DAN

    # test CS1, CS2 and control odors after FOC training
    if con_timing == 'test':
        if con_neuron == 'KCall':
            KC_CS1 = np.zeros((num_KC))
            KC_CS2 = np.zeros((num_KC))
            KC_ctl = np.zeros((num_KC))
        elif con_neuron == 'PPL1gamma1':
            DAN = 0
        elif con_neuron == 'PPL1gamma2alpha1':
            DAN_gamma_alpha = 0
        elif con_neuron == 'MBONalpha2':
            MBONalpha = 0

    MBONgamma_act1 = np.dot(KC_CS1, wKC_MBONgamma)
    MBONgamma_act2 = np.dot(KC_CS2, wKC_MBONgamma)
    MBONgamma_ctl = np.dot(KC_ctl, wKC_MBONgamma)

    if (con_neuron == 'MBONgamma1') and (con_timing == 'test'):
        MBONgamma_act1 = 0
        MBONgamma_act2 = 0
        MBONgamma_ctl = 0

    test_CS1 = np.append(test_CS1, MBONgamma_act1)
    test_CS2 = np.append(test_CS2, MBONgamma_act2)
    test_ctl = np.append(test_ctl, MBONgamma_ctl)


    if con_timing=='soc':
        ko_trial=con_neuron
    else:
        ko_trial=False

    for i, trial in enumerate(np.arange(trials_phase1+1, trials+ 1)):

        KC, DAN, DAN_gamma_alpha, CO, MBONalpha, MBONgamma, MBONgamma5, weights = single_trials_sim(num_KC=num_KC, KC_baseline=KC_baseline,
                                                                   odor_activation=odor_activation, odor=odor_phase2,
                                                                   w1_init=w1_init, w2_init=w2_init, w3_init=w3_init,
                                                                   w1_connection=w1_connection,
                                                                   w2_connection=w2_connection, KC=KC,
                                                                   wKC_MBONalpha=wKC_MBONalpha,
                                                                   wKC_MBONgamma=wKC_MBONgamma,
                                                                   wKC_MBONgamma5=wKC_MBONgamma5,
                                                                   wMBONg1_MBONg5=wMBONg1_MBONg5,
                                                                   wMBONg5_MBONa2=wMBONg5_MBONa2,
                                                                   wMBONalpha_DAN=wMBONalpha_DAN,
                                                                   wMBONgamma_DAN=wMBONgamma_DAN,
                                                                   MBONalpha=MBONalpha, MBONgamma=MBONgamma, DAN=DAN,DAN_gamma_alpha=DAN_gamma_alpha,CO=CO,
                                                                   lr_wKC_gamma=lr_wKC_gamma,lr_wKC_alpha=lr_wKC_alpha,CO_factor=CO_factor,
                                                                   DAN_activation=False,DAN_activation_gamma_alpha=False,
                                                                   ko_trial=ko_trial)


        # save values
        KC_rate[:,trial] = KC
        DAN_rate[trial] = DAN
        DAN_gamma_alpha_rate[trial] = DAN_gamma_alpha
        DAN_CO[trial] = CO
        MBONalpha_rate[trial] = MBONalpha
        MBONgamma_rate[trial] = MBONgamma
        MBONgamma5_rate[trial] = MBONgamma5
        KC_MBONalpha_weights[:, trial] = wKC_MBONalpha
        KC_MBONgamma_weights[:, trial] = wKC_MBONgamma
        MBONalpha_DAN_weights[trial] = wMBONalpha_DAN


    #test CS1, CS2 and control odors after training
    if con_timing == 'test':
        if con_neuron == 'KCall':
            KC_CS1 = np.zeros((num_KC))
            KC_CS2 = np.zeros((num_KC))
            KC_ctl = np.zeros((num_KC))
        elif con_neuron == 'PPL1gamma1':
            DAN=0
        elif con_neuron == 'PPL1gamma2alpha1':
            DAN_gamma_alpha = 0
        elif con_neuron == 'MBONalpha2':
            MBONalpha=0

    MBONgamma_act1 = np.dot(KC_CS1, wKC_MBONgamma)
    MBONgamma_act2 = np.dot(KC_CS2, wKC_MBONgamma)
    MBONgamma_ctl = np.dot(KC_ctl, wKC_MBONgamma)

    if (con_neuron == 'MBONgamma1') and (con_timing == 'test'):
        MBONgamma_act1 = 0
        MBONgamma_act2 = 0
        MBONgamma_ctl = 0

    test_CS1 = np.append(test_CS1,MBONgamma_act1)
    test_CS2 = np.append(test_CS2, MBONgamma_act2)
    test_ctl = np.append(test_ctl, MBONgamma_ctl)


    return KC_rate ,DAN_rate, DAN_gamma_alpha_rate, DAN_CO, MBONalpha_rate, MBONgamma_rate, MBONgamma5_rate, KC_MBONalpha_weights, KC_MBONgamma_weights,  MBONalpha_DAN_weights,test_CS1 ,test_CS2, test_ctl


KC_rate, DAN_rate, DAN_gamma_alpha_rate, DAN_CO, MBONalpha_rate, MBONgamma_rate, MBONgamma5_rate, KC_MBONalpha_weights, KC_MBONgamma_weights, MBONalpha_DAN_weights,test_CS1 ,test_CS2,test_ctl = run_sim(trials_phase1=3,
                                                                                                                                                                               trials_phase2=3,
                                                                                                                                                                              num_KC=2000,
                                                                                                                                                                              KC_baseline=0,
                                                                                                                                                                              odor_activation=10,

                                                                                                                                                                              odor_phase1='odor1',
                                                                                                                                                                              odor_phase2='odor1_2',
                                                                                                                                                                              w1_init=0.035,
                                                                                                                                                                              w2_init=0.035,

                                                                                                                                                                              w3_init=0.1,
                                                                                                                                                                              w5_init=0.15,

                                                                                                                                                                              w6_init=0.035,

                                                                                                                                                                              w7_init=2,

                                                                                                                                                                              w8_init=2,

                                                                                                                                                                              w1_connection=2000,
                                                                                                                                                                              w2_connection=2000,
                                                                                                                                                                              lr_wKC_gamma=0.0009,
                                                                                                                                                                              lr_wKC_alpha=0.0009,
                                                                                                                                                                              CO_factor=0.7,
                                                                                                                                                                              DAN_activation=20,

                                                                                                                                                                              DAN_activation_gamma_alpha=20)
                                                                                                                                                                              #con_timing='soc',
                                                                                                                                                                              #con_neuron='MBONgamma1')


cm=1/2.54

fig_test, ((ax0)) = plt.subplots(1, 1,figsize=(12*cm,10*cm), sharex=True,dpi=600)
ax0.set_xlim(-0.5, 10.5)
ax0.set_ylabel('γ1-MBON activation rate (Hz)',fontsize=10)
ax0.set_xticks(np.array([0,1,2,4,5,6,8,9,10]), ['CS–','CS1','CS2','CS–','CS1','CS2','CS–','CS1','CS2'],fontsize=10)
ax0.set_yticks([0,40,80])
ax0.bar([0,1,2],[test_ctl[0],test_CS1[0],test_CS2[0]],color=['#E15500','#DB8E00','#887800'])
ax0.bar([4,5,6],[test_ctl[1],test_CS1[1],test_CS2[1]],color=['#E15500','#DB8E00','#887800'],hatch='/')
ax0.bar([8,9,10],[test_ctl[2],test_CS1[2],test_CS2[2]],color=['#E15500','#DB8E00','#887800'],hatch='X')
ax0.text(-0.5,75,'before learning',fontsize=10)
ax0.text(4,75,'after FOC',fontsize=10)
ax0.text(8,75,'after SOC',fontsize=10)
ax0.set_ylim(0, 80)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('Figure_Model_learning_effect_new.png')
plt.show()

fig_w, ((ax0, ax1, ax2,ax3)) = plt.subplots(4, 1,figsize=(15*cm,15*cm), sharex=True,dpi=600)
ax0.set_title('Population average synaptic weights2',fontsize=12)
ax0.set_xlim(-1, 7)
ax0.set_ylabel('KC_MBONα`2\n(w1)',fontsize=10)
ax0.set_xticks(np.arange(len(DAN_rate)), ['initial\nvalue', 1, 2, 3, 4, 5, 6])
ax0.plot(np.arange(len(KC_MBONalpha_weights[0])),np.mean(KC_MBONalpha_weights, axis=0), 'x',color= [0,0.58,0.26])
ax1.set_ylabel('KC_MBONγ1\n(w2)',fontsize=10)
ax1.plot(np.arange(len(KC_MBONgamma_weights[0])),np.mean(KC_MBONgamma_weights, axis=0), 'x',color= [0,0.58,0.26])
ax2.set_ylabel('MBONα`2_\nDANγ1(w3)',fontsize=10)
ax2.plot(np.arange(len(MBONalpha_DAN_weights)), MBONalpha_DAN_weights, 'x',color=[0.93,0.11,0.14])
ax3.set_ylabel('MBONγ1_\nDANγ1(w5)',fontsize=10)
ax3.plot(np.arange(len(MBONalpha_DAN_weights)), MBONalpha_DAN_weights, 'x',color=[0.93,0.11,0.14])
ax3.set_xlabel('trial')
plt.tight_layout()
plt.show()

#figure for activation
fig_w, ((ax0, ax1, ax2,ax3,ax4, ax5)) = plt.subplots(6, 1,figsize=(15*cm,18*cm), sharex=True,dpi=600)
ax0.set_title('Population average neuron activation2',fontsize=12)
ax0.set_xlim(-1, 7)
ax0.set_ylabel('DAN gamma1',fontsize=10)
ax0.set_xticks(np.arange(len(DAN_rate)), ['initial\nvalue', 1, 2, 3, 4, 5, 6])
ax0.plot(np.arange(len(DAN_rate)),DAN_rate, 'x',color= [0,0.58,0.26])

ax1.set_ylabel('DAN gamma\nalpha',fontsize=10)
ax1.plot(np.arange(len(DAN_gamma_alpha_rate)),DAN_gamma_alpha_rate, 'x',color= [0,0.58,0.26])

ax2.set_ylabel('DAN CO',fontsize=10)
ax2.plot(np.arange(len(MBONalpha_DAN_weights)), DAN_CO, 'x',color=[0.93,0.11,0.14])

ax3.set_ylabel('MBONgamma',fontsize=10)
ax3.plot(np.arange(len(MBONalpha_DAN_weights)), MBONgamma_rate, 'x',color=[0.93,0.11,0.14])

ax4.set_ylabel('MBONalpha',fontsize=10)
ax4.plot(np.arange(len(MBONalpha_DAN_weights)), MBONalpha_rate, 'x',color=[0.93,0.11,0.14])

ax5.set_ylabel('MBONgamma5',fontsize=10)
ax5.plot(np.arange(len(MBONalpha_DAN_weights)), MBONgamma5_rate, 'x',color=[0.93,0.11,0.14])


ax5.set_xlabel('trial')
plt.tight_layout()
plt.show()


#  #addition Masti Figure to test odors before and after training
#
# fig_test, ((ax0)) = plt.subplots(1, 1,figsize=(8*cm,7*cm), sharex=True,dpi=600)
# ax0.set_title('Tests for Odor CS1 and CS2',fontsize=12)
# ax0.set_xlim(-0.5, 3.5)
# ax0.set_ylabel('γ1-MBON activation rate',fontsize=10)
# ax0.set_xticks(np.arange(4), ['innate\nCS1','innate\nCS2','learned\nCS1','learned\nCS2'])
# ax0.set_yticks([0,40,80])
# ax0.bar([0,2],test_CS1,color=[0.3,0.3,0.3])
# ax0.bar([1,3],test_CS2,color=[0.7,0.7,0.7])
# ax0.set_ylim(0, 80)
# ax0.spines['right'].set_visible(False)
# ax0.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('Figure_Model_learning_effect.pdf')
# plt.show()
#
#
# #parallelization
# #sample = np.arange(1)
# #Parallel(n_jobs=len(sample))(delayed(run_sim)(trials=1,num_KC=2000,KC_baseline=1,odor_activation=10,odor='odor1',w1_init=0.5,w2_init=0.5,w3_init=0.5,w4_init=0.5,w1_connection=1500,w2_connection=1500,lr_wKC_gamma=0.1,DAN_activation=1) for i in sample)
#
# #Add new polished Figure to show learning effect
# # New Figures also covers MBON activation rates for CS–


