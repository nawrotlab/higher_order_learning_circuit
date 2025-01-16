import numpy as np
from input import network_input



def sigmoid_DAN(x,max_activation):
    z = 2000*np.exp(-x * 2)
    sig_DAN = (1 / (1 + z)) * max_activation
    return sig_DAN

def sigmoid_CT(x,max_activation):
    z = 10000000*np.exp(-x * 0.5)
    sig_CT = (1 / (1 + z)) * max_activation
    return sig_CT
'''
x = np.arange(0,40)
plt.plot(x,sigmoid_CT(x,40))
plt.show()
'''

def single_trials_sim(num_KC,KC_baseline,odor_activation,odor,w1_init,w2_init,w3_init,w1_connection,w2_connection,KC, wKC_MBONalpha, wKC_MBONgamma,wKC_MBONgamma5, wMBONg1_MBONg5, wMBONg5_MBONa2, wMBONalpha_DAN,wMBONgamma_DAN,MBONalpha, MBONgamma, DAN,DAN_gamma_alpha, CT, lr_wKC_gamma,lr_wKC_alpha,CT_factor,DAN_activation,DAN_activation_gamma_alpha,ko_trial=False):


    '''

    simulates a single trial

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
    :param KC: (np.array) KC activation pattern
    :param wKC_MBONalpha: (np.array) synaptic weight matrix
    :param wKC_MBONgamma: (np.array) synaptic weight matrix
    :param wMBONalpha_DAN: (np.array) synaptic weight matrix
    :param wMBONgamma_MBONalpha: (np.array) synaptic weight matrix
    :param MBONalpha: (num) MBON activation
    :param MBONgamma: (num) MBON activation
    :param DAN: (num) DAN activation
    :param DAN_gamma_alpha: (num) DAN activation
    :param lr_wKC_gamma: (num) learning rate
    :param lr_wKC_alpha: (num) learning rate
    :param DAN_activation: (num) DAN spike rate
    :param DAN_activation_a+: (num) DAN spike rate

    :return:
    KC (np.array) KC spike rates
    DAN (np.array) DAN spike rates
    DAN_gamma_alpha (np.array) DAN spike rates
    MBONalpha (np.array) MBON spike rates
    MBONgamma (np.array) MBON spike rates
    weights (dict) all synaptic weights

    '''

    # compute KC odor activation
    odors = network_input(num_KC, KC_baseline, odor_activation)


    if ko_trial == 'KCall':
        KC = odors[odor]
        KC_output = np.zeros((num_KC))
    else:
        KC = odors[odor]
        KC_output = odors[odor]


    # compute MBON output
    if ko_trial == 'MBONgamma1':
        MBONgamma = 0
        MBONalpha = np.dot(KC_output, wKC_MBONalpha)
    elif ko_trial == 'MBONalpha2':
        MBONalpha = 0
        MBONgamma = np.dot(KC_output, wKC_MBONgamma)
    else:
        MBONgamma = np.dot(KC_output, wKC_MBONgamma)
        MBONalpha = np.dot(KC_output, wKC_MBONalpha)

    MBONgamma5 = np.dot(KC_output, wKC_MBONgamma5)

    # inhibition from MBONgamma1 to MBONgamma5
    MBONgamma5 -= MBONgamma * wMBONg1_MBONg5
    if MBONgamma5 < 0:
        MBONgamma5 = 0

    # inhibition from MBONgamma5 to MBONalpha'2
    MBONalpha -= MBONgamma5 * wMBONg5_MBONa2
    if MBONalpha < 0:
        MBONalpha = 0

    # compute DAN activation
    if ko_trial=='PPLgamma1':
        DAN = 0
    else:
        DAN = DAN_activation + np.dot(MBONalpha, wMBONalpha_DAN) - np.dot(MBONgamma, wMBONgamma_DAN)
        DAN = sigmoid_DAN(DAN,10)
    if DAN < 0:
        DAN = 0

    # compute DAN_gamma_alpha activation
    if ko_trial == 'PPLgamma2alpha1':
        DAN_gamma_alpha = 0
        CT = CT
        CT_act = sigmoid_CT(CT, 40)
    else:
        DAN_gamma_alpha = DAN_activation_gamma_alpha
        CT = CT + (DAN_gamma_alpha) * CT_factor
        CT_act = sigmoid_CT(CT, 40)
    if DAN_gamma_alpha < 0:
        DAN_gamma_alpha = 0




    # plasticity rules
    active_KC = np.where(KC > KC_baseline)[0]

    for neuron in active_KC:
        # plasticity wKC_MBONgamma (coincidence of odor and reinforcement triggers plasticity)
        if wKC_MBONgamma[neuron] > (0 + lr_wKC_gamma * DAN):
            wKC_MBONgamma[neuron] -= lr_wKC_gamma * DAN
        else:
            wKC_MBONgamma[neuron] = 0

        # Plasticity wKC_MBONalpha (coincidence of odor (KC activation) and reinforcement (alpha DAN activation) triggers plasticity (facilitation))
        if wKC_MBONalpha[neuron] < (2 - lr_wKC_alpha * (DAN_gamma_alpha- CT_act)):
            wKC_MBONalpha[neuron] = max(0, wKC_MBONalpha[neuron] - lr_wKC_alpha * (DAN_gamma_alpha - CT_act))
            #wKC_MBONalpha[neuron] += lr_wKC_alpha * DAN_gamma_alpha
        else:
            wKC_MBONalpha[neuron] = 2

            

    weights = {'wKC_alpha': wKC_MBONalpha,'wKC_MBONgamma': wKC_MBONgamma, 'wMBONalpha_DAN': wMBONalpha_DAN}

    return KC, DAN, DAN_gamma_alpha, CT, MBONalpha, MBONgamma, MBONgamma5, weights


