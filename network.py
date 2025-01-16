import numpy as np



def create_network(num_KC,w1_init,w2_init,w3_init,w5_init,w6_init, w7_init, w8_init, w1_connection,w2_connection):
    '''

    Creates KC matrix, MBON_alpha, MBON_gamma, DAN_gamma and initializes wKC_aplha and wKC_gamma

    :param num_KC: size of KC population (int)
    :param w1_init: initial weight for all active connection (num)
    :param w2_init: initial weight for all active connection (num)
    :param w1_connection: number of active connections (int)
    :param w2_connection: number of active connections (int)

    :return: KC, wKC_alpha, wKC_gamma, MBON_gamma, MBON_gamma, DAN, DAN_gamma_alpha

    '''

    # create network structure
    KC = np.zeros((num_KC))
    MBONgamma = 0
    MBONalpha = 0
    MBONgamma5 = 0
    DAN = 0
    DAN_gamma_alpha = 0
    DAN_CT = 0

    wKC_MBONalpha = np.zeros((num_KC))
    wKC_MBONgamma = np.zeros((num_KC))
    wKC_MBONgamma5 = np.zeros((num_KC))
    wMBONalpha_DAN = 0
    wMBONgamma_DAN = 0



    # initialize weights
    wKC_MBONalpha[np.random.choice(num_KC,w1_connection,replace=False)] = w1_init
    wKC_MBONgamma[np.random.choice(num_KC, w2_connection, replace=False)] = w2_init
    wKC_MBONgamma5[np.random.choice(num_KC, w2_connection, replace=False)] = w6_init
    wMBONalpha_DAN = w3_init
    wMBONgamma_DAN = w5_init
    wMBONg1_MBONg5 = w7_init
    wMBONg5_MBONa2 = w8_init


    return KC, wKC_MBONalpha, wKC_MBONgamma, wKC_MBONgamma5,wMBONg1_MBONg5,wMBONg5_MBONa2, wMBONalpha_DAN,wMBONgamma_DAN, MBONalpha, MBONgamma, MBONgamma5, DAN, DAN_gamma_alpha, DAN_CT


