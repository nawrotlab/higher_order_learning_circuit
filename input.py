import numpy as np


def network_input(num_KC,KC_baseline, odor_activation):
  '''
  Creates KC odor activation patterns 
  
  :param num_KC: size of the KC population (int)
  :param KC_baseline: baseline activation rate (num)
  :param odor_activation: odor activation rate (num)
  
  :return: odors (baseline + odor) (dict)
  '''

  odor1 = np.zeros((num_KC)) + KC_baseline
  odor1[0:200] = odor_activation

  odor2 = np.zeros((num_KC)) + KC_baseline
  odor2[200:400] = odor_activation

  odor1_2 = np.zeros((num_KC)) + KC_baseline
  odor1_2[0:100] = odor_activation
  odor1_2[200:300] = odor_activation

  odor3 = np.zeros((num_KC)) + KC_baseline #control odor / CS-
  odor3[400:600] = odor_activation  

  odors = {'odor1':odor1,'odor2':odor2,'odor1_2':odor1_2,'odor3':odor3}

  return odors






