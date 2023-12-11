# Visualize different odors and odor mixtures as representations in the KC layer


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from input import network_input
odors = network_input(num_KC=2000, KC_baseline=0, odor_activation=10)

KC_all = np.zeros((len(odors),len(odors['odor1'])))
KC_all[0,:] = odors['odor1']
KC_all[1,:] = odors['odor2']
KC_all[2,:] = odors['odor1_2']
KC_all[3,:] = odors['odor3']

cm = 1/2.54
fig = plt.figure(figsize = (13*cm, 4*cm), dpi = 600)
ax = sns.heatmap(KC_all,cmap = 'Greys',cbar=False,linewidth=0)
ax.axhline(1,color=[0.5,0.5,0.5],linewidth=1)
ax.axhline(2,color=[0.5,0.5,0.5],linewidth=1)
ax.axhline(3,color=[0.5,0.5,0.5],linewidth=1)

# Drawing the frame

ax.axhline(0,color=[0.5,0.5,0.5],linewidth=1)
ax.axhline(4,color=[0.5,0.5,0.5],linewidth=1)
ax.axvline(0,color=[0.5,0.5,0.5],linewidth=1)
ax.axvline(2000,color=[0.5,0.5,0.5],linewidth=1)




plt.yticks(np.array([0.5,1.5,2.5,3.5]),['CS1','CS2','CS1+2','CS–'],rotation = 0,fontsize=10)
plt.xticks([0,200,400,600,1000,2000],['0','200','400','600','1000','2000'],rotation = 0,fontsize=8)

plt.xlabel('# KC',fontsize=10)
plt.title('Odor representation in KCs',fontsize=12)
plt.tight_layout()

plt.savefig('Odor_representation.eps')
plt.show()


