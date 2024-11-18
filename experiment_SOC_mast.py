import numpy as np
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
                    KC_rate, DAN_rate, DAN_gamma_alpha_rate, DAN_CO, MBONalpha_rate, MBONgamma_rate, MBONgamma5_rate, KC_MBONalpha_weights, KC_MBONgamma_weights, MBONalpha_DAN_weights, test_CS1, test_CS2, test_ctl = run_sim(
                        trials_phase1=3,
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
                        DAN_activation_gamma_alpha=20,
                        con_timing = con_timing[i],
                        con_neuron = con_neuron[e])




                    test_all_CS2.append(test_CS2[2])
                    test_all_ctl.append(test_ctl[2])


np.seterr(invalid='ignore')  # Supress/hide the warning


behav_bias = np.subtract(np.array(test_all_CS2),np.array(test_all_ctl)) / np.sum([test_all_CS2,test_all_ctl],axis=0)
behav_bias = np.nan_to_num(behav_bias) #in test KO condition two nans bc 0 divided by 0

#Figure 

cm=1/2.54
afont= {'family' : 'Arial',
        'size'   : 12}

bfont= {'family' : 'Arial',
        'size'   : 10}

cfont= {'family' : 'Arial',
        'size'   : 8}

dfont= {'family' : 'Arial',
        'size'   : 6}

y = 0.1 #value for learned behavior (downregulation of y1-MBON activation levels)
n = 0.9 #value for impaired learning


xval = ['ctrl','KC','MBONγ1\npedc>α/β','MBONα´2','PPL1\nγ1pedc','PPL1\nγ2α´1']


grey = [0.5,0.5,0.5] #ctrl
blue= [0.11,0.46,0.74] #DAN
green= [0,0.58,0.26] #KC
red = [0.93,0.11,0.14] #MBON

blue= '#5c8aa6ff' #DAN
green= '#7aab61ff' #KC
red = '#bc4752ff' #MBON

xcol = [grey,green,red,red,blue,blue]

fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(12*cm,4.7*cm),sharey=True,sharex=True)

ax1.set_title('Block during training 1',**cfont)
ax2.set_title('Block during training 2',**cfont)
ax3.set_title('Block during test',**cfont)
ax1.axhline(y=0, color='k', linestyle='-',linewidth=0.5,zorder=1)
ax2.axhline(y=0, color='k', linestyle='-',linewidth=0.5,zorder=1)
ax3.axhline(y=0, color='k', linestyle='-',linewidth=0.5,zorder=1)
ax1.bar(np.arange(0,len(xval)),behav_bias[0:6],color=xcol,width=0.7,edgecolor = xcol)
ax2.bar(np.arange(0,len(xval)),behav_bias[6:12],color=xcol,width=0.7,edgecolor = xcol)
ax3.bar(np.arange(0,len(xval)),behav_bias[12:18],color=xcol,width=0.7,edgecolor = xcol)
for i in np.arange(0,len(xval)):
    ax1.text(i,0.03,u'\u2713',horizontalalignment='center')
for i in np.arange(0,len(xval)):
    ax3.text(i,0.03,u'\u2713',horizontalalignment='center')
model = '1' #1 = without extra mbon, 2 = with extra mbon
if model == '2':
    for i in np.arange(0, len(xval)):
        ax2.text(i, 0.03, u'\u2713', horizontalalignment='center')
elif model == '1':
    for i in [0,1,3,4,5]:
        ax2.text(i, 0.03, u'\u2713', horizontalalignment='center')
        ax2.text(2, 0.03, u'\u2717', horizontalalignment='center',size=8)

xp = 0.05
lab = ['A','B','C']
for i in np.arange(1,4):
    eval("ax" + str(i)).set_position([xp,0.15,0.3,0.6])
    eval("ax" + str(i)).spines['right'].set_visible(False)
    eval("ax" + str(i)).spines['top'].set_visible(False)
    eval("ax" + str(i)).set_xticks(np.arange(0,len(xval)))
    eval("ax" + str(i)).set_xticklabels(xval,**dfont,rotation=90,linespacing=0.9)
    eval("ax" + str(i)).tick_params(axis='x', pad=-0.5)
    eval("ax" + str(i)).set_yticks([-0.3,-0.2,-0.1,0,0.1])
    xp +=0.315
for i in np.arange(2,4):
    eval("ax" + str(i)).spines['right'].set_visible(False)
    eval("ax" + str(i)).spines['top'].set_visible(False)
    eval("ax" + str(i)).set_yticklabels([])
ax1.set_yticklabels(['','-0.2','-0.1','0',''],rotation=90,**dfont)
ax1.set_ylabel('Behavioral bias towards\nCS2 after SOC',**dfont)
ax1.set_position([0.11,0.23,0.28,0.6])
ax2.set_position([0.41,0.23,0.28,0.6])
ax3.set_position([0.71,0.23,0.28,0.6])
plt.savefig('Figure_Model_SOC_1.png',dpi=1200)
plt.show()

