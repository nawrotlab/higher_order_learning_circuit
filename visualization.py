import numpy as np
import matplotlib.pyplot as plt

cm=1/2.54
afont= {'family' : 'Arial',
        'size'   : 12}

bfont= {'family' : 'Arial',
        'size'   : 10}

cfont= {'family' : 'Arial',
        'size'   : 8}

y = 0.1 #value for learned behavior (downregulation of y1-MBON activation levels)
n = 0.9 #value for impaired learning


xval = ['ctrl','KC','MBON\nγ1','MBON\nα´2','PPL1\nγ1']

#color coding (according to microcircuit figure of Yazid
grey = [0.5,0.5,0.5] #ctrl
blue= [0.11,0.46,0.74] #DAN
green= [0,0.58,0.26] #KC
red = [0.93,0.11,0.14] #MBON
xcol = [grey,green,red,red,blue]



fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(17.2*cm,6*cm))

#all the titles
fig.suptitle('Activation levels of γ1-MBON towards CS2 after SOC (according to Yazid)',**afont)
ax1.set_title('KO during FOC',**bfont)
ax2.set_title('KO during SOC',**bfont)
ax3.set_title('KO during test',**bfont)

#plot data
ax1.bar(np.arange(0,len(xval)),np.array([y,y,y,y,n/2]),color=xcol,alpha = 0.8,edgecolor = 'k')
ax2.bar(np.arange(0,len(xval)),np.array([y,n,n,n,n]),color=xcol,alpha = 0.8,edgecolor = 'k')
ax3.bar(np.arange(0,len(xval)),np.array([y,n,n,y,y]),color=xcol,alpha = 0.8,edgecolor = 'k')


xp = 0.05
lab = ['A','B','C']
for i in np.arange(1,4):
    eval("ax" + str(i)).set_position([xp,0.15,0.3,0.6])
    eval("ax" + str(i)).spines['right'].set_visible(False)
    eval("ax" + str(i)).spines['top'].set_visible(False)
    eval("ax" + str(i)).set_xticks(np.arange(0,len(xval)))
    eval("ax" + str(i)).set_xticklabels(xval,**cfont)
    eval("ax" + str(i)).set_yticks([0,0.5,1])
    eval("ax" + str(i)).text(-1,1.1,lab[i-1],**afont)
    xp +=0.315

ax1.set_yticklabels(['0','0.5','1'])

for i in np.arange(2,4):
    eval("ax" + str(i)).spines['right'].set_visible(False)
    eval("ax" + str(i)).spines['top'].set_visible(False)
    eval("ax" + str(i)).set_yticklabels([])
    
plt.savefig('Figure_Experiment_example.pdf')
plt.savefig('Figure_Experiment_example.jpg',dpi=800)


