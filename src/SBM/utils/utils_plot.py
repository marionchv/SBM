#! /usr/bin/env python3
"""
@author: Marion CHAUVEAU

:On:  October 2022
"""

####################### MODULES #######################
import SBM.utils.utils as ut
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.stats import gaussian_kde
import scipy.io
from sklearn import metrics
import plotly.graph_objects as go
import os
import ot

import scipy.io as sio
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

##########################################################

####################### PLOT STATISTICS #######################

def plot_stats(output,Stats,plot = 'Freq',ma=None):
	if plot=='Freq':
		if ma is None: ma = 1
		fig = plt.figure(figsize = (12,4))
		fig.add_subplot(1,2,1)
		Pears = np.round(np.corrcoef(Stats['Test']['Freq'].flatten(),Stats['Artificial']['Freq'].flatten())[0,1],2)
		plt.plot([],[],'o',markersize = 3, color = 'white',label='1st order statistics\n Pearson: '+str(Pears))
		plt.plot(Stats['Test']['Freq'].flatten(),Stats['Artificial']['Freq'].flatten(),'o',markersize = 3)
		plt.xlabel('Test set')
		plt.ylabel('Artificial set')
		plt.plot([0,ma],[0,ma])
		#plt.xticks(fontsize = 16)
		plt.legend()
		plt.grid()
		plt.title('Artificial VS Test')

		fig.add_subplot(1,2,2)
		Pears = np.round(np.corrcoef(Stats['Test']['Freq'].flatten(),Stats['Train']['Freq'].flatten())[0,1],2)
		plt.plot([],[],'o',markersize = 3, color = 'white',label='1st order statistics\n Pearson: '+str(Pears))
		plt.plot(Stats['Test']['Freq'].flatten(),Stats['Train']['Freq'].flatten(),'o',markersize = 3)
		plt.xlabel('Test set')
		plt.ylabel('Training set')
		plt.plot([0,ma],[0,ma])
		#plt.xticks(fontsize = 16)
		plt.legend()
		plt.grid()
		plt.title('Train VS Test')

		fig.tight_layout()
	
	if plot=='Pair_freq':
		BINS = 40
		if ma is None: ma = 0.4
		fig = plt.figure(figsize = (12,4))
		ax1 = fig.add_subplot(1,2,1)
		ind = np.triu_indices(output['align_mod'].shape[1],1)
		Pears = np.round(np.corrcoef(Stats['Test']['Pair_freq'][ind].flatten(),Stats['Artificial']['Pair_freq'][ind].flatten())[0,1],2)
		plt.plot([],[],'o',markersize = 3, color = 'white',label='Pairwise Corr\n Pearson: '+str(Pears))
		plt.plot(Stats['Test']['Pair_freq'][ind].flatten(),Stats['Artificial']['Pair_freq'][ind].flatten(),'o',markersize = 3)
		plt.xlabel('Test set')
		plt.ylabel('Artificial set')
		#plt.plot([0,ma],[0,ma])
		plt.plot([-ma,ma],[-ma,ma])
		#plt.xticks(fontsize = 16)
		plt.legend()
		plt.grid()
		plt.title('Artificial VS Test')

		ax2 = fig.add_subplot(1,2,2)
		Pears = np.round(np.corrcoef(Stats['Test']['Pair_freq'][ind].flatten(),Stats['Train']['Pair_freq'][ind].flatten())[0,1],2)
		plt.plot([],[],'o',markersize = 3, color = 'white',label='Pairwise Corr\n Pearson: '+str(Pears))
		plt.plot(Stats['Test']['Pair_freq'][ind].flatten(),Stats['Train']['Pair_freq'][ind].flatten(),'o',markersize = 3)
		plt.xlabel('Test set')
		plt.ylabel('Training set')
		#plt.plot([0,ma],[0,ma])
		plt.plot([-ma,ma],[-ma,ma])
		#plt.xticks(fontsize = 16)
		plt.legend()
		plt.grid()
		plt.title('Train VS Test')

		fig.tight_layout()

	if plot=='Corr3':
		if ma is None: ma = 0.1
		fig = plt.figure(figsize = (12,4))
		fig.add_subplot(1,2,1)
		Pears = np.round(np.corrcoef(Stats['Test']['Three_corr'].flatten(),Stats['Artificial']['Three_corr'].flatten())[0,1],2)
		plt.plot([],[],'o',markersize = 3, color = 'white',label='3rd order correlations\n Pearson: '+str(Pears))
		plt.plot(Stats['Test']['Three_corr'].flatten(),Stats['Artificial']['Three_corr'].flatten(),'o',markersize = 3)
		plt.xlabel('Test set')
		plt.ylabel('Artificial set')
		plt.plot([np.amin(Stats['Test']['Three_corr']),np.amax(Stats['Test']['Three_corr'])],
						[np.amin(Stats['Test']['Three_corr']),np.amax(Stats['Test']['Three_corr'])])
		#plt.xticks(fontsize = 14)
		#plt.plot([-ma,ma],[-ma,ma])
		plt.legend(loc = 'upper left')
		plt.grid()
		plt.title('Artificial VS Test')

		fig.add_subplot(1,2,2)
		Pears = np.round(np.corrcoef(Stats['Test']['Three_corr'].flatten(),Stats['Train']['Three_corr'].flatten())[0,1],2)
		plt.plot([],[],'o',markersize = 3, color = 'white',label='3rd order correlations\n Pearson: '+str(Pears))
		plt.plot(Stats['Test']['Three_corr'].flatten(),Stats['Train']['Three_corr'].flatten(),'o',markersize = 3)
		plt.xlabel('Test set')
		plt.ylabel('Training set')
		plt.plot([np.amin(Stats['Test']['Three_corr']),np.amax(Stats['Test']['Three_corr'])],
						[np.amin(Stats['Test']['Three_corr']),np.amax(Stats['Test']['Three_corr'])])
		#plt.xticks(fontsize = 14)
		plt.legend(loc = 'upper left')
		plt.grid()
		plt.title('Train VS Test')
		
		fig.tight_layout()

	if plot=='PCA':
		axis_font = {'size':'17'}
		Max=0.15
		align_nat = output['align']
		M = min(align_nat.shape[0],output['align_mod'].shape[0])
		sub_align_PCA =align_nat[np.random.choice(align_nat.shape[0],M,replace=False)]
		sub_align_mod = output['align_mod'][np.random.choice(output['align_mod'].shape[0],M,replace=False)]
		bin_align = ut.alg2bin(sub_align_PCA, N_aa=20)
		bin_align_mod = ut.alg2bin(sub_align_mod,N_aa=20)

		X,X_mod = ut.PCA_comparison(bin_align,bin_align_mod,Pears=0,Mask = 1)

		shift=0.4
		ma1, mi1 = np.amax(X[:,0])+shift, np.amin(X[:,0])-shift
		ma2, mi2 = np.amax(X[:,1])+shift, np.amin(X[:,1])-shift
		Wass_dist = ot.sliced_wasserstein_distance(X, X_mod, n_projections=500)
		print('Dist:',Wass_dist)

		density_scatter(X[:,0],X[:,1],Max=Max,markersize=18)
		plt.xlim([mi1,ma1])
		plt.ylim([mi2,ma2])
		plt.xlabel('PC 1',**axis_font)
		plt.ylabel('PC 2',**axis_font)
		plt.title('Natural sequences',**axis_font)
		plt.grid(color='gray',linestyle=(0, (5, 10)))
		plt.gca().spines[['right', 'top','left','bottom']].set_visible(False)
		
		density_scatter(X_mod[:,0],X_mod[:,1],Max=Max,markersize=18)
		plt.xlim([mi1,ma1])
		plt.ylim([mi2,ma2])
		plt.xlabel('PC 1',**axis_font)
		plt.ylabel('PC 2',**axis_font)
		plt.title('Artificial sequences',**axis_font)
		plt.grid(color='gray',linestyle=(0, (5, 10)))
		plt.gca().spines[['right', 'top','left','bottom']].set_visible(False)

	if plot=='Energy':
		fig = plt.figure(figsize = (8,4))
		Bins = 60
		# Random Sequences
		rand = np.round(np.random.random(output['align_mod'].shape)).astype('int32')

		Erand_SBM = ut.compute_energies(rand,output['h'],output['J'])
		Etest_SBM = ut.compute_energies(output['Test'],output['h'],output['J'])
		Etrain_SBM = ut.compute_energies(output['Train'],output['h'],output['J'])
		Emod_SBM = ut.compute_energies(output['align_mod'],output['h'],output['J'])
		Mean_SBM = np.mean(Etrain_SBM)
		STD_SBM = np.std(Etrain_SBM)
		Erand_SBM,Etest_SBM,Etrain_SBM,Emod_SBM = (Erand_SBM-Mean_SBM)/STD_SBM,(Etest_SBM-Mean_SBM)/STD_SBM,(Etrain_SBM-Mean_SBM)/STD_SBM,(Emod_SBM-Mean_SBM)/STD_SBM

		fig.add_subplot(1,1,1)
		c1, c2, c3 = 'rgb(0.279,0.681,0.901)','rgb(0.616,0.341,0.157)','rgb(0.092,0.239,0.404)'
		mi = np.amin(np.concatenate((Etest_SBM,Etrain_SBM,Emod_SBM,Erand_SBM)))
		ma = np.amax(np.concatenate((Etest_SBM,Etrain_SBM,Emod_SBM,Erand_SBM)))
		plt.hist(Etest_SBM, Bins, range=(mi-.5,ma+.5), alpha=.4, label='Test',color = c1,density=True)
		plt.hist(Emod_SBM,  Bins, range=(mi-.5,ma+.5), alpha=.4, label='Artificial',color = c2, density=True)
		plt.hist(Etrain_SBM, Bins, range=(mi-.5,ma+.5), alpha=.4, label='Train',color = c3,density=True)
		plt.hist(Erand_SBM,  Bins, range=(mi-.5,ma+.5), alpha=.4, label='Random', color = 'grey',density=True)
		plt.legend()
		plt.xlabel('Statistical energy') 
		plt.ylabel('probability den.')
		plt.grid()

		fig.tight_layout()

	if plot=='Similarity':
		fig = plt.figure(figsize = (8,4))
		Bins = 80 #25
		#align_train = ut.RemoveCloseSeqs(output['Train'],0.2)$
		Sim_SBM = ut.compute_similarities(output['align_mod'],output['Train'])
		Sim_train = ut.compute_similarities(output['Train'])
		Sim_test = ut.compute_similarities(output['Test'],output['Train'])

		fig.add_subplot(1,1,1)
		plt.hist(Sim_test,  Bins, range=(0,1), alpha=.4,  density=True, label = 'Test')
		plt.hist(Sim_SBM,  Bins, range=(0,1), alpha=.4,  density=True,label  ='Artificial')
		plt.hist(Sim_train,  Bins, range=(0,1), alpha=.4,  density=True, label = 'Train')
		plt.xlabel('Distance to closest natural seq') 
		plt.ylabel('probability den.')
		plt.legend()
		plt.grid()
		fig.tight_layout()
	
	if plot=='Diversity':
		Bins = 60
		fig = plt.figure(figsize = (8,4))
		Div_SBM = ut.compute_diversity(output['align_mod'])
		Div_train = ut.compute_diversity(output['Train'])
		Div_test = ut.compute_diversity(output['Test'])

		fig.add_subplot(1,1,1)
		plt.hist(Div_train,  Bins, range=(0,1),label='Train', alpha=.3, density=True)
		plt.hist(Div_test,  Bins, range=(0,1),label='Test', alpha=.3, density=True)
		plt.hist(Div_SBM, Bins, range=(0,1), alpha=.3, label='Artificial',density=True)
		plt.legend()
		plt.xlabel('Diversity') 
		plt.ylabel('probability den.')
		plt.grid()
		fig.tight_layout()

	if plot=='Length':
		Bins = 80
		fig = plt.figure(figsize = (8,4))
		Length_SBM = np.sum(output['align_mod'],axis=1)
		Length_train = np.sum(output['Train'],axis=1)
		Length_test = np.sum(output['Test'],axis=1)

		fig.add_subplot(1,1,1)
		mi, ma = np.amin(np.concatenate((Length_SBM,Length_train,Length_test))),np.amax(np.concatenate((Length_SBM,Length_train,Length_test)))
		plt.hist(Length_train,  Bins, range=(mi,ma),label='Train', alpha=.3, density=True)
		plt.hist(Length_test,  Bins, range=(mi,ma),label='Test', alpha=.3, density=True)
		plt.hist(Length_SBM, Bins, range=(mi,ma), alpha=.3, label='Artificial',density=True)
		plt.legend()
		plt.xlabel('Genome Length') 
		plt.ylabel('probability den.')
		plt.grid()
		fig.tight_layout()

	if plot=='Coupling_evol':
		fig = plt.figure(figsize = (5,4))
		plt.plot(output['J_norm'],'o',markersize = 2,color = 'tab:blue')
		plt.xlabel('Iterations')
		plt.ylabel('Couplings norm')
		#plt.xticks(fontsize = 16)
		plt.grid()
		#plt.yscale('log')
		plt.title('SBM, n_states='+str(output['options']['n_states'])+' m='+str(output['options']['m']))

def plot_2Dhistogram_SimEnergy(output,Titre = '',theta=0.2,ITER='',Low_T_mod=None,Ylimit=None,RAND=False,
                              remove_close = True):

	rand1 = np.random.randint(output['options']['q'],size = output['align_mod'].shape)
	align_train= output['Train']
	align_test= output['Test']
	align_mod= output['align_mod']
	SBM_J,SBM_h = output['J'+str(ITER)],output['h'+str(ITER)]
	SBM_J,SBM_h = ut.Zero_Sum_Gauge(SBM_J,SBM_h)
	#align_train = align_train[np.random.choice(np.arange(align_train.shape[0]),align_test.shape[0],replace=False)]
	align_train_theta = align_train
	if remove_close:
		align_train_theta = ut.RemoveCloseSeqs(align_train,0.2)
	rand2 = ut.shuff_column(align_train)
	Erand1 = ut.compute_energies(rand1,SBM_h,SBM_J)
	Erand2 = ut.compute_energies(rand2,SBM_h,SBM_J)
	Etest = ut.compute_energies(align_test,SBM_h,SBM_J)
	Etrain = ut.compute_energies(align_train_theta,SBM_h,SBM_J)
	Emod = ut.compute_energies(align_mod,SBM_h,SBM_J)
	Mean = np.median(Etrain)

	Sim_train = ut.compute_similarities(align_train_theta)
	Sim_test = ut.compute_similarities(align_test,align_train)
	Sim_mod = ut.compute_similarities(align_mod,align_train)
	Sim_rand1 = ut.compute_similarities(rand1,align_train)
	Sim_rand2 = ut.compute_similarities(rand2,align_train)

	E = np.concatenate((Etrain-Mean,Etest-Mean,Emod-Mean))
	if math.isinf(np.std(Etrain)): E/= 1e300
	Sim = np.concatenate((Sim_train,Sim_test,Sim_mod))
	Type = np.concatenate((['Train']*Etrain.shape[0],['Test']*Etest.shape[0],['Artificial T=1']*Emod.shape[0]))

	if Low_T_mod is not None:
		Emod2 = ut.compute_energies(Low_T_mod['align_mod'],SBM_h,SBM_J)
		Sim_mod2 = ut.compute_similarities(Low_T_mod['align_mod'],align_train)
		E = np.concatenate((E,Emod2-Mean))
		Sim = np.concatenate((Sim,Sim_mod2))
		Type = np.concatenate((Type,['Artificial T='+str(Low_T_mod['options']['T'])]*Emod2.shape[0]))

	if RAND:
		E = np.concatenate((E,Erand1-Mean))
		Sim = np.concatenate((Sim,Sim_rand1))
		Type = np.concatenate((Type,['Random']*rand1.shape[0]))
		
	E = np.concatenate((E,Erand2-Mean))
	Sim = np.concatenate((Sim,Sim_rand2))
	Type = np.concatenate((Type,['Random col.']*rand2.shape[0]))
	
	df = pd.DataFrame({Titre:Type,
						'Statistical energy':E,
						'ID to nearest natural':1-Sim}, 
						index=np.arange(E.shape[0]))
	
	sns.set_theme(style="ticks",rc={'axes.labelsize':16,"legend.fontsize":14,"xtick.labelsize":13,"ytick.labelsize":13})
	c1, c2, c3,c4 = (0.092,0.239,0.404),(0.279,0.681,0.901),(0.616,0.341,0.157),(0.986,0.544,0.248)
	color_palette = [c1,c2,c3]
	if Low_T_mod is not None:
		color_palette.append(c4)
	if RAND: 
		color_palette.append('black')
	color_palette.append('grey')
	

	g = sns.jointplot(
		data=df,
		x='ID to nearest natural', 
		y="Statistical energy",
		hue = Titre,
		#kind="scatter",
		#fill=True,
		#common_norm = False,
		xlim = [0,1],
		ylim = Ylimit,
		palette=color_palette,
		marginal_kws = {'common_norm':False},
		joint_kws={'alpha': 0.5})
	
	#g.plot_marginals()


def density_scatter( x , y,Max,markersize=10)   :
	"""
	Scatter plot colored by 2d histogram
	"""
	# Calculate the point density
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	# Sort the points by density, so that the densest points are plotted last
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	print(np.min(z),np.max(z))
	z = np.concatenate((np.array([0]),z,np.array([Max])))
	x = np.concatenate((np.array([-10]),x,np.array([-10])))
	y = np.concatenate((np.array([-10]),y,np.array([-10])))
	#print(x.shape,y.shape,z.shape)
	fig, ax = plt.subplots()
	ax.scatter(x, y, c=z, s=markersize,cmap='magma')

	norm = Normalize(vmin = 0,vmax=Max)
	cbar = fig.colorbar(cm.ScalarMappable(norm = norm,cmap='magma'), ax=ax)
	#cbar.ax.set_ylabel('Density')q
	return ax


def BARplot_distances(file_name,Mod_list,fam,delta_t = 1e4,ITER='',Temp=1):
	Distances = {
		'y_diversity' : [],'Dist_diversity' : [],
		'y_similarity' : [],'Dist_similarity' : [],
		'y_train' : [],'Dist_train' : [],
		'y_test' : [],'Dist_test' : [],
		'y_rand1' : [],'Dist_rand1' : [],
		'y_rand2' : [],'Dist_rand2' : [],}

	for i in range(len(file_name)):
		f = file_name[i]
		output = np.load('results/'+Mod_list[i]+'/'+fam+'/'+f,allow_pickle=True)[()]
		if output['Test'] is None:
			if fam[:2]=='TM':
				output['Test'] = np.load('data/MSA_array/MSA_test_ToyModel.npy')
			else: output['Test']=output['Train']
		if 'align_mod' not in list(output.keys()):
			print('generate align_mod')
			N = output['Train'].shape[0]
			align_mod=ut.Create_modAlign(output,N = N,delta_t = delta_t,ITER=ITER,temperature=Temp)
			output['align_mod'] = align_mod

		if output['Test'] is None: align_test=output['Train']
		else:align_test= output['Test']
		align_test= output['Test']
		align_train= output['Train']
		align_mod= output['align_mod']

		rand1 = np.random.randint(output['options']['q'],size = align_mod.shape)
		rand2 = ut.shuff_column(align_train)

		# if fam=='CM':
		# 	Pos_rem = np.array([65,1, 2, 3, 4, 64, 67]) - 1
		# 	Pos_keep = np.delete(np.arange(output['J'].shape[0]),Pos_rem)
		# 	align_test,align_train,align_mod = align_test[:,Pos_keep],align_train[:,Pos_keep], align_mod[:,Pos_keep]

		if Mod_list[i]=='SBM':
			lab = r'SBM'+str(i)
		else:
			lab = r'BM'+str(i)
		
		Sim_rand1 = ut.compute_similarities(rand1,align_train)
		Distances['Dist_rand1'].extend(1 - Sim_rand1)
		Distances['y_rand1'].extend([lab]*len(Sim_rand1))
		Sim_rand2 = ut.compute_similarities(rand2,align_train)
		Distances['Dist_rand2'].extend(1 - Sim_rand2)
		Distances['y_rand2'].extend([lab]*len(Sim_rand2))

		Sim_train = ut.compute_similarities(align_train)
		Distances['Dist_train'].extend(1 - Sim_train)
		Distances['y_train'].extend([lab]*len(Sim_train))

		Sim_test = ut.compute_similarities(align_test,align_train)
		Distances['Dist_test'].extend(1 - Sim_test)
		Distances['y_test'].extend([lab]*len(Sim_test))
		
		Sim_mod = ut.compute_similarities(align_mod,align_train)
		Distances['Dist_similarity'].extend(1-Sim_mod)
		Distances['y_similarity'].extend([lab]*len(Sim_mod))

		# Div_mod = ut.compute_diversity(align_mod)
		# Distances['Dist_diversity'].extend(1-Div_mod)
		# Distances['y_diversity'].extend([lab]*len(Div_mod))
	print('Mod: ',np.mean(Distances['Dist_similarity']))
	print('Test: ',np.mean(Distances['Dist_test']))
	#sprint(np.mean(Distances['Dist_test']))
	if True:
		fig = go.Figure()
		#c1, c2 = 'rgb(0.616,0.341,0.157)','rgb(0.035,0.125,0.239)'
		c1, c2, c3 = 'rgb(0.279,0.681,0.901)','rgb(0.616,0.341,0.157)','rgb(0.092,0.239,0.404)'
		fig.add_trace(go.Box(x= Distances['y_train'], y = Distances['Dist_train'],name = 'Train', marker_color = c3,boxpoints = False,boxmean=True))
		fig.add_trace(go.Box(x= Distances['y_test'], y = Distances['Dist_test'],name = 'Test', marker_color = c1,boxpoints = False,boxmean=True))
		fig.add_trace(go.Box(x= Distances['y_similarity'], y = Distances['Dist_similarity'],name = 'Artificial T='+str(Temp), marker_color = c2,boxpoints = False,boxmean=True))
		fig.add_trace(go.Box(x= Distances['y_rand1'], y = Distances['Dist_rand1'],name = 'Random', marker_color = 'black',boxpoints = False,boxmean=True))
		fig.add_trace(go.Box(x= Distances['y_rand2'], y = Distances['Dist_rand2'],name = 'Random col.', marker_color = 'grey',boxpoints = False,boxmean=True))
		#fig.add_trace(go.Box(x= Distances['y_diversity'], y = Distances['Dist_diversity'],name = 'Similarity inter alignement',marker_color = c2,boxpoints = False))
		

		fig.update_layout(yaxis=dict(title='ID to Nearest Natural', zeroline=False),boxmode='group',boxgroupgap=0,boxgap=0.5,plot_bgcolor='white',
		    																		#height=600, width=500,font_size=20,legend_font_size=20)
																					#height=800, width=600,font_size=20,legend_font_size=20)
																					height=550, width=560,font_size=20,legend_font_size=20)
		
		fig.update_xaxes(
			mirror=True,
			showline=True,
			linecolor='black'
		)
		fig.update_yaxes(
			mirror=True,
			ticks='outside',
			showline=True,
			linecolor='black',
			gridcolor='lightgrey'
		)
		fig.update_yaxes(range =[0,1])
		#fig.update_yaxes(range =[0.3,0.6])
		fig.show()
	return 


def BARplot_distances_BM_T(file_name,Mod_list,fam,delta_t = 1e4):
	Distances = {
		'y_diversity' : [],
		'Dist_diversity' : [],
		'y_similarity1' : [],
		'Dist_similarity1' : [],
		'y_similarity75' : [],
		'Dist_similarity75' : [],
		'y_train' : [],
		'Dist_train' : [],
		'y_test' : [],
		'Dist_test' : [],
	}

	for i in range(len(file_name)):
		f = file_name[i]
		output = np.load('results/Article/'+Mod_list[i]+'/'+fam+'/'+f,allow_pickle=True)[()]
		if output['Test'] is None:
			if fam[:2]=='TM':
				output['Test'] = np.load('data/MSA_array/MSA_test_ToyModel.npy')
			else: output['Test']=output['Train']
		if 'align_mod' not in list(output.keys()):
			align_mod=ut.Create_modAlign(output,N = output['Test'].shape[0],delta_t = delta_t)
			output['align_mod'] = align_mod

		if output['Test'] is None: align_test=output['Train']
		else:align_test= output['Test']
		align_test= output['Test']
		align_train= output['Train']
		align_mod= output['align_mod']

		if fam=='CM':
			Pos_rem = np.array([65,1, 2, 3, 4, 64, 67]) - 1
			Pos_keep = np.delete(np.arange(output['J'].shape[0]),Pos_rem)
			align_test,align_train,align_mod = align_test[:,Pos_keep],align_train[:,Pos_keep], align_mod[:,Pos_keep]

		if Mod_list[i]=='SBM':
			lab = r'SBM'+str(i)
		else:
			lab = r'BM'+str(i) 
		
		if i==0:
			Sim_train = ut.compute_similarities(align_train)
			Distances['Dist_train'].extend(1 - Sim_train)
			Distances['y_train'].extend([lab]*len(Sim_train))

			Sim_test = ut.compute_similarities(align_test,align_train)
			Distances['Dist_test'].extend(1 - Sim_test)
			Distances['y_test'].extend([lab]*len(Sim_test))
			
			Sim_mod = ut.compute_similarities(align_mod,align_train)
			Distances['Dist_similarity1'].extend(1-Sim_mod)
			Distances['y_similarity1'].extend([lab]*len(Sim_mod))
		else:
			Sim_mod = ut.compute_similarities(align_mod,align_train)
			Distances['Dist_similarity75'].extend(1-Sim_mod)
			Distances['y_similarity75'].extend([lab]*len(Sim_mod))

	if True:
		fig = go.Figure()
		c1, c2, c3 = 'rgb(0.279,0.681,0.901)','rgb(0.616,0.341,0.157)','rgb(0.092,0.239,0.404)'
		c4 = 'rgb(0.986,0.544,0.248)'
		fig.add_trace(go.Box(x= Distances['y_train'], y = Distances['Dist_train'],name = 'Train', marker_color = c3,boxpoints = False))
		fig.add_trace(go.Box(x= Distances['y_test'], y = Distances['Dist_test'],name = 'Test', marker_color = c1,boxpoints = False))
		fig.add_trace(go.Box(x= Distances['y_similarity1'], y = Distances['Dist_similarity1'],name = 'Artificial (T=1)', marker_color = c2,boxpoints = False))
		fig.add_trace(go.Box(x= Distances['y_similarity75'], y = Distances['Dist_similarity75'],name = 'Artificial (T=0.75)', marker_color = c4,boxpoints = False))

		fig.update_layout(yaxis=dict(title='Similarity to Nearest Natural', zeroline=False),boxmode='group',boxgroupgap=0,boxgap=0.5,plot_bgcolor='white',
		    																		#height=600, width=520,font_size=20,legend_font_size=20)
																					height=550, width=580,font_size=20,legend_font_size=20)
		
		fig.update_xaxes(
			mirror=True,
			showline=True,
			linecolor='black'
		)
		fig.update_yaxes(
			mirror=True,
			ticks='outside',
			showline=True,
			linecolor='black',
			gridcolor='lightgrey'
		)
		fig.update_yaxes(range =[0,1])
		fig.show()


#################################################################################

####################### CONTACT PREDICTION #######################

def plot_Contact_Pred(Score,align,align_mod,struct_mat):
	"""
	Function to generate a plot that includes PPV versus predicted pairs and a contact map

	Args:
	- Score (numpy.array): A 2D numpy array representing the score computed with "compute_score_contacts"
	- align (numpy.array): A 2D numpy array representing the natural alignment.
	- align_mod (numpy.array): A 2D numpy array representing the artificial alignement.
	- struct_mat (numpy.array): A 2D numpy array representing the matrix of the alignment 3D structure.

	Returns: None
	"""
	L = Score.shape[0]
	fig = plt.figure(figsize = (12,8))

	#PPV versus predicted pairs
	fig.add_subplot(1,2,2)
	struct_mat =  (struct_mat < 8).astype('int') - np.tri(L,L)
	distij = np.abs(np.expand_dims(np.arange(L),axis = 1)*np.ones((L,L)) - np.expand_dims(np.arange(L),axis = 0)*np.ones((L,L)))
	mask_dist = (distij > 4).astype('int')
	struct_mat = struct_mat*mask_dist
	Score *= mask_dist

	argsort_S = np.argsort(Score[np.triu_indices(L,k = 1)])[::-1]
	sort_struct = struct_mat[np.triu_indices(L,k=1)]
	sort_struct = sort_struct[argsort_S]
	PPV = np.cumsum(sort_struct)/np.arange(1,len(sort_struct)+1)
	plt.plot(np.arange(1,len(sort_struct)+1)[:1000],PPV[:1000])
	plt.xscale('log')
	plt.xlabel('Number of predicted pairs',fontsize = 15)
	plt.ylabel('PPV',fontsize = 15)
	plt.yticks(fontsize = 14)
	plt.xticks(fontsize = 14)

	# Contact map
	fig.add_subplot(1,2,3)
	Pred_contact = np.where((Score*(np.tri(L,L,k = -1).T)).T[::-1] > 0.2)
	plt.imshow(struct_mat.T[::-1]/3,vmin = 0,vmax =1,cmap = 'Greys')
	plt.xticks(ticks = np.arange(struct_mat.shape[0])[::10],labels = np.arange(struct_mat.shape[0])[::10].astype('str'),fontsize=14)
	plt.yticks(ticks = np.arange(struct_mat.shape[0])[::10],labels = np.arange(struct_mat.shape[0])[::10][::-1].astype('str'),fontsize=14)
	plt.plot(Pred_contact[1],Pred_contact[0],'o',markersize = 4,color = 'blue',label = 'lbfgsDCA')
	plt.legend()

	fig.tight_layout()

def PPV_plot(J_inp,J_inf,N_contact,ZSG = True,APC = True,plot=False):
	L = J_inp.shape[0]
	J_inp = ut.Zero_Sum_Gauge(J_inp)
	J_inf = ut.Zero_Sum_Gauge(J_inf)
	Score_inf = ut.compute_score(J_inf,APC = APC)
	Score_inp = ut.compute_score(J_inp,APC = APC)
	argsort_inf = np.argsort(Score_inf[np.triu_indices(L,k = 1)])[::-1]
	argsort_inp = np.argsort(Score_inp[np.triu_indices(L,k = 1)])[::-1]

	Contacts = np.zeros(int(L*(L-1)/2))
	Contacts[argsort_inp[:N_contact]] = 1
	Contacts = Contacts[argsort_inf]
	PPV = np.cumsum(Contacts)/np.arange(1,len(Contacts)+1)
	if plot:
		plt.plot(np.arange(1,len(Contacts)+1)[:1000],PPV[:1000])
		plt.xscale('log')
		plt.xlabel('Number of predicted pairs',fontsize = 15)
		plt.ylabel('PPV',fontsize = 15)
		plt.yticks(fontsize = 14)
		plt.xticks(fontsize = 14)
	return PPV


def plot_AUC_contacts(struct_mat,Ns,N_iter,Rep_list,m_list=None,lambd_list=None,k_list =None,xaxis='n_states',fam='CM',Mod='SBM'):
	Z_value = 1.96
	AUC = np.zeros((len(Rep_list),len(Ns)))

	L = struct_mat.shape[0]
	struct_mat =  (struct_mat < 8).astype('int') - np.tri(L,L)
	distij = np.abs(np.expand_dims(np.arange(L),axis = 1)*np.ones((L,L)) - np.expand_dims(np.arange(L),axis = 0)*np.ones((L,L)))
	mask_dist = (distij > 4).astype('int')
	struct_mat = struct_mat*mask_dist
	sort_struct = struct_mat[np.triu_indices(L,k=1)]
	
	axis_font = {'size':'22'}
	#fig = plt.figure(figsize = (10,8))
	fig = plt.figure(figsize = (13,6))

	for i in range(len(Ns)):
		for r in range(len(Rep_list)):
			rep = Rep_list[r]
			if Mod=='BM':
				J,h = Mean_J_features_CM(Ns[i],N_iter[i],[rep],lambd=lambd_list[i],k_list=None,fam=fam,Mod='BM')
			elif Mod=='SBM':
				J,h = Mean_J_features_CM(Ns[i],N_iter[i],[rep],m=m_list[i],k_list=None,fam=fam,Mod='SBM')

			Score = ut.compute_score_contacts(J,APC = True)
			Score *= mask_dist
			Score = Score[np.triu_indices(L,k = 1)]

			fpr, tpr,_ = metrics.roc_curve(sort_struct,Score, pos_label=1)
			#plt.plot(fpr,tpr)
			#plt.show()
			#assert 0==1
			AUC[r,i] = metrics.auc(fpr, tpr)

	mean_AUC = np.mean(AUC,axis=0)
	eb_AUC = Z_value*np.std(AUC,axis=0)/max(np.sqrt(AUC.shape[0]-1),1)

	c1 = 'tab:blue'
	if xaxis=='n_states':
		plt.plot(Ns,mean_AUC,'--o',color=c1)
		plt.fill_between(Ns,mean_AUC-eb_AUC,mean_AUC+eb_AUC,alpha=0.2,color = c1)
	elif xaxis=='lambda_J':
		plt.plot(lambd_list,mean_AUC,'--o',color=c1)
		plt.fill_between(lambd_list,mean_AUC-eb_AUC,mean_AUC+eb_AUC,alpha=0.2,color = c1)
	plt.xscale('log')
	plt.ylim([0.5,0.62])
	if xaxis=='n_states':
		plt.xlabel(r'$N_{chains}$')
		xlim = [.8,160]
		plt.xlim(xlim)
	elif xaxis=='lambda_J':
		plt.xlabel(r'$\lambda_J$')
		xlim = [.8e-3,2.2]
		plt.xlim(xlim)
		plt.gca().invert_xaxis()
	plt.ylabel('AUC')
	plt.grid()
	plt.show()
	
#################################################################################

####################### MUTATIONAL EFFECTS PREDICTION #######################

def plot_DMS(DMS_mat,WT_seq=None):
	if DMS_mat.shape[0]==21:
		DMS_mat = DMS_mat[1:]
	plt.figure(figsize = (15,13))
	q,L = DMS_mat.shape[0],DMS_mat.shape[1]
	ma = np.amax(np.abs(DMS_mat))
	plt.imshow(DMS_mat,vmin=-ma,vmax=ma,cmap = 'bwr')
	plt.colorbar(location='right',fraction=0.03,aspect =5)
	if WT_seq is not None:
		WT_CM = np.full(DMS_mat.shape, np.nan)
		WT_CM[WT_seq-1,np.arange(DMS_mat.shape[1])] = 0.8
		print(WT_CM[:,0].shape)
		plt.imshow(WT_CM[1:],vmin = 0,vmax = 1,cmap = 'Greens')
	plt.yticks(np.arange(DMS_mat.shape[0]),['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'],fontsize = 8)
	plt.xticks(np.arange(DMS_mat.shape[1])[::5],np.arange(2,DMS_mat.shape[1])[::5],fontsize = 10)
	plt.hlines(y=np.arange(0, q)+0.5, xmin=np.full(q, 0)-0.5, xmax=np.full(q, L)-0.5, linewidth = 0.3,color="grey")
	plt.vlines(x=np.arange(0, L)+0.5, ymin=np.full(L, 0)-0.5, ymax=np.full(L,q)-0.5,linewidth = 0.3, color="grey")
	plt.gca().yaxis.set_tick_params(bottom = False)
	plt.show()

def plot_AUC_DMS_meanJ(Ns,N_iter,Rep_list,m_list=None,lambd_list=None,k_list =None,xaxis='n_states',fam='CM',Mod='SBM'):
	Z_value = 1.96
	AUC = np.zeros(len(Ns))

	axis_font = {'size':'22'}
	#fig = plt.figure(figsize = (10,8))
	fig = plt.figure(figsize = (13,6))

	if fam=='CM':
		mat = scipy.io.loadmat('data/CM_DMS.mat')
		DMS_results = mat['dms']
		Mut_DMS = DMS_results[0][0][1]
		WT_seq = np.load('data/MSA_array/MSA_CM.npy')[0]
		Pos_rem = np.array([1,65]) -1 #,1, 2, 3, 4, 64, 67]) - 1
		Pos_keep = np.delete(np.arange(96),Pos_rem)
		WT_seq = WT_seq[Pos_keep]
		Binary_DMS = (Mut_DMS[:-1].flatten()<-0.3)
	if fam[:2]=='TM':
		Mut_DMS = 0
	
	for i in range(len(Ns)):
		if Mod=='BM':
			J,h = Mean_J_features_CM(Ns[i],N_iter[i],Rep_list,lambd=lambd_list[i],k_list=None,Mod='BM')
		elif Mod=='SBM':
			J,h = Mean_J_features_CM(Ns[i],N_iter[i],Rep_list,m=m_list[i],k_list=None,Mod='SBM')
		Mut_CM = ut.Mutational_effect(WT_seq,h,J)
		fpr, tpr,_ = metrics.roc_curve( Binary_DMS.astype('int'),Mut_CM[1:].flatten(), pos_label=1)
		AUC[i]=metrics.auc(fpr, tpr)
	if xaxis=='n_states':
		plt.plot(Ns,AUC,'--o')
	elif xaxis=='lambda_J':
		plt.plot(lambd_list,AUC,'--o')
	plt.xscale('log')
	plt.ylim([0.75,0.9])
	if xaxis=='n_states':
		plt.xlabel(r'$N_{chains}$')
		xlim = [.8,90]
		plt.xlim(xlim)
	elif xaxis=='lambda_J':
		plt.xlabel(r'$\lambda_J$')
		xlim = [.8e-3,2.2]
		plt.xlim(xlim)
		plt.gca().invert_xaxis()

	J_prof, h_prof = Mean_J_features_CM(None,None,[''],lambd=None,m=None,k_list=None,fam='CM',Mod = 'Profile')
	Mut_CM = ut.Mutational_effect(WT_seq,h_prof,J_prof)
	fpr, tpr,_ = metrics.roc_curve(Binary_DMS.astype('int'),Mut_CM[1:].flatten(), pos_label=1)
	AUC_prof = metrics.auc(fpr, tpr)
	plt.plot(xlim,[AUC_prof,AUC_prof],'--o',color='tab:red',label='Profile model')
	plt.legend()
	plt.ylabel('AUC')
	plt.grid()
	plt.show()

def plot_AUC_DMS_meanPred(Ns,N_iter,Rep_list,m_list=None,lambd_list=None,k_list =None,xaxis='n_states',fam='CM',Mod='SBM'):
	Z_value = 1.96
	AUC = np.zeros((len(Rep_list),len(Ns)))

	axis_font = {'size':'22'}
	#fig = plt.figure(figsize = (10,8))
	fig = plt.figure(figsize = (13,6))

	if fam=='CM':
		mat = scipy.io.loadmat('data/CM_DMS.mat')
		DMS_results = mat['dms']
		Mut_DMS = DMS_results[0][0][1]
		WT_seq = np.load('data/MSA_array/MSA_CM.npy')[0]
		Pos_rem = np.array([1,65]) -1 #,1, 2, 3, 4, 64, 67]) - 1
		Pos_keep = np.delete(np.arange(96),Pos_rem)
		WT_seq = WT_seq[Pos_keep]
		Binary_DMS = (Mut_DMS[:-1].flatten()<-0.3)
	if fam[:2]=='TM':
		Mut_DMS = 0
	
	for i in range(len(Ns)):
		for r in range(len(Rep_list)):
			rep = Rep_list[r]
			if Mod=='BM':
				J,h = Mean_J_features_CM(Ns[i],N_iter[i],[rep],lambd=lambd_list[i],k_list=None,Mod='BM')
			elif Mod=='SBM':
				J,h = Mean_J_features_CM(Ns[i],N_iter[i],[rep],m=m_list[i],k_list=None,Mod='SBM')
			Mut_CM = ut.Mutational_effect(WT_seq,h,J)
			fpr, tpr,_ = metrics.roc_curve( Binary_DMS.astype('int'),Mut_CM[1:].flatten(), pos_label=1)
			AUC[r,i] = metrics.auc(fpr, tpr)
	mean_AUC = np.mean(AUC,axis=0)
	eb_AUC = Z_value*np.std(AUC,axis=0)/max(np.sqrt(AUC.shape[0]-1),1)

	c1 = 'tab:blue'
	if xaxis=='n_states':
		plt.plot(Ns,mean_AUC,'--o',color=c1)
		plt.fill_between(Ns,mean_AUC-eb_AUC,mean_AUC+eb_AUC,alpha=0.2,color = c1)
	elif xaxis=='lambda_J':
		plt.plot(lambd_list,mean_AUC,'--o',color=c1)
		plt.fill_between(lambd_list,mean_AUC-eb_AUC,mean_AUC+eb_AUC,alpha=0.2,color = c1)
	plt.xscale('log')
	plt.ylim([0.75,0.9])
	if xaxis=='n_states':
		plt.xlabel(r'$N_{chains}$')
		xlim = [.8,90]
		plt.xlim(xlim)
	elif xaxis=='lambda_J':
		plt.xlabel(r'$\lambda_J$')
		xlim = [.8e-3,2.2]
		plt.xlim(xlim)
		plt.gca().invert_xaxis()

	J_prof, h_prof = Mean_J_features_CM(None,None,[''],lambd=None,m=None,k_list=None,fam=fam,Mod = 'Profile')
	Mut_CM = ut.Mutational_effect(WT_seq,h_prof,J_prof)
	fpr, tpr,_ = metrics.roc_curve(Binary_DMS.astype('int'),Mut_CM[1:].flatten(), pos_label=1)
	AUC_prof = metrics.auc(fpr, tpr)
	plt.plot(xlim,[AUC_prof,AUC_prof],'--o',color='tab:red',label='Profile model')
	plt.legend()
	plt.ylabel('AUC')
	plt.grid()
	plt.show()
#################################################################################

####################### PLOT COUPLINGS #######################


def SBM_compute_J_features(ITER,folder,ns_list,Ni_list,m_list,Rep_list,k_list=None,batch_list=None,TM_nb = 1):
	#ITER = 100 #''#500
	val = 2
	L, q = 20, 10
	J_ab = np.zeros((L,L))
	J_ab[[4,6,8],[1,3,5]] = val
	J_ab = J_ab + J_ab.T
	J_ab[10:14,10:14],J_ab[14:,14:] = val, val
	np.fill_diagonal(J_ab,0)
	J_inp = np.zeros((L,L,q,q))
	for i in range(q):
		J_inp[:,:,i,i] = J_ab
	h_inp = np.zeros((L,q)) 

	J_inp,_ = ut.Zero_Sum_Gauge(J_inp, h_inp)
	
	
	J0 = np.linalg.norm(J_inp,'fro',axis = (2,3))
	J0 = J0 + (J0==0).astype('int')*np.amax(J0)
	mask = (np.linalg.norm(J_inp,'fro',axis = (2,3))==0)
	#mask_int = (np.linalg.norm(J_inp,'fro',axis = (2,3))!=0)

	J_iso = np.zeros((len(Rep_list),len(ns_list)))*np.nan
	J_small = np.zeros((len(Rep_list),len(ns_list)))*np.nan
	J_large = np.zeros((len(Rep_list),len(ns_list)))*np.nan
	J_back  = np.zeros((len(Rep_list),len(ns_list)))*np.nan
	#J_int = np.zeros((len(Rep_list),len(ns_list)))*np.nan

	for r in range(len(Rep_list)):
		for i in range(len(ns_list)):
			if k_list is None:
				if batch_list is None:
					f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+Rep_list[r]+'.npy'
				else:
					f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'Bf'+str(batch_list[i])+Rep_list[r]+'.npy'
			else:
				f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'k'+str(k_list[i])+Rep_list[r]+'.npy'
			
			path = 'results/Article/SBM/'+folder+'/'+f
			if os.path.exists(path):
				output_SBM = np.load('results/Article/SBM/'+folder+'/'+f,allow_pickle=True)[()]

				J,_ = ut.Zero_Sum_Gauge(output_SBM['J'+str(ITER)], output_SBM['h'+str(ITER)])

				Jinf_norm = np.linalg.norm(J,'fro',axis = (2,3))/J0
				
				norm_iso = np.mean(Jinf_norm[[4,6,8],[1,3,5]])
				norm_small = np.mean(Jinf_norm[10:14,10:14])
				norm_large = np.mean(Jinf_norm[14:,14:])
				norm_back = np.mean(Jinf_norm[mask])
				#norm_int=  np.mean(Jinf_norm[mask_int])

				J_iso[r,i] = norm_iso
				J_small[r,i] = norm_small
				J_large[r,i] = norm_large
				J_back[r,i] = norm_back

	J_features = {'J_iso':J_iso,
	       		  'J_small':J_small,
			      'J_large':J_large,
			      'J_back':J_back}
	return J_features


def Compute_J_features_STM(ns_list,Ni_list,m_list,Rep_list,k_list=None,batch_list=None,TM_nb = 1):
	folder =  'STM_Init_Random' #'SBM_batch' #
	val = 4
	L, q = 6, 2
	J_ab = np.zeros((L,L))
	J_ab[0,1] = val
	J_ab = J_ab + J_ab.T
	J_ab[2:,2:]= val
	np.fill_diagonal(J_ab,0)
	J_inp = np.zeros((L,L,q,q))
	for i in range(q):
		J_inp[:,:,i,i] = J_ab
	h_inp = np.zeros((L,q))
	
	#print(J_inp[0,1,0,0])
	#J_inp,_ = ut.Zero_Sum_Gauge(J_inp, h_inp)
	#print(J_inp[0,1,0,0],J_inp[2,3,0,0],J_inp[0,4,0,0])
	
	J0 = np.linalg.norm(J_inp,'fro',axis = (2,3))
	J0 = J0 + (J0==0).astype('int')*np.amax(J0)
	mask = (np.linalg.norm(J_inp,'fro',axis = (2,3))==0)

	J_iso = np.zeros((len(Rep_list),len(ns_list)))*np.nan
	J_large = np.zeros((len(Rep_list),len(ns_list)))*np.nan
	J_back  = np.zeros((len(Rep_list),len(ns_list)))*np.nan

	for r in range(len(Rep_list)):
		for i in range(len(ns_list)):
			if k_list is None:
				if batch_list is None:
					f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+Rep_list[r]+'.npy'
				else:
					f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'Bf'+str(batch_list[i])+Rep_list[r]+'.npy'
			else:
				f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'k'+str(k_list[i])+Rep_list[r]+'.npy'
			
			path = 'results/Article/SBM/'+folder+'/'+f
			if os.path.exists(path):
				output_SBM = np.load(path,allow_pickle=True)[()]
				J = output_SBM['J']
				#J,_ = ut.Zero_Sum_Gauge(output_SBM['J'], output_SBM['h'])
				Jinf_norm = np.linalg.norm(J,'fro',axis = (2,3))
				#print(J[0,1,0,0],J[0,1,0,1],J_inp[0,1,0,0],J_inp[0,1,0,0],Jinf_norm[0,1],J0[0,1])
				Jinf_norm = Jinf_norm/J0
				
				norm_iso = np.mean(Jinf_norm[0,1])
				norm_large = np.mean(Jinf_norm[2:,2:])
				norm_back = np.mean(Jinf_norm[mask])

				J_iso[r,i] = norm_iso
				J_large[r,i] = norm_large
				J_back[r,i] = norm_back

	J_features = {'J_iso':J_iso,
			      'J_large':J_large,
			      'J_back':J_back}
	return J_features

def BM_compute_J_features(folder,ns_list,Ni_list,lb_list,Rep_list,k_list=None,batch_list=None,TM_nb = 1):
	ITER = ''
	val = 2
	L, q = 20, 10
	J_ab = np.zeros((L,L))
	J_ab[[4,6,8],[1,3,5]] = val
	J_ab = J_ab + J_ab.T
	J_ab[10:14,10:14],J_ab[14:,14:] = val, val
	np.fill_diagonal(J_ab,0)
	J_inp = np.zeros((L,L,q,q))
	for i in range(q):
		J_inp[:,:,i,i] = J_ab
	h_inp = np.zeros((L,q))

	J_inp,_ = ut.Zero_Sum_Gauge(J_inp, h_inp)

	J0 = np.linalg.norm(J_inp,'fro',axis = (2,3))
	J0 = J0 + (J0==0).astype('int')*np.amax(J0)
	mask = (np.linalg.norm(J_inp,'fro',axis = (2,3))==0)
	mask_int = (np.linalg.norm(J_inp,'fro',axis = (2,3))!=0)

	lb_plot = []
	nstates_plot = []
	Ni_plot = []

	#J_score = np.zeros((len(Rep_list),len(ns_list)))
	J_iso = np.zeros((len(Rep_list),len(ns_list)))
	J_small = np.zeros((len(Rep_list),len(ns_list)))
	J_large = np.zeros((len(Rep_list),len(ns_list)))
	J_back  = np.zeros((len(Rep_list),len(ns_list)))
	#J_int = np.zeros((len(Rep_list),len(ns_list)))

	for r in range(len(Rep_list)):
		for i in range(len(ns_list)):
			if k_list is None:
				if batch_list is None:
					f = 'BM_TM'+str(TM_nb)+'_Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'lb'+str(lb_list[i])+Rep_list[r]+'.npy'
				else:
					f = 'BM_TM'+str(TM_nb)+'_Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'lb'+str(lb_list[i])+'Bf'+str(batch_list[i])+Rep_list[r]+'.npy'
				
			else:
				f = 'BM_TM'+str(TM_nb)+'_Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'lb'+str(lb_list[i])+'k'+str(int(k_list[i]))+Rep_list[r]+'.npy'
			
			output_SBM = np.load('results/Article/BM/'+folder+'/'+f,allow_pickle=True)[()]

			J,_ = ut.Zero_Sum_Gauge(output_SBM['J'], output_SBM['h'])
			Jinf_norm = np.linalg.norm(J,'fro',axis = (2,3))/J0
			
			norm_iso = np.mean(Jinf_norm[[4,6,8],[1,3,5]])
			norm_small = np.mean(Jinf_norm[10:14,10:14])
			norm_large = np.mean(Jinf_norm[14:,14:])
			norm_back = np.mean(Jinf_norm[mask])
			#norm_int=  np.mean(Jinf_norm[mask_int])

			J_iso[r,i] = norm_iso
			J_small[r,i] = norm_small
			J_large[r,i] = norm_large
			J_back[r,i] = norm_back
			#J_int[r,i] = norm_int

			if r==0:
				nstates_plot.append(output_SBM['options']['n_states'])
				lb_plot.append(output_SBM['options']['lambda_J'])
				Ni_plot.append(output_SBM['options']['maxIter'])
	lb_plot,nstates_plot,Ni_plot = np.array(lb_plot),np.array(nstates_plot),np.array(Ni_plot)

	J_features = {'J_iso':J_iso,
	       		  'J_small':J_small,
			      'J_large':J_large,
			      'J_back':J_back}
	return J_features,lb_plot,nstates_plot,Ni_plot

def plot_J_features(J_features,Xaxis_list,Xlim,xaxis = 'n_states',Reverse_xaxis=False):
	Z_value = 1.96
	c4 = (0.8,0.8,0.8)
	#c1,c2,c3 = (0.929,0.737,0.255), (0.592,0.122,0.086),(0.318,0.051,0.043)
	c1,c2,c3 = (0.818,0.650,0.223),(0.8,0.170,0.121),(0.318,0.051,0.043)
	axis_font = {'size':'22'}
	fig = plt.figure(figsize = (9,7))
	#fig.patch.set_facecolor((0.921,0.921,0.921))

	plt.plot(Xlim,[1,1],'--',color = 'black',linewidth = 1.3)

	mean_iso = np.nanmean(J_features['J_iso'],axis=0)
	mean_small = np.nanmean(J_features['J_small'],axis=0)
	mean_large = np.nanmean(J_features['J_large'],axis=0)
	mean_back = np.nanmean(J_features['J_back'],axis=0)

	eb_iso = Z_value*np.nanstd(J_features['J_iso'],axis=0)/max(np.sqrt(J_features['J_iso'].shape[0]-1),1)
	eb_small = Z_value*np.nanstd(J_features['J_small'],axis=0)/max(np.sqrt(J_features['J_small'].shape[0]-1),1)
	eb_large = Z_value*np.nanstd(J_features['J_large'],axis=0)/max(np.sqrt(J_features['J_large'].shape[0]-1),1)
	eb_back = Z_value*np.nanstd(J_features['J_back'],axis=0)/max(np.sqrt(J_features['J_back'].shape[0]-1),1)

	plt.loglog(Xaxis_list,mean_iso,'--o',linewidth = 2.4,color = c1,label = 'Pairwise int.')
	plt.fill_between(Xaxis_list,np.maximum(mean_iso-eb_iso,4e-3*np.ones(mean_iso.shape)),mean_iso+eb_iso,color = c1,alpha=0.2)

	plt.loglog(Xaxis_list,mean_small,'--o',linewidth = 2.4,color = c2,label = 'Small collective')
	plt.fill_between(Xaxis_list,mean_small-eb_small,mean_small+eb_small,color = c2,alpha=0.2)
	
	plt.loglog(Xaxis_list,mean_large,'--o',linewidth = 2.4,color = c3,label = 'Large collective')
	plt.fill_between(Xaxis_list,mean_large-eb_large,mean_large+eb_large,color = c3,alpha=0.2)

	plt.loglog(Xaxis_list,mean_back,'--o',linewidth = 2.4,color = c4,label = 'Non interacting')
	plt.fill_between(Xaxis_list,np.maximum(mean_back-eb_back,4e-3*np.ones(mean_back.shape)),mean_back+eb_back,color = 'gray',alpha=0.2)

	plt.ylabel(r'$||\hat{J}_{ij}||\ / \ J_0$',**axis_font)
	plt.ylim([4e-3,3])

	if xaxis=='n_states':
		plt.xlabel(r'$N_{chains}$',**axis_font)
	elif xaxis=='N_iter':
		plt.xlabel(r'$N_{iter}$',**axis_font)
	elif xaxis=='m':
		plt.xlabel(r'$m$',**axis_font)
	elif xaxis=='k':
		plt.xlabel(r'$k$',**axis_font)
	elif xaxis=='lambda_J':
		plt.xlabel(r'$\lambda_J$',**axis_font)
	elif xaxis=='batch_size':
		plt.xlabel(r'Batch size',**axis_font)
	
	plt.xlim(Xlim)

	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	if Reverse_xaxis:
		plt.gca().invert_xaxis()
	plt.legend(loc = 'lower left',fontsize = 17)
	plt.grid(True,which="both", ls="-", color='0.9')
	plt.show()

def plot_J_features_STM(J_features,Xaxis_list,Xlim,xaxis = 'n_states',Reverse_xaxis=False):
	Z_value = 1.96
	c4 = (0.8,0.8,0.8)
	c1,c2,c3 = (0.929,0.737,0.255), (0.592,0.122,0.086),(0.318,0.051,0.043)
	axis_font = {'size':'22'}
	fig = plt.figure(figsize = (9,7))
	#fig.patch.set_facecolor((0.921,0.921,0.921))

	plt.plot(Xlim,[1,1],'--',color = 'black',linewidth = 1.3)

	mean_iso = np.nanmean(J_features['J_iso'],axis=0)
	mean_large = np.nanmean(J_features['J_large'],axis=0)
	mean_back = np.nanmean(J_features['J_back'],axis=0)

	eb_iso = Z_value*np.nanstd(J_features['J_iso'],axis=0)/max(np.sqrt(J_features['J_iso'].shape[0]-1),1)
	eb_large = Z_value*np.nanstd(J_features['J_large'],axis=0)/max(np.sqrt(J_features['J_large'].shape[0]-1),1)
	eb_back = Z_value*np.nanstd(J_features['J_back'],axis=0)/max(np.sqrt(J_features['J_back'].shape[0]-1),1)

	plt.loglog(Xaxis_list,mean_iso,'--o',linewidth = 2.4,color = c1,label = 'Pairwise int.')
	plt.fill_between(Xaxis_list,np.maximum(mean_iso-eb_iso,4e-3*np.ones(mean_iso.shape)),mean_iso+eb_iso,color = c1,alpha=0.2)
	
	plt.loglog(Xaxis_list,mean_large,'--o',linewidth = 2.4,color = c3,label = 'Large collective')
	plt.fill_between(Xaxis_list,mean_large-eb_large,mean_large+eb_large,color = c3,alpha=0.2)

	plt.loglog(Xaxis_list,mean_back,'--o',linewidth = 2.4,color = c4,label = 'Non interacting')
	plt.fill_between(Xaxis_list,np.maximum(mean_back-eb_back,4e-3*np.ones(mean_back.shape)),mean_back+eb_back,color = 'gray',alpha=0.2)

	plt.ylabel(r'$||\hat{J}_{ij}||\ / \ J_0$',**axis_font)
	plt.ylim([4e-3,3])

	if xaxis=='n_states':
		plt.xlabel(r'$N_{chains}$',**axis_font)
	elif xaxis=='N_iter':
		plt.xlabel(r'$N_{iter}$',**axis_font)
	elif xaxis=='m':
		plt.xlabel(r'$m$',**axis_font)
	elif xaxis=='k':
		plt.xlabel(r'$k$',**axis_font)
	elif xaxis=='lambda_J':
		plt.xlabel(r'$\lambda_J$',**axis_font)
	elif xaxis=='batch_size':
		plt.xlabel(r'Batch size',**axis_font)
	
	plt.xlim(Xlim)

	plt.xticks(fontsize = 18)
	plt.yticks(fontsize = 18)
	if Reverse_xaxis:
		plt.gca().invert_xaxis()
	plt.legend(loc = 'lower left',fontsize = 17)
	plt.grid(True,which="both", ls="-", color='0.9')
	plt.show()


def Mean_J_features_CM(n_states,N_iter,Rep_list,lambd=None,m=None,k_list=None,fam='CM',Mod = 'SBM'):
	if Mod=='SBM':M=''
	else:M=Mod+'_'
	for r in range(len(Rep_list)):
		if k_list is None:
			if Mod=='BM':
				f = M+fam+'_Ns'+str(n_states)+'Ni'+str(N_iter)+'lb'+str(lambd)+Rep_list[r]+'.npy'
			elif Mod=='SBM':
				f = M+fam+'_m'+str(m)+'Ns'+str(n_states)+'Ni'+str(N_iter)+Rep_list[r]+'.npy'
			elif Mod=='Profile':
				f = fam+'_'+Mod+'.npy'
		else:
			f = 'CM_m'+str(m)+'Ns'+str(n_states)+'Ni'+str(N_iter)+'k'+str(k_list)+Rep_list[r]+'.npy'
		
		output_SBM = np.load('results/Article/'+Mod+'/'+fam+'/'+f,allow_pickle=True)[()]

		J_SBM = output_SBM['J']
		h_SBM = output_SBM['h']

		if fam=='CM':
			Pos_rem = np.array([1,65]) -1 #,1, 2, 3, 4, 64, 67]) - 1
			Pos_keep = np.delete(np.arange(output_SBM['J'].shape[0]),Pos_rem)
			J_SBM = J_SBM[Pos_keep]
			J_SBM = J_SBM[:,Pos_keep]
			h_SBM = h_SBM[Pos_keep]

		J_SBM,h_SBM = ut.Zero_Sum_Gauge(J_SBM,h_SBM)
		
		if r==0:
			J_mean = np.copy(J_SBM)
			h_mean = np.copy(h_SBM)
		else:
			J_mean += J_SBM
			h_mean += h_SBM
	return J_mean/len(Rep_list), h_mean/len(Rep_list)

####################### OTHER SCORES #######################
		
def SBM_compute_several_scores(ns_list,Ni_list,m_list,Rep_list,k_list=None,TM_nb = 1,delta_t = 1e4):
	E_G = np.zeros((len(Rep_list),len(ns_list)))
	E_F = np.zeros((len(Rep_list),len(ns_list)))

	for r in range(len(Rep_list)):
		for i in range(len(ns_list)):
			if k_list is None:
				f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+Rep_list[r]+'.npy'
			else:
				f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'k'+str(k_list[i])+Rep_list[r]+'.npy'
			MSA_test = np.load('data/MSA_array/MSA_test_ToyModel.npy')

			output_SBM = np.load('results/Toy_model/SBM_TM/'+f,allow_pickle=True)[()]
			align_SBM=ut.Create_modAlign(output_SBM,N = output_SBM['align'].shape[0],delta_t = delta_t)
			output_SBM['align_mod'] = align_SBM
			output_SBM['Test'] = MSA_test

			e_g,e_f,_,_,_ = ut.compute_CustomScore(output_SBM)
			E_F[r,i] = e_f
			E_G[r,i] = e_g
	return E_G, E_F
	
def SBM_plot_scores(Xaxis_list,E_G,E_F,E_R,Xlim,xaxis='n_states'):
	axis_font = {'size':'18'}
	fig = plt.figure(figsize = (9,3))
	#fig.patch.set_facecolor((0.921,0.921,0.921))

	mean_eG = np.mean(E_G,axis=0)
	mean_eF = np.mean(E_F,axis=0)
	mean_eR = np.mean(E_R,axis=0)
	eb_eG = 1.96*np.std(E_G,axis=0)/max(np.sqrt(E_G.shape[0]-1),1)
	eb_eF = 1.96*np.std(E_F,axis=0)/max(np.sqrt(E_F.shape[0]-1),1)
	eb_eR = 1.96*np.std(E_R,axis=0)/max(np.sqrt(E_R.shape[0]-1),1)

	plt.plot(Xaxis_list,mean_eG,'--o',linewidth = 2.4,color = 'tab:blue',label=r'$W(E_{train},E_{test})$')
	plt.fill_between(Xaxis_list,mean_eG-eb_eG,mean_eG+eb_eG,color = 'tab:blue',alpha=0.2)

	plt.plot(Xaxis_list,mean_eF,'--o',linewidth = 2.4,color = 'tab:orange',label=r'$W(E_{train},E_{model})$')
	plt.fill_between(Xaxis_list,mean_eF-eb_eF,mean_eF+eb_eF,color = 'tab:orange',alpha=0.2)
	plt.ylim([0-0.03,np.max(mean_eG)+1])
	plt.grid(True,axis='both',which="both", ls="-", color='0.9')
	plt.legend(loc='upper right')

	if xaxis=='n_states':
		plt.xlabel(r'$N_{chains}$',**axis_font)
		#plt.xlim([0.8,3000])
		plt.xlim(Xlim)
	elif xaxis=='N_iter':
		plt.xlabel(r'$N_{iter}$',**axis_font)
		#plt.xlim([0.8,1100])
		plt.xlim(Xlim)
	elif xaxis=='m':
		plt.xlabel(r'$m$',**axis_font)
		#plt.xlim([0.8,60])
		plt.xlim(Xlim)
	elif xaxis=='k':
		plt.xlabel(r'$k$',**axis_font)
		plt.xlim(Xlim)
	elif xaxis=='lambda':
		plt.xlabel(r'$\lambda_J$',**axis_font)
		plt.xlim(Xlim)

	#plt.ylim([0-0.03,1+0.03])
	#plt.ylim([0-0.03,0.45])

	plt.xscale('log')
	plt.xticks(size = 13)
	plt.yticks(size = 13)
	plt.gca().invert_xaxis()
	ax1 = plt.gca()
	ax1.set_ylabel(r'$W(E_{train},E_{model}),\ W(E_{train},E_{test})$',**{'size':'10'})#,color = 'tab:blue')

	ax2 = ax1.twinx()
	ax2.set_ylabel(r'$W(E_{train},E_{rand})$',**{'size':'14'},color = 'gray')
	plt.plot(Xaxis_list,mean_eR,'--o',linewidth = 2.4,color = 'gray',label=r'$W(E_{train},E_{rand})$')
	plt.fill_between(Xaxis_list,mean_eR-eb_eR,mean_eR+eb_eR,color = 'gray',alpha=0.2)
	plt.ylim([0-0.03,np.max(mean_eR)+1])
	#plt.legend()
	plt.show()


def SBM_compute_eps_score(ns_list,Ni_list,m_list,Rep_list,k_list=None,batch_list=None,theta=0.2,delta_t=10000,TM_nb = 1):
	folder = 'SBM_batch'
	EPS2_art = np.zeros((len(Rep_list),len(ns_list)))
	EPS2_nat = np.zeros((len(Rep_list),len(ns_list)))
	EPS2_rand = np.zeros((len(Rep_list),len(ns_list)))
	EPS_art = np.zeros((len(Rep_list),len(ns_list)))
	EPS_nat = np.zeros((len(Rep_list),len(ns_list)))
	EPS_rand = np.zeros((len(Rep_list),len(ns_list)))
	for r in range(len(Rep_list)):
		for i in range(len(ns_list)):
			if k_list is None:
				if batch_list is None:
					f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+Rep_list[r]+'.npy'
				else:
					f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'Bf'+str(batch_list[i])+Rep_list[r]+'.npy'
			else:
				f = 'TM'+str(TM_nb)+'_m'+str(m_list[i])+'Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'k'+str(k_list[i])+Rep_list[r]+'.npy'
			output_SBM = np.load('results/Toy_model/'+folder+'/'+f,allow_pickle=True)[()]
			Gen_train = output_SBM['Train']
			Gen_test = np.load('data/MSA_array/MSA_test_ToyModel.npy')
			Gen_rand = np.random.randint(output_SBM['options']['q'],size = output_SBM['Train'].shape)
			Gen_mod = ut.Create_modAlign(output_SBM,output_SBM['Train'].shape[0],delta_t=delta_t)
			Gen_test = Gen_test[np.random.choice(np.arange(Gen_test.shape[0]),Gen_train.shape[0],replace=False)]

			EPS2_art[r,i] = ut.compute_eps2(Gen_test,Gen_mod,output_SBM['options']['q'],theta)
			EPS2_nat[r,i] = ut.compute_eps2(Gen_test,Gen_train,output_SBM['options']['q'],theta)
			EPS2_rand[r,i] = ut.compute_eps2(Gen_test,Gen_rand,output_SBM['options']['q'],theta)

			EPS_art[r,i] = ut.compute_eps(Gen_test,Gen_mod,output_SBM['options']['q'],theta)
			EPS_nat[r,i] = ut.compute_eps(Gen_test,Gen_train,output_SBM['options']['q'],theta)
			EPS_rand[r,i] = ut.compute_eps(Gen_test,Gen_rand,output_SBM['options']['q'],theta)

	EPS2 = {'EPS2_art':EPS2_art,
			'EPS2_nat':EPS2_nat,
			'EPS2_rand':EPS2_rand}
	EPS = {'EPS_art':EPS_art,
			'EPS_nat':EPS_nat,
			'EPS_rand':EPS_rand}
	return EPS2,EPS

def plot_EPS2(Xaxis_list,EPS2,EPS2_ref,EPS2_rand,Xlim,xaxis='n_states',Reverse_xaxis = False):
	axis_font = {'size':'18'}
	fig = plt.figure(figsize = (9,3))
	#fig.patch.set_facecolor((0.921,0.921,0.921))

	mean_EPS2 = np.mean(EPS2,axis=0)
	eb_EPS2 = 1.96*np.std(EPS2,axis=0)/max(np.sqrt(EPS2.shape[0]-1),1)
	mean_EPS2r = np.mean(EPS2_ref,axis=0)
	eb_EPS2r = 1.96*np.std(EPS2_ref,axis=0)/max(np.sqrt(EPS2_ref.shape[0]-1),1)
	mean_EPS2_rand = np.mean(EPS2_rand,axis=0)
	eb_EPS2_rand = 1.96*np.std(EPS2_rand,axis=0)/max(np.sqrt(EPS2_rand.shape[0]-1),1)

	plt.plot(Xaxis_list,mean_EPS2,'--o',linewidth = 2.4,color = 'tab:blue',label='Artificial VS Test')
	plt.fill_between(Xaxis_list,mean_EPS2-eb_EPS2,mean_EPS2+eb_EPS2,color = 'tab:blue',alpha=0.2)

	plt.plot(Xaxis_list,mean_EPS2r,'--o',linewidth = 2.4,color = 'tab:green',label='Train VS Test')
	plt.fill_between(Xaxis_list,mean_EPS2r-eb_EPS2r,mean_EPS2r+eb_EPS2r,color = 'tab:green',alpha=0.2)

	plt.plot(Xaxis_list,mean_EPS2_rand,'--o',linewidth = 2.4,color = 'grey',label='Random VS Test')
	plt.fill_between(Xaxis_list,mean_EPS2_rand-eb_EPS2_rand,mean_EPS2_rand+eb_EPS2_rand,color = 'grey',alpha=0.2)

	if xaxis=='n_states':
		plt.xlabel(r'$N_{chains}$',**axis_font)
		plt.xlim(Xlim)
	elif xaxis=='N_iter':
		plt.xlabel(r'$N_{iter}$',**axis_font)
		plt.xlim(Xlim)
	elif xaxis=='m':
		plt.xlabel(r'$m$',**axis_font)
		plt.xlim(Xlim)
	elif xaxis=='k':
		plt.xlabel(r'$k$',**axis_font)
		plt.xlim(Xlim)
	elif xaxis=='lambda_J':
		plt.xlabel(r'$\lambda_J$',**axis_font)
		plt.xlim(Xlim)

	#plt.ylim([0.5e-4,1.6e-4])
	plt.ylim([0,1e-3])
	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	plt.xscale('log')
	plt.xticks(size = 13)
	plt.yticks(size = 13)
	if Reverse_xaxis:	
		plt.gca().invert_xaxis()
	ax1 = plt.gca()
	ax1.set_ylabel(r'$\epsilon^{(2)}$',**axis_font,color = 'tab:blue')
	plt.grid(True,which="both", ls="-", color='0.9')
	plt.legend()
	plt.show()


def BM_compute_several_scores(ns_list,Ni_list,lb_list,Rep_list,k_list=None,TM_nb = 1,delta_t = 1e4):
	E_G = np.zeros((len(Rep_list),len(ns_list)))
	E_F = np.zeros((len(Rep_list),len(ns_list)))

	for r in range(len(Rep_list)):
		for i in range(len(ns_list)):
			if k_list is None:
				f = 'BM_TM'+str(TM_nb)+'_Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'lb'+str(lb_list[i])+Rep_list[r]+'.npy'
			else:
				f = 'BM_TM'+str(TM_nb)+'_Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'lb'+str(lb_list[i])+'k'+str(int(k_list[i]))+Rep_list[r]+'.npy'
			MSA_test = np.load('data/MSA_array/MSA_test_ToyModel.npy')

			output_SBM = np.load('results/Toy_model/BM_TM/'+f,allow_pickle=True)[()]
			align_SBM=ut.Create_modAlign(output_SBM,N = output_SBM['align'].shape[0],delta_t = delta_t)
			output_SBM['align_mod'] = align_SBM
			output_SBM['Test'] = MSA_test

			e_g,e_f,_,_,_ = ut.compute_CustomScore(output_SBM)
			E_F[r,i] = e_f
			E_G[r,i] = e_g
			#print(output_SBM['options']['maxIter'])
	return E_G, E_F


def BM_compute_eps2_score(ns_list,Ni_list,lb_list,Rep_list,k_list=None,theta=0.2,delta_t=10000,TM_nb = 1):
	EPS2_art = np.zeros((len(Rep_list),len(ns_list)))
	EPS2_nat = np.zeros((len(Rep_list),len(ns_list)))
	EPS2_rand = np.zeros((len(Rep_list),len(ns_list)))
	EPS_art = np.zeros((len(Rep_list),len(ns_list)))
	EPS_nat = np.zeros((len(Rep_list),len(ns_list)))
	EPS_rand = np.zeros((len(Rep_list),len(ns_list)))

	for r in range(len(Rep_list)):
		for i in range(len(ns_list)):
			if k_list is None:
				f = 'BM_TM'+str(TM_nb)+'_Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'lb'+str(lb_list[i])+Rep_list[r]+'.npy'
			else:
				f = 'BM_TM'+str(TM_nb)+'_Ns'+str(ns_list[i])+'Ni'+str(Ni_list[i])+'lb'+str(lb_list[i])+'k'+str(int(k_list[i]))+Rep_list[r]+'.npy'
			output_SBM = np.load('results/Toy_model/BM_TM/'+f,allow_pickle=True)[()]
			Gen_train = output_SBM['Train']
			Gen_test = np.load('data/MSA_array/MSA_test_ToyModel.npy')
			Gen_rand = np.random.randint(output_SBM['options']['q'],size = output_SBM['Train'].shape)
			Gen_mod = ut.Create_modAlign(output_SBM,output_SBM['Train'].shape[0],delta_t=delta_t)
			Gen_test = Gen_test[np.random.choice(np.arange(Gen_test.shape[0]),Gen_train.shape[0],replace=False)]
			EPS2_art[r,i] = ut.compute_eps2(Gen_test,Gen_mod,output_SBM['options']['q'],theta)
			EPS2_nat[r,i] = ut.compute_eps2(Gen_test,Gen_train,output_SBM['options']['q'],theta)
			EPS2_rand[r,i] = ut.compute_eps2(Gen_test,Gen_rand,output_SBM['options']['q'],theta)

			EPS_art[r,i] = ut.compute_eps(Gen_test,Gen_mod,output_SBM['options']['q'],theta)
			EPS_nat[r,i] = ut.compute_eps(Gen_test,Gen_train,output_SBM['options']['q'],theta)
			EPS_rand[r,i] = ut.compute_eps(Gen_test,Gen_rand,output_SBM['options']['q'],theta)

	EPS2 = {'EPS2_art':EPS2_art,
			'EPS2_nat':EPS2_nat,
			'EPS2_rand':EPS2_rand}
	EPS = {'EPS_art':EPS_art,
			'EPS_nat':EPS_nat,
			'EPS_rand':EPS_rand}
	return EPS2,EPS

################## ADDED BY EMILY HINDS #######################

def compare_to_dms_regression(h, J, wt, dms_file = "/Users/emily/Downloads/CM_DMS.mat",
                              outfile_name='', per_position=True,
                              include_wt=False, char_skip={0}):
    """
    For a provided Potts model, compare the in silico deep mutational scan (DMS)
    to provided experimental data, and calculate both spearman rho and pearson r.
    
    NOTE: While in theory one could input non-chorismate mutase DMSes,
          currently all of the file processing in this function is hard-coded
          to the format of the chorismate mutase DMS data distribution, which is
          in a .mat file and not following any particular standard (as far as I'm
          aware). In the future it would perhaps be beneficial to identify a standard
          file format for the DMS data and use that as input to this function.
          
    Args:
    - h: numpy array, L x Q matrix of fields (single-site parameters) of Potts model
    - J: numpy array, L x L x Q x Q matrix of couplings (pairwise-site parameters) of Potts model
    - wt: numpy array of reference sequence in numerical format. Will be used to generate in silico DMS.
    - dms_file: string with .mat file specifying location of experimental DMS data. Default is E. coli CM.
    - outfile_name: optional string, file to which the plot should be saved (.svg, .png, or other file formats
                          supported by matplotlib). If empty string, do not save plot, just print
                          to console. Default is empty string.
    - per_position: optional bool, average mutational effects (both experimental and in silico) by position.
                          Default True
    - include_wt: optional bool, include the wild-type sequence in comparisons. If per_position is also true,
                        will be included in each position-average; if per_position is false, will be included once
                        in entire comparison. Default False
    - char_skip: optional set (ideal) or list, values in set Q that, if present in wt,
                       should not be mutated to other residues and to which other
                       residues should not be mutated (e.g. gap characters for proteins).
                       Default: {0} (number 0, not empty set) corresponding to gap character in
                       protein alignments
           
    Returns:
    - float r2 corresponding to the squared Pearson correleation coefficient
    """

    # Read in experimental DMS data excluding the stop codons
    expt_data = sio.loadmat(dms_file)['dms'][0,0][1][:-1,:].T.flatten()
    
    # Get the in silico energiesCM_YaakovMOD.npy of the DMS under provided model
    dms_alg = ut.get_dms(wt=wt, Q=h.shape[1], char_skip=char_skip, include_wt=True)
    pred_data = ut.compute_energies(dms_alg, h, J).flatten()
    
    # Process out WT sequence if you don't want to include it in comparisons
    # This is messy but it seems to work ok
    wt_indices = np.array([x for x in range(dms_alg.shape[0]) if np.all(dms_alg[x]==wt)])
    pred_data -= pred_data[wt_indices[0]] # normalize energies to WT sequence
    
    if not include_wt:
        expt_data[wt_indices] = np.nan
        pred_data[wt_indices] = np.nan
    elif not per_position: 
        # include_wt is implicit if we hit this code_block
        # but we only want to include one WT point, not L points
        # but only if we're not averaging per-position
        expt_data[wt_indices[1:]] = np.nan
        pred_data[wt_indices[1:]] = np.nan
    
    # Take the position averages, if you want
    if per_position:
        """
        I think even though expt data is log relative enrichment, I do want to be doing
        the averages over the log space rather than over exp(data)? Average over log10(r.e.)
        was what was done in Kleeorin et. al. 2023 Fig 7C (including stop codons, which are
        excluded here)
        """
        expt_data = np.nanmean(expt_data.reshape(-1,20), axis=1).reshape(-1,1)
        
        """
        Note: uncomment the below line if you want to take averages in non-log space.
        Since the reported data are in log base 10, we'll stick with that
        """
        #expt_data = np.log10(np.nanmean(np.power(10,expt_data.reshape(-1,20)), axis=1)).reshape(-1,1)
        
        pred_data = np.nanmean(pred_data.reshape(-1,20), axis=1).reshape(-1,1)
    else:           
        expt_data = np.hstack([x for x in expt_data if not np.isnan(x)]).reshape(-1,1)
        pred_data = np.hstack([x for x in pred_data if not np.isnan(x)]).reshape(-1,1)

    # Compare model dEs to experimental data
    model = LinearRegression(fit_intercept = True)
    model.fit(expt_data.reshape(-1,1), pred_data.reshape(-1,1))

    sp_r, pval = spearmanr(pred_data.flatten(), expt_data.flatten())
    r_sq = np.corrcoef(pred_data.flatten(), expt_data.flatten())[0,1]**2

    # Plotting
    x_new = np.array([expt_data.min(), expt_data.max()]).reshape(-1,1)
    y_new = model.predict(x_new)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5,3.5))
    
    ax.scatter(expt_data, pred_data, color='black', alpha=0.7)
    ax.plot(x_new, y_new, 'r--')   
    ax.text(0.01, 0.02, '$\\rho=%.3f$\n$r^2=%.3f$'%(sp_r, r_sq), 
            ha='left', va='bottom', transform=ax.transAxes,fontsize=12)

    plt.xlabel("Relative Enrichment", fontsize=14)
    plt.ylabel("In silico $\Delta E$", fontsize=14)

    if per_position:
        plt.title("Position Averaged Effects", fontsize=15)
    else:
        plt.title("Individual Effects", fontsize=15)

    plt.setp(ax.get_xticklabels(), fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    plt.grid(alpha=0.6)
    plt.tight_layout()
    
    if len(outfile_name)>0:
        plt.savefig(outfile_name)

    plt.show()

    return r_sq


