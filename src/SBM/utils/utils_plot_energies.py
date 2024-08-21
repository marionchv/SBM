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
from tqdm import tqdm
import plotly.graph_objects as go
import itertools

##########################################################

####################### PLOT STATISTICAL ENERGY DISTRIBUTIONS #######################

def BARplot_energies(file_name,Mod_list,fam,Ylim=None,delta_t=1e4,ITER='',Temp=1):
	Energies = {
		'x_Etrain' : [],'Energies_train' : [],
		'x_Etest' : [],'Energies_test' : [],
		'x_Emod' : [],'Energies_mod' : [],
		'x_Erand1' : [],'Energies_rand1' : [],
		'x_Erand2' : [],'Energies_rand2' : []}
	
	for i in range(len(file_name)):
		f = file_name[i]
		output = np.load('results/'+Mod_list[i]+'/'+fam+'/'+f,allow_pickle=True)[()]
		if 'align_mod' not in list(output.keys()):
			print('generate align_mod')
			N = output['Train'].shape[0]
			align_mod=ut.Create_modAlign(output,N = N,delta_t = delta_t,ITER=ITER,temperature=Temp)
			output['align_mod'] = align_mod

		if fam in ['TM1','TM1_Init_Profile']:
			output['Test'] = np.load('data/MSA_array/MSA_test_ToyModel.npy')

		if output['Test'] is None: align_test=output['Train']
		else:align_test= output['Test']
		align_train= output['Train']
		align_mod= output['align_mod']
		SBM_J,SBM_h = output['J'+str(ITER)],output['h'+str(ITER)]
		SBM_J,SBM_h = ut.Zero_Sum_Gauge(SBM_J,SBM_h)

		# if fam=='CM':
		# 	Pos_rem = np.array([65,1, 2, 3, 4, 64, 67]) - 1
		# 	Pos_keep = np.delete(np.arange(output['J'].shape[0]),Pos_rem)
		# 	align_test,align_train,align_mod = align_test[:,Pos_keep],align_train[:,Pos_keep], align_mod[:,Pos_keep]
		# 	SBM_J = SBM_J[Pos_keep]
		# 	SBM_J = SBM_J[:,Pos_keep]
		# 	SBM_h = SBM_h[Pos_keep]

		rand1 = np.random.randint(output['options']['q'],size = align_mod.shape)
		rand2 = ut.shuff_column(align_train)

		if Mod_list[i]=='SBM':
			lab = r'SBM'+str(i) 
		else:
			lab = r'BM'+str(i) 


		Erand1 = ut.compute_energies(rand1,SBM_h,SBM_J)
		Erand2 = ut.compute_energies(rand2,SBM_h,SBM_J)
		Etest = ut.compute_energies(align_test,SBM_h,SBM_J)
		#align_train_energies = align_train[np.random.choice(np.arange(align_train.shape[0]),align_test.shape[0],replace=False)]
		Etrain = ut.compute_energies(align_train,SBM_h,SBM_J)#ut.RemoveCloseSeqs(align_train,0.2),SBM_h,SBM_J)
		Emod = ut.compute_energies(align_mod,SBM_h,SBM_J)
		Mean = np.median(Etrain)
		print(np.std(Etrain))
		STD = 1 #np.std(Etrain)
		Erand1,Erand2,Etrain,Emod,Etest = (Erand1-Mean)/STD,(Erand2-Mean)/STD,(Etrain-Mean)/STD,(Emod-Mean)/STD,(Etest-Mean)/STD
		
		Energies['Energies_rand1'].extend(Erand1)
		Energies['Energies_rand2'].extend(Erand2)
		Energies['x_Erand1'].extend([lab]*len(Erand1))
		Energies['x_Erand2'].extend([lab]*len(Erand2))

		Energies['Energies_train'].extend(Etrain)
		Energies['x_Etrain'].extend([lab]*len(Etrain))

		Energies['Energies_test'].extend(Etest)
		Energies['x_Etest'].extend([lab]*len(Etest))

		Energies['Energies_mod'].extend(Emod)
		Energies['x_Emod'].extend([lab]*len(Emod))

	Pairs_TrainArt = list(itertools.product(Energies['Energies_train'],Energies['Energies_mod']))
	A = np.array([d[0] for d in Pairs_TrainArt])
	B = np.array([d[1] for d in Pairs_TrainArt])
	print('Overlap score: ',np.sum(((A-B)>0).astype('int'))/len(A))


	Pairs_TrainArt = list(itertools.product(Energies['Energies_rand2'],Energies['Energies_mod']))
	A = np.array([d[0] for d in Pairs_TrainArt])
	B = np.array([d[1] for d in Pairs_TrainArt])
	print('Overlap score Rc/Art: ',np.sum(((A-B)>0).astype('int'))/len(A))

	c1, c2, c3 = 'rgb(0.279,0.681,0.901)','rgb(0.616,0.341,0.157)','rgb(0.092,0.239,0.404)'
	fig = go.Figure()

	fig.add_trace(go.Box(x= Energies['x_Etrain'], y = Energies['Energies_train'],name = 'Train', marker_color = c3,boxpoints = False,boxmean=True))
	fig.add_trace(go.Box(x= Energies['x_Etest'], y = Energies['Energies_test'],name = 'Test', marker_color = c1,boxpoints = False))
	fig.add_trace(go.Box(x= Energies['x_Emod'], y = Energies['Energies_mod'],name = 'Artificial (T=1)', marker_color = c2,boxpoints = False,boxmean=True))
	fig.add_trace(go.Box(x= Energies['x_Erand1'], y = Energies['Energies_rand1'],name = 'Random', marker_color = 'black',boxpoints = False,boxmean=True))
	fig.add_trace(go.Box(x= Energies['x_Erand2'], y = Energies['Energies_rand2'],name = 'Random col.', marker_color = 'grey',boxpoints = False,boxmean=True))
	
	fig.update_layout(yaxis=dict(title='Statistical Energies', zeroline=False),boxmode='group',
				boxgroupgap=0,boxgap=0.5,plot_bgcolor='white',
				height=550, width=560,font_size=20,legend_font_size=20)
	
	fig.update_xaxes(mirror=True,showline=True,linecolor='black')
	fig.update_yaxes(mirror=True,ticks='outside',showline=True,
		linecolor='black',gridcolor='lightgrey')
	if Ylim is not None:
		fig.update_yaxes(range =Ylim)
	fig.show()

def BARplot_energies_BM_T(file_name,Mod_list,fam,delta_t=1e4):
	Energies = {
		'x_Etrain' : [],'Energies_train' : [],
		'x_Etest' : [],'Energies_test' : [],
		'x_Emod1' : [],'Energies_mod1' : [],
		'x_EmodlowT' : [],'Energies_modlowT' : [],
		'x_Erand' : [],'Energies_rand' : []}
	
	for i in range(len(file_name)):
		f = file_name[i]
		output = np.load('results/Article/'+Mod_list[i]+'/'+fam+'/'+f,allow_pickle=True)[()]
		if 'align_mod' not in list(output.keys()):
			align_mod=ut.Create_modAlign(output,N = output['Train'].shape[0],delta_t = delta_t)
			output['align_mod'] = align_mod

		if fam=='TM1':
			output['Test'] = np.load('data/MSA_array/MSA_test_ToyModel.npy')

		if output['Test'] is None: align_test=output['Train']
		else:align_test= output['Test']
		align_train= output['Train']
		align_mod= output['align_mod']
		SBM_J,SBM_h = output['J'],output['h']
		SBM_J,SBM_h = ut.Zero_Sum_Gauge(SBM_J,SBM_h)

		if fam=='CM':
			Pos_rem = np.array([65,1, 2, 3, 4, 64, 67]) - 1
			Pos_keep = np.delete(np.arange(output['J'].shape[0]),Pos_rem)
			align_test,align_train,align_mod = align_test[:,Pos_keep],align_train[:,Pos_keep], align_mod[:,Pos_keep]
			SBM_J = SBM_J[Pos_keep]
			SBM_J = SBM_J[:,Pos_keep]
			SBM_h = SBM_h[Pos_keep]

		rand = np.random.randint(output['options']['q'],size = align_mod.shape)

		if Mod_list[i]=='SBM':
			lab = r'SBM'
		else:
			lab = r'BM'

		Erand = ut.compute_energies(rand,SBM_h,SBM_J)
		Etest = ut.compute_energies(align_test,SBM_h,SBM_J)
		#align_train_energies = align_train[np.random.choice(np.arange(align_train.shape[0]),align_test.shape[0],replace=False)]
		Etrain = ut.compute_energies(ut.RemoveCloseSeqs(align_train,0.2),SBM_h,SBM_J)
		Mean = np.median(Etrain)
		print(np.std(Etrain))
		STD = 1 #np.std(Etrain)
		if i==0:
			Erand,Etrain,Etest = (Erand-Mean)/STD,(Etrain-Mean)/STD,(Etest-Mean)/STD
			Energies['Energies_rand'].extend(Erand)
			Energies['x_Erand'].extend([lab]*len(Erand))

			Energies['Energies_train'].extend(Etrain)
			Energies['x_Etrain'].extend([lab]*len(Etrain))

			Energies['Energies_test'].extend(Etest)
			Energies['x_Etest'].extend([lab]*len(Etest))
		Emod = ut.compute_energies(align_mod,SBM_h,SBM_J)
		Emod = (Emod-Mean)/STD
		if i==0:
			Energies['Energies_mod1'].extend(Emod)
			Energies['x_Emod1'].extend([lab]*len(Emod))
		else:
			Energies['Energies_modlowT'].extend(Emod)
			Energies['x_EmodlowT'].extend([lab]*len(Emod))

	c1, c2, c3 = 'rgb(0.279,0.681,0.901)','rgb(0.616,0.341,0.157)','rgb(0.092,0.239,0.404)'
	c4 = 'rgb(0.986,0.544,0.248)'
	fig = go.Figure()

	fig.add_trace(go.Box(x= Energies['x_Etrain'], y = Energies['Energies_train'],name = 'Train', marker_color = c3,boxpoints = False,boxmean=True))
	fig.add_trace(go.Box(x= Energies['x_Etest'], y = Energies['Energies_test'],name = 'Test', marker_color = c1,boxpoints = False))
	fig.add_trace(go.Box(x= Energies['x_Emod1'], y = Energies['Energies_mod1'],name = 'Artificial (T=1)', marker_color = c2,boxpoints = False,boxmean=True))
	fig.add_trace(go.Box(x= Energies['x_EmodlowT'], y = Energies['Energies_modlowT'],name = 'Artificial (T=0.75)', marker_color = c4,boxpoints = False,boxmean=True))
	fig.add_trace(go.Box(x= Energies['x_Erand'], y = Energies['Energies_rand'],name = 'Random', marker_color = 'grey',boxpoints = False,boxmean=True))

	fig.update_layout(yaxis=dict(title='Statistical Energies', zeroline=False),boxmode='group',
				boxgroupgap=0,boxgap=0.5,plot_bgcolor='white',
				height=550, width=580,font_size=20,legend_font_size=20)
	
	fig.update_xaxes(mirror=True,showline=True,linecolor='black')
	fig.update_yaxes(mirror=True,ticks='outside',showline=True,
		linecolor='black',gridcolor='lightgrey')
	#fig.update_yaxes(range =[-100,420])
	fig.show()



def Test_averaged_model(file_name,Mod_list,fam):
	Energies = {
		'x_Etrain' : [],
		'Energies_train' : [],
		'x_Etest' : [],
		'Energies_test' : []
	}
	for i in range(len(file_name)):
		f = file_name[i]
		output = np.load('results/Article/'+Mod_list[i]+'/'+fam+'/'+f,allow_pickle=True)[()]

		align_test = np.load('data/MSA_array/MSA_test_ToyModel.npy')
		align_train= output['Train']
		SBM_J,SBM_h = output['J'],output['h']

		lab=str(i)
		
		Etrain = ut.compute_energies(align_train,SBM_h,SBM_J)
		Etest=ut.compute_energies(align_test,SBM_h,SBM_J)
		if i==0:
			Etest_true=ut.compute_energies(align_test,SBM_h,SBM_J)
			Etrain_true = ut.compute_energies(align_train,SBM_h,SBM_J)
			Mean = np.mean(Etrain_true)
		Etrain = Etrain - Mean #Etrain_true
		Etest = Etest - Mean #Etest_true

		Energies['Energies_train'].extend(Etrain)
		Energies['x_Etrain'].extend([lab]*len(Etrain))
		Energies['Energies_test'].extend(Etest)
		Energies['x_Etest'].extend([lab]*len(Etest))

	if True:
		c1, c2, c3 = 'rgb(0.279,0.681,0.901)','rgb(0.616,0.341,0.157)','rgb(0.092,0.239,0.404)'
		fig = go.Figure()

		fig.add_trace(go.Box(x= Energies['x_Etrain'], y = Energies['Energies_train'],name = 'Train', marker_color = c3,boxpoints = False))
		fig.add_trace(go.Box(x= Energies['x_Etest'], y = Energies['Energies_test'],name = 'Test', marker_color = c1,boxpoints = False))

		fig.update_layout(yaxis=dict(title='Statistical Energies', zeroline=False),boxmode='group',boxgroupgap=0,boxgap=0.5,plot_bgcolor='white',
																					height=600, width=1200,font_size=20,legend_font_size=20)
		
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
		#fig.update_yaxes(range =[-45,60])
		fig.show()


def Test_SepEnergies_model(file_name,Mod_list,fam):
	Energies = {
		'x_Eiso' : [],
		'Energies_iso' : [],
		'x_Esmall' : [],
		'Energies_small' : [],
		'x_Elarge' : [],
		'Energies_large' : [],
		'x_Eback' : [],
		'Energies_back' : []
	}
	output = np.load('results/Article/'+Mod_list[0]+'/'+fam+'/'+file_name[0],allow_pickle=True)[()]
	L,q = output['J'].shape[0],output['J'].shape[2]
	iso,small,large = np.zeros((L,L)),np.zeros((L,L)),np.zeros((L,L))
	iso[[4,6,8],[1,3,5]]=1
	iso = iso + iso.T
	small[10:14,10:14] = 1
	large[14:,14:] = 1
	mask_iso,mask_small,mask_large = np.zeros(output['J'].shape),np.zeros(output['J'].shape),np.zeros(output['J'].shape)
	for i in range(q):
		mask_iso[:,:,i,i],mask_small[:,:,i,i],mask_large[:,:,i,i] = iso,small,large
	mask_back = ((mask_iso+mask_small+mask_large)==0).astype('int')

	mask_isoh, mask_smallh, mask_largeh = np.zeros((L,q)),np.zeros((L,q)),np.zeros((L,q))
	mask_isoh[[4,6,8,1,3,5],:]= 1
	mask_smallh[10:14,:] = 1
	mask_largeh[14:,:] =1
	mask_backh = ((mask_isoh+mask_smallh+mask_largeh)==0).astype('int')

	for i in range(len(file_name)):
		f = file_name[i]
		output = np.load('results/Article/'+Mod_list[i]+'/'+fam+'/'+f,allow_pickle=True)[()]
		
		if Mod_list[i]=='BM':
			if i==0: lab='True'
			else:lab = str(i)+': '+str(output['options']['lambda_J'])
		elif Mod_list[i]=='SBM':
			if i==0: lab='True'
			else:lab = str(i)+': '+str(output['options']['n_states'])

		#align_test = np.load('data/MSA_array/MSA_test_ToyModel.npy')
		align_train= np.load('data/MSA_array/MSA_test_ToyModel.npy') #output['Train']#
		SBM_J,SBM_h = output['J'],output['h']
		SBM_J,SBM_h = ut.Zero_Sum_Gauge(SBM_J,SBM_h)
		#SBM_h = SBM_h*0
		SBM_J = SBM_J*0

		E_iso = ut.compute_energies(align_train,SBM_h*mask_isoh,SBM_J*mask_iso)
		if i==0:
			E_iso_true = ut.compute_energies(align_train,SBM_h*mask_isoh,SBM_J*mask_iso)
			Mean_iso = np.mean(E_iso)
		E_iso = E_iso #- Mean_iso #E_iso_true
		Energies['Energies_iso'].extend(E_iso)
		Energies['x_Eiso'].extend([lab]*len(E_iso))

		E_small = ut.compute_energies(align_train,SBM_h*mask_smallh,SBM_J*mask_small)
		if i==0:
			E_small_true = ut.compute_energies(align_train,SBM_h*mask_smallh,SBM_J*mask_small)
			Mean_small = np.mean(E_small)
		E_small = E_small #- Mean_small #E_small_true
		Energies['Energies_small'].extend(E_small)
		Energies['x_Esmall'].extend([lab]*len(E_small))

		E_large = ut.compute_energies(align_train,SBM_h*mask_largeh,SBM_J*mask_large)
		if i==0:
			E_large_true = ut.compute_energies(align_train,SBM_h*mask_largeh,SBM_J*mask_large)
			Mean_large = np.mean(E_large)
		E_large = E_large #- Mean_large #E_large_true
		Energies['Energies_large'].extend(E_large)
		Energies['x_Elarge'].extend([lab]*len(E_large))

		E_back = ut.compute_energies(align_train,SBM_h*mask_backh,SBM_J*mask_back)
		if i==0:
			E_back_true = ut.compute_energies(align_train,SBM_h*mask_backh,SBM_J*mask_back)
			Mean_back = np.mean(E_back)
		E_back = E_back #- Mean_back #E_back_true
		Energies['Energies_back'].extend(E_back)
		Energies['x_Eback'].extend([lab]*len(E_back))


	if True:
		c1, c2, c3 = 'rgb(0.957,0.643,0.376)','rgb(0.545,0.271,0.075)','rgb(0.275,0.510,0.706)'
		fig = go.Figure()

		fig.add_trace(go.Box(x= Energies['x_Eiso'], y = Energies['Energies_iso'],name = 'Isolated couplings', marker_color = c3,boxpoints = False))
		fig.add_trace(go.Box(x= Energies['x_Esmall'], y = Energies['Energies_small'],name = 'small', marker_color = c1,boxpoints = False))
		fig.add_trace(go.Box(x= Energies['x_Elarge'], y = Energies['Energies_large'],name = 'Large', marker_color = c2,boxpoints = False))
		fig.add_trace(go.Box(x= Energies['x_Eback'], y = Energies['Energies_back'],name = 'Background', marker_color = 'grey',boxpoints = False))


		fig.update_layout(yaxis=dict(title='Statistical Energies', zeroline=False),boxmode='group',boxgroupgap=0,boxgap=0.5,plot_bgcolor='white',
																					height=600, width=1200,font_size=20,legend_font_size=20)
		
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
		fig.update_yaxes(range =[-60,35])
		fig.show()

##########################################################