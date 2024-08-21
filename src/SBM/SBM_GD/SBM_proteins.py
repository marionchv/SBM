#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:25:00 2020

@author: yaakov KLEEORIN
Modified by: Marion CHAUVEAU
"""

####################### MODULES #######################

import numpy as np
import itertools as it
from scipy.spatial.distance import pdist,squareform
import C_MonteCarlo # type: ignore
from scipy import stats
from tqdm import tqdm
import time
import more_itertools as mit
import SBM.utils.utils as ut
import SBM.SCA.sca_pruning as prun

##########################################################

####################### OPTIONS #######################

def ParseOptions(options):
    assert options['Model'] in ['BM','SBM']
    Opt = [
        ('N_iter', 300), #nb of GD iterations
        ('N_chains', 1000),  #nb of states used to compute statistics
        ('m', 1), # Rank of the Hessian matrix (only for SBM)
        ('theta',0.2),
        ('k_MCMC',10000),

        ('alpha',0.2),  #Learning rate for the BM method
        ('Learning_rate',None),

        ('lambda_h', 0),   # regularization for the fields
        ('lambda_J', 0),    # regularization for the couplings

        ('Pruning', False),
        ('Pruning_perc',0.97),
        ('Pruning Mask', None),
        
        ('Param_init', 'Profile'), # Zero, Profile, Custom

        ('Test/Train', True), #If True and 'Train sequences' is None: the MSA is randomly splitted in a 80% training set / 20% test set
        ('Train sequences',None), #indices of sequences used for training

        ('Weights',None),

        ('Shuffle Columns',False),

        ('SGD',None), #for classic stochastic gradient descent

        ('Seed',None),

        ('Zero Fields',False), # True to impose zero fields

        ('Store Parameters', None) #if not None store couplings and fields every ** iterations ('Store Couplings', **)
    ]
    for k, v in Opt:
        if k not in options.keys():
            if options['Model']=='SBM':
                if k not in ['alpha','Learning rate']:
                    options[k] = v
            else:
                if k not in ['m']:
                    options[k] = v
    return options

##############################################################################

####################### FUNCTIONS for Initialization #######################

def Init_options(options,align):
    options=ParseOptions(options)

    options['q'] = np.max(align) + 1
    options['L'] = align.shape[1]

    ############# SEED #############
    if options['Seed'] is None:
        t = time.time()
        options['Seed'] = int(t)
    np.random.seed(options['Seed'])
    ################################

    return options

def Init_TestTrain(options,align):
    N = align.shape[0]

    ########## TEST/TRAIN ##########
    if options['Test/Train']:
        if options['Train sequences'] is None:
            ind_train = np.random.choice(N,int(0.80*N),replace = False)
        else:
            ind_train = options['Train sequences']
        train_align = align[ind_train]
        test_align = align[np.delete(np.arange(N),ind_train)]
        sim_test = ut.compute_similarities(test_align,train_align)
        test_align = test_align[(sim_test>0.2)]
    else:
        train_align = align
        test_align = None
    
    if options['Shuffle Columns']:
        print('Shuffle Columns...')
        train_align = ut.shuff_column(train_align)
    ################################
    return train_align,test_align

def Init_SGD(options,train_align):

    ########## SGD OPTIONS #########
    if options['SGD'] is not None:
        assert options['SGD'] <= train_align.shape[0]
        ind = np.arange(train_align.shape[0])
        np.random.shuffle(ind)
        options['Batches'] = list(mit.chunked(ind, options['SGD']))
        options['Num_batch'] = 0
    ################################

    if options['SGD'] is not None:
            align_subsamp = train_align
    else: align_subsamp = None

    return align_subsamp

def Init_statistics(options,train_align):

    ###### EVALUATE GOAL STATS #####
    print('Compute the statistics from the database....')
    if options['Weights'] is None:
        W,N_eff=ut.CalcWeights(train_align,options['theta'])
        print('Training size: ',train_align.shape[0])
        print('Effective training size: ',N_eff)
    else:
        assert len(options['Weights'])==train_align.shape[0]
        W = options['Weights']
        N_eff = np.sum(W)
    fi,fij=ut.CalcStatsWeighted(options['q'],train_align,W/N_eff)
    ################################

    return fi,fij,N_eff

def Init_Pruning(options,shape_fij,train_align):

    ############ PRUNING ###########
    if options['Pruning']:
        if options['Pruning Mask'] is None:

            if options['Pruning_perc'] ==0:
                options['Pruning Mask'] = np.ones(shape_fij).astype('bool')
            elif options['Pruning_perc']==1:
                options['Pruning Mask'] = np.zeros(shape_fij).astype('bool')
            else:
                # size = len(fij.flatten())
                # Mask = np.zeros(size)
                # ind_1 = np.random.choice(np.arange(size),int(size*(1-options['Pruning_perc'])),replace=False)
                # Mask[ind_1] = 1
                # Mask = Mask.reshape(fij.shape)

                Mask = prun.prune_params(train_align,cij=True, weight=0.8, pct = 100*options['Pruning_perc'])
                assert shape_fij == Mask.shape
                print('Pruning pct: ',1-np.sum(Mask)/np.sum(np.ones(Mask.shape)))

                options['Pruning Mask'] = Mask.astype('bool')
            
            # p_val = ut.compute_p_values(train_align,options['q'])
            # thresh = np.sort((p_val+p_val.T).flatten())[int((1-options['Pruning_perc'])*p_val.size)]
            # options['Pruning Mask'] = np.expand_dims(((p_val+p_val.T)<thresh),axis=(2,3))

            # Cij = ut.CalcCorr2(fi,fij)
            # thresh = np.sort(Cij.flatten())[int(0.97*Cij.size)] 
            # options['Pruning Mask'] = (Cij>thresh)
    ################################

def Init_Param(options,J0,h0,N_eff,fi):

    ########### PARAM INIT #########
    if options['Param_init'] == 'Zero':
        Jinit = np.zeros((options['L'],options['L'],options['q'],options['q']))
        hinit = np.zeros((options['L'],options['q']))
    elif options['Param_init'] == 'Profile':
        Jinit = np.zeros((options['L'],options['L'],options['q'],options['q']))
        alpha = 1/N_eff
        fi_init = (1-alpha)*fi + alpha/options['q']
        hinit = np.log(fi_init) #- np.log(1 - fi_init)
    elif options['Param_init'] == 'Custom':
        assert J0 is not None and h0 is not None
        Jinit = J0
        hinit = h0
    elif options['Param_init']=='Random':
        ma = 1
        Jinit = np.random.uniform(0,ma,(options['L'],options['L'],options['q'],options['q']))
        hinit = np.random.uniform(0,ma,(options['L'],options['q']))
    else:
        print('This "Param_init" option is not available')
        assert 0==1
    ################################

    ################################
    if options['Pruning']:Jinit *= options['Pruning Mask']
    if options['Zero Fields']:hinit*=0
        
    w0=Wj(Jinit,hinit)
    
    ################################

    return w0

##############################################################################

####################### FUNCTIONS OUTSIDE THE MINIMIZER #######################

def SBM(align,options,J0 = None,h0 = None):

    ###### SBM Initialization ######
    options = Init_options(options,align)
    train_align,test_align = Init_TestTrain(options,align)
    align_subsamp = Init_SGD(options,train_align)
    fi,fij,N_eff = Init_statistics(options,train_align)
    Init_Pruning(options,fij.shape,train_align)
    w0 = Init_Param(options,J0,h0,N_eff,fi)
    ################################
    
    ###### OBJECTIVE FUNCTION ######
    lamJ, lamh = options['lambda_J'], options['lambda_h']
    f=lambda x: GradLogLike(x,lamJ,lamh,fi,fij,options,align_subsamp=align_subsamp)
    ################################
    
    ####### GRADIENT DESCENT #######
    Ex_time=time.time()
    w,output=Minimizer(f,w0,options)
    output['Execution time'] = time.time()-Ex_time
    print('Execution time: ',time.time()-Ex_time)
    ################################

    output['options'] = options
    J,h=Jw(w,options['q'])
    output['J'] = J; output['h'] = h

    output['align'] = align
    output['Test'] = test_align; output['Train'] = train_align
    
    return output

def GradLogLike(w,lambdaJ,lambdah,fi,fij,options,align_subsamp=None):
    [J,h]=Jw(w,options['q'])
    
    ########## MODEL STATS #########
    if options['Zero Fields']:h*=0
    align_mod=CreateAlign(options['N_chains'],Wj(J,h),options['L'],options['q'],options['k_MCMC'])     
    p=np.zeros(options['N_chains'])+1/options['N_chains']
    fi_mod,fij_mod=ut.CalcStatsWeighted(options['q'],align_mod,p)
    ################################
    
    ########## SGD OPTIONS #########
    if options['SGD'] is not None:
        Batch = options['Batches'][options['Num_batch']]
        sub = align_subsamp[Batch]
        if options['Num_batch']==len(options['Batches'])-1:
            options['Num_batch']=0
        else:options['Num_batch'] = options['Num_batch'] + 1
        W,N_eff=ut.CalcWeights(sub,options['theta'])
        fi,fij=ut.CalcStatsWeighted(options['q'],sub,W/N_eff)
    ################################

    ####### COMPUTE GRADIENTS ######
    gradh=fi_mod-fi+2*lambdah*h
    gradJ=fij_mod-fij+2*lambdaJ*J
    ################################
    
    if options['Zero Fields']:gradh*=0
    if options['Pruning']:gradJ*=options['Pruning Mask']

    grad=Wj(gradJ,gradh)
    return grad

##########################################################
    
####################### FUNCTIONS INSIDE THE MINIMIZER #######################
        
def Minimizer(fun,x0,options):    
    x=x0
    output={'skipping':0,'J_norm':0}
    for i in tqdm(range(options['N_iter'])):
        
        ########## SBM METHOD #########
        if options['Model']=='SBM':
            if i==0:
                g = fun(x)
                h=-g
                s=np.zeros((x.shape[0],options['m']))
                y=np.zeros((x.shape[0],options['m']))
                ys=np.zeros((options['m']))
                diag=1
                gtd=np.dot(-g,h)
                ind=np.zeros(options['m'])-1;ind[0]=0
                t=1/np.sum(g**2)**0.7
            else:t=1
            x,h,g,gtd,s,y,ys,diag,ind,output['skipping']=AdvanceSearch(x,t,h,g,fun,gtd,s,y,ys,diag,ind,output['skipping'],options)
        ################################
        
        ########## BM METHOD ###########
        else:
            if options['Learning_rate'] is not None: t = options['Learning_rate'][i]
            else:t = 1/((i+1)**options['alpha'])
            grad = fun(x)
            x -= t*grad
        ################################

        ################################

        [J,h_field]=Jw(x,options['q'])
        if options['Store Parameters'] is not None:
            if i%options['Store Parameters']==0:
                output['J'+str(i)] = J
                output['h'+str(i)] = h_field
        if i%10==0:
            J_norm = np.mean(np.linalg.norm(J,'fro',axis = (2,3)))
            output['J_norm'] = np.append(output['J_norm'],np.round(J_norm,3))
        
    return x,output

def AdvanceSearch(x,t,h,g,fun,gtd,s,y,ys,diag,ind,skipping,options):
    count = 0
    while True:
        x_out=x+t*h
        # calculate the gradient
        g_out = fun(x_out)
        # and use it to update the h=hessian*gradient
        h_out,s_out,y_out,ys_out,diag_out,ind_out,skipping_out=UpdateHessian(g_out,g_out-g,x_out-x,s,y,ys,diag,ind,skipping,options)
        gtd_out=np.dot(-g_out,h_out)
        # sometimes this can be an irrelevant step, retry it if it strays too far
        if abs(gtd_out)<abs(50*gtd): break
        else:
            count += 1
            if count == 5000:
                print('too much irrelevant steps')
                break
    #hessian_save = h/(g_out+(g_out==0).astype('int'))
    return x_out,h_out,g_out,gtd_out,s_out,y_out,ys_out,diag_out,ind_out,skipping_out

def UpdateHessian(g,y,s,s_out,y_out,ys_out,diag,ind,skipping,options):
    ys=np.dot(y,s)
    # if this is a meaningful step
    if ys>10**(-10):
        y_out[:,ind==max(ind)]=y.reshape(-1,1)
        s_out[:,ind==max(ind)]=s.reshape(-1,1)
        ys_out[ind==max(ind)]=ys
        diag=ys/np.dot(y,y)
    # or if not meaningful, the update will be skipped
    else:
        skipping=skipping+1
    # here the hessian*gradient is calculated
    h_out=-g
    order=np.argsort(ind[ind>-1])
    alpha=np.zeros(order.shape[0])
    beta=np.zeros(order.shape[0])
    for i in order[::-1]:
        alpha[i]=np.dot(s_out[:,i],h_out)/ys_out[i]
        h_out=h_out-alpha[i]*y_out[:,i]
    h_out=diag*h_out
    for i in order:
        beta[i]=np.dot(y_out[:,i],h_out)/ys_out[i]
        h_out=h_out+s_out[:,i]*(alpha[i]-beta[i])
        
    # update the memory steps (indices) only if it is meaningful
    if ys>10**(-10):
        if ind[options['m']-1]==-1:
            ind=ind
            ind[(ind==max(ind)).nonzero()[0]+1]=max(ind)+1
        else:
            ind=np.roll(ind,1)    
    return h_out,s_out,y_out,ys_out,diag,ind,skipping

##########################################################
    
####################### TOOLS #######################

def Wj(J,h):
    """
    Function to translate J (L*L*q*q) and h (L*q)  into a vector of L*q + (L-1)*L*q*q/2 variables.

    Parameters:
    J : numpy array
        A 4D array representing J with dimensions (L, L, q, q).
    h : numpy array
        An array representing h with dimensions (L, q).

    Returns:
    W : numpy array
        A 1D numpy array of independent variables derived from J and h.
    """
    q=J.shape[2]
    L=J.shape[0]
    W=np.zeros(int((q*L+q*q*L*(L-1)/2),))
    x=np.array([[i,j] for i,j in it.combinations(range(L),2)])
    for a in range(q):
        for b in range(q):   
            W[(q**2*((x[:,0])*(2*L-x[:,0]-1)/2+(x[:,1]-x[:,0]-1))+(a)*q+b).astype(int)]=J[x[:,0],x[:,1],a,b]
    x=np.array(range(L))
    for a in range(q):
        W[(q**2*L*(L-1)/2+q*x[:]+a).astype(int)]=h[x[:],a]
    return W

def Jw(W,q):
    L=int(((q*q-2*q)+((2*q-q*q)**2+8*W.shape[0]*q*q)**(1/2))/2/q/q)
    J=np.zeros((L,L,q,q))
    h=np.zeros((L,q))
    x=np.array([[i,j] for i,j in it.combinations(range(L),2)])
    for a in range(q):
        for b in range(q):
            J[x[:,0],x[:,1],a,b]=W[(q**2*((x[:,0])*(2*L-x[:,0]-1)/2+(x[:,1]-x[:,0]-1))+(a)*q+b).astype(int)]
            J[x[:,1],x[:,0],b,a]=J[x[:,0],x[:,1],a,b]
    x=np.array(range(L))
    for a in range(q):
        h[x[:],a]=W[(q**2*L*(L-1)/2+q*x[:]+a).astype(int)]
    return J,h

def CreateAlign(N_chains,w,L,q,delta_t):
    """
	Function to create a alignment based on the provided parameters stored in w

	Args:
    - N_chains (int): Number of MCMC chains
	- w (numpy.array): A 1D numpy array containing 'h' and 'J' values.
    - L (int): Length of the sequences
    - q (int): Number of MCMC chains
	- delta_t (int): Number of MCMC steps

	Returns:
	- numpy.array: A 2D numpy array with the created alignment.
	"""
    seed = int(time.time())
    MSA=np.array(C_MonteCarlo.MC(np.array([x for x in w]),N_chains,L,int(q),delta_t,seed)).reshape(-1,L)
    return np.array(MSA,dtype = 'int64')

def errors(fi_mod, fi_msa, fij_mod, fij_msa):
    eps_f = np.sum(np.abs(fi_mod - fi_msa))/(fi_mod.shape[0]*fi_mod.shape[1])
    eps_s = np.sum(np.abs(fij_mod - fij_msa))/(fi_mod.shape[0]*fi_mod.shape[1])**2
    Cij_mod = fij_mod - np.reshape(np.expand_dims(fi_mod.flatten(),axis=1) @ np.expand_dims(fi_mod.flatten(),axis=1).T, fij_mod.shape)
    Cij_msa = fij_msa - np.reshape(np.expand_dims(fi_msa.flatten(),axis=1) @ np.expand_dims(fi_msa.flatten(),axis=1).T, fij_msa.shape)
    eps_c = np.amax(np.abs(Cij_mod - Cij_msa))
    Pc,_ = stats.pearsonr(fij_mod.flatten(), fij_msa.flatten())
    return (eps_f,eps_s,eps_c,Pc)

def CalcContingency(q,MSA):
    L=MSA.shape[1];    
    fi=np.zeros([L,q])
    x=np.array([i for i in range(L)])
    for m in range(MSA.shape[0]):
        fi[x[:],MSA[m,x[:]]]+= 1
    fij=np.zeros([L,L,q,q])
    x=np.array([[i,j] for i,j in it.product(range(L),range(L))])  
    for m in range(MSA.shape[0]):
        fij[x[:,0],x[:,1],MSA[m,x[:,0]],MSA[m,x[:,1]]]+= 1
    return fi,fij
        
##########################################################
