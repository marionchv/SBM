#! /usr/bin/env python3
"""
@author: Marion CHAUVEAU

:On:  October 2022
"""

####################### MODULES #######################
import itertools as it
#import C_MonteCarlo # type: ignore
#import MonteCarlo_Potts # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import math
import csv as csv
import time
from scipy.stats import wasserstein_distance
from scipy import stats
import pcmap as cmap
from sklearn import metrics
import scipy.io as sio

##########################################################
####################### LOAD FILES #######################

def csv_to_fasta(csv_path,fasta_path):
	"""
    Function to convert a CSV file to a FASTA file.
    
    Args:
    - csv_path (str): The path to the input CSV file.
    - fasta_path (str): The path to the output FASTA file.
    
    Returns: None
    """
	list_seq = []
	list_name = []
	with open(csv_path, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			list_seq.append(row['sequence'])
			list_name.append(row['sequence_id'])
	ofile = open(fasta_path, "w")
	for i in range(len(list_seq)):
		ofile.write(">" + list_name[i] + "\n" +list_seq[i] + "\n")
	ofile.close()

def load_fasta(file):
	"""
    Function to load a FASTA file and convert the sequences into a numerical 
	representation stored in a numpy array.

    Args:
    - file (str): The path to the FASTA file.

    Returns:
    - numpy.array: A 2D numpy array representing the numerical sequences.
    """
	code = "-ACDEFGHIKLMNPQRSTVWY"
	q=len(code)
	AA_to_num=dict([(code[i],i) for i in range(len(code))])
	errs = "BJOUXZabcdefghijklmonpqrstuvwxyz"
	AA_to_num.update(dict([(errs[i],-1) for i in range(len(errs))]))

	MSA=np.array([])
	print('Nb of sequences: ',len(list(SeqIO.parse(file, "fasta"))))
	count = 0
	for record in SeqIO.parse(file, "fasta"):
		if count%10000 == 0 and count!=0:
			print(count)
		seq=np.array([[AA_to_num[record.seq[i]] for i in range(len(record.seq))]])
		if MSA.shape[0]==0:
			MSA=seq
		else:
			MSA=np.append(MSA,seq,axis=0)
		count +=1
            
    # Remove all errornous sequences (contain '-1')
	print(MSA.shape,np.min(MSA))
	if np.min(MSA)<0:
		SUM = np.sum((MSA==-1),axis=1)
		MSA = MSA[(SUM==0)]
		#MSA=np.delete(MSA,(np.sum(MSA==-1,axis=1)).nonzero()[0][0],axis=0)
	print(MSA.shape)
	return MSA

##########################################################

####################### CREATE ARTICIAL ALIGNEMENT #######################

import MonteCarlo_Potts # type: ignore
def Create_modAlign(output,N,delta_t = 10000,ITER='',temperature=1):
	"""
	Function to create a alignment based on the provided parameters
	using the C_MonteCarlo module implemented in cython

	Args:
	- output (dict): A dictionary containing 'h' and 'J' values.
	- N (int): The value of N.
	- delta_t (int): Number of MCMC steps
	- ITER (str): Optional parameter (if several 'h' and 'J' values are stored in the output dictionary
	we can choose 'h' and 'J' values at a specific iteration)
    - temperature (float): Optional parameter to specify the temperature at which sampling should proceed

	Returns:
	- numpy.array: A 2D numpy array with the created alignment.
	"""
	h,J = output['h'+str(ITER)]/temperature,output['J'+str(ITER)]/temperature
	L,q = h.shape
	w = np.array(Wj(J,h))
	states = np.random.randint(q,size=(N,L)).astype('int32')
	MonteCarlo_Potts.MC(w,states,int(delta_t),int(q))
	MSA = np.copy(states)
	return np.array(MSA,dtype = 'int64')

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

def states_rand(samples):
	"""
	Function to randomize states in the input samples

	Args:
	- samples (numpy.array): A 2D numpy array representing the input samples.

	Returns:
	- numpy.array: A 2D numpy array representing the states after randomization without preserving correlations.
	"""
	#output states randomised (without correlations)
	Cop = np.copy(samples)
	np.apply_along_axis(np.random.shuffle, 0, Cop)
	return Cop

# def Create_modAlign(output,N,delta_t = 10000,ITER='',temperature=1):
# 	h,J = output['h'+str(ITER)]/temperature,output['J'+str(ITER)]/temperature
# 	L,q = h.shape
# 	w = Wj(J,h)
# 	w = np.array(w)
# 	states = np.zeros((N,L)).astype('int32')
# 	MonteCarlo_Potts.MC(w,states,int(delta_t),int(q))
# 	return states

# import MonteCarlo_Ising # type: ignore
# def Create_modAlign_Ising(output,N,delta_t = 10000,ITER='',temperature=1):
# 	h,J = output['h'+str(ITER)]/temperature,output['J'+str(ITER)]/temperature
# 	L = h.shape[0]
# 	states = np.zeros((N,L)).astype('int32')
# 	MonteCarlo_Ising.MC(h,J,states,int(delta_t))
# 	return states

##########################################################

####################### COMPUTE STATISTICS #######################

def CalcWeights(align,theta):
	"""
	Function to compute the weights and effective count for a given alignment and threshold.

	Parameters:
	align : numpy array
		The alignment data.
	theta : float
		The threshold value for distance (if dist < theta sequences are considered to be "the same")

	Returns:
	W : numpy array
		An array of weights calculated using the Hamming distance.
	N_eff : float
		The effective count derived from the sum of weights.
	"""
	#W = 1/(np.sum(squareform(pdist(align, 'hamming'))<theta,axis=0))
	W = 1/(np.sum(squareform(compute_diversity(align))<theta,axis=0))
	N_eff=sum(W)
	return W,N_eff

def CalcStatsWeighted(q,MSA,p=None):
	"""
	Function to calculate the statistics of a given weighted multiple sequence alignment,
	including the frequencies and pairwise frequencies.

	Parameters:
	q : int
		The number of amino acids.
	MSA : numpy array
		A 2D numpy array representing the Multiple Sequence Alignment.
	p : numpy array
		An array representing the weights. If None, it is set to an array of equal weights.

	Returns:
	fi : numpy array
		A 2D numpy array with the calculated frequencies.
	fij : numpy array
		A 4D numpy array with the pairwise frequencies.
	"""
	if p is None:
		p= np.zeros(MSA.shape[0])+1/MSA.shape[0]
	L=MSA.shape[1]  
	fi=np.zeros([L,q])
	x=np.array([i for i in range(L)])
	for m in range(MSA.shape[0]):
		fi[x[:],MSA[m,x[:]]]+= p[m]

	fij=np.zeros([L,L,q,q])
	x=np.array([[i,j] for i,j in it.product(range(L),range(L))])
			
	for m in range(MSA.shape[0]):
		fij[x[:,0],x[:,1],MSA[m,x[:,0]],MSA[m,x[:,1]]]+=p[m]
	return fi,fij

def CalcThreeCorrWeighted(MSA,fi,fij,p=None,ind_L = None):
	"""
	Function to calculate the three-point correlation of a given MSA.

	Args:
	- MSA (numpy.array): A 2D numpy array representing the Multiple Sequence Alignment.
	- fi (numpy.array): A numpy array with frequencies fi(a)
	- fij (numpy.array): A numpy array with pairwise frequencies fij(a,b)
	- p (numpy.array): An optional parameter representing weights. If None, it is set to an array of equal weights.
	- ind_L (numpy.array): An optional array representing indices for which we compute the three-point correlations. 
	If None, it is set to all indices.

	Returns:
	- numpy.array: A 6D numpy array representing the three-point correlation.
		
	"""
	L,q = fi.shape
	Np = MSA.shape[0]
	if p is None:
		p = np.zeros(Np)+1/Np
	if ind_L is None:
		ind_L = np.arange(L)
	l = len(ind_L)
	fijk = np.zeros((l,l,l,q,q,q))
	x = np.array([[i,j,k] for i,j,k in it.product(ind_L,ind_L,ind_L)])
	x2 = np.array([[i,j,k] for i,j,k in it.product(range(l),range(l),range(l))])
	for m in tqdm(range(Np)):
		fijk[x2[:,0],x2[:,1],x2[:,2],MSA[m,x[:,0]],MSA[m,x[:,1]],MSA[m,x[:,2]]] += p[m]
	fij_l, fi_l = fij[ind_L], fi[ind_L]
	fij_l = fij_l[:,ind_L]
	C3 = fijk - (fij_l.reshape(l,l,1,q,q,1)*fi_l.reshape(1,1,l,1,1,q))
	C3 -= (fij_l.reshape(l,1,l,q,1,q)*fi_l.reshape(1,l,1,1,q,1))
	C3 -= (fij_l.reshape(1,l,l,1,q,q)*fi_l.reshape(l,1,1,q,1,1))
	C3 += 2*(fi_l.reshape(l,1,1,q,1,1)*fi_l.reshape(1,l,1,1,q,1)*fi_l.reshape(1,1,l,1,1,q))
	return C3

def compute_stats(output,align_mod):
	"""
	Function to compute various statistics (frequency, pairwise frequency, and three-point correlation)
	for the test, training and artificial sets stored in the output dictionary. 

	Args:
	- output (dict): A dictionary containing various data including 'Test', 'options', 'Train', and 'align_mod'.
	- align_mod (numpy.array): A 2D numpy array representing the alignment data.

	Returns:
	- dict: A dictionary containing different statistics calculated from the input data.
	"""
	Stats = {}
	test_align = output['Test']
	options = output['options']
	train_align = output['Train']
	M = min(output['Train'].shape[0],output['Test'].shape[0],align_mod.shape[0])
	train_align = train_align[np.sort(np.random.choice(output['Train'].shape[0],M,replace=False))]
	align_mod = align_mod[np.sort(np.random.choice(align_mod.shape[0],M,replace=False))]
	test_align = test_align[np.sort(np.random.choice(test_align.shape[0],M,replace=False))]

	ind_L = np.random.choice(options['L'],10,replace=False)
	# Artificial stats
	art = {}
	W,N_eff=CalcWeights(align_mod,options['theta'])
	fi_s,fij_s=CalcStatsWeighted(options['q'],align_mod,W/N_eff)
	C3_s = CalcThreeCorrWeighted(align_mod,fi_s,fij_s,p=W/N_eff,ind_L=ind_L)
	art['Freq'] = fi_s
	art['Pair_freq'] = CalcCorr2(fi_s,fij_s) #fij_s#
	art['Three_corr'] = C3_s

	#Train stats
	train = {}
	W,N_eff=CalcWeights(train_align,options['theta'])
	fi,fij=CalcStatsWeighted(options['q'],train_align,W/N_eff)
	C3 = CalcThreeCorrWeighted(train_align,fi,fij,p = W/N_eff,ind_L=ind_L)
	train['Freq'] = fi
	train['Pair_freq'] = CalcCorr2(fi,fij)#fij#
	train['Three_corr'] = C3

	#Test stats
	test = {}
	W,N_eff=CalcWeights(test_align,options['theta'])
	fi,fij=CalcStatsWeighted(options['q'],test_align,W/N_eff)
	C3 = CalcThreeCorrWeighted(test_align,fi,fij,p = W/N_eff,ind_L=ind_L)
	test['Freq'] = fi
	test['Pair_freq'] = CalcCorr2(fi,fij) #fij #
	test['Three_corr'] = C3

	Stats['Train'] = train
	Stats['Test'] = test
	Stats['Artificial'] = art

	return Stats

def CalcContingency(q,MSA):
    # input MSA in amino acid form
    # output the unweighted freqs fi, co-occurnces fij and correlations Cij=fij-fi*fj
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

def compute_p_values(align,q=21):
	L = align.shape[1]
	fi,fij = CalcContingency(q,align)
	P_val = np.zeros((L,L)) + np.diag(np.ones(L))
	for i in range(L):
		for j in range(i):
			N = align.shape[0] #np.sum((align[:,i]*align[:,j]!=0))

			E_ij = (np.expand_dims(fi[i],axis=1) @ np.expand_dims(fi[j],axis = 0))/N
			O_ij = fij[i,j]

			Norm = E_ij 
			Norm[(E_ij==0)] = np.nan

			T_val = np.nansum(((O_ij - E_ij)**2)/Norm)
			P_val[i,j] = stats.chi2.sf(T_val,(O_ij.shape[0]-1)*(O_ij.shape[0]-1))
	return P_val

def shuff_column(align):
	align_rand = np.zeros(align.shape)
	for i in range(align.shape[1]):
		col = np.copy(align[:,i])
		np.random.shuffle(col)
		align_rand[:,i] = col
	align_rand = align_rand.astype('int')
	return(align_rand)

def Zero_Sum_Gauge(J,h):
	"""
	Function to apply a zero-sum gauge transformation to J and h matrices.

	Args:
	- J (numpy.array): A 4D numpy with couplings.
	- h (numpy.array): A 2D numpy array with fields.

	Returns:
	- J_zg (numpy.array): Updated J matrix after applying the zero-sum gauge transformation.
	- h_zg (numpy.array): Updated h vector after applying the zero-sum gauge transformation.
	"""
	J_zg = np.copy(J)
	h_zg = np.copy(h)

	J_zg -= np.expand_dims(np.mean(J,axis = 2),axis = 2) 
	J_zg -= np.expand_dims(np.mean(J,axis=3),axis =3) 
	J_zg += np.expand_dims(np.mean(J,axis=(2,3)),axis=(2,3))

	h_zg -= np.expand_dims(np.mean(h,axis = 1),axis = 1) 
	h_zg += np.sum(np.mean(J,axis=3)+np.expand_dims(np.mean(J,axis=(2,3)),axis=2),axis=1)
	return J_zg, h_zg

def compute_energies(seqs,h,J):
	"""
	Function to compute energies for an alignment based on the provided parameters provided h and J values.

	Args:
	- align (numpy.array): A 2D or 1D numpy array representing the input alignment.
	- h (numpy.array): A 2D numpy array representing the h values (fields).
	- J (numpy.array): A 4D numpy array representing the J values (couplings).

	Returns:
	- numpy.array: A 1D numpy array representing the computed energies for the input alignment.
	"""
	if len(seqs.shape)==2:
		L=seqs.shape[1]
		N=seqs.shape[0]
	elif len(seqs.shape)==1:
		L=seqs.shape[0]
		N=1
		seqs=seqs.reshape((1,L))
	energy=np.sum(np.array([h[i,seqs[:,i]] for i in range(L)]),axis=0)
	energy=energy+(np.sum(np.array([[J[i,j,seqs[:,i],seqs[:,j]] for j in range(L)] for i in range(L)]),axis=(0,1))/2)
	return -energy
				
# def compute_diversity(align):
# 	"""
# 	Function to compute distances between each pair of sequences of the provided alignment on the Hamming distance.

# 	Args:
# 	- align (numpy.array): A 2D numpy array representing the input alignment.

# 	Returns:
# 	- numpy.array: A 1D numpy array representing the computed diversity based on the Hamming distance.
# 	"""
# 	Div = pdist(align, 'hamming')
# 	return Div


def compute_similarities(Gen1, Gen2=None, N_aa=20):
    N1 = Gen1.shape[0]
    Gen1_2d = alg2bin(Gen1, N_aa=N_aa)
    
    if Gen2 is None:
        Sim = np.zeros(N1)
        for i in range(N1):
            a2d = Gen1_2d[i:i+1] 
            simMat = a2d.dot(Gen1_2d.T)
            SUM = Gen1[i] + Gen1
            norm = np.sum(SUM != 0, axis=1)
            d = 1 - simMat[0] / norm
            Sim[i] = np.amin(np.sort(d)[1:])
    else:
        N2 = Gen2.shape[0]
        Sim = np.zeros(N1)
        Gen2_2d = alg2bin(Gen2, N_aa=N_aa)
        for i in range(N1):
            a2d = Gen1_2d[i:i+1] 
            simMat = a2d.dot(Gen2_2d.T)
            SUM = Gen1[i] + Gen2
            norm = np.sum(SUM != 0, axis=1)
            d = 1 - simMat[0] / norm
            Sim[i] = np.amin(d)
    return Sim

def compute_diversity(alg, N_aa=20):
    Nseq = alg.shape[0]
    X2d = alg2bin(alg, N_aa=N_aa)
    simMat = X2d.dot(X2d.T)
    Dist = simMat[np.triu_indices(Nseq, k=1)]
    NORM = np.zeros(Dist.size)
    idx = 0
    for i in range(Nseq - 1):
        a = alg[i]
        align_rm = alg[i + 1:]
        SUM = a + align_rm
        NORM[idx:idx + align_rm.shape[0]] = np.sum(SUM != 0, axis=1)
        idx += align_rm.shape[0]
    Dist = 1 - Dist / NORM
    return Dist


# def compute_diversity(alg,N_aa=20):
# 	Nseq = alg.shape[0]
# 	X2d = alg2bin(alg,N_aa=N_aa)
# 	simMat = X2d.dot(X2d.T)
# 	Dist = simMat[np.triu_indices(simMat.shape[0],k=1)]

# 	align_rm = np.copy(alg)
# 	NORM = np.array([])
# 	for i in range(Nseq-1):
# 		a = alg[i]
# 		align_rm = np.delete(align_rm,0,0)
# 		SUM=a+align_rm
# 		NORM = np.concatenate((NORM,np.sum((SUM!=0),axis=1)))
# 	Dist = 1 - Dist/NORM
# 	return Dist

# def compute_similarities(Gen1,Gen2 = None):
# 	"""
# 	Function to compute for each sequence of Gen1 the distances to the nearest sequence of Gen2 (based on the Hamming distance).
# 	If Gen2 is not provided, it computes distances within Gen1.

# 	Args:
# 	- Gen1 (numpy.array): A 2D numpy array representing the first set of sequences.
# 	- Gen2 (numpy.array): An optional 2D numpy array representing the second set of sequences. 
# 	If None, it calculates the distance within Gen1.

# 	Returns:
# 	- numpy.array: A 1D numpy array representing the computed distances between the sequences.
# 	"""
# 	if Gen2 is None:
# 		Div = squareform(pdist(Gen1, 'hamming')) + np.eye(Gen1.shape[0])
# 		Sim = np.amin(Div,axis = 0)
# 	else:
# 		N = Gen1.shape[0]
# 		L = Gen1.shape[1]
# 		Sim = np.zeros(N)
# 		for i in range(N):
# 			a = Gen1[i]
# 			a = np.expand_dims(a,axis = 0)*np.ones(Gen2.shape)
# 			s = 1 - np.sum((a == Gen2),axis = 1)/L
# 			Sim[i] = np.amin(s)
# 	return Sim

def CalcCorr2(fi,fij):
	"""
	Function to calculate pairwise correlations based on the provided fi and fij values.

	Args:
	fi (numpy.array): A 2D numpy array representing the frequencies.
	fij (numpy.array): A 4D numpy array representing the pairwise frequencies.

	Returns:
	numpy.array: A 4D numpy array representing the calculated pairwise correlations.
	"""
	L,q = fi.shape
	Cij=fij-(fi.reshape([L,1,q,1])*fi.reshape([1,L,1,q]))
	for i in range(L):
		Cij[i,i,:,:]=0
	return Cij

def compute_eps(Gen_nat,Gen_mod,q,theta):
	"""
	Function to compute epsilon score between a natural MSA and a artificial MSA 
	(Pearson correlation between natural and artificial frequencies)

	Args:
	- Gen_nat (numpy.array): A 2D numpy array representing the natural generation data.
	- Gen_mod (numpy.array): A 2D numpy array representing the modified generation data.
	- q (int): Number of amino acids. Default value is 21.
	- theta (float)

	Returns:
	- float: The computed epsilon^2 score
	"""
	W,N_eff=CalcWeights(Gen_mod,theta)
	fi_mod,_=CalcStatsWeighted(q,Gen_mod,W/N_eff)
	W,N_eff=CalcWeights(Gen_nat,theta)
	fi_nat,_=CalcStatsWeighted(q,Gen_nat,W/N_eff)
	Pears = np.corrcoef(fi_nat.flatten(),fi_mod.flatten())[0,1]
	#eps = np.sum((fi_nat - fi_mod)**2)/fi_nat.size
	return Pears

def compute_eps2(MSA_nat,MSA_mod,theta,q=21):
	"""
	Function to compute epsilon^2 score between a natural MSA and a artificial MSA 
	(Pearson correlation between natural and artificial pairwise frequencies)

	Args:
	- Gen_nat (numpy.array): A 2D numpy array representing the natural generation data.
	- Gen_mod (numpy.array): A 2D numpy array representing the modified generation data.
	- q (int): Number of amino acids. Default value is 21.
	- theta (float)

	Returns:
	- float: The computed epsilon^2 score
	"""
	W,N_eff=CalcWeights(MSA_mod,theta)
	_,fij_mod=CalcStatsWeighted(q,MSA_mod,W/N_eff)
	W,N_eff=CalcWeights(MSA_nat,theta)
	_,fij_nat=CalcStatsWeighted(q,MSA_nat,W/N_eff)
	#Cij_mod = CalcCorr2(fi_mod,fij_mod)
	#Cij_nat = CalcCorr2(fi_nat,fij_nat)
	Pears = np.corrcoef(fij_nat.flatten(),fij_mod.flatten())[0,1]
	# Nv = Cij_nat.shape[0]
	# mask = np.tril(np.ones((Nv,Nv)), k = -1)
	# mask = np.expand_dims(mask,axis = (2,3))
	# eps_2 = 2*np.sum(((Cij_nat - Cij_mod)*mask)**2)/(Nv*(Nv-1)*(q**2))
	return Pears

def compute_epsAAI(Gen_nat,Gen_mod,theta=0.2):
	"""
	Function to compute epsilon AAI score from this article:
	@article{decelle2021equilibrium,
	title={Equilibrium and non-equilibrium regimes in the learning of restricted Boltzmann machines},
	author={Decelle, Aur{\'e}lien and Furtlehner, Cyril and Seoane, Beatriz},
	journal={Advances in Neural Information Processing Systems},
	volume={34},
	pages={5345--5359},
	year={2021}
	}

	Args:
	- Gen_nat (numpy.array): A 2D numpy array representing the natural generation data.
	- Gen_mod (numpy.array): A 2D numpy array representing the modified generation data.
	- theta (float)

	Returns:
	- float: The computed epsilon AAI score
	"""
	d_TT = compute_similarities(Gen_mod)
	d_SS = compute_similarities(Gen_nat)
	d_ST = compute_similarities(Gen_nat,Gen_mod)
	d_TS = compute_similarities(Gen_mod,Gen_nat)

	# W,N_eff = CalcWeights(Gen_nat,theta)
	# AS = np.sum((d_SS < d_ST)*W)/N_eff #d_SS.shape[0]
	# #print(Gen_nat.shape[0],N_eff)
	# W,N_eff = CalcWeights(Gen_mod,theta)
	# #print(Gen_mod.shape[0],N_eff)
	# AT = np.sum((d_TT < d_TS)*W)/N_eff #d_TT.shape[0]

	AS = np.sum((d_SS < d_ST))/d_SS.shape[0]
	AT = np.sum((d_TT < d_TS))/d_TT.shape[0]
	eps_AAI = (AS - 0.5)**2 + (AT - 0.5)**2
	return eps_AAI,AS, AT

##########################################################

####################### TRAINING/TEST SETS #######################

def sampling_TestTrain(MSA, distance_threshold=0.2, target_percentage=0.2):
	M = MSA.shape[0]
	num_test_sequences = int(M * target_percentage)

	# Compute the distance matrix
	dist_matrix = compute_diversity(MSA)
	dist_matrix = squareform(dist_matrix)

	# Initialize sets for indices
	ind_train = set(range(M))
	ind_test = set()

	# Select a random initial sequence for the test set
	initial_index = np.random.choice(list(ind_train))
	ind_test.add(initial_index)
	ind_train.remove(initial_index)

	for t in tqdm(range(num_test_sequences)):
		far_enough_candidates = []
		for i in ind_train:
			if np.sum(dist_matrix[i, j] < distance_threshold for j in ind_train)==1:
				far_enough_candidates.append(i)
		
		if not far_enough_candidates:
			print("Impossible de trouver plus de séquences pour le test set avec le seuil de distance donné.")
			return np.array(ind_train), np.array(ind_test)
		
		# Randomly select one of the candidates
		chosen_index = np.random.choice(far_enough_candidates)
		ind_test.add(chosen_index)
		ind_train.remove(chosen_index)

	return np.array(list(ind_train)), np.array(list(ind_test))

# def sampling_TestTrain(MSA, similarity_threshold=0.2, target_percentage=0.2, max_iterations=100000):
#     ind_train = np.arange(MSA.shape[0])
#     ind_test = np.array([], dtype=int)
#     per = 0
#     count = 0

#     while per < target_percentage:
#         ind = np.random.choice(ind_train)
#         msa_test = MSA[ind].reshape((1, MSA.shape[1]))
        
#         ind_temp = np.concatenate((ind_test, [ind]))
#         ind_temp2 = np.delete(np.arange(MSA.shape[0]), ind_temp)
        
#         sim_test = compute_similarities(msa_test, MSA[ind_temp2])
        
#         if sim_test[0] > similarity_threshold:
#             ind_test = ind_temp
#             ind_train = np.delete(ind_train, np.where(ind_train == ind))
        
#         per = len(ind_test) / MSA.shape[0]
#         count += 1
        
#         if count % 5000 == 0:
#             print(per)
        
#         if count == max_iterations:
#             break

#     return ind_train, ind_test
	
def RemoveCloseSeqs(align,theta):
	"""
	Function to remove closely related sequences from an alignment based on the provided threshold for sequence similarity.

	Args:
	- align (numpy.array): A 2D numpy array representing the input alignment.
	- theta (float): The threshold value for sequence similarity.

	Returns:
	- numpy.array: A 2D numpy array representing the alignment with closely related sequences removed.
	"""
	Dist = squareform(pdist(align, 'hamming')) + np.eye(align.shape[0])
	M = Dist.shape[0]
	Ind = np.arange(M)
	Ind_del = np.arange(M)
	Ind_unk = np.arange(M)
	sim_del = np.ones(M) #compute_similarities(align[Ind],align[Ind])
	while np.sum((sim_del>=0.2))>1:
		Ind_del_add = []
		align_unk = align[Ind_unk]
		Dist = squareform(pdist(align_unk, 'hamming')) + np.eye(align_unk.shape[0])
		for i in range(Dist.shape[0]):
			dist = Dist[i]
			if np.amin(dist) < theta:
				Dist[i,:] = 1
				Dist[:,i] = 1
				Ind_del_add.append(Ind_unk[i])
		Ind_del = np.concatenate((Ind_del[(sim_del<0.2)],Ind_del_add)).astype('int')
		if len(Ind_del)==0:
			return align
		Ind = np.delete(np.arange(M),Ind_del)
		sim_del = compute_similarities(align[Ind_del],align[Ind])
		Ind_unk = Ind_del[(sim_del>=0.2)]

	Ind_del = Ind_del[(sim_del<0.2)]
	Ind = np.delete(np.arange(M),Ind_del)
	Dist_align = align[np.sort(Ind)]
	return Dist_align


def custom_MSA(MSA, distance_threshold, ref_indices):
	ref_sequences = MSA[ref_indices]

	selected_sequences = []

	# Parcourir chaque séquence dans MSA
	for seq in MSA:
		# Vérifier la distance avec chaque séquence de référence
		for ref_seq in ref_sequences:
			MSA_2 = np.concatenate((np.expand_dims(seq,axis=0),np.expand_dims(ref_seq,axis=0)))
			dist = compute_diversity(MSA_2)[0]
			if dist < distance_threshold:
				selected_sequences.append(seq)
				break

	# Convertir la liste des séquences sélectionnées en un numpy array
	new_MSA = np.array(selected_sequences)
	sum0 = np.sum(new_MSA,axis=0)
	new_MSA = new_MSA[:,(sum0!=0)]
	return new_MSA

##########################################################

####################### CONTACT PREDICTION #######################

def read_structdat_file(file_path):
	struct = np.loadtxt(file_path)
	L = int(np.amax(struct[:,:2]))
	struct_mat = np.zeros((L,L))
	ind1, ind2 = (struct[:,0]-1).astype('int'), (struct[:,1]-1).astype('int')
	struct_mat[ind1,ind2] = struct[:,-1]
	return struct_mat

def compute_score_contacts(J,APC = True):
	L = J.shape[0]
	F = np.zeros((L,L))
	for i in range(L):
		for j in range(i,L):
			F_ij = np.sqrt(np.sum(J[i,j,1:,1:]**2))
			F[i,j],F[j,i] = F_ij, F_ij
	#F = np.linalg.norm(J,'fro',axis = (2,3))
	if APC:
		F -= np.expand_dims(np.mean(F,axis=0),axis=1) @ np.expand_dims(np.mean(F,axis=1),axis=0)/np.mean(F)	
	return F
			
##########################################################

####################### MUTATIONAL EFFECTS PREDICTION #######################

def Mutational_effect(WT_seq,h,J):
	"""
	Function to compute mutational effects based on the provided wild-type sequence, h, and J values, 
	considering the changes in energy resulting from single amino acid substitutions.

	Args:
	- WT_seq (numpy.array): A 1D numpy array representing the wild-type sequence.
	- h (numpy.array): A 2D numpy array representing the h values.
	- J (numpy.array): A 4D numpy array representing the J values.

	Returns:
	- numpy.array: A 2D numpy array representing the computed mutational effects.
	"""
	L,q = h.shape
	Mut = np.zeros((q,L))
	for k in range(L):
		for b in range(q):
			if b != WT_seq[k]:
				Delta_E = h[k,WT_seq[k]] - h[k,b] - np.sum(J[k,np.arange(L),b,WT_seq]) + np.sum(J[k,np.arange(L),WT_seq[k],WT_seq]) 
				Mut[b,k] = Delta_E
	return Mut

##########################################################

####################### PCA #######################

def PCA_comparison(COG_samp,COG_model,Pears=0,Mask = 1):
	assert COG_samp.shape[1] == COG_model.shape[1]

	if Pears!=0:
		Cov_Ising = np.corrcoef(COG_samp.T)*Mask
	else:
		Cov_Ising = np.cov(COG_samp.T)*Mask

	W_cov,V_cov = np.linalg.eigh(Cov_Ising)

	ind_sort = np.argsort(W_cov)[::-1]
	W_cov = W_cov[ind_sort]
	V_cov = V_cov[:,ind_sort]
	w1,w2 = W_cov[0],W_cov[1]
	v1,v2 = V_cov[:,0],V_cov[:,1]

	v1_norm,v2_norm = v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)

	ProjX_samp,ProjY_samp = COG_samp@ v1_norm,COG_samp@v2_norm
	ProjX_model,ProjY_model = COG_model@v1_norm,COG_model@v2_norm

	#Efficiency
	conserved_var = (w1 + w2)/np.sum(W_cov)
	print(conserved_var*100, '%')

	X_samp = np.concatenate((np.expand_dims(ProjX_samp,axis=1),np.expand_dims(ProjY_samp,axis=1)),axis = 1)
	X_model = np.concatenate((np.expand_dims(ProjX_model,axis=1),np.expand_dims(ProjY_model,axis=1)),axis = 1)
	
	return X_samp,X_model

##########################################################
		
####################### OTHER FUNCTIONS #######################

def alg2bin(alg, N_aa=20):
	"""
	Function to convert an alignment into a binary representation.

	Args:
	- alg (numpy.array): A 2D numpy array representing the input alignment.
	- N_aa (int): The number of amino acids. Default value is 20.

	Returns:
	- numpy.array: A 2D numpy array representing the binary representation of the input alignment.
	"""
	[N_seq, N_pos] = alg.shape
	Abin_tensor = np.zeros((N_aa, N_pos, N_seq))
	for ia in range(N_aa):
		Abin_tensor[ia, :, :] = (alg == ia+1).T
	Abin = Abin_tensor.reshape(N_aa * N_pos, N_seq, order="F").T
	return Abin

def averaged_model(file_names,fam,Model,ITER=''):
	"""
	Function to compute the averaged model based on the provided file names, family, and Model.
	It saves the results with the appropriate naming convention.

	Args:
	- file_names (list): A list of file names.
	- fam (str): A string representing the family.
	- Model (str): A string representing the model.
	- ITER (str): An optional string representing the iteration.

	Returns: None
	"""
	AVG_model = np.load('results/Article/'+Model+'/'+fam+'/'+file_names[0],allow_pickle=True)[()]
	J_avg = np.zeros(AVG_model['J'+str(ITER)].shape)
	h_avg = np.zeros(AVG_model['h'+str(ITER)].shape)
	for f in file_names:
		mod = np.load('results/Article/'+Model+'/'+fam+'/'+f,allow_pickle=True)[()]
		J_avg += mod['J'+str(ITER)]
		h_avg += mod['h'+str(ITER)]
	AVG_model['J'] = J_avg/len(file_names)
	AVG_model['h'] = h_avg/len(file_names)
	if Model=='SBM':
		if ITER=='':nb_it = AVG_model['options']['maxIter']
		else: nb_it=ITER
		#np.save('results/Article/'+Model+'/'+fam+'/'+fam+'_avgMod_m'+str(AVG_model['options']['m'])+'Ns'+str(AVG_model['options']['n_states'])+'Ni'+str(nb_it)+'.npy',AVG_model)
		np.save('results/Article/'+Model+'/'+fam+'/'+'TM1_avgMod_m'+str(AVG_model['options']['m'])+'Ns'+str(AVG_model['options']['n_states'])+'Ni'+str(nb_it)+'.npy',AVG_model)

def rho(X):
	"""
	Function to compute  the empirical autocorrelation function for the input array X, 
	which measures how much the fluctuations in X are correlated over a time distance t.

	Args:
	- X (numpy.array): A 2D numpy array representing the input data.

	Returns:
	- numpy.array: A 1D numpy array representing the empirical autocorrelation function for the input array X.
	"""

	#empirical autocorrelation function (tells us 
	x,y = np.shape(X)
	N=x
	A = np.zeros(N)
	for t in range(0,N):
		mean1 = np.sum(X[0:N-t,:],axis = 0)/(N-t)
		mean2 = np.sum(X[1+t:N,:],axis = 0)/(N-t)
		Num = 0
		Den1 = 0
		Den2 = 0
		for j in range(0,N-t):
			Num += np.dot((X[j,:]-mean1),(X[j+t,:]-mean2))
			Den1 += np.dot((X[j,:]-mean1),(X[j,:]-mean1))
			Den2 += np.dot((X[j+t,:]-mean2),(X[j+t,:]-mean2))
		if Num == 0:  #to not divide by zero
			autocorr = 0
		else:
			autocorr = Num/np.sqrt(Den1*Den2)
		A[t]=autocorr
	return A

##########################################################

####################### MUTATIONAL EFFECTS PREDICTION #######################
###################### functions added by Emily Hinds #######################

def get_dms(wt, Q, char_skip=set(), include_wt=False):
    """
    Produce the deep mutational scan alignment (comprehensive single-site
    mutations) for a provided protein sequence in numerical format.
    
    NOTE: If include_wt = True, the index of the sequence that is guaranteed to have letter q
          and position i is i*(Q-size(char_skip))+q, regardless of if q is wild-type or not. 
          If include_wt = False, this indexing is more complicated and will be i*(Q-size(char_skip)-1)+q
          if q < wild-type value, and will be i*(Q-size(char_skip)-1)+q-1 if q > wild-type value.
          To make things easier I usually run this function with include_wt=True and process out
          the WT sequences out later if I want to.    
    
    Args:
    - wt: numpy array defining the wild-type sequence (in numerical format)
    - Q: int, size of alphabet
    - char_skip: optional set (ideal) or list, values in set Q that, if present in wt,
                       should not be mutated to other residues and to which other
                       residues should not be mutated (e.g. gap characters for proteins).
                       Default: empty set, so do not skip any characters
    - include_wt: optional bool corresponding to whether or not you want to include
                        the wild-type residue at each position. If True/1, note that WT
                        sequence will be present in output alignment L (# positions) times
                        (each position will have Q-(size char_skip) sequences corresponding
                        to comprehensive mutagenesis at that position, including the WT).
                        Default: False/0
             
    Returns:
    - numpy array of (L*(Q - size(char_skip) - include_wt)) sequences x L positions
      corresponding to an alignment of the deep mutational scan sequences
                      
    """

    dms = []
    for i in range(wt.size):
        curr_seq = wt.copy()
        wt_res = wt[i]
        if wt_res in char_skip:
            continue
        for a in range(Q):
            if a not in char_skip:
                if a != wt_res or include_wt:
                    curr_seq[i] = a
                    dms.append(curr_seq.copy())
            
    return np.vstack(dms)

def get_matlab_info(filename):
    """
    Load Potts models parameters and other key information from the MATLAB implementation 
    of the SBM.

    Args:
    - filename: string with file path to .mat file

    Returns:
    - outputs: dictionary of model features with the same keys and corresponding
               information as the SBM_py output dictionary
    """

    outputs = dict()
    
    mat = loadmat(filename)
    outputs['J'] = mat['J_av'].transpose(2,3,0,1)
    outputs['h'] = mat['h_av'].transpose(1,0)
    outputs['align'] = mat['align'] - 1
    outputs['Train'] = mat['align'] - 1
    outputs['Test'] = mat['align_cv'] - 1

    # almost certainly a more elegant way
    matlab_options_kws = ['maxIter',
                          'maxCycle',
                          'TolX',
                          'N',
                          'delta_t',
                          'theta',
                          'm',
                          'skip',
                          'gauge',
                          'prune',
                          'q',
                          'L']
    n_options = len(mat['options'][0][0])
    if n_options < 10 or n_options > 12:
        print("MATLAB file has %d options specified, expected 10, 11, or 12"%n_options)
        raise Exception
    elif n_options == 10:
        del matlab_options_kws[8:10] # no prune param or gauge param
    elif n_options == 11:
        del matlab_options_kws[8] # no prune param
    
    outputs['options'] = dict(zip(matlab_options_kws, mat['options'][0][0]))

    # for some reason all of the values read in from the MATLAB file are
    # single elements in deeply nested arrays, so we need to get the actual element
    for k, v in outputs['options'].items():
        if k == 'prune': # want this one to stay an array, but we need to reshape it
            outputs['options'][k] = v.transpose(2,3,0,1)
        else:
            while type(v) == np.ndarray:
                v = v[0]
            outputs['options'][k] = v

    return outputs


#######################################################################

def other_metrics(Models):
	Mod_list = list(Models.keys())
	for m in range(len(Mod_list)):
		mod = Mod_list[m]
		print(mod)
		MOD = Models[mod]
		N_mod = len(MOD['Models'])

		MOD['AUC_contacts'] = np.zeros(N_mod)
		MOD['AUC_DMS'] = np.zeros(N_mod)
		MOD['Freq Pearson'] = np.zeros(N_mod)
		MOD['Pair Freq Pearson'] = np.zeros(N_mod)

		for ind in range(N_mod+1):
			if ind == N_mod: output = np.load('results/'+MOD['Folder']+'/'+MOD['Avg Mod'],allow_pickle=True)[()]
			else: output = np.load('results/'+MOD['Folder']+'/'+MOD['Models'][ind],allow_pickle=True)[()]

			k_MCMC = output['options']['k_MCMC']
			if 'align_mod' not in list(output.keys()):
				N = output['Train'].shape[0]
				align_mod=Create_modAlign(output,N = N,delta_t = k_MCMC,ITER='',temperature=MOD['Temperature'])
				output['align_mod'] = align_mod
			
			Stats = compute_stats(output,align_mod)

			SBM_J,SBM_h = output['J'],output['h']
			SBM_J,SBM_h = Zero_Sum_Gauge(SBM_J,SBM_h)

			#### AUC_DMS ####
			AUC = compare_to_dms_AUC(SBM_h,SBM_J,dms_file='data/CM_DMS.mat',wt = wt, per_position=True,include_wt=False)
			if ind==N_mod: MOD['Avgmod AUC_DMS'] = AUC
			else: MOD['AUC_DMS'][ind] = AUC
			#################################

			#### Freq Pearson ########
			Pears = np.corrcoef(Stats['Test']['Freq'].flatten(),Stats['Artificial']['Freq'].flatten())[0,1]
			if ind==N_mod: MOD['Avgmod Freq Pearson'] = Pears
			else: MOD['Freq Pearson'][ind] = Pears
			#################################

			#### Pair Freq Pearson ###
			Pears = np.corrcoef(Stats['Test']['Pair_freq'][ind].flatten(),Stats['Artificial']['Pair_freq'][ind].flatten())[0,1]
			if ind==N_mod: MOD['Avgmod Pair Freq Pearson'] = Pears
			else: MOD['Pair Freq Pearson'][ind] = Pears
			#################################

			#### AUC_contacts ####
			AUC = compute_AUC_contacts(SBM_J,pdb_file = 'data/1ecm.pdb')
			if ind == N_mod: MOD['Avgmod AUC_contacts'] = AUC
			else: MOD['AUC_contacts'][ind] = AUC
			#################################
	
	return Models

wt = np.array([ 0, 17, 16,  4, 12, 13, 10, 10,  1, 10, 15,  4,  9,  8, 16,  1, 10,
        3,  4,  9, 10, 10,  1, 10, 10,  1,  4, 15, 15,  4, 10,  1, 18,  4,
       18,  6,  9,  1,  9, 10, 10, 16,  7, 15, 13, 18, 15,  3,  8,  3, 15,
        4, 15,  3, 10, 10,  4, 15, 10,  8, 17, 10,  6,  9,  0,  1,  7,  7,
       10,  3,  1,  7, 20,  8, 17, 15, 10,  5, 14, 10,  8,  8,  4,  3, 16,
       18, 10, 17, 14, 14,  1, 10, 10, 14, 14,  7])


def get_dms(wt, Q, char_skip=set(), include_wt=False):
    dms = []
    for i in range(wt.size):
        curr_seq = wt.copy()
        wt_res = wt[i]
        if wt_res in char_skip:
            continue
        for a in range(Q):
            if a not in char_skip:
                if a != wt_res or include_wt:
                    curr_seq[i] = a
                    dms.append(curr_seq.copy())
            
    return np.vstack(dms)


def compare_to_dms_AUC(h, J, wt,dms_file = "/Users/emily/Downloads/CM_DMS.mat", per_position=True,include_wt=False, char_skip={0}):

    # Read in experimental DMS data excluding the stop codons
    expt_data = sio.loadmat(dms_file)['dms'][0,0][1][:-1,:].T.flatten()
    
    # Get the in silico energies of the DMS under provided model
    dms_alg = get_dms(wt=wt, Q=h.shape[1], char_skip=char_skip, include_wt=True)
    pred_data = compute_energies(dms_alg, h, J).flatten()
    
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

        expt_data = np.nanmean(expt_data.reshape(-1,20), axis=1).reshape(-1,1)
        
        pred_data = np.nanmean(pred_data.reshape(-1,20), axis=1).reshape(-1,1)
    else:           
        expt_data = np.hstack([x for x in expt_data if not np.isnan(x)]).reshape(-1,1)
        pred_data = np.hstack([x for x in pred_data if not np.isnan(x)]).reshape(-1,1)

    Binary_DMS = (expt_data<-0.3).astype('int')
    fpr, tpr,_ = metrics.roc_curve(Binary_DMS,pred_data,pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def compute_AUC_contacts(SBM_J,struct_mat = 'data/StructMat_CM.npy'):

	L = struct_mat.shape[0]
	distij = np.abs(np.expand_dims(np.arange(L),axis = 1)*np.ones((L,L)) - np.expand_dims(np.arange(L),axis = 0)*np.ones((L,L)))
	mask_dist = (distij > 4).astype('int')
	struct_mat = struct_mat*mask_dist
	sort_struct = struct_mat[np.triu_indices(L,k=1)]

	Pos_rem = np.array([1, 2, 3, 4,96]) - 1
	Pos_keep = np.delete(np.arange(SBM_J.shape[0]),Pos_rem)

	SBM_J_contacts = SBM_J[Pos_keep]
	SBM_J_contacts = SBM_J_contacts[:,Pos_keep]

	Score = compute_score_contacts(SBM_J,APC = True)
	Score *= mask_dist
	Score = Score[np.triu_indices(L,k = 1)]

	
	fpr, tpr,_ = metrics.roc_curve(sort_struct,Score, pos_label=1)
	AUC = metrics.auc(fpr, tpr)
	return AUC


INDICES_CMAP = {'5 ': 0,
				'6 ': 1,
				'7 ': 2,
				'8 ': 3,
				'9 ': 4,
				'10 ': 5,
				'11 ': 6,
				'12 ': 7,
				'13 ': 8,
				'14 ': 9,
				'15 ': 10,
				'16 ': 11,
				'17 ': 12,
				'18 ': 13,
				'19 ': 14,
				'20 ': 15,
				'21 ': 16,
				'22 ': 17,
				'23 ': 18,
				'24 ': 19,
				'25 ': 20,
				'26 ': 21,
				'27 ': 22,
				'28 ': 23,
				'29 ': 24,
				'30 ': 25,
				'31 ': 26,
				'32 ': 27,
				'33 ': 28,
				'34 ': 29,
				'35 ': 30,
				'36 ': 31,
				'37 ': 32,
				'38 ': 33,
				'39 ': 34,
				'40 ': 35,
				'41 ': 36,
				'42 ': 37,
				'43 ': 38,
				'44 ': 39,
				'45 ': 40,
				'46 ': 41,
				'47 ': 42,
				'48 ': 43,
				'49 ': 44,
				'50 ': 45,
				'51 ': 46,
				'52 ': 47,
				'53 ': 48,
				'54 ': 49,
				'55 ': 50,
				'56 ': 51,
				'57 ': 52,
				'58 ': 53,
				'59 ': 54,
				'60 ': 55,
				'61 ': 56,
				'62 ': 57,
				'63 ': 58,
				'64 ': 59,
				'65 ': 60,
				'66 ': 61,
				'67 ': 62,
				'68 ': 63,
				'69 ': 64,
				'70 ': 65,
				'71 ': 66,
				'72 ': 67,
				'73 ': 68,
				'74 ': 69,
				'75 ': 70,
				'76 ': 71,
				'77 ': 72,
				'78 ': 73,
				'79 ': 74,
				'80 ': 75,
				'81 ': 76,
				'82 ': 77,
				'83 ': 78,
				'84 ': 79,
				'85 ': 80,
				'86 ': 81,
				'87 ': 82,
				'88 ': 83,
				'89 ': 84,
				'90 ': 85,
				'91 ': 86,
				'92 ': 87,
				'93 ': 88,
				'94 ': 89,
				'95 ': 90}

def create_Struct_mat(struct_file='data/1ecm.pdb',dist=8):
	struct_mat = np.zeros((91,91))
	Contacts = cmap.contactMap(struct_file,dist=dist)
	Contacts = Contacts['data']
	for c in range(len(Contacts)):
		if Contacts[c]['root']['chainID']=='A':
			root = Contacts[c]['root']['resID']
			partners = Contacts[c]['partners']
			for p in range(len(partners)):
				if partners[p]['chainID']=='A':
					part = partners[p]['resID']
					i1 = INDICES_CMAP[root]
					i2 = INDICES_CMAP[part]
					struct_mat[i1,i2],struct_mat[i2,i1]=1,1
	np.save('data/StructMat_CM_5.npy',struct_mat)


# Contacts = cmap.contactMap('data/1ecm.pdb',dist=8)
# Contacts = Contacts['data']
# list_res = {}
# test_list = [0]
# for c in range(len(Contacts)):
# 	new_res= int(Contacts[c]['root']['resID'])
# 	if test_list[-1] > new_res: break
# 	else: 
# 		test_list.append(new_res)
# 		list_res[str(new_res)] = c

def all_metrics_V2(Models):
	Mod_list = list(Models.keys())
	for m in range(len(Mod_list)):
		mod = Mod_list[m]
		print(mod)
		MOD = Models[mod]
		if MOD['Temperature'] == 1:
			averaged_model(MOD['Models'],MOD['Folder'],ITER='')
			MOD['Avg Mod'] = mod[:-1]+'_avg.npy'
		else: MOD['Avg Mod'] = MOD['Models'][0][:-6] +'_avg.npy'
		print(MOD['Avg Mod'])
		N_mod = len(MOD['Models'])

		#MOD['E_score Train/Art'] = np.zeros(N_mod)
		MOD['E_score Train/Test'] = np.zeros(N_mod)
		MOD['E_score Nat/RandCol'] = np.zeros(N_mod)


		for ind in range(N_mod+1):
			print(ind)
			if ind == N_mod: output = np.load('results/'+MOD['Folder']+'/'+MOD['Avg Mod'],allow_pickle=True)[()]
			else: output = np.load('results/'+MOD['Folder']+'/'+MOD['Models'][ind],allow_pickle=True)[()]
			if ind==0: MOD['options'] = output['options']

			k_MCMC = output['options']['k_MCMC']
			# if 'align_mod' not in list(output.keys()):
			# 	N = output['Train'].shape[0]
			# 	align_mod=Create_modAlign(output,N = N,delta_t = k_MCMC,ITER='',temperature=MOD['Temperature'])
			# 	output['align_mod'] = align_mod
			align_train= output['Train']
			align_test= output['Test']
			rand2 = shuff_column(output['align'])

			SBM_J,SBM_h = output['J'],output['h']
			SBM_J,SBM_h = Zero_Sum_Gauge(SBM_J,SBM_h)

			Erand2 = compute_energies(rand2,SBM_h,SBM_J)
			Etest = compute_energies(align_test,SBM_h,SBM_J)
			Etrain = compute_energies(align_train,SBM_h,SBM_J)
			Mean = np.median(Etrain)
			STD = 1 #np.std(Etrain)
			Erand2,Etrain,Etest = (Erand2-Mean)/STD,(Etrain-Mean)/STD,(Etest-Mean)/STD

			#### Nat/randcol overlap score ####
			Enat = np.concatenate((Etest,Etrain))
			Pairs = list(it.itertools.product(Enat,Erand2))
			A = np.array([d[0] for d in Pairs])
			B = np.array([d[1] for d in Pairs])
			if ind == N_mod: MOD['Avgmod E_score Nat/RandCol'] = np.sum(((A-B)>0).astype('int'))/len(A)
			else: MOD['E_score Nat/RandCol'][ind] = np.sum(((A-B)>0).astype('int'))/len(A)
			#################################

			#### Train/Test overlap score ####
			Pairs = list(it.itertools.product(Etrain,Etest))
			A = np.array([d[0] for d in Pairs])
			B = np.array([d[1] for d in Pairs])
			if ind==N_mod: MOD['Avgmod E_score Train/Test'] = 0.5 - np.sum(((A-B)>0).astype('int'))/len(A)
			else: MOD['E_score Train/Test'][ind] = 0.5 - np.sum(((A-B)>0).astype('int'))/len(A)
			#################################
	
	return Models


def all_metrics_1mod(Model_file):
	Scores = {}

	output = np.load(Model_file,allow_pickle=True)[()]

	align_train= output['Train']
	align_test= output['Test']
	rand2 = shuff_column(output['align'])

	SBM_J,SBM_h = output['J'],output['h']
	SBM_J,SBM_h = Zero_Sum_Gauge(SBM_J,SBM_h)

	Erand2 = compute_energies(rand2,SBM_h,SBM_J)
	Etest = compute_energies(align_test,SBM_h,SBM_J)
	Etrain = compute_energies(align_train,SBM_h,SBM_J)
	Mean = np.median(Etrain)
	STD = 1 #np.std(Etrain)
	Erand2,Etrain,Etest = (Erand2-Mean)/STD,(Etrain-Mean)/STD,(Etest-Mean)/STD

	#### Nat/randcol overlap score ####
	Enat = np.concatenate((Etest,Etrain))
	Pairs = list(it.itertools.product(Enat,Erand2))
	A = np.array([d[0] for d in Pairs])
	B = np.array([d[1] for d in Pairs])
	Scores['E_score Nat/RandCol'] = np.sum(((A-B)>0).astype('int'))/len(A)
	#################################

	#### Train/Test overlap score ####
	Pairs = list(it.itertools.product(Etrain,Etest))
	A = np.array([d[0] for d in Pairs])
	B = np.array([d[1] for d in Pairs])
	Scores['E_score Train/Test'] = 0.5 - np.sum(((A-B)>0).astype('int'))/len(A)
	#################################
	
	return Scores


def Heat_capacity(Model_file,T=[1]):
	output = np.load(Model_file,allow_pickle=True)[()]

	if 'k_MCMC' in list(output['options'].keys()):
		k_mc = output['options']['k_MCMC']
	else: k_mc = output['options']['delta_t']

	J_mod,h_mod = output['J'],output['h']

	Ch = np.zeros(len(T))

	for i in range(len(T)):
		temp = T[i]
		align_temp=Create_modAlign(output,N = 10000,delta_t = 10000,ITER='',temperature=temp)
		Energies = compute_energies(align_temp,h_mod,J_mod)

		Ch[i] = (np.mean(Energies**2)-np.mean(Energies)**2)/(temp**2)
	return Ch

def entropy_with_AIS(Model_file,T=[1],Nb_runs = 100,M=int(1e5)):
	output = np.load(Model_file,allow_pickle=True)[()]

	if 'k_MCMC' in list(output['options'].keys()):
		k_mc = output['options']['k_MCMC']
	else: k_mc = output['options']['delta_t']

	J_mod,h_mod = output['J'],output['h']

	Entropy = np.zeros(len(T))

	for i in range(len(T)):
		temp = T[i] 
		align_temp=Create_modAlign(output,N = 50000,delta_t = 10000,ITER='',temperature=temp)
		Energies = compute_energies(align_temp,h_mod,J_mod)
		U = np.mean(Energies)
		WAIS = np.ones(Nb_runs)
		############## AIS ################
		
		### sample Nb_runs from the indepedant model ###
		samp = Create_modAlign(output,N = Nb_runs,delta_t = 5000,ITER='',temperature=temp)
		### Compute the unormalized probabilities of these samples ###
		E0 = compute_energies(samp,h_mod,np.zeros(J_mod.shape))
		Proba_0 = np.exp(-E0/temp)
  
		for m in tqdm(range(1,M+1)):
			### Compute the unormalized proba of the previous samples with the model m ###
			E1 = compute_energies(samp,h_mod,J_mod*m/M)
			Proba_1 = np.exp(-E1/temp)

			WAIS *= Proba_1/Proba_0
			samp = Montecarlo(h_mod,J_mod*m/M,1,samp)
			E0 = compute_energies(samp,h_mod,J_mod*m/M)
			Proba_0 = np.exp(-E0/temp)

		Z0 = np.prod(np.sum(np.exp(h_mod/temp),axis=1))
		ZM = Z0*np.mean(WAIS)
		F = -temp*np.log(ZM)
		###################################
		Entropy[i] = (U-F)/temp
	return Entropy

def Montecarlo(h,J,delta_t,Init_states):
	seed = int(time.time())
	np.random.seed(seed)
	q = h.shape[1]
	n_states,L = Init_states.shape
	new_states = np.copy(Init_states)
	for k in range(delta_t):
		for s in range(n_states):
			sigma = new_states[s]
			i = np.random.randint(0,L)
			aa_i = int(sigma[i])
			aa = np.random.choice(np.delete(np.arange(q),aa_i),1)
			sigma_bis = np.copy(sigma)
			sigma_bis[i] = aa
			delta_E = compute_energies(sigma_bis,h,J) - compute_energies(sigma,h,J)
			if delta_E<=-700: alpha=1
			else: alpha = np.exp(-delta_E)
			if np.random.random() < alpha:
				new_states[s] = sigma_bis
	return new_states