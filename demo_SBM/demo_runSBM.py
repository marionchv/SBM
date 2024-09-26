"""
Created in 2024

@author: Marion CHAUVEAU
"""

####################### MODULES #######################
import numpy as np
import SBM.SBM_GD.SBM_proteins as sbm
import argparse
import os
if os.getcwd().split('/').pop() == 'Grid_search_Nchains': 
	os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
##########################################################

def run_SBM(Input_MSA,fam,Model,train_file,N_iter, m, N_chains_list,Nb_rep,k_MCMC,TestTrain,ParamInit,lambdJ,lambdh,theta):
    fam = str(fam)

    for N_chains in N_chains_list:
        for repetition in range(Nb_rep):
            align = np.load(str(Input_MSA))
            if train_file is not None:
                ind_train = np.load(train_file)
                print('Database size: ', align.shape, ' & Training set size: ', len(ind_train))
            else:
                ind_train = None
                print('Database size: ', align.shape)

            options = dict([('Model', Model),
                            ('N_iter', N_iter), ('N_chains', N_chains), ('m', m), 
                            ('skip_log', 1), ('theta', theta), ('k_MCMC', k_MCMC),
                            ('lambda_h', lambdh), ('lambda_J', lambdJ),
                            ('Pruning', False), ('Pruning Mask', None),
                            ('Param_init', ParamInit),
                            ('Test/Train', TestTrain==1), ('Train sequences', ind_train),
                            ('Weights', None), ('SGD', None),
                            ('Seed', None), ('Zero Fields', False), 
                            ('Store Parameters', None)])

            output = sbm.SBM(align, options)
            dossier = "results/"+fam+"/"

            if not os.path.exists(dossier):
                os.makedirs(dossier)
            
            r = 0
            while os.path.exists(dossier+fam+'_m'+str(m)+'Ns'+str(N_chains)+'Ni'+str(N_iter)+'R'+str(r)+'.npy'):
                r+=1
            np.save(dossier+fam+'_m'+str(m)+'Ns'+str(N_chains)+'Ni'+str(N_iter)+'R'+str(r)+'.npy', output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SBM parameters.')
    parser.add_argument('fam',help='Protein family name in a numpy format')
    parser.add_argument('--train_file', type=str, default=None, help='Ind_train filename')
    parser.add_argument('--TestTrain', type=int, default=1, help='1 if you want Test/Train sets, 0 otherwise')
    parser.add_argument('--rep', type=int, default=3, help='Number of repetitions')
    parser.add_argument('--mod', type=str, default='SBM', help='Model')
    parser.add_argument('--N_iter', type=int, default=500, help='Number of iterations')
    parser.add_argument('--m', type=int, default=1, help='Parameter m')
    parser.add_argument('--N_chains', type=int, nargs='+', help='List of N_chains values')
    parser.add_argument('--ParamInit', type=str, default='Profile', help='Init of fields and couplings')
    parser.add_argument('--k_MCMC', type=int, default=10000, help='Number of MCMC steps')
    parser.add_argument('--lambdJ', type=int, default=0, help='lambda J')
    parser.add_argument('--lambdh', type=int, default=0, help='lambda h')
    parser.add_argument('--theta', type=int, default=0.2, help='threshold to compute the effective number of sequences')
    parser.add_argument('Input_MSA')

    args = parser.parse_args()
    run_SBM(args.Input_MSA,args.fam,args.mod,args.train_file,args.N_iter, 
            args.m, args.N_chains,args.rep,args.k_MCMC,args.TestTrain,
            args.ParamInit,args.lambdJ,args.lambdh,args.theta)