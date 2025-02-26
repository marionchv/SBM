"""
Created in 2024

@author: Marion CHAUVEAU
"""

####################### MODULES #######################
import numpy as np
import SBM.SBM_GD.SBM_proteins as sbm
import SBM.utils.utils as ut
import argparse
import os
if os.getcwd().split('/').pop() == 'demo_SBM': 
	os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))
##########################################################

def run_SBM(Input_MSA,fam,Model,train_file,N_iter, m, N_chains_list,Nb_rep,Nb_av,k_MCMC,TestTrain,ParamInit,lambdJ,lambdh,theta,PseudoCount,InfiniteMaskFields):
    fam = str(fam)

    for repetition in range(Nb_rep):
        for N_chains in N_chains_list:
            W_rep = np.array([[]])
            Jnorm_rep = np.array([[]])
            Seeds_rep = np.zeros(Nb_av)
            Extime_rep = np.zeros(Nb_av)
            for n_av in range(Nb_av):
                print('AVG: ',n_av)
                align = np.load(str(Input_MSA))
                if train_file is not None:
                    ind_train = np.load(train_file)
                    print('Database size: ', align.shape, ' & Training set size: ', len(ind_train))
                else:
                    ind_train = None
                    print('Database size: ', align.shape)
                if InfiniteMaskFields is not None:
                      InfiniteMaskFields_mat = np.load(InfiniteMaskFields)

                options = dict([('Model', Model),
                                ('N_iter', N_iter), ('N_chains', N_chains), ('m', m), 
                                ('skip_log', 1), ('theta', theta), ('k_MCMC', k_MCMC),
                                ('lambda_h', lambdh), ('lambda_J', lambdJ),
                                ('Pruning', False), ('Pruning Mask', None),
                                ('Param_init', ParamInit),
                                ('PseudoCount', PseudoCount==1),
                                ('Infinite Mask Fields',InfiniteMaskFields_mat),
                                ('Test/Train', TestTrain==1), ('Train sequences', ind_train),
                                ('Weights', None), ('SGD', None),
                                ('Seed', None), ('Zero Fields', False), 
                                ('Store Parameters', None)])

                output = sbm.SBM(align, options)
                
                J_out,h_out = ut.Zero_Sum_Gauge(output['J'],output['h'])
                W_out = ut.Wj(J_out,h_out)
                W_rep = np.concatenate((W_rep,np.expand_dims(W_out,axis=0)),axis=int((n_av==0)))

                Jnorm_rep = np.concatenate((Jnorm_rep,np.expand_dims(output['J_norm'],axis=0)),axis=int((n_av==0)))

                Seeds_rep[n_av] = output['options']['Seed']
                Extime_rep[n_av] = output['Execution time']



            W_av = np.mean(W_rep,axis=0)
            J_av,h_av = ut.Jw(W_av,output['options']['q'])
            output_av = {'J':J_av,'h':h_av,'W_all':W_rep,'Seeds':Seeds_rep,'Execution times':Extime_rep,'J_norm':Jnorm_rep,'align':output['align'],'Test':output['Test'],'Train':output['Train']}

            output_av['options0'] = {'Model':output['options']['Model'], 
                                     'N_iter':output['options']['N_iter'],
                                     'N_chains':output['options']['N_chains'],
                                     'm':output['options']['m'],
                                     'theta':output['options']['theta'],
                                     'k_MCMC':output['options']['k_MCMC'], 
                                     'lambda_h':output['options']['lambda_h'], 
                                     'lambda_J':output['options']['lambda_J'],  
                                     'Param_init':output['options']['Param_init'],
                                     'PseudoCount':output['options']['PseudoCount']}

            output_av['options1'] = {'skip_log':output['options']['skip_log'],
                                     'Pruning':output['options']['Pruning'],
                                     'Pruning Mask':output['options']['Pruning Mask'],
                                     'Test/Train':output['options']['Test/Train'],
                                     'Train sequences':output['options']['Train sequences'],
                                     'Weights':output['options']['Weights'],
                                     'SGD':output['options']['SGD'],
                                     'Seed':output['options']['Seed'],
                                     'Zero Fields':output['options']['Zero Fields'],
                                     'Store Parameters':output['options']['Store Parameters'],
                                     'Learning_rate':output['options']['Learning_rate'],
                                     'Pruning_perc':output['options']['Pruning_perc'],
                                     'Shuffle Columns':output['options']['Shuffle Columns'],
                                     'q':output['options']['q'],
                                     'L':output['options']['L']}

            dossier = "results/"+fam+"/"

            if not os.path.exists(dossier):
                os.makedirs(dossier)
            
            r = 0
            file_name = fam
            key_list = list(output_av['options0'].keys())
            key_list = np.sort(key_list)
            for k in key_list:
                file_name+='_'+k+str(output_av['options0'][k])
            file_name+='_N_Av'+str(Nb_av)
            
            while os.path.exists(dossier+file_name+'_R'+str(r)+'.npy'):
                r+=1
            np.save(dossier+file_name+'_R'+str(r)+'.npy', output_av)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SBM parameters.')
    parser.add_argument('fam',help='Protein family name in a numpy format')
    parser.add_argument('--train_file', type=str, default=None, help='Ind_train filename')
    parser.add_argument('--TestTrain', type=int, default=0, help='1 if you want Test/Train sets, 0 otherwise')
    parser.add_argument('--PseudoCount', type=int, default=0, help='1 if you want a 1/Neff pseudo count, 0 otherwise')
    parser.add_argument('--rep', type=int, default=1, help='Number of repetitions')
    parser.add_argument('--N_av', type=int, default=1, help='Number of averaged models')
    parser.add_argument('--mod', type=str, default='SBM', help='Model')
    parser.add_argument('--N_iter', type=int, default=400, help='Number of iterations')
    parser.add_argument('--m', type=int, default=20, help='Parameter m')
    parser.add_argument('--N_chains', type=int, nargs='+', help='List of N_chains values')
    parser.add_argument('--ParamInit', type=str, default='Zero', help='Init of fields and couplings')
    parser.add_argument('--k_MCMC', type=int, default=10000, help='Number of MCMC steps')
    parser.add_argument('--lambdJ', type=float, default=0, help='lambda J')
    parser.add_argument('--lambdh', type=float, default=0, help='lambda h')
    parser.add_argument('--theta', type=int, default=-1, help='threshold to compute the effective number of sequences')
    parser.add_argument('--InfiniteMaskFields', type=str, default=None, help='Mask to impose infinite fields where entries are 0')
    parser.add_argument('Input_MSA')

    args = parser.parse_args()
    run_SBM(args.Input_MSA,args.fam,args.mod,args.train_file,args.N_iter, 
            args.m, args.N_chains,args.rep,args.N_av,args.k_MCMC,args.TestTrain,
            args.ParamInit,args.lambdJ,args.lambdh,args.theta,args.PseudoCount,args.InfiniteMaskFields)
