#!/usr/bin/env python
# coding: utf-8
#!conda install -c anaconda joblib -y
#!conda install -c anaconda scipy -y
#!conda install -c anaconda seaborn -y
#!conda install -c anaconda sklearn -y
#!conda install -c anaconda statsmodels -y


import numpy as np
import gzip, pickle
import scipy
import pandas as pd
import joblib
from scipy import stats, special, linalg
import argparse
import sys
import datetime
import gibbs
import statsmodels.api as sm
#compute by Y = X1*beta1+X2*X2
# beta3hat \sim N(R3 eta beta3,R3 N3-1,aI



def simulate_model(n, p, pi_k, sigma_3k, rho_1, rho_2, rho_3,seed = None):
    seed_value = 43
    np.random.seed(seed_value)

    # Print the seed value
    print(f"The seed value for beta0s is: {seed_value}")

    # Generate covariance matrices for each component
    cov_matrices = []
    for sigma in sigma_3k:
        cov_matrix = sigma * np.array([[1, rho_1, rho_2],
                                          [rho_1, 1, rho_3],
                                          [rho_2, rho_3, 1]])
        cov_matrices.append(cov_matrix)
    #np.random.seed(seed)
    # Function to simulate one sample
    def simulate_sample():
        if np.random.rand() < p:
            return np.zeros(3)  # delta_0 part
        else:
            component = np.random.choice(len(pi_k), p=pi_k)
            mean = np.zeros(3)
            cov = cov_matrices[component]
            return np.random.multivariate_normal(mean, cov)

    # Generate samples
    samples = np.array([simulate_sample() for _ in range(n)])
    return samples

def simulate_Y(samples,X1, X2, eta, R3,ld_boundaries,N3,  seed=None):
    # Simulate beta coefficients
    betas = samples
    sample_assig = (betas.sum(axis=1) != 0).astype(int)
    # Extract beta1 and beta2
    beta1 = betas[:, 0]
    beta2 = betas[:, 1]
    beta3 = betas[:, 2]
    seed_value = 43
    np.random.seed(seed_value)

    # Print the seed value
    print(f"The seed value for Y and beta3hat is: {seed_value}")

    
    # Compute Y
    Y0 = np.dot(beta1,X1)  + np.dot( beta2,X2) 
    eps = np.var(Y0)*(1/0.3-1)
    print('eps',eps,'Y0',np.var(Y0))
    Y = Y0 +  np.random.normal(loc=0, scale=np.sqrt(eps), size=len(Y0))
    

    # Compute beta3_hat
    beta3_hat = np.zeros(ld_boundaries[len(ld_boundaries)-1][1])
    for j in range(len(ld_boundaries)):
        start_i = ld_boundaries[j][0]
        end_i = ld_boundaries[j][1]
        ref_ld3 = R3[j]
        beta3_i = beta3[start_i:end_i]
        mean_beta3_hat = eta * np.dot(beta3_i, ref_ld3)
        cov_beta3_hat = ref_ld3/N3 
        beta3_hat[start_i:end_i] = np.mean(np.random.multivariate_normal(mean_beta3_hat, cov_beta3_hat, 100), axis=0)

    return Y, beta3_hat , sample_assig,eps

# app
def SDPRX_m_gibbs(Y, X1,X2, beta_margin3,p,N3,  rho, ld_boundaries,  ref_ld_mat3, var_pos ,mcmc_samples, 
    burn,  tau ,sample_assig,samples,save_mcmc = None, n_threads=4, VS=True):

    l = len(beta_margin3)
    trace = { 'beta1':np.zeros(shape=(mcmc_samples, l )),
        'beta2':np.zeros(shape=(mcmc_samples, l)),
        'beta3':np.zeros(shape=(mcmc_samples, l)),
        'suffstats':[], 'h2_1':[], 'h2_3':[]}#'alpha':[], 'num_cluster':[],

    # initialize
    state = gibbs.initial_state(Y=Y, data1=X1, data2=X2,data3 = beta_margin3,p = p,N3=N3,ld_boundaries=ld_boundaries,tau = tau,var_pos=var_pos)
    #state['beta1'] = samples[:,0]
    #state['beta2'] = samples[:,1]
    #state['beta3'] = samples[:,2]
    state['residual'] = state['A']-np.dot(state['beta1'],state['X1_'])-np.dot(state['beta2'],state['X2_'])
    state['sample_assig'] = sample_assig
    rho_1,rho_2,rho_3 = rho
    state['det_sigma_'] = 1-rho_1**2-rho_2**2-rho_3**2+2*rho_1*rho_2*rho_3
    for j in range(len(ld_boundaries)):
        start_i = ld_boundaries[j][0]
        end_i = ld_boundaries[j][1]
        ref_ld3 = ref_ld_mat3[j]
        X1_i = state['X1_'][start_i:end_i]
        X2_i = state['X2_'][start_i:end_i]
        
        state['A3'].append( np.linalg.solve(ref_ld3+ state['N3_']*state['a']*np.identity(ref_ld3.shape[0]), ref_ld3))
        state['B3'].append( np.dot(ref_ld3, state['A3'][j]) )
        state['B1'].append(np.dot(X1_i, X1_i.T))
        state['B2'].append(np.dot(X2_i, X2_i.T))
        state['C'].append(np.dot(X1_i, X2_i.T))
        
    for i in range(mcmc_samples):
        # update everything
        gibbs.gibbs_stick_break(state, rho, ld_boundaries=ld_boundaries, ref_ld_mat3=ref_ld_mat3, sample_assig=sample_assig,
             n_threads=n_threads, VS=VS)

        if (i > burn):
            trace['h2_1'].append(state['h2_1'])
            trace['h2_3'].append(state['h2_3']*state['eta']**2)
        print(i,'h2_1: ', state['h2_1'], 'h2_3: ', state['h2_3']*state['eta']**2)
        # record the result
        trace['beta1'][i,] = state['beta1']
        trace['beta2'][i,] = state['beta2']
        trace['beta3'][i,] = state['beta3']*state['eta']
        # trace['pi'][i,:] = np.array(state['pi'].values())
        # trace['cluster_var'][i,:] = np.array(state['cluster_var'].values())
        # trace['alpha'].append(state['alpha'])
        # trace['num_cluster'].append( np.sum(np.array(state['pi'].values()) > .0001) )
        # trace['suffstats'].append(state['suffstats'])

        #util.progressBar(value=i+1, endvalue=mcmc_samples)

    # calculate posterior average
    poster_mean1 = np.mean(trace['beta1'][burn:mcmc_samples], axis=0)
    poster_mean2 = np.mean(trace['beta2'][burn:mcmc_samples], axis=0)
    poster_mean3 = np.mean(trace['beta3'][burn:mcmc_samples], axis=0)
    
    print ('m_h2_1: ',np.median(trace['h2_1']),' m_h2_3: ' ,np.median(trace['h2_3']))
    #print('mean same assignment',np.mean(state['same_assig']))

    #print state['pi_pop']

    if save_mcmc is not None:
        df_beta1 = pd.DataFrame(trace['beta1'], columns=[f'beta1_{i}' for i in range(l)])
        df_beta2 = pd.DataFrame(trace['beta2'], columns=[f'beta2_{i}' for i in range(l)])
        df_beta3 = pd.DataFrame(trace['beta3'], columns=[f'beta3_{i}' for i in range(l)])
        df_beta1.to_csv('beta1.csv', index=False)
        df_beta2.to_csv('beta2.csv', index=False)
        df_beta3.to_csv('beta3.csv', index=False)



    return poster_mean1, poster_mean2, poster_mean3



def silumations(X1,X2,ld_dict,N,N3,p_sparse,pi_k,rho,n_sim,burn,ID):
    
    ld_boundaries = ld_dict[0]
    R3 = ld_dict[1]
    # ld_dict[0] is the LD boundary and ld_dict[1] is the LD matrix
    #ld_boundaries = ld_boundaries[:3]########note it?
    num_snp = np.max(np.max(ld_boundaries))
    
    rho_1,rho_2,rho_3=rho

    sigma_3k = [0.3/(num_snp*(1-p_sparse))]

    samples = simulate_model(num_snp, p_sparse, pi_k, sigma_3k, rho_1, rho_2, rho_3)

    #15357 #snp 10000 #patients
    #to subset
    X1_sub = X1[0:num_snp,0:N]
    X2_sub = X2[0:num_snp,0:N]
    
    eta = 1
    
    Y, beta3_hat ,sample_assig,eps= simulate_Y(samples,X1_sub, X2_sub, eta, R3,ld_boundaries, N3)
    pheno = pd.DataFrame({'ID': list(range(1, len(Y) + 1)),'ID2': list(range(1, len(Y) + 1)),'Y':Y})
    pheno.to_csv('pheno_'+ID+'.txt',header=False, index=False, sep=' ')
    print("Store simulated pheno data")
    #print(sum(beta3_hat==0))
    #print('sample_assig',)
    #print(np.var(np.dot(samples[:,0],X1_sub))/np.var(Y)) print(np.var(np.dot(samples[:,1],X2_sub))/np.var(Y))
    #print(np.var(np.dot(samples[:,0],X1_sub)+np.dot(samples[:,1],X2_sub))/np.var(Y))
    #Y_new = pd.DataFrame(Y)
    #Y_new.to_csv("simdata.csv")
    burn = 500
    rho = [rho_1,rho_2,rho_3]
    tau = 1/eps

    beta_1,beta_2,beta_3 = SDPRX_m_gibbs(Y, X1_sub,X2_sub, beta3_hat,p_sparse,N3, rho, ld_boundaries,  R3,sigma_3k, n_sim, burn,tau,sample_assig,samples)
    
    #X_combined = np.vstack((X1_sub, X2_sub))
    # Fit the regression model
    #model = sm.OLS(Y, X_combined.T)
    #results = model.fit()
    #coefficients = results.params
    #coef_X1 = coefficients[:X1_sub.shape[0] ]
    #coef_X2 = coefficients[X1_sub.shape[0]:]

    return beta_1,beta_2,beta_3,beta3_hat,samples#,coef_X1,coef_X2




def pipeline(args):
    
    # sanity check

    if args.bfile is not None and args.load_ld is not None:
        raise ValueError('Both --bfile and --load_ld flags were set. \
            Please use only one of them.')

    if args.bfile is None and args.load_ld is None:
        raise ValueError('Both --bfile and --load_ld flags were not set. \
            Please use one of them.')
        
    print('Load individual data from {}'.format(args.ss1))
    X1 = np.loadtxt(args.ss1)
    print('Load individual data from {}'.format(args.ss2))
    X2 = np.loadtxt(args.ss2)
    
    if args.load_ld is not None:
        print('Load pre-computed reference LD from {}'.format(args.load_ld))
        with gzip.open(args.load_ld, 'rb') as f:
            ld_dict = pickle.load(f, encoding='latin1')
        f.close()
    else:
        print('Calculating reference LD. May take ~ 2 hours ...')
        ld_boundaries = ld.parse_ld_boundaries(min_block_wd=100, ss=ss, block_path=args.block)
        ref_ld_mat = Parallel(n_jobs=args.threads)(delayed(ld.calc_ref_ld)(i, ref_path=args.bfile, 
                ld_boundaries=ld_boundaries) for i in range(len(ld_boundaries))) 
        if args.save_ld is not None:
            print('Save computed reference LD to {}'.format(args.save_ld))
            f = gzip.open(args.save_ld, 'wb')
            pickle.dump([ld_boundaries, ref_ld_mat], f, protocol=2)
            f.close()
    rho1,rho2,rho3 = args.rho
    print('rho is ',rho1,rho2,rho3)
    print('Start MCMC ...')
    beta_1,beta_2,beta_3, beta3_hat, samples = silumations(X1,X2,ld_dict,args.N,args.N3,args.p_sparse,args.pi_k, args.rho,args.n_sim,args.burn,args.ID)
   
    #beta_1,beta_2,beta_3, beta3_hat, samples,coef_X1,coef_X2 = silumations(X1,X2,ld_dict,args.N,args.N3,args.p_sparse,args.pi_k, args.rho,args.n_sim,args.burn)
    print('Done!\nWrite output to {}'.format(args.out+'.txt'))
    #print(len(beta_1),len(beta_2),len(beta_3),len(beta3_hat),len(coef_X1),len(coef_X2))
    
    results = pd.DataFrame({'beta1':beta_1,'beta2':beta_2,'beta3':beta_3,               'beta10':samples[:,0],'beta20':samples[:,1],'beta30':samples[:,2],'beta3_hat':beta3_hat})
    #results = pd.DataFrame({'beta1':beta_1,'beta2':beta_2,'beta3':beta_3, 'beta10':samples[:,0],'beta20':samples[:,1],'beta30':samples[:,2],'beta3_hat':beta3_hat,'coef_X1':coef_X1,'coef_X2':coef_X2})
    results.to_csv(args.out+'sim_Results_'+args.ID+'.csv', index=False)
    

parser = argparse.ArgumentParser(prog='SDPR',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Version 0.0.1 Test Only")

parser.add_argument('--ss1', type=str, required=True,
                        help='Path to individual statistics 1. e.g. /home/tutor/myss.txt')

parser.add_argument('--ss2', type=str, required=True,
                        help='Path to individual statistics 2. e.g. /home/tutor/myss.txt')

parser.add_argument('--load_ld', type=str, default=None,required=True,
                        help='Prefix of the location to load calculated LD Reference file \
                        in pickled and gzipped format.')

parser.add_argument('--N', type=int, default=None, required = True,
                        help='Number of patients in population with individual data. e.g. 10000')

parser.add_argument('--N3', type=int, default=None, required=True,
                        help='Number of individuals in summary statistic sile 3.')

parser.add_argument('--n_sim', type=int, default=None, required=True,
                        help='Number of simulations')

parser.add_argument('--rho', type=float, nargs=3, default=[0.8,0.5,0.3], 
                        help='Transethnic genetic correlation.')

parser.add_argument('--p_sparse', type=float, default=None,required=True,
                        help='Proportion of non-causal snps in simulation data.')

parser.add_argument('--pi_k', type=list, default=[1],
                        help='Distribution of sigma_k.')

parser.add_argument('--VS', type=bool, default=True, 
                        help='Whether to perform variable selection.')
parser.add_argument('--bfile', type=str, default=None,
                        help='Path to reference LD file. Prefix for plink .bed/.bim/.fam.')

parser.add_argument('--threads', type=int, default=1, 
                        help='Number of Threads used.')

parser.add_argument('--seed', type=int, 
                        help='Specify the seed for numpy random number generation.')

parser.add_argument('--burn', type=int, default=None,required=True,
                        help='Specify the total number of iterations to be discarded before \
                        Markov Chain approached the stationary distribution.')

parser.add_argument('--save_ld', type=str, default=None,
                        help='Prefix of the location to save calculated LD Reference file \
                        in pickled and gzipped format.')

parser.add_argument('--save_mcmc', type=str, default=None,
                        help='Prefix of the location to save intermediate output of MCMC \
                        in pickled and gzipped format.')

parser.add_argument('--out', type=str, required=True,
                        help='Prefix of the location for the output tab deliminated .txt file.')

parser.add_argument('--ID', type=str, required=True,
                        help='Profix of the name for the output .csv file.')


def main():
    if sys.version_info[0] != 3:
        print(sys.version_info[0],'ERROR: SDPR currently does not support Python 3')
        sys.exit(1)
    pipeline(parser.parse_args())

if __name__ == '__main__':
    main()





