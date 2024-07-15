import numpy as np
from scipy import stats, special, linalg
from collections import Counter
from joblib import Parallel, delayed


def initial_state(Y, data1, data2,data3,p, N3, ld_boundaries,var_pos, tau, alpha=1.0, a0k=.5, b0k=.5):
    num_clusters = len(var_pos)
    n_snp = len(data3)
    n_idv = len(Y)
    state = {
        'Y_':Y,
        'X1_': data1,
        'X2_': data2,
        'beta_margin3_': data3,
        'N3_':N3,
        'b1': np.zeros(n_snp),
        'b2': np.zeros(n_snp),
        'b3': np.zeros(n_snp),
        'B1': [],
        'B2': [],
        'B3': [],
        'A3':[],
        'C':[],
        'same_assig':[],
        'beta1': np.zeros(n_snp),
        'beta2': np.zeros(n_snp),
        'beta3': np.zeros(n_snp),
        'num_clusters_': num_clusters,
        'hyperparameters_': {
            "a0k": a0k,"b0k": b0k,
            "a0": 0.1,"b0": 0.1,
        },
        'det_sigma_':0,
	    'assignment': np.zeros(n_snp),
        'pi': np.array([alpha / num_clusters]*num_clusters),
        'p': np.array([p,1-p]),
        'var': var_pos,# possible variance
        'h2_1': 0,
        'h2_3': 0,
        'eta': 1,
	    'tau':tau,
        'alpha':0,
        'W1':np.zeros(n_idv),
        'residual':np.zeros(n_idv)
    }

    # define indexes
    state['a'] = 0.1/N3; state['c'] = 1
    state['A'] = Y-state['alpha']*state['W1']*np.ones(len(Y))
    print('start assignment',sum(state['assignment']))
    
    return state   
    
   
def calc_b(j, state, ld_boundaries, ref_ld_mat3):
    
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    X1_i = state['X1_'][start_i:end_i]
    X2_i = state['X2_'][start_i:end_i]
    C=np.array(np.diag(state['C'][j]))
    ref_ld3 = ref_ld_mat3[j]
    B1 = np.diag(state['B1'][j]);B2 = np.diag(state['B2'][j]);B3 =state['B3'][j]
    residual = state['residual']
    b1 = np.dot(X1_i,residual) \
     + C*np.array(state['beta2'][start_i:end_i])+ B1*np.array(state['beta1'][start_i:end_i])
    b1 = b1*state['tau']
    
    b2 = np.dot(X2_i,residual) \
    + C*np.array(state['beta1'][start_i:end_i])+ B2*np.array(state['beta2'][start_i:end_i])
    b2 = b2*state['tau']
    
    b3 = state['eta']*np.dot(state['A3'][j], state['beta_margin3_'][start_i:end_i]) - state['eta']**2 * \
    (np.dot(B3, state['beta3'][start_i:end_i]) - np.diag(B3)*state['beta3'][start_i:end_i])
    state['b1'][start_i:end_i] = b1
    state['b2'][start_i:end_i] = b2
    state['b3'][start_i:end_i] = b3

  
   
def vectorized_random_choice(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    return k
    
def sample_assignment(j, ld_boundaries,ref_ld_mat3, state, VS, rho):
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    rho_1,rho_2,rho_3 = rho
    det_sigma = state['det_sigma_']
    tau = state['tau']
    N3 = state['N3_']
    num_snp = end_i - start_i
    
    b1 = state['b1'][start_i:end_i]
    b2 = state['b2'][start_i:end_i]
    b3 = state['b3'][start_i:end_i]
    
    B1 = state['B1'][j]; B2 = state['B2'][j];B3 = state['B3'][j]
    C = state['C'][j]
    
    A = np.array([np.empty((3, 3)) for _ in range(num_snp)])
    B = np.empty((C.shape[0],3))
    log_prob = np.zeros((2,num_snp))
    # shared with correlation
    var_k = state['var'][0]
    deno_k = det_sigma*var_k
    ak1 = np.add.outer(.5*state['tau']*np.diag(B1), .5*(1-rho_3**2)/deno_k)
    ak2 = np.add.outer(.5*state['tau']*np.diag(B2), .5*(1-rho_2**2)/deno_k)
    ak3 = np.add.outer(.5*N3*state['eta']**2*np.diag(B3), .5*(1-rho_1**2)/deno_k)
    ck1 = (rho_3-rho_1*rho_2) / deno_k
    ck2 = (rho_2-rho_1*rho_3) / deno_k
    ck3 = np.subtract.outer((rho_1-rho_2*rho_3) / deno_k ,state['tau']*np.diag(C))
    for i in range(num_snp):
        A_i = np.array([
                [2 * ak1[i], -ck3[i], -ck2],
                [-ck3[i], 2 * ak2[i], -ck1],
                [-ck2, -ck1, 2 * ak3[i]]
                ])
        B_i = np.array([b1[i], b2[i], N3*b3[i]])
        exp_ele = 0.5 * np.dot(np.dot(B_i.T, np.linalg.inv(A_i)), B_i)
        non_exp = -0.5*np.log(np.linalg.det(A_i)) - 1.5*np.log(state['var'][0])  - .5*np.log(det_sigma) +np.log(state['p'][1]/state['p'][0])+ np.log( state['pi'][0]+1e-40)
        log_prob[1,i] = exp_ele + non_exp
    logexpsum = special.logsumexp(log_prob, axis=0,keepdims=True)
    prob_mat = np.exp(log_prob - logexpsum)
    state['assignment'][start_i:end_i] = vectorized_random_choice(prob_mat,range(1))
    


def sample_beta(j, state,  ld_boundaries,  ref_ld_mat3, rho, VS=True):
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    X1 = state['X1_'][start_i:end_i];X2 = state['X2_'][start_i:end_i]
    A = state['residual']+np.dot(state['beta1'][start_i:end_i],X1)+np.dot(state['beta2'][start_i:end_i],X2)
    B3 = state['B3'][j];A3 = state['A3'][j]

    assignment = state['assignment'][start_i:end_i]

    num_snp = end_i - start_i
    
    beta1 = np.zeros(num_snp ); beta2 =np.zeros(num_snp);beta3 = np.zeros(num_snp )

    # null
    # shared with correlation
    idx = assignment != 0
    nonzero = sum(idx)
    zeros = np.zeros((nonzero, nonzero))
    if sum(idx)==0 :
        # all SNPs in this block are non-causal
        print("non causal")
        pass
    else:
        var = [state['var'][0] for i in assignment[idx]]
        # population LD matrix
        shrink_ld = np.block([[state['tau']*np.dot(X1[idx],X1[idx].T),     state['tau']*np.dot(X1[idx],X2[idx].T),           zeros],
                              [state['tau']*np.dot(X2[idx],X1[idx].T),      state['tau']*np.dot(X2[idx],X2[idx].T),      zeros],
                             [zeros,                             zeros,   state['eta']**2*state['N3_']*B3[idx,:][:,idx]]])
       
        #print(shrink_ld)
        diag = np.diag([x for x in var])
        #print('diag',diag)
        
        rho_1,rho_2,rho_3 = rho
        cov_matrix =  np.array([[1, rho_1, rho_3],
                                [rho_1, 1, rho_2],
                                [rho_3, rho_2, 1]])

        

        var_mat = np.zeros((3*nonzero,3*nonzero))
        for i in range(3):
            for j in range(3):
                var_mat[i*nonzero:(i+1)*nonzero, j*nonzero:(j+1)*nonzero] = cov_matrix[i, j] *diag
        var_mat2 = np.linalg.inv(var_mat)
  
        mat = shrink_ld + var_mat2 + np.eye(var_mat.shape[0]) * 1e-10
        try:
            cov_mat = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            print("Matrix is not invertible.")
        
        # A matrix
        A_gamma = np.concatenate([state['tau']*np.dot(X1[idx],A),
                                  state['tau']*np.dot(X2[idx],A),
               state['eta']*state['N3_']*np.dot(A3[idx,:], state['beta_margin3_'][start_i:end_i])])
        
        mu = np.dot(cov_mat, A_gamma)
        beta_tmp = np.random.multivariate_normal(mu, cov_mat, size=1)
        beta1[idx] = beta_tmp[0,:nonzero]
        beta2[idx] = beta_tmp[0,nonzero:2*nonzero]
        beta3[idx] = beta_tmp[0,2*nonzero:3*nonzero]
    state['beta1'][start_i:end_i] = beta1
    state['beta2'][start_i:end_i] = beta2
    state['beta3'][start_i:end_i] = beta3


def gibbs_stick_break(state, rho, ld_boundaries, ref_ld_mat3,sample_assig, n_threads=4, VS=True):
    #condition = (state['assignment']==state['sample_assig']) & (state['assignment']==1)
    #print('assignment same:',sum(state['assignment']==state['sample_assig']), 'one',sum(state['assignment']==1),'true',sum(state['sample_assig']),'both',sum(condition))   
    #print('variance of residual is ',np.var(state['residual']))
    var1 = np.zeros(state['N3_'])
    var3 = 0
    #state['assignment'] = sample_assig
    for j in range(len(ld_boundaries)):
        calc_b(j, state, ld_boundaries, ref_ld_mat3)
        sample_assignment(j,ld_boundaries, ref_ld_mat3, state, rho=rho, VS=True)
        sample_beta(j, state, ld_boundaries,ref_ld_mat3 , rho=rho, VS=True)
        state['residual'] = state['A']-np.dot(state['beta1'],state['X1_'])-np.dot(state['beta2'],state['X2_']) 
        #print(j,'max(residual)',max(state['residual']),min(state['residual']), np.var(state['residual']))
              
    #state['assignment'] = np.concatenate(Parallel(n_jobs=n_threads, require='sharedmem')(delayed(sample_assignment)(j=j, ld_boundaries= ld_boundaries, ref_ld_mat3=ref_ld_mat3, state=state, rho=rho, VS=True) for j in range(len(ld_boundaries))))
    for j in range(len(ld_boundaries)):    
        start_i = ld_boundaries[j][0]
        end_i = ld_boundaries[j][1]
    
        X1_i = state['X1_'][start_i:end_i]
        X2_i = state['X2_'][start_i:end_i]
        
        ref_ld3 = ref_ld_mat3[j]
        var3 = var3+np.sum(state['beta3'][start_i:end_i] * np.dot(ref_ld3, state['beta3'][start_i:end_i]))

    state['h2_1'] = np.var(np.dot(state['beta1'],state['X1_'])+np.dot(state['beta2'],state['X2_']))/np.var(state['Y_'])
    state['h2_3'] = var3
