import numpy as np


def sample_chain(T, dc, pi, ttr):
    """
    Sample a random sequence from the prior.
    Arguments:
    pi - array of initial probabilities pi[j] = P(x(0)=j)
    ttr - transition table, ttr[i, j] = P(x(t)=j|x(t-1)=i)
    Return:
    x - sequence
    """
    c = np.zeros((T, dc), 'd')
    # t=0
    c[0,:] = np.argmax(np.random.multinomial(1, pi, size=(dc)))
    # t>0
    for t in range(1,T):
        for i in range(dc):
            p = ttr[int(c[t-1,i]), :]
            tmp = np.random.multinomial(1, p, size=(1))
            c[t, i] = np.argmax(tmp[0])
    return c


def get_prior(T, pi, ttr):
    y = np.zeros((T, 1), dtype='d')
    def output_distr(y, state, t):
        return 1. 
    p, L, eps = forward_backward(y, pi, ttr, output_distr)
    return p, L, eps


def forward_backward(y, pi, ttr, output_distr, store_A=True, *args):
    """
    Forward-backward algorithm. It infers P(x(t)=state) from observations
    y(1)...y(T) for a Markov chain x(t) with output distribution
    'output_distr'.

    Arguments:
    y - array observations; variables are on columns, observations on rows.
        E.g., y[t,i] is the t-th observation of the i-th variable
    pi - array of initial probabilities pi[j] = P(x(0)=j)
    ttr - transition table, ttr[i, j] = P(x(t)=j|x(t-1)=i)
    output_distr - function returning the output probability of an observation
                given a hidden state. It is called like this:
                 output_distr(y[t,:], state, t, *args)
                *args can contain various parameters of the distribution
    store_A - if True, stores the intermediate values of the output distr;
               it is faster but requires additional memory (T*nstates array)
    *args - other arguments that will be passed to the out_distr function

    Returns:
    gamma - array of state probabilities
            gamma[t,i] = P(x(t)=i|y(1), ..., y(T))
    L - log-likelihood
    """
    
    nstates = pi.shape[0]
    T = y.shape[0]

    ##########
    # forward
    ##########
    # alpha[t,j] = P(y_1...y_t,x_t=j) (this formula does not hold
    # strictly because of the scaling-trick)
    alpha = np.zeros((T, nstates), 'd')
    # scaling variable used for the scaling trick in order to avoid
    # underflow problems
    rho = np.zeros((T), 'd')
    ### t=0
    if store_A: A = np.zeros((T, nstates), 'd')
    for i in range(nstates):
        # compute A(i) = P(y_0 | x_0=i)
        A_i = output_distr(y[0,:], i, 0, *args)
        if store_A: A[0, i] = A_i
        alpha[0, i] = pi[i]*A_i
    rho[0] = np.sum(alpha[0,:])
    if rho[0]==0.:
        print 'aiutino',0
        alpha[0,:] = 1.e-300
        rho[0] = np.sum(alpha[0,:])
    alpha[0,:] /= rho[0]
    ### t>0
    for t in range(1,T):
        # cn is the state number, c is the corresponding content
        for i in range(nstates):
            # compute A(i) = P(y_t | x_t=i)
            A_i = output_distr(y[t,:], i, t, *args)
            if store_A: A[t, i] = A_i
            alpha[t,i] = A_i * np.sum(alpha[t-1,:]*ttr[:,i])
        # scaling trick
        rho[t] = np.sum(alpha[t,:])
        if rho[t]==0.:
            print 'aiutino',t
            alpha[t,:] = 1.e-300
            rho[t] = np.sum(alpha[t,:])
        alpha[t,:] /= rho[t]

    #print 'rho', rho[:6]
    
    ###########
    # backward
    ###########
    # beta[t,j] = P(y_t+1|x_t=j) (this formula does not hold
    # strictly because of the scaling-trick)
    beta = np.zeros((T, nstates), 'd')
    # t=T-1
    beta[-1,:] = 1
    # t<T-1
    for t in range(T-2, -1, -1):
        if store_A:
            A_t1 = A[t+1,:]
        else :
            A_t1 = [output_distr(y[t+1,:], i, t+1, *args)
                    for i in range(nstates)]
        for j in range(nstates):
            beta[t,j] = np.sum(ttr[j,:]*beta[t+1,:]*A_t1[:])
        # scaling trick
        beta[t,:] /= rho[t+1]

    # total probability
    # gamma[t,i] = P(x_t=i|y_1 ... y_T)
    gamma = alpha*beta
        
    # log likelihood
    L = np.sum(np.log(rho))
    # epsilon[t,i,j] = P(x_t=i, x_t+1=j|y_1 ... y_T)
    epsilon = np.zeros((T-1, nstates, nstates), 'd')
    for t in range(T-1):
        for j in range(nstates):
            if store_A:
                A_j = A[t+1,j]
            else:
                A_j = output_distr(y[t+1,:], j, t+1, *args)
            epsilon[t,:,j] = alpha[t,:]*ttr[:,j]*A_j*beta[t+1,j]
        epsilon[t,:,:] /= rho[t+1]

    return gamma, L, epsilon
