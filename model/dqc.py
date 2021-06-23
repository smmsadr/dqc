
# coding: utf-8

# In[1]:

import numpy  as np
import pandas as pd


# In[6]:

class DQC():
    """
    This class is to perform dynamic quantum clustering.
    Version 0.1.2
    """
    
    def __init__(self,data):
        """
        This function is intialed data from dataset. The input should be pandas DataFrame.
        
        """
        self._data = data
        
        self.NUMBER_OF_SAMPLE = len(data)
        self.NUMBER_OF_FEATURE = len(data.iloc[0])
        
        DATA_MIN, DATA_MAX = data.min(),data.max()
        
        # Normalize data and convert it to numpy narray
        data_normalize = ((data - DATA_MIN)/(DATA_MAX- DATA_MIN)).values
        
        #SVD
        data_fit = np.linalg.svd(data_normalize,full_matrices=True)
        S = np.zeros((self.NUMBER_OF_SAMPLE,self.NUMBER_OF_FEATURE))
        S[:self.NUMBER_OF_FEATURE,:self.NUMBER_OF_FEATURE] = np.diag(data_fit[1])
        self.data_fit = np.matrix(np.dot(data_fit[0],S))

        
    def calculate_N(self, sigma):
        '''
        Computes the N matrix of <psi_j|psi_i> values with standard deviation sigma.
        '''
        self.SIGMA = sigma
        
        N = np.matrix(np.zeros((self.NUMBER_OF_SAMPLE, self.NUMBER_OF_SAMPLE)))
        for i in range(self.NUMBER_OF_SAMPLE):

            x_diff = self.data_fit[i] - self.data_fit
            N[i] = (x_diff * x_diff.T).diagonal()

        self.N = np.exp((-np.float64(self.SIGMA)**-2)*N/4)  
    
    def calculate_H(self, m):
    
        H = np.matrix(np.empty((self.NUMBER_OF_SAMPLE,self.NUMBER_OF_SAMPLE)))
        
        xtx = (self.data_fit*self.data_fit.T)
        k = -1/(2*m)*xtx * 1 / (crab.SIGMA ** 2) * np.exp(- xtx / (2 * crab.SIGMA ** 2 ))
        
        s_sq=np.float64(self.SIGMA)**2
        a = (2*s_sq)**-1

        vt = total_potential(data_array,self.SIGMA,m)
        
        
         
#         def p_sq_expectation(x,y,sigma):
            #return np.dot(x-y,x-y)/(2*sigma**2) * np.exp(-np.dot(x-y,x-y)/(4*sigma**2))
#             return np.divide(1,2*m,dtype=np.float64)*(euclidean(x,y)**2)* 0.5*a* np.exp(-0.5*a*(euclidean(x,y)**2))
        p_vec = np.vectorize(p_sq_expectation)
        def v_expectation(x,y,sigma):
            return np.exp(-0.5*a*(euclidean(x,y)**2)*vt(0.5*(x+y)))
        #v_vec = np.vectorize(v_expectation)
        for i in range(dim):
            for j in range(dim):
                x_i = data_array[i]
                x_j = data_array[j]
#                 p_term = p_sq_expectation(x_i,x_j,self.SIGMA)
                v_term = np.exp(-a*(euclidean(x_i,x_j)**2))*vt(0.5*(x_i+x_j))
              #  if p_term == np.nan or math.isnan(v_term) is True:
              #      print(x_i)
    #                print(x_j)
                term = v_term.astype(np.float64)
                H[i][j] = term
        self.H = H + k
#         return H

        
    def plot(self):
        """
        This function will plot the spot for the underlying.
        """
        get_ipython().magic('matplotlib inline')
        import matplotlib.pyplot as plt
        self.feed.plot(figsize=(13,10))
        
    def win_call(self, spot, duration):
        """
        This function are calculating the call option will be win or lose. The values will be store in "call" attribute.
        """
        
        timestamps = self.feed.index.tolist()
        for i in range(0, len(timestamps),duration):
            self.call.set_value(timestamps[i], spot+'_'+str(duration), win(timestamps[i], duration, self.feed, spot, 'call'))
        self.call.sort_index(inplace=True)
        
    def epoch_indexed(self):
        """
        This function will return feed data with epoch indexed format.
        """

        self.feed['epoch'] = self.feed.index.map(timestamp_2_epoch)

        return self.feed.set_index('epoch')


# In[ ]:

def total_potential(data_array,m):
    '''
    Calculates the DQC Schrodinger potential for a given point x and given initial value x_i. 

    From Horn and Gotlieb (2001), the potential should be:
    E - d/2 + 1/2(sigma)(psi)* sum((x-x_i)**2 * exp(-(x-x_i)**2/2(sigma)**2)). 
    The second and third terms are referred to as the 2nd and 3rd terms in the code.
    '''
    

    kts = total_kinetic_term(data_array,self.SIGMA,m)
    wfs = total_wavefunction(data_array,self.SIGMA)

    d=(np.shape(data_array[0]))[0]

    term_3 = lambda x: (wfs(x))**-1 * kts(x)

    #print(term_3(data_array[0]))
    data_sum = np.zeros(np.shape(data_array[0])) 
    for data in data_array:
        data_sum+=data
    #print(term_3(zero))

    E = d/2 #- term_3(data_sum)
    #print(E)
    #return lambda x:  wfs(x,data_array,sigma), lambda x: kts(x,data_array,sigma,m)
    return lambda x: E - d/2 + term_3(x)   


# In[3]:

def total_kinetic_term(data_array,sigma,m):
    '''
    Returns the the kinetic term function which evaluates the total kinetic term as a sum of all kinetic term components.
    '''

    kts = multi_kterm(data_array,sigma,m)
    def call(x):
        term = 0
        for ks in kts:
            term+=ks(x)
        return term
    return lambda x: call(x)
    #return lambda x: apply_vectorized(kfs, x)


# In[4]:

def multi_kterm(data_array,sigma,m):

    ks = []
    for i in range(len(data_array)):
        ks.append(kinetic_term(data_array[i],sigma,m))
    return ks


# In[5]:

def kinetic_term(x_i,sigma,m):
    '''
    Returns the kinetic term of the Schrodinger equation with a Gaussian wavefunction input.
    '''
    s_sq=np.float64(sigma)**2
    a = (2*s_sq)**-1
        
    return lambda x: np.divide(1,2*m,dtype=np.float64)*(4*a**2) *  (euclidean(x,x_i))**2 * np.exp(-a*(euclidean(x,x_i)**2))


# In[ ]:



