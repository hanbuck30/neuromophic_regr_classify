import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
import torch
import sklearn
from sklearn import linear_model

class ESN():
    def __init__(self, n_readout, 
                 resSize, damping=0.5,spectral_radius=0.1,
                 weight_scaling=1.25,initLen=0, random_state=42):
        
        self.resSize=resSize
        self.n_readout=n_readout
        self.damping = damping
        self.spectral_radius=spectral_radius
        self.weight_scaling=weight_scaling
        self.initLen=initLen
        self.random_state=random_state
        torch.manual_seed(random_state)
        self.initmodel()
        
    def init_fit(self,input):
        W = torch.rand(self.resSize,self.resSize, dtype=torch.double) - 0.5
        self.Win = (torch.rand(self.resSize,1+self.inSize, dtype=torch.double) - 0.5) * 1
        print('Computing spectral radius...')
        #spectral_radius = max(abs(linalg.eig(W)[0]))
        print('done.')
        self.W= W*(self.weight_scaling/self.spectral_radius)
        
        n_input, n_feature = input.shape
        X = torch.zeros((1+self.inSize+self.resSize,n_input)).type(torch.double)
        U = torch.cat(torch.ones([n_input,1]),torch.DoubleTensor(input) ,axis=1)
        x = torch.zeros((self.resSize,1)).type(torch.double)
        for t in range(n_input):
            u = U[t,:].T
            x = (1-self.damping)*x + self.damping*torch.tanh(torch.matmul( self.Win, u + torch.matmul( self.W, x )))
            #if t >= 100:
            X[:,t] = torch.vstack([torch.Tensor([1]),u,x])[:,0]
        
        self.X=X
       
        #### train the output by ridge regression
        #reg = 1e-8  # regularization coefficient
        #### direct equations from texts:
        #X_T = X.T
        #Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
        #    reg*np.eye(1+inSize+resSize) ) )
        # using scipy.linalg.solve:
        '''
        reg = 1e-8
        Wout = linalg.solve(torch.matmul(X,X.T) + reg*torch.eye(1+self.inSize+self.resSize), torch.matmul(X,Yt.T)).T
        Wout=np.array(Wout)
        Wout=torch.DoubleTensor(Wout)
        self.Wout=torch.DoubleTensor(Wout)
        outputs=np.array(outputs[len(inputs)-1]).reshape(1,-1)
        self.u = torch.DoubleTensor(outputs)
        '''
        return self
    def fit(self,input):
        output=self.init_fit(input)
        return output
    
    def predict(self,inputs):    
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        Y = torch.zeros((self.outSize,len(inputs)))
        u=self.u
        x=self.x

        for t in range(len(inputs)):
            
            x = (1-self.a)*x + self.a*torch.tanh( torch.matmul( self.Win, torch.vstack([torch.DoubleTensor([1]),u]) ) + torch.matmul( self.W, x ) )
            y = torch.matmul( self.Wout, torch.vstack([torch.DoubleTensor([1]),u,x])) 

            Y[:,t] = y
            # generative mode:
            u = y
        self.Y=Y
        return self.Y

