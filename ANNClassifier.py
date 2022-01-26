import numpy as np
class ANN(object):
    
    '''
    Initialize the ANN;
    HiddenLayer vector : will contain the Layers' info
    w, b, phi = (empty) arrays that will contain all the w, b and activation functions for all the Layers
    mu = cost function
    eta = a standard learning rate initialization. It can be modified by the 'set_learning_rate' method
    '''
    def __init__(self) :
        np.random.seed(1000000)
        self.HiddenLayer = []
        self.w = []
        self.b = []
        self.phi = []
        self.mu = []

    
    '''
    add method: to add layers to the network
    '''
    def add(self, lay = (4, 'ReLU') ):
        self.HiddenLayer.append(lay)
    
    '''
    FeedForward method: as explained before. 
    '''
    @staticmethod
    def FeedForward(w, b, phi, x):
        return phi(np.dot(w, x) + b)
        
    '''
    BackPropagation algorithm implementing the Gradient Descent 
    '''
    def BackPropagation(self, x, z, Y, w, b, phi, eta):
        self.delta = []
        
        # We initialize ausiliar w and b that are used only inside the backpropagation algorithm once called        
        self.W = []
        self.B = []
        
        # We start computing the LAST error, the one for the OutPut Layer 
        self.delta.append(  (z[len(z)-1] - Y) * phi[len(z)-1](z[len(z)-1], der=True) )
        
        '''Now we BACKpropagate'''
        # We thus compute from next-to-last to first
        for i in range(0, len(z)-1):
            self.delta.append( np.dot( self.delta[i], w[len(z)- 1 - i] ) * phi[len(z)- 2 - i](z[len(z)- 2 - i], der=True) )
        
        # We have the error array ordered from last to first; we flip it to order it from first to last
        self.delta = np.flip(self.delta, 0)  
        
        # Now we define the delta as the error divided by the number of training samples
        self.delta = self.delta/self.X.shape[0] 
        
        '''GRADIENT DESCENT'''
        # We start from the first layer that is special, since it is connected to the Input Layer
        self.W.append( w[0] - eta * np.kron(self.delta[0], x).reshape( len(z[0]), x.shape[0] ) )
        self.B.append( b[0] - eta * self.delta[0] )
        
        # We now descend for all the other Hidden Layers + OutPut Layer
        for i in range(1, len(z)):
            self.W.append( w[i] - eta * np.kron(self.delta[i], z[i-1]).reshape(len(z[i]), len(z[i-1])) )
            self.B.append( b[i] - eta * self.delta[i] )
        
        # We return the descended parameters w, b
        return np.array(self.W), np.array(self.B)
    
    
    '''
    Fit method: it calls FeedForward and Backpropagation methods
    '''
    def fit_once(self, X_train, Y_train, eta,c):            
#         print('Start fitting...')
        '''
        Input layer
        '''
        self.X = X_train
        self.Y = Y_train  
        
        '''
        We now initialize the Network by retrieving the Hidden Layers and concatenating them 
        ''' 
        if c ==0: #For the first time 
            for i in range(0, len(self.HiddenLayer)) :
                if i==0:
                    # We now try to use the He et al. Initialization from ArXiv:1502.01852
                    self.w.append( np.random.randn(self.HiddenLayer[i][0] , self.X.shape[1])/np.sqrt(2/self.X.shape[1]) )
                    self.b.append( np.random.randn(self.HiddenLayer[i][0])/np.sqrt(2/self.X.shape[1]))
                    # Old initialization
                    #self.w.append(2 * np.random.rand(self.HiddenLayer[i][0] , self.X.shape[1]) - 0.5)
                    #self.b.append(np.random.rand(self.HiddenLayer[i][0]))

                    # Initialize the Activation function
                    for act in Activation_function.list_act():
                        if self.HiddenLayer[i][1] == act :
                            self.phi.append(Activation_function.get_act(act))
    #                         print('\tActivation: ', act)

                else :
                    # We now try to use the He et al. Initialization from ArXiv:1502.01852
                    self.w.append( np.random.randn(self.HiddenLayer[i][0] , self.HiddenLayer[i-1][0] )/np.sqrt(2/self.HiddenLayer[i-1][0]))
                    self.b.append( np.random.randn(self.HiddenLayer[i][0])/np.sqrt(2/self.HiddenLayer[i-1][0]))
                    # Old initialization
                    #self.w.append(2*np.random.rand(self.HiddenLayer[i][0] , self.HiddenLayer[i-1][0] ) - 0.5)
                    #self.b.append(np.random.rand(self.HiddenLayer[i][0]))

                    # Initialize the Activation function
                    for act in Activation_function.list_act():
                        if self.HiddenLayer[i][1] == act :
                            self.phi.append(Activation_function.get_act(act))
         
        '''
        Now we start the Loop over the training dataset
        '''  
        for I in range(0, self.X.shape[0]): # loop over the training set
            '''
            Now we start the feed forward
            '''  
            self.z = []
            self.z.append( self.FeedForward(self.w[0], self.b[0], self.phi[0], self.X[I]) ) # First layers
            
            for i in range(1, len(self.HiddenLayer)): #Looping over layers
                self.z.append( self.FeedForward(self.w[i] , self.b[i], self.phi[i], self.z[i-1] ) )
        
            
            '''
            Here we backpropagate
            '''      
            self.w, self.b  = self.BackPropagation(self.X[I], self.z, self.Y, self.w, self.b, self.phi, eta)
            self.w = list(self.w)
            self.b = list(self.b)         
            
            '''
            Compute cost function
            ''' 
            self.mu.append(
                (1/2) * np.dot(self.z[len(self.z)-1] - self.Y, self.z[len(self.z)-1] - self.Y) 
            )        
            
        return self 
        

    
    '''
    predict method
    '''
    def predict_once(self, X_test, c):
        
        self.pred = []
        self.XX = X_test
        if c ==0:
            val =np.array([0])
        else:   
        
            for I in range(0, self.XX.shape[0]): # loop over the training set

                '''
                Now we start the feed forward
                '''  
                self.z = []

                self.z.append(self.FeedForward(self.w[0] , self.b[0], self.phi[0], self.XX[I])) #First layer

                for i in range(1, len(self.HiddenLayer)) : # loop over the layers
                    self.z.append( self.FeedForward(self.w[i] , self.b[i], self.phi[i], self.z[i-1]))

                # Append the prediction;
                # We now need a binary classifier; we this apply an Heaviside Theta and we set to 0.5 the threshold
                # if y < 0.5 the output is zero, otherwise is 1
                self.pred.append( np.heaviside(  self.z[-1] - 0.5, 1)[0] ) # NB: self.z[-1]  is the last element of the self.z list

            val = np.array(self.pred)

        return val




# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    
    if (der==True) : #derivative of the sigmoid
        f = 1/(1+ np.exp(- x))*(1-1/(1+ np.exp(- x)))
    else : # sigmoid
        f = 1/(1+ np.exp(- x))
    
    return f

# We may employ the Rectifier Linear Unit (ReLU)
def ReLU_act(x, der=False):
    
    if (der == True): # the derivative of the ReLU is the Heaviside Theta
        f = np.heaviside(x, 1)
    else :
        f = np.maximum(x, 0)
    
    return f

# Define the tanh activator; we ask if we want the sigmoid or its derivative
def tanh_act(x, der=False):
    
    if (der==True) : #derivative of the tanh
        f = 1 - pow(((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))), 2)
    else : # sigmoid
        f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    return f


'''
layers class
'''
class layers():
    '''
    Layer method: used to call standar layers to add. 
    Easily generalizable to more general layers (Pooling and Convolutional layers)
    '''        
    def layer(p=4, activation = 'ReLU'):
        return (p, activation)

'''
Activation functions class
'''
class Activation_function(ANN):

    def __init__(self) :
        super().__init__()

    def list_act():
        return ['sigmoid', 'ReLU', 'tanh']

    def get_act(string = 'ReLU'):
        if string == 'ReLU':
            return ReLU_act
        
        elif string == 'sigmoid':
            return sigmoid_act

        elif string == 'tanh':
            return tanh_act

        else :
            return sigmoid_act