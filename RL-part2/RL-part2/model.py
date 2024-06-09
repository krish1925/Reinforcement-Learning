import torch as torch 
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    #a helper function to initialize the weights of the model
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    """Actor model for the DDPG algorithm.
    
    Layer 1: 400 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 2: 300 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 3: 1 unit, tanh activation, intialized with uniform weights in the range of -0.003 to 0.003
    
    """
    def __init__(self, input_size:tuple[int], action_size:int,CNN = None):
        """
        input: tuple[int]
            The input size, as a tuple of dimensions, for the DoubleInvertedPendulum environment, of shape (11,)
        action_size: int
            The number of actions
        """
        super(Actor, self).__init__()
        #TODO: Define the fully connected layers for the actor
        raise NotImplementedError
        
    def init_weights(self,init_w=3e-3):
        """

        Args:
            init_w (float, optional): the onesided range of the uniform distribution for the final layer. Defaults to 3e-3.

        """
        #initialize the weights of the model
        raise NotImplementedError
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        #TODO: 
        raise NotImplementedError

    
    
class Critic(nn.Module):
    """Critic model for the DDPG algorithm.
    Layer 1: 400 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in)
    Layer 2: 300 units, ReLU activation, Fan-in weight initialization, ie each weight is initialized with a uniform distribution in the range of -1/sqrt(fan_in) to 1/sqrt(fan_in). Input is the concatenation of the 400 dimension embedding from the state, and the action taken.
    Layer 3: 1 unit, intialized with uniform weights in the range of -0.003 to 0.003
    """
    def __init__(self,input_size:tuple[int],action_size:int):
        """
        input: tuple[int]
            The input size, as a tuple of dimensions, for the DoubleInvertedPendulum environment, of shape (11,)
        action_size: int
            The number of actions
        """
        super(Critic, self).__init__()
        #TODO: Define the fully connected layers for the critic and initialize the weights
        raise NotImplementedError
        
    def init_weights(self,init_w=3e-3):
        #initialize the weights of the model
        #TODO:
        raise NotImplementedError
        
    def forward(self, x:torch.Tensor, a:torch.Tensor)->torch.Tensor:
        #TODO:
        raise NotImplementedError