################################################################################
# Description: Module with the pre-processors for the deep learning models' inputs
# Author:      Sergio Iommi
################################################################################

import numpy as np
from game2048 import Game2048Env
from rl.core import Processor

class Log2NNInputProcessor(Processor):
    """
    Log2NNInputProcessor normalizes the neural network input (i.e., the observation/environment state represented
    with a single 2048 board-matrix) with the following equation (we assume the max achievable in the 2048 game is
    65536 as suggested at https://puzzling.stackexchange.com/questions/48/what-is-the-largest-tile-possible-in-2048):
       state = matrix S (elements s_ij, i,j=1,2,3,4)
       state_preprocessed = matrix P (elements p_ij, i,j=1,2,3,4)
       Matrix P is defined by:
           if s_ij != 0: p_ij = log2(s_ij)/log2(65536)
           if s_ij == 0: p_ij = log2(s_ij+1)/log2(65536) = 0
       In this way the matrix elements will be limited in the range [0,1] and we avoid training problems if we use
       the ReLU activation function in the neural networks.
       If we use instead the Sigmoid activation function we need another normalization scheme to avoid the neuron
       saturation issue (i.e., the case where the inputs activate the Sigmoid in the flat regions).
    
    Example:
        S: np.array([[ 8,  4,  8,  2],
                     [ 0,  0,  0,  4],
                     [ 0,  0,  0,  8],
                     [ 0,  0,  0, 32]])
        S_temp = np.where(A <= 0, 1, A)
        S_temp: array([[ 8,  4,  8,  2],
                       [ 1,  1,  1,  4],
                       [ 1,  1,  1,  8],
                       [ 1,  1,  1, 32]])
        P = np.log2(B)/np.log2(65536)
        P: array([[0.25      , 0.16666667, 0.25      , 0.08333333],
                  [0.        , 0.        , 0.        , 0.16666667],
                  [0.        , 0.        , 0.        , 0.25      ],
                  [0.        , 0.        , 0.        , 0.41666667]])
    """
    def process_observation(self, observation):
        """
        process_observation is the interface function called by Keras-RL to pre-process each observation of the environment state
        before passing it to the DQN/neural network agent.
        
        Args:
            observation: numpy.array representing the board-matrix of the game 2048 
        
        Returns:
            grid (numpy.array) representing the pre-processed board-matrix of the game where each element s!=0 of the matrix is
            normalized as log2(s)/log2(65536)
        """
        # We reshape the observation (i.e., the grid representing the board-matrix of the game 2048) to make sure we have a 4x4 numpy.array
        observation = np.reshape(observation, (4, 4))
        observation_temp = np.where(observation <= 0, 1, observation) # observation = S, observation_temp = S_temp
        processed_observation = np.log2(observation_temp)/np.log2(65536) # processed_observation = P
        return processed_observation
    
class OneHotNNInputProcessor(Processor):
    """
    OneHotNNInputProcessor is a pre-processor for the neural network input (i.e., the observation/environment state represented
    with a single 2048 board-matrix).
    It pre-processes an observation/grid by returning all the possible grids representing the board-matrices of the game in the next
    2 steps (4+16=20 grids). In particular it encodes each of these 20 grids with a one-hot encoding method, that is it represents each
    grid with a number of matrices made of 1s and 0s equal to num_one_hot_matrices (parameter passed to the constructor).
    
    Example of one hot encoding for a single grid (in our case we will have 20 grids to encode in this way):
    S: np.array([[ 2,      2,      0,      4],
                 [ 0,      0,      8,      0],
                 [ 0,      16,     0,      0],
                 [ 65536,  0,      0,      0]])
    S_onehot: np.array([[[ 0,      0,      1,      0], # Matrix 0: 1s for grid elements s=0, 0s otherwise 
                         [ 1,      1,      0,      1],
                         [ 1,      0,      1,      1],
                         [ 0,      1,      1,      1]],
                         
                         [ 1,      1,      0,      0], # Matrix 1: 1s for grid elements s=2^1=2, 0s otherwise
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0]],
                         
                         [ 0,      0,      0,      1], # Matrix 2: 1s for grid elements s=2^2=4, 0s otherwise
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0]],
                         
                         [ 0,      0,      0,      0], # Matrix 3: 1s for grid elements s=2^3=8, 0s otherwise
                         [ 0,      0,      1,      0],
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0]],
                         
                         [ 0,      0,      0,      0], # Matrix 4: 1s for grid elements s=2^4=16, 0s otherwise
                         [ 0,      0,      0,      0],
                         [ 0,      1,      0,      0],
                         [ 0,      0,      0,      0]],
                         
                         ...
                         
                         [ 0,      0,      0,      0], # Matrix 16: 1s for grid elements s=2^16=65536, 0s otherwise
                         [ 0,      0,      0,      0],
                         [ 0,      0,      0,      0],
                         [ 1,      0,      0,      0]]])
    """
    
    def __init__(self, num_one_hot_matrices=16, window_length=1, model="dnn"):
        """
        Check description of OneHotNNInputProcessor class
        
        Args:
             num_one_hot_matrices: number of matrices to use to encode (via one-hot encoding) each game grid.
                 Assuming that the max achievable in the 2048 game is 65536 (as suggested at https://puzzling.stackexchange.com/questions/48/what-is-the-largest-tile-possible-in-2048)
                 this number should normally be 16.
        """
        self.num_one_hot_matrices = num_one_hot_matrices
        self.window_length = window_length
        self.model = model
        
        self.game_env = Game2048Env() # we make an instance of the game environment so that we can use the functions implementing the game logic
        
        # Variables used by one_hot_encoding() function:
        self.table = {2**i:i for i in range(1,self.num_one_hot_matrices)} # dictionary storing powers of 2: {2: 1, 4: 2, 8: 3, ..., 16384: 14, 32768: 15, 65536: 16}
        self.table[0] = 0 # Add element {0: 0} to the dictionary
    
    def one_hot_encoding(self, grid):
        """
        one_hot_encoding receives a grid representing the board-matrix of the game 2048 and returns a one-hot encoding
        representation.
        Check the description of the OneHotNNInputProcessor class to see an example of such encoding.
        
        Args:
            grid: 4x4 numpy.array representing the board-matrix of the game 2048
            
        Returns:
            numpy.array containing a number equal to num_one_hot_matrices of 4x4 numpy.arrays
        """
        # grid_onehot: contains a number (normally 16) equal to num_one_hot_matrices matrices 4x4 of 0s and 1s where each matrix
        #     stores information on a single power of 2
        #
        #     matrix in position 0 stores information for elements 0s in the matrix-tiles-grid
        #     matrix in position 1 stores information for elements 2^1=2 in the matrix-tiles-grid
        #     matrix in position 2 stores information for elements 2^2=4 in the matrix-tiles-grid
        #     ...
        #     In particular each matrix stores the information by using 0s and 1s (1s to indicate the presence of elements, 0s for the absence)
        grid_onehot = np.zeros(shape=(self.num_one_hot_matrices, 4, 4))
        for i in range(4):
            for j in range(4):
                grid_element = grid[i, j]
                grid_onehot[self.table[grid_element],i, j]=1
        return grid_onehot

    def get_grids_next_step(self, grid):
        """
        get_grids_next_step receives a grid representing the board-matrix of the game 2048 and returns
        a list of grids representing the 4 possible grids at the next step, one for each possible movement.
        
        Args:
            grid: 4x4 numpy.array representing the board-matrix of the game 2048 
        
        Returns:
            list of 4 numpy.arrays representing the 4 possible grids at the next step, one for each possible
            movement.
        """
        grids_list = [] # list storing the 4 possible grids at the next step, one for each possible movement
        for movement in range(4): # the possible movements (up/left/down/right) are represented with a number (0,1,2,3)
            grid_before = grid.copy()
            self.game_env.set_board(grid_before)
            try:
                # move() raises an exception in case of an illegal movement hence we capture it so we can append an unchanged
                # grid to grids_list.
                # It is necessary to have a grid for each of the possible movements (even the illegal ones) because with
                # Tensorflow (used as a backed of Keras) we cannot pass a variable number (e.g., via None) in the input_shape
                # (differently from Theano) hence we cannot select and pass as input to the neural networks only the grids originating
                # from legal moves.
                _ = self.game_env.move(movement) # move() returns a score which is useless in this case
            except:
                pass
            grid_after = self.game_env.get_board()
            grids_list.append(grid_after)
        return grids_list

    def process_observation(self, observation):
        """
        process_observation is the interface function called by Keras-RL to pre-process each observation of the environment state
        before passing it to the DQN/neural network agent. 
        
        Args:
            observation: numpy.array representing the board-matrix of the game 2048 
        
        Returns:
            list of numpy.arrays with all the possible grids representing the board-matrices of the game in the next 2 steps, with
            each grid encoded with a one-hot encoding method.
        """
        # We reshape the observation (i.e., the grid representing the board-matrix of the game 2048) to make sure we have a 4x4 numpy.array
        observation = np.reshape(observation, (4, 4))
        
        # POSSIBLE IMPROVEMENT:
        # This implementation contemplates only the case where we return all and only the grids representing the board-matrices of the game
        # in the next 2 steps.
        # It would be interesting to implement a more general framework where it is possible to specify the number of future steps of the game
        # for which we want to calculate the board-matrices. In that case we could compare the performance of the DQN/neural network agent in
        # solving the game using a different number of future step grids provided as an input.
        
        grids_list_step1 = self.get_grids_next_step(observation)
        grids_list_step2 =[]
        for grid in grids_list_step1:
            grids_list_step2.append(grid) # In the NN input I give both, the 1-step and 2-step grids
            grids_temp = self.get_grids_next_step(grid)
            for grid_temp in grids_temp:
                grids_list_step2.append(grid_temp)
        grids_list = np.array([self.one_hot_encoding(grid) for grid in grids_list_step2])
        
        return grids_list
    
    def process_state_batch(self, batch):
        """
        process_state_batch processes an entire batch of states and returns it.
        It is required to reshape the NN input in case we want to use a CNN model with the one-hot encoding.
        The implementation contemplates only the case where we look at the grids of the 2 next steps (for a total
        of 4+4*4=20 grids).
        Check the comments in dqn2048.py regarding the input shape of the CNNs.

        Args:
            batch (list): List of states
        
        Returns:
            Processed list of states
        """
        if self.model == "cnn": # batch pre-processing only required for the cnn models
            try:
                batch = np.reshape(batch, (self.window_length, self.window_length*(4+4*4)*self.num_one_hot_matrices, 4, 4))
            except:
                batch = np.reshape(batch, (np.shape(batch)[0], self.window_length*(4+4*4)*self.num_one_hot_matrices, 4, 4))
                pass
        return batch