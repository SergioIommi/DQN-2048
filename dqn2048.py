################################################################################
# Description: Main script to train and/or test the deep Q-network (DQN) with
#              the definitions of the deep learning models 
# Author:      Sergio Iommi
################################################################################

import os, fnmatch, pickle

import numpy as np
import random

from game2048 import Game2048Env # module with game logic for 2048

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.layers.merge import concatenate
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from callbacks2048 import TrainEpisodeLogger2048, TestLogger2048
from processors2048 import Log2NNInputProcessor, OneHotNNInputProcessor

######################################################################
# SYSTEM PARAMETERS:

# Set the filesystem path containing the project script/files:
prj_path = '/Users/sergioiommi/Workspace/Eclipse/DQN_2048'
data_filepath = prj_path + '/data' # set the path for the folder where to store the data (log files, CSVs, neural network trained weights, etc.) 
if not os.path.exists(data_filepath): # check if the folder exists otherwise create it
    os.makedirs(data_filepath)

# Decide to either train or test the Deep Q-Network:
TRAIN_TEST_MODE = 'train'
#TRAIN_TEST_MODE = 'test'

######################################################################
# MODEL HYPERPARMETERS:

# Set the DQN/Neural Network type and some hyperparameters (further ones can be specified later):

# DQN/Neural Network type:
MODEL_TYPE = 'dnn' # 3-layer dnn (deep neural network / multi-layer perceptron / feed-forward neural network / dense neural network)
#MODEL_TYPE = 'cnn1' # convolutional neural network (vanilla)
#MODEL_TYPE = 'cnn2' # convolutional neural network (with "orthogonal" strides)

# Input/output layers parameters:
NUM_ACTIONS_OUTPUT_NN = 4 # number of agent's actions (corresponds to the number of neurons for the neural network's output layer)
# WINDOW_LENGTH:
#    - number of historical states (observations) from the environment to provide to the neural network/DQN agent
#    - the process, describing the evolution of the game 2048, is fully observable because each board-matrix fully describes the state
#      of the process at each step, hence there is no need for WINDOW_LENGTH > 1. For the same reason there is no need to use a 
#      recurrent neural network (RNN), in place of a deep neural network or a convolutional neural network, because the state of the RNN
#      would probably memorize useless information.
#    - ATTENTION: the code has not been tested for values of WINDOW_LENGTH different from 1
WINDOW_LENGTH = 1
INPUT_SHAPE = (4, 4) # game-grid/matrix size (corresponds to the input shape for the neural network)

# Decide the pre-processing method for the neural network inputs:
#PREPROC="log2" # Log2 Normalization; explained in detail in module "processors2048.py"
PREPROC="onehot2steps" # One-Hot Encoding; explained in detail in module "processors2048.py"
NUM_ONE_HOT_MAT = 16 # number of matrices to use for encoding each game-grid in the one-hot encoding pre-processing

# Set the training hyperparameters:
#BATCH_SIZE = 256 # neural network's batch size (number of examples in each batch) 
NB_STEPS_TRAINING = int(5e6) # number of steps used for training the model
NB_STEPS_ANNEALED = int(1e5) # number of steps used in LinearAnnealedPolicy()
NB_STEPS_WARMUP = 5000 # number of steps to fill memory before training
MEMORY_SIZE = 6000 # used in SequentialMemory()
TARGET_MODEL_UPDATE = 1000 # used in DQNAgent(); https://github.com/keras-rl/keras-rl/issues/55

######################################################################
# ENVIRONMENT:

# Create the environment for the DQN agent:
ENV_NAME = '2048'
env = Game2048Env()

# Set a specific seed for the pseudo-random number generators to obtain reproducible results:
random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
env.seed(random_seed)

######################################################################
# NEURAL NETWORK MODELS:

# Load Keras model (if already trained/partially trained and saved) otherwise build a new one
# ISSUE TO SOLVE - SAVING MODEL'S OPTIMIZER/TRAINING STATE:
#    - the neural network model is defined using Keras but it is built and trained using Keras-RL library (DQNAgent() and fit() functions)
#      hence we cannot use the Keras function save() (https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) to save the 
#      model's optimizer/training state (e.g., state of the learning rate in the Adam optimization method) in case we want to resume the 
#      training and keep improving the model.
#      We can still save the model architecture using Keras function save() and the trained weights using Keras-RL callback function
#      ModelIntervalCheckpoint() but these, on their own, are useful only for the testing phase.
#    - Attempt to save the model's optimizer state & pickle error:
#      I have tried to serialize the object (built with DQNAgent()) storing the model's architecture and optimizer state using the pickle
#      library but I encountered the error "TypeError: can't pickle _thread.RLock objects" which seems caused by the fact that Python
#      cannot pickle lambda expressions like the ones used in Tensorflow (https://github.com/keras-team/keras/issues/8343,
#      https://stackoverflow.com/questions/44855603/typeerror-cant-pickle-thread-lock-objects-in-seq2seq). 
#      Solving this issue will probably require to modify the implementation of DQNAgent().
try:
    print("Loading Keras model to resume the training")
    # model_filename example: dqn_2048_dnn_onehot2steps_5000000_model.h5
    #    5000000 is the number of training steps already executed (stored in nb_training_steps_pickle with the next statement)
    model_filename = fnmatch.filter(os.listdir(data_filepath), 'dqn_{}_{}_{}_'.format(ENV_NAME, MODEL_TYPE, PREPROC) + '*_model.h5')[-1]    
    nb_training_steps_pickle = int(model_filename.split("_")[4]) # nb_training_steps_pickle: stores the number of training steps from previously trained Keras model
    NB_STEPS_TRAINING = NB_STEPS_TRAINING - nb_training_steps_pickle # We adjust the number of training steps considering that we have already trained the model with a number of training steps equal to nb_training_steps_pickle 
    model_filepath = data_filepath + '/' + model_filename
    model = load_model(model_filepath)
except:
    print("Define Keras model for training/testing")
    # We could have enclosed the 3 types of NN (dnn, cnn1, cnn2) in a separated Python module but we have preferred to keep them in the main
    # script so that the modification of the hyperparameters is more agile.
    if MODEL_TYPE=='dnn':
        # Choose the pre-processing method and consequently the input shape for the NN
        if PREPROC=="onehot2steps": # Data Pre-Processing
            processor = OneHotNNInputProcessor(num_one_hot_matrices=NUM_ONE_HOT_MAT)
            INPUT_SHAPE_DNN = (WINDOW_LENGTH, 4+4*4, NUM_ONE_HOT_MAT,) + INPUT_SHAPE # Ex. (1,20,16,4,4); 4+4*4 is the total number of grids in the next 2 steps (4 for the 1st step, 4*4 for the 2nd step)
        elif PREPROC=="log2":
            processor = Log2NNInputProcessor()
            INPUT_SHAPE_DNN = (WINDOW_LENGTH*1,) + INPUT_SHAPE # Ex. (1,4,4); in WINDOW_LENGTH*1 the *1 is to underline that we give only 1 grid to the NN input 
        
        # Set the DNN hyperparameters:
        NUM_DENSE_NEURONS_DNN_L1 = 1024 # number of neurons in 1st layer 
        NUM_DENSE_NEURONS_DNN_L2 = 512  # number of neurons in 2nd layer
        NUM_DENSE_NEURONS_DNN_L3 = 256  # number of neurons in 3rd layer
        ACTIVATION_FTN_DNN = 'relu'
        ACTIVATION_FTN_DNN_OUTPUT = 'linear'
        
        # DNN model definition:
        model = Sequential()
        model.add(Flatten(input_shape=INPUT_SHAPE_DNN))
        model.add(Dense(units=NUM_DENSE_NEURONS_DNN_L1, activation=ACTIVATION_FTN_DNN))#, batch_size=BATCH_SIZE))
        model.add(Dense(units=NUM_DENSE_NEURONS_DNN_L2, activation=ACTIVATION_FTN_DNN))
        model.add(Dense(units=NUM_DENSE_NEURONS_DNN_L3, activation=ACTIVATION_FTN_DNN))
        model.add(Dense(units=NUM_ACTIONS_OUTPUT_NN, activation=ACTIVATION_FTN_DNN_OUTPUT))
        print(model.summary())
    elif MODEL_TYPE=='cnn1':
        # Choose the pre-processing method and consequently the input shape for the NN
        if PREPROC=="onehot2steps": # Data Pre-Processing
            processor = OneHotNNInputProcessor(num_one_hot_matrices=NUM_ONE_HOT_MAT, window_length=WINDOW_LENGTH, model="cnn")
            # CNN input_shape using the one-hot encoding:
            #    - the 2D CNN input shape is always (channel size, dim1, dim2)
            #    - (dim1, dim2) = (4, 4) size of the board-matrix
            #    - (channel size) in the one-hot encoding is the number of all the matrices (made of 1s and 0s) that we need to give
            #      as an input to the CNN, i.e.:
            #        WINDOW_LENGTH*(4+4*4)*NUM_ONE_HOT_MAT = 1*(4+4*4)*16 = 320
            #        where (4+4*4)=20 is the number of grids of the next 2 steps (4 for the 1st step + 4*4 for the 2nd step)
            INPUT_SHAPE_CNN = (WINDOW_LENGTH*(4+4*4)*NUM_ONE_HOT_MAT,) + INPUT_SHAPE # Ex. (320,4,4)
        elif PREPROC=="log2":
            processor = Log2NNInputProcessor()
            INPUT_SHAPE_CNN = (WINDOW_LENGTH*1,) + INPUT_SHAPE # in WINDOW_LENGTH*1 the *1 is to underline that we give only 1 grid to the NN input 
        
        # Tell to Keras the input_shape ordering used to pass the data to the CNN input.
        #    This step is necessary to avoid any conflict with the backend (Tensorflow or Theano), given that each
        #    one uses its own standard.
        #    Theano is 'channels_first':    (channel, n, m)
        #    Tensorflow is 'channels_last': (n, m, channels)
        K.set_image_dim_ordering('th') # th: theano, tf: tensorflow
        # Otherwise we can use the data_format parameter to be passed to the Conv2D() functions:
        #    data_format = 'channels_first'
        #    data_format = 'channels_last'     
        
        # Set the CNN hyperparameters:
        # CNN Layers:
        #NUM_FILTERS_LAYER_1 = 512
        #NUM_FILTERS_LAYER_2 = 4096
        NUM_FILTERS_LAYER_1 = 32 # number of filters in 1st layer
        NUM_FILTERS_LAYER_2 = 64 # number of filters in 2nd layer
        FILTERS_SIZE_LAYER_1 = 3 # Filter Size = 3 x 3
        FILTERS_SIZE_LAYER_2 = 1 # Filter Size = 1 x 1
        STRIDES_LAYER_1 = (2, 2)
        STRIDES_LAYER_2 = (1, 1)
        ACTIVATION_FTN_CNN = 'relu'
        # Dense Layers:
        NUM_DENSE_NEURONS = 512
        ACTIVATION_FTN_DENSE = 'relu'
        ACTIVATION_FTN_OUTPUT = 'linear'
        
        # CNN model definition:
        model = Sequential()
        model.add(Conv2D(filters=NUM_FILTERS_LAYER_1,
                         kernel_size=FILTERS_SIZE_LAYER_1,
                         strides=STRIDES_LAYER_1,
                         padding='valid',
                         activation=ACTIVATION_FTN_CNN,
                         input_shape=INPUT_SHAPE_CNN)) #, data_format=data_format)) 
        model.add(Conv2D(filters=NUM_FILTERS_LAYER_2,
                         kernel_size=FILTERS_SIZE_LAYER_2,
                         strides=STRIDES_LAYER_2,
                         padding='valid',
                         activation=ACTIVATION_FTN_CNN,
                         input_shape=INPUT_SHAPE_CNN)) #, data_format=data_format))
        model.add(Flatten())
        model.add(Dense(units=NUM_DENSE_NEURONS, activation=ACTIVATION_FTN_DENSE))
        model.add(Dense(units=NUM_ACTIONS_OUTPUT_NN, activation=ACTIVATION_FTN_OUTPUT))
        print(model.summary())
    elif MODEL_TYPE == 'cnn2': 
        if PREPROC=="onehot2steps": # Data Pre-Processing
            processor = OneHotNNInputProcessor(num_one_hot_matrices=NUM_ONE_HOT_MAT, window_length=WINDOW_LENGTH, model="cnn")
            # CNN input_shape using the one-hot encoding:
            #    - the 2D CNN input shape is always (channel size, dim1, dim2)
            #    - (dim1, dim2) = (4, 4) size of the board-matrix
            #    - (channel size) in the one-hot encoding is the number of all the matrices (made of 1s and 0s) that we need to give
            #      as an input to the CNN, i.e.:
            #        WINDOW_LENGTH*(4+4*4)*NUM_ONE_HOT_MAT = 1*(4+4*4)*16 = 320
            #        where (4+4*4)=20 is the number of grids of the next 2 steps (4 for the 1st step + 4*4 for the 2nd step)
            INPUT_SHAPE_CNN = (WINDOW_LENGTH*(4+4*4)*NUM_ONE_HOT_MAT,) + INPUT_SHAPE # Ex. (320,4,4)
        elif PREPROC=="log2":
            processor = Log2NNInputProcessor()
            INPUT_SHAPE_CNN = (WINDOW_LENGTH*1,) + INPUT_SHAPE # in WINDOW_LENGTH*1 the *1 is to underline that we give only 1 grid to the NN input
        
        # Tell to Keras the input_shape ordering used to pass the data to the CNN input.
        #    This step is necessary to avoid any conflict with the backend (Tensorflow or Theano), given that each
        #    one uses its own standard.
        #    Theano is 'channels_first':    (channel, n, m)
        #    Tensorflow is 'channels_last': (n, m, channels)
        K.set_image_dim_ordering('th') # th: theano, tf: tensorflow
        # Otherwise we can use the data_format parameter to be passed to the Conv2D() functions:
        #    data_format = 'channels_first'
        #    data_format = 'channels_last'
        
        # Set the CNN hyperparameters:
        # CNN Layers:
        NUM_FILTERS_LAYER_1 = 512   # number of filters in 1st layer
        NUM_FILTERS_LAYER_2 = 4096  # number of filters in 2nd layer
        # NUM_FILTERS_LAYER_1 = 32
        # NUM_FILTERS_LAYER_2 = 64
        FILTERS_SIZE_LAYER_1 = 3 # Filter Size = 3 x 3
        FILTERS_SIZE_LAYER_2 = 1 # Filter Size = 1 x 1
        ACTIVATION_FTN_CNN = 'relu'
        # Dense/Output Layers:
        NUM_DENSE_NEURONS = 512
        ACTIVATION_FTN_DENSE = 'relu'
        ACTIVATION_FTN_OUTPUT = 'linear'
          
        # CNN model definition:
        #    We use the Functional API of Keras (https://keras.io/getting-started/functional-api-guide/) 
        _input = Input(shape=INPUT_SHAPE_CNN)
        conv_a = Conv2D(filters=NUM_FILTERS_LAYER_1, kernel_size=FILTERS_SIZE_LAYER_1, strides=(2,1), padding='valid', activation=ACTIVATION_FTN_CNN)(_input)
        conv_b = Conv2D(filters=NUM_FILTERS_LAYER_1, kernel_size=FILTERS_SIZE_LAYER_1, strides=(1,2), padding='valid', activation=ACTIVATION_FTN_CNN)(_input)
        conv_aa = Conv2D(filters=NUM_FILTERS_LAYER_2, kernel_size=FILTERS_SIZE_LAYER_2, strides=(2,1), padding='valid', activation=ACTIVATION_FTN_CNN)(conv_a)
        conv_ab = Conv2D(filters=NUM_FILTERS_LAYER_2, kernel_size=FILTERS_SIZE_LAYER_2, strides=(1,2), padding='valid', activation=ACTIVATION_FTN_CNN)(conv_a)
        conv_ba = Conv2D(filters=NUM_FILTERS_LAYER_2, kernel_size=FILTERS_SIZE_LAYER_2, strides=(2,1), padding='valid', activation=ACTIVATION_FTN_CNN)(conv_b)
        conv_bb = Conv2D(filters=NUM_FILTERS_LAYER_2, kernel_size=FILTERS_SIZE_LAYER_2, strides=(1,2), padding='valid', activation=ACTIVATION_FTN_CNN)(conv_b)
        merge = concatenate([Flatten()(x) for x in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]])
        _output = Dense(units=NUM_ACTIONS_OUTPUT_NN, activation='linear')(merge)
        model = Model(inputs=_input, outputs=_output)
        print(model.summary())

######################################################################
# Q-LEARNING (NN-Agent training via fitting of Q-learning equation)

# Training Memory:
#    SequentialMemory: data structure used to store the agents' state observations for training.
#    In the following block we check if we can load the training memory (SequentialMemory) from previously stored training.
#    This is useful in case we want to train the DQN agent in multiple occasions. Otherwise initialize SequentialMemory.
#    Please, solve the issue previously mentioned: "ISSUE TO SOLVE - SAVING MODEL'S OPTIMIZER/TRAINING STATE" 
try:
    # Load Keras-RL agent training memory
    print("Loading Keras-RL agent training memory")
    # agentmem_filename example: dqn_2048_dnn_onehot2steps_5000000_agentmem.pkl
    #    5000000 is the number of training steps already executed (stored in nb_training_steps_pickle with the next statement)
    agentmem_filename = fnmatch.filter(os.listdir(data_filepath), 'dqn_{}_{}_{}_'.format(ENV_NAME, MODEL_TYPE, PREPROC) + '*_agentmem.pkl')[-1]
    nb_training_steps_pickle = int(agentmem_filename.split("_")[4]) # nb_training_steps_pickle: stores the number of training steps from previously trained Keras model
    NB_STEPS_TRAINING = NB_STEPS_TRAINING - nb_training_steps_pickle # We adjust the number of training steps considering that we have already trained the model with a number of training steps equal to nb_training_steps_pickle 
    pickle_filepath = data_filepath + '/' + agentmem_filename 
    (memory, memory.actions, memory.rewards, memory.terminals, memory.observations) = pickle.load( open(pickle_filepath, "rb")) # https://github.com/keras-rl/keras-rl/issues/186#issuecomment-385200010
except:
    memory = SequentialMemory(limit=MEMORY_SIZE, window_length=WINDOW_LENGTH)

# Policy:
# train_policy vs test_policy (issues): https://github.com/keras-rl/keras-rl/issues/18
# 2048 Game: Policy & Illegal Moves
#    With GreedyQPolicy() there are too many illegal moves (commands on the game-grid that are not available) and
#    the game/episode never ends (both in training and testing) meanwhile with LinearAnnealedPolicy() and
#    EpsGreedyQPolicy() the number of illegal moves is considerably reduced.
#    Hence I decided to avoid using GreedyQPolicy(). 

try:
    # Load Keras-RL agent
    print("Loading Keras-RL agent")
    agent_filename = fnmatch.filter(os.listdir(data_filepath), 'dqn_{}_{}_{}_'.format(ENV_NAME, MODEL_TYPE, PREPROC) + '*_agent.pkl')[-1]
    nb_training_steps_pickle = int(agent_filename.split("_")[4]) # nb_training_steps_pickle: stores the number of training steps from previously stored training  
    NB_STEPS_TRAINING = NB_STEPS_TRAINING - nb_training_steps_pickle # We Please, solve the issue the number of training steps considering that we have already trained the model with a number of training steps equal to nb_training_steps_pickle 
    agent_filepath = data_filepath + '/' + agent_filename
    # Please, solve the issue previously mentioned: "ISSUE TO SOLVE - SAVING MODEL'S OPTIMIZER/TRAINING STATE" 
    dqn = pickle.load( open(agent_filepath, "rb")) # https://github.com/keras-rl/keras-rl/issues/186#issuecomment-385200010
except:
    TRAIN_POLICY = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05, value_min=0.05, value_test=0.01, nb_steps=NB_STEPS_ANNEALED)
    TEST_POLICY = EpsGreedyQPolicy(eps=.01)
    #TRAIN_POLICY = GreedyQPolicy()
    #TEST_POLICY = GreedyQPolicy()
    
    # DQNAgent:
    #    We use a DQN agent with the neural network previously defined.
    dqn = DQNAgent(model=model, nb_actions=NUM_ACTIONS_OUTPUT_NN, test_policy=TEST_POLICY, policy=TRAIN_POLICY, memory=memory, processor=processor,
                    nb_steps_warmup=NB_STEPS_WARMUP, gamma=.99, target_model_update=TARGET_MODEL_UPDATE, train_interval=4, delta_clip=1.) #, batch_size=BATCH_SIZE)
    
    # Training Method & Metric:
    #    We use the Adam learning method with MSE (mean squared error) metric.
    dqn.compile(Adam(lr=.00025), metrics=['mse'])

######################################################################
# TRAINING + TESTING:
if TRAIN_TEST_MODE == 'train':
    # Filepaths:
    weights_filepath = data_filepath + '/dqn_{}_{}_{}_weights.h5f'.format(ENV_NAME, MODEL_TYPE, PREPROC)
    checkpoint_weights_filepath = data_filepath + '/dqn_' + ENV_NAME + '_' + MODEL_TYPE + '_' + PREPROC + '_weights_' + '{step}.h5f'
    csv_filepath = data_filepath + '/dqn_{}_{}_{}_train.csv'.format(ENV_NAME, MODEL_TYPE, PREPROC)
    # Callbacks: Stored with a list of Callbacks() functions
    # ModelIntervalCheckpoint is a callback used to save the NN weights every few steps.
    _callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filepath, interval=250000)]
    # TrainEpisodeLogger2048 is a callback used to plot some charts, updated in real-time, that show the NN training results and to save the results in a CSV file.
    _callbacks += [TrainEpisodeLogger2048(csv_filepath)]
    # Training:
    #     visualize = True
    #     verbose = 0/1/2
    #        0 for no logging to stdout
    #        1 for progress bar logging
    #        2 for one log line per epoch
    
    # fit() starts the actual training of the model.
    dqn.fit(env, callbacks=_callbacks, nb_steps=NB_STEPS_TRAINING, visualize=False, verbose=0) #, nb_max_episode_steps=20000, log_interval=10000)

    # Save the final weights one more time after the training is completed.
    dqn.save_weights(weights_filepath, overwrite=True)

    # Save training memory on pickle file and Keras model (architecture + weights + optimizer state) on h5 file
    memory = (memory, memory.actions, memory.rewards, memory.terminals, memory.observations)
    if 'nb_training_steps_pickle' in locals(): # if nb_training_steps_pickle exists among the local variables it means that we have already loaded pre-existing files storing the model's architecture, weights and optimizer state  
        # Save Keras-RL agent training memory
        agentmem_filepath = data_filepath + '/dqn_{}_{}_{}_{}_agentmem.pkl'.format(ENV_NAME, MODEL_TYPE, PREPROC, NB_STEPS_TRAINING + nb_training_steps_pickle)
        pickle.dump(memory, open(agentmem_filepath, "wb"))
        # Save Keras-RL agent
        # Please, solve the issue previously mentioned: "ISSUE TO SOLVE - SAVING MODEL'S OPTIMIZER/TRAINING STATE"
        #     If you uncomment the following lines you will incur the "TypeError: can't pickle _thread.RLock objects" error
        #agent_filepath = data_filepath + '/dqn_{}_{}_{}_{}_agent.pkl'.format(ENV_NAME, MODEL_TYPE, PREPROC, NB_STEPS_TRAINING + nb_training_steps_pickle)
        #pickle.dump(dqn, open(agent_filepath, "wb"))
        # Save Keras model
        model_filepath = data_filepath + '/dqn_{}_{}_{}_{}_model.h5'.format(ENV_NAME, MODEL_TYPE, PREPROC, NB_STEPS_TRAINING + nb_training_steps_pickle)
        model.save(model_filepath)  # creates a HDF5 file 'my_model.h5'
    else:
        # Save Keras-RL agent training memory
        agentmem_filepath = data_filepath + '/dqn_{}_{}_{}_{}_agentmem.pkl'.format(ENV_NAME, MODEL_TYPE, PREPROC, NB_STEPS_TRAINING)
        pickle.dump(memory, open(agentmem_filepath, "wb"), protocol=-1) # protocol=-1 means the the highest protocol version available will be used (binary, etc.)
        # Save Keras-RL agent
        agent_filepath = data_filepath + '/dqn_{}_{}_{}_{}_agent.pkl'.format(ENV_NAME, MODEL_TYPE, PREPROC, NB_STEPS_TRAINING)
        pickle.dump(dqn, open(agent_filepath, "wb"), protocol=-1) # protocol=-1 means the the highest protocol version available will be used (binary, etc.)
        # Save Keras model
        model_filepath = data_filepath + '/dqn_{}_{}_{}_{}_model.h5'.format(ENV_NAME, MODEL_TYPE, PREPROC, NB_STEPS_TRAINING)
        model.save(model_filepath)  # creates a HDF5 file 'my_model.h5'

    # Evaluate the trained model on few episodes/games
    env.reset() # reset the game environment before testing the model
    # TestLogger2048 is a callback used to show the game result (final grid and max tile) and other info at test time for each episode/match of the game. 
    _callbacks = [TestLogger2048()] # List of Callbacks() functions
    # nb_episodes: number to game episodes/matches to test
    dqn.test(env, nb_episodes=5, visualize=False, verbose=0, callbacks=_callbacks) 

elif TRAIN_TEST_MODE == 'test':
    # Load the weights previously estimated and saved in h5f files
    weights_filepath = data_filepath + '/dqn_{}_{}_{}_weights.h5f'.format(ENV_NAME, MODEL_TYPE, PREPROC)
    dqn.load_weights(weights_filepath)
    
    # TestLogger2048 is a callback used to show the game result (final grid and max tile) and other info at test time for each episode/match of the game.
    _callbacks = [TestLogger2048()] 
    # Testing: 
    #     visualize [= True/False]:
    #        True: shows the evolution of each (episode of the) game
    #     verbose [= 0/1/2]:
    #        1: shows the result of each (episode of) game
    dqn.test(env, nb_episodes=100, visualize=False, verbose=0, callbacks=_callbacks)
