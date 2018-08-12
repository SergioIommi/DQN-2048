################################################################################
# Description: Module with Keras-RL callbacks used to monitor and log the NN training
# Author:      Sergio Iommi
################################################################################

from os.path import exists
import csv
import numpy as np
import matplotlib.pyplot as plt
from rl.callbacks import Callback, TestLogger
    
class TestLogger2048(TestLogger):
    def on_episode_end(self, episode, logs):
        """ TestLogger2048 is a callback function that prints the logs at end of each episode/game match """
        
        grid = self.env.get_board()
        template = 'episode: {episode}, max tile: {max_tile}, episode reward: {episode_reward:.3f}, episode steps: {nb_steps}'
        variables = {
            'episode': episode + 1,
            'max_tile': np.amax(grid),
            'episode_reward': logs['episode_reward'],
            'nb_steps': logs['nb_steps']
        }
        print(template.format(*variables))
        print("Final Grid: \n{0}\n".format(grid))

class TrainEpisodeLogger2048(Callback):
    """
    TrainEpisodeLogger2048 is a callback function used to plot some charts, updated in real-time, that show the
    NN training results (max tile reached, average reward, etc.) and to save the results in a CSV file.
    """
    # Code modified from TrainEpisodeLogger
    
    def __init__(self, filePath):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.observations = {}
        self.rewards = {}
        self.max_tile = {}
        self.step = 0
                
        self.episodes = []
        
        # Max Tile/Episode Rewards: initialize variables and figures
        self.max_tiles = []
        self.episodes_rewards = []
        self.fig_max_tile = plt.figure()
        self.ax1 = self.fig_max_tile.add_subplot(1,1,1) # 1 row, 1 column, 1st plot
        self.fig_reward = plt.figure()
        self.ax2 = self.fig_reward.add_subplot(1,1,1) # 1 row, 1 column, 1st plot
        
        # Max Tiles Means/Episode Rewards Means: initialize variables and figures
        self.max_tiles_means = 0
        self.episodes_rewards_means = 0
        self.fig_max_tile_mean = plt.figure()
        self.ax3 = self.fig_max_tile_mean.add_subplot(1,1,1) # 1 row, 1 column, 1st plot
        self.fig_reward_mean = plt.figure()
        self.ax4 = self.fig_reward_mean.add_subplot(1,1,1) # 1 row, 1 column, 1st plot
        
        self.nb_episodes_for_mean = 50 # calculate means after this amount of episodes 
        self.episode_counter = 0 # Used to count the episodes and to decide when to calculate the average and plot it

        # CSV file:
        if exists(filePath):
            csv_file = open(filePath, "a") # a = append
            self.csv_writer = csv.writer(csv_file, delimiter=',')
        else:
            csv_file = open(filePath, "w") # w = write (clear and restart)
            self.csv_writer = csv.writer(csv_file, delimiter=',')
            headers = ['episode', 'episode_steps', 'episode_reward', 'max_tile']
            self.csv_writer.writerow(headers)
        
    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.observations[episode] = []
        self.rewards[episode] = []
        self.max_tile[episode] = 0
    
    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done but also plot training charts and save the data in a CSV file. """
        self.episode_counter += 1
        self.episodes = np.append(self.episodes, episode + 1)
        self.max_tiles = np.append(self.max_tiles, self.max_tile[episode])
        self.episodes_rewards = np.append(self.episodes_rewards, np.sum(self.rewards[episode]))
        
        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1) # used to monitor how many training steps are left to end the training phase
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps} - episode: {episode}, episode steps: {episode_steps}, episode reward: {episode_reward:.3f}, max tile: {max_tile}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'episode_steps': len(self.observations[episode]),
            'episode_reward': self.episodes_rewards[-1],
            'max_tile': self.max_tiles[-1] # self.max_tile[episode]
        }
        print(template.format(**variables))
        
        # Save CSV:
        self.csv_writer.writerow((episode + 1, len(self.observations[episode]), self.episodes_rewards[-1], self.max_tiles[-1]))

        # Figures: Means
        # Graphs for values averaged across more episodes (e.g., averages over 50 episodes, etc.)
        if self.episode_counter % self.nb_episodes_for_mean == 0 :
            self.max_tiles_means = np.append(self.max_tiles_means,np.mean(self.max_tiles[-self.nb_episodes_for_mean:]))
            self.fig_max_tile_mean.clear()
            plt.figure(self.fig_max_tile_mean.number)
            plt.plot(np.arange(0,self.episode_counter+self.nb_episodes_for_mean,self.nb_episodes_for_mean), self.max_tiles_means)
            plt.title("max tiles means (over last {} episodes)".format(self.nb_episodes_for_mean))
            plt.xlabel("episode #")
            plt.ylabel("max tiles mean")
            plt.pause(0.01) # https://github.com/matplotlib/matplotlib/issues/7759/
            
            self.episodes_rewards_means = np.append(self.episodes_rewards_means,np.mean(self.episodes_rewards[-self.nb_episodes_for_mean:]))
            self.fig_reward_mean.clear()
            plt.figure(self.fig_reward_mean.number)            
            plt.plot(np.arange(0,self.episode_counter+self.nb_episodes_for_mean,self.nb_episodes_for_mean), self.episodes_rewards_means)
            plt.title("rewards means (over last {} episodes)".format(self.nb_episodes_for_mean))
            plt.xlabel("episode #")
            plt.ylabel("rewards mean")
            plt.pause(0.01)
        
        # Figures: Points
        self.fig_max_tile.clear()
        plt.figure(self.fig_max_tile.number)
        plt.scatter(self.episodes, self.max_tiles, s=1) # scatterplot
        #plt.plot(self.episodes, self.max_tiles) # line plot
        plt.title("max tile (per episode)")
        plt.xlabel("episode #")
        plt.ylabel("max tile")
        plt.pause(0.01)

        self.fig_reward.clear()
        plt.figure(self.fig_reward.number)
        plt.scatter(self.episodes, self.episodes_rewards, s=1) # scatterplot
        #plt.plot(self.episodes, self.episodes_rewards) # line plot
        plt.title("reward (per episode)")
        plt.xlabel("episode #")
        plt.ylabel("reward")
        plt.pause(0.01)

        # Free up resources.
        del self.observations[episode]
        del self.rewards[episode]
        del self.max_tile[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.max_tile[episode] = logs['info']['max_tile']
        self.step += 1