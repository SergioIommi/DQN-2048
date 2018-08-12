################################################################################
# Description: Module with game logic of 2048 (using OpenAI Gym interface
#              https://gym.openai.com/)
# Author:      Sergio Iommi's mod of https://github.com/rgal/gym-2048
################################################################################

from __future__ import print_function

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

import itertools
import logging
from six import StringIO
import sys

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        # Definitions for game. Board-matrix must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards.
        self.score = 0

        # Members for gym implementation:
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board-matrix.
        self.observation_space = spaces.Box(0, 2**squares, (self.w * self.h, ), dtype=np.int)
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        self.reward_range = (0., float(2**squares))

        # Initialise the random seed of the gym environment.
        self.seed()

        # Reset the board-matrix, ready for a new game.
        self.reset()

    def seed(self, seed=None):
        """Set the random seed for the gym environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Implementation of gym interface:
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        try:
            score = float(self.move(action))
            self.score += score
            assert score <= 2**(self.w*self.h)
            self.add_tile()
            done = self.isend()
            reward = float(score)
        except IllegalMove as e:
            logging.debug("Illegal move")
            done = False
            reward = 0. # No reward for an illegal move. We could even set a negative value to penalize the agent.

        #print("Am I done? {}".format(done))
        observation = self.Matrix
        
        # info (dictionary):
        #    - can be used to store further information to the caller after executing each step/movement in the game
        #    - it is useful for testing and for monitoring the agent (via callback functions) while it is training
        info = {"max_tile": self.highest()}        
        
        # Return observation (board-matrix state), reward, done and info dictionary
        return observation, reward, done, info

    def reset(self):
        """Reset the game board-matrix and add 2 tiles."""
        self.Matrix = np.zeros((self.h, self.w), np.int)
        self.score = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return self.Matrix

    def render(self, mode='human'):
        """Rendering for standard output of score, highest tile reached and
        board-matrix of game."""
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implementation of game logic for 2048:
    def add_tile(self):
        """Add a tile with value 2 or 4 with different probabilities."""
        val = 0
        if self.np_random.random_sample() > 0.8:
            val = 4
        else:
            val = 2
        empties = self.empties()
        assert empties
        empty_idx = self.np_random.choice(len(empties))
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Get the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a list of tuples of the location of empty squares."""
        empties = list()
        for y in range(self.h):
            for x in range(self.w):
                if self.get(x, y) == 0:
                    empties.append((x, y))
        return empties

    def highest(self):
        """Report the highest tile on the board-matrix."""
        highest = 0
        for y in range(self.h):
            for x in range(self.w):
                highest = max(highest, self.get(x, y))
        return highest

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Check if the game is ended. Game ends if there is a 2048 tile or
        there are no legal moves. If there are empty spaces then there must
        be legal moves."""

        if self.highest() == 2048:
            return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Get the whole board-matrix, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Set the whole board-matrix, useful for testing."""
        self.Matrix = new_board
