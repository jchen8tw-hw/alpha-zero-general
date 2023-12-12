import Arena
from MCTS import MCTS
# from othello.OthelloGame import OthelloGame
# from othello.OthelloPlayers import *
# from othello.pytorch.NNet import NNetWrapper as NNet
from ewn.EWNGame import EWNGame
from ewn.EWNPlayers import *
from ewn.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

g = EWNGame(board_size=5, cube_layer=3)


# nnet players
n1 = NNet(g)

n1.load_checkpoint('./temp', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
def n1p(x): return np.argmax(mcts1.getActionProb(x, temp=0))


player2 = HumanEWNPlayer(g).play


arena = Arena.Arena(player2, n1p, g, display=g.display)

print(arena.playGames(2, verbose=True))
