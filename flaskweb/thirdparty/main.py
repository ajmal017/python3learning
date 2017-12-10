#!/home/hangyu5/anaconda2/envs/py3dl/bin/python
import argparse
# import argh
from time import time
from contextlib import contextmanager
import os
import random
import re
import sys
from collections import namedtuple

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

from flaskweb.thirdparty.config import FLAGS, HPS
from flaskweb.thirdparty.utils.load_data_sets import DataSet
from flaskweb.thirdparty.model.SelfPlayWorker import SelfPlayWorker, RivalWorker
from flaskweb.thirdparty.Network import Network

@contextmanager
def timer(message):
    tick = time()
    yield
    tock = time()
    logger.info(f"{message}: {(tock - tick):.3f} seconds")


'''
params:
    @ train_step: total number of mini-batch updates
    @ usage: learning rate annealling
'''


def schedule_lrn_rate(train_step):
    """train_step equals total number of min_batch updates"""
    f = 1  # rl schedule factor
    lr = 1e-3
    if train_step < 1 * f:
        lr = 1e-3  # 1e-1 blows up, sometimes 1e-2 blows up too.
    elif train_step < 2 * f:
        lr = 1e-4
    elif train_step < 3 * f:
        lr = 1e-4
    elif train_step < 4 * f:
        lr = 1e-4
    elif train_step < 5 * f:
        lr = 1e-5
    else:
        lr = 1e-5
    return lr


'''
params:
    @ usage: Go text protocol to play in Sabaki
'''
# Credit: Brain Lee



'''
params:
    @ usage: self play with search pipeline
'''


def selfplay(flags=FLAGS, hps=HPS):


    test_dataset = DataSet.read(os.path.join(flags.processed_dir, "test.chunk.gz"))
    #test_dataset = None

    """set the batch size to -1==None"""
    flags.n_batch = -1
    flags.n_epoch = 1
    flags.selfplay_games_per_epoch = 1
    flags.num_playouts = 1
    flags.load_model_path = os.path.join("","savedmodels/large20")

    net = Network(flags, hps)
    Worker = SelfPlayWorker(net, flags)


    def train(epoch: int):
        lr = schedule_lrn_rate(epoch)
        Worker.run(lr=lr)

    # TODO: consider tensorflow copy_to_graph
    def get_best_model():
        return Network(flags, hps)

    def evaluate_generations():
        best_model = get_best_model()
        Worker.evaluate_model(best_model)

    def evaluate_testset():
        Worker.evaluate_testset(test_dataset)

    """Self Play Pipeline starts here"""
    for g_epoch in range(flags.global_epoch):
        logger.info(f'Global epoch {g_epoch} start.')

        """Train"""
        train(g_epoch)

        """Evaluate on test dataset"""
        evaluate_testset()

        """Evaluate against best model"""
        evaluate_generations()

        logger.info(f'Global epoch {g_epoch} finish.')


def gyySelfplay():
    hps = HPS
    flags = FLAGS

    flags.n_batch = None
    flags.n_epoch = 1
    flags.selfplay_games_per_epoch = 1
    flags.num_playouts = 1600
    flags.load_model_path = os.path.join(
        "/Users/kyoka/Documents/coding/pycharm_workplace/python3learning/flaskweb/thirdparty/",
        "savedmodels/large20")

    net = Network(flags, hps)
    selfplayAgent = RivalWorker(net, flags)

    isContinue = True
    while isContinue:
        isEnd, x, y = selfplayAgent.getMove()
        if isEnd:
            break
        logger.debug(f'Game #{1} Final Position:\n{selfplayAgent.position}')



if __name__ == '__main__':

    gyySelfplay()
