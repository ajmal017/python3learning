from flaskweb.thirdparty.config import FLAGS, HPS
from flaskweb.thirdparty.Network import Network

from flaskweb.thirdparty.model.SelfPlayWorker import RivalWorker
import os

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance

class SelfPlayAgent():
    def __init__(self):

        hps = HPS
        flags = FLAGS

        flags.n_batch = None
        flags.n_epoch = 1
        flags.selfplay_games_per_epoch = 1
        flags.num_playouts = 1000
        flags.load_model_path = os.path.join("/Users/kyoka/Documents/coding/pycharm_workplace/python3learning/flaskweb/thirdparty/",
                                             "savedmodels/large20")

        net = Network(flags, hps)
        self.selfplayAgent = RivalWorker(net, flags)

    def getMove(self):
        return self.selfplayAgent.getMove()