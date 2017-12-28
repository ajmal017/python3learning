
import logging
import random
import tensorflow as tf
import socket
import time
import requests

def getIndex():
    """
        use a index system to decide worker index
    :return:
    """
    time.sleep(random.random() * 0.1)
    api = "http://127.0.0.1:5000/getindex"
    result = eval(requests.get(api).text)
    return int(result["index"])

def getLogger():

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        # filename='/home/gyy/pysparkTry.log',
                        # filemode='w',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )
    return logging.getLogger(__name__)


def isPortAvaliable(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    res = sock.connect_ex((host, port))
    if res == 0:
        sock.close()
        return False
    return True

def createPort():
    """

    :return:
    """
    basePort = random.randint(3000, 4000)
    while not isPortAvaliable("localhost", basePort):
        print("port %s is not available" % basePort)
    return basePort

def getClusterAndServer():
    """
        The spec is fix for this demo,
        It's need to design a spec provider
    :param jobName:
    :param taskIndex:
    :return:
    """

    # ports = []
    # for _ in range(4):
    #     port = createPort()
    #     if port not in ports:
    #         ports.append(port)
    # print("prots ars %s"%str(ports))
    ports = ["3222","3223","3224","3225"]

    def f(jobName, taskIndex):
        cluster_spec = {
            "worker": [
                "localhost:%s" % ports[0],
                "localhost:%s" % ports[1],
            ],
            "ps": [
                "localhost:%s" % ports[2],
                "localhost:%s" % ports[3]
            ]}
        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec(cluster_spec)

        # Create and start a server for the local task.
        server = tf.train.Server(cluster, jobName, taskIndex)

        return cluster, server
    return f