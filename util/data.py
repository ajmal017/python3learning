
import numpy as np
import os

def getSeparator():
    '''
        获取不同平台下的斜杠符号
    :return
    Created by Wu Yongcong 2017-8-17
    '''
    import platform
    if 'Windows' == platform.system():
        separator = '\\'
    else:
        separator = '/'
    return separator

def getRootDir():
    """
    :return:
    """
    sep = getSeparator()
    wkDir = os.getcwd()
    items = wkDir.split(sep)
    for i, item in enumerate(reversed(items)):
        if item == "python3learning":
            rootDir = sep.join(items[:len(items)-i])

    return rootDir

def getMnist():
    root = getRootDir()
    dataX = np.loadtxt(os.path.join(root, 'resource/digitInput.txt'))
    dataY = np.loadtxt(os.path.join(root, 'resource/digitOutput.txt'))

    return dataX, dataY

if __name__ == '__main__':
    dataX, dataY = getMnist()
    print(dataX.shape, dataY.shape)