
import multiprocessing

import time
from grpcTry.server import serve
from grpcTry.client import run


def main():
    mp = multiprocessing.Process(target=serve)
    mp.start()
    time.sleep(1)
    run()
    if mp.is_alive():
        mp.terminate()

if __name__ == '__main__':
    main()