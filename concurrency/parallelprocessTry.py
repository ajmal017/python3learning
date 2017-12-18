
import multiprocessing
import asyncio
import requests
import numpy as np
import aiohttp
import util.log
import time

from util.timer import timer
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import JoinableQueue
from multiprocessing.managers import BaseManager


urls = ['http://www.baidu.com',
            'http://www.taobao.com',
            'http://www.weibo.com',
            'http://www.163.com']

logger = util.log.logging.getLogger(__name__)


@timer
def multiProcessTask():
    def requestTask(sem, urls, i):
        with sem:
            print("start task %s" % i)
            results = [requests.get(url) for url in urls]
            print("end task %s" % i)

    sem = multiprocessing.Semaphore(10)

    for i in range(50):
        p = multiprocessing.Process(target=requestTask, args=(sem, urls, i, ))
        p.start()



@timer
def asyncSemaphore():
    """
        If there is no asyn i/o request in the coroutine end,
        it will be the same as serial

        I used requests first to find the problem above
    :return:
    """
    async def requestTask(sem, urls, i):
        with (await sem):
            # print("start task %s" % i)
            results = await getRequests(urls)
            # print("end task %s" % i)
            return results

    async def getRequests(urls):
        r = []
        for url in urls:
            r.append(await get(url))
        return r

    async def get(url):
        async with aiohttp.request("GET", url) as response:
            return await response.read()

    sem = asyncio.Semaphore(10)
    loop = asyncio.get_event_loop()
    futures = [requestTask(sem, urls, i) for i in range(20)]
    loop.run_until_complete(asyncio.gather(*futures))


@timer
def asyncSemaphoreWithThreadPool():
    """
        In order to get high performance, it's need to use ThreadPoolExecutor to create a multi thread work

        coroutine is just with no thread pool, is just like serial programming
    :return:
    """
    async def getRequests(urls):
        r = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {executor.submit(requests.get, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    r.append(data)
                except:
                    pass
            return r

    async def requestTask(sem, urls, i):
        with (await sem):
            # print("start task %s" % i)
            results = await getRequests(urls)
            # print("end task %s" % i)
            return results

    sem = asyncio.Semaphore(10)
    loop = asyncio.get_event_loop()
    futures = [requestTask(sem, urls, i) for i in range(20)]
    loop.run_until_complete(asyncio.gather(*futures))


@timer
def normal():
    for i in range(20):
        results = [requests.get(url) for url in urls]

################### Joinablequeue test
def processwait(mgrlist, i):
    mgrlist.append(i)
    logger.info(" process sleep start")
    time.sleep(10)
    logger.info(" process sleep end")

def JoinqueueTest():
    queue = JoinableQueue()
    done = False

    p = multiprocessing.Process(target=processwait)
    p.start()
    while not done:
        msg = queue.get(block=True)
        logger.info("Got msg: {0}".format(msg))
        if msg is None:
            logger.info("Terminating PS")
            done = True
        queue.task_done()




def managerTest():
    mgr = multiprocessing.Manager()
    mgrlist = mgr.list()
    for i in range(20):
        if len(mgrlist) < 10:
            p = multiprocessing.Process(target=processwait, args=(mgrlist, i))
            p.start()
        else:
            pass
    logger.info("mgr shutdown end")
    time.sleep(10)

if __name__ == '__main__':
    # asyncSemaphore()
    # asyncSemaphoreWithThreadPool()
    # normal()
    managerTest()