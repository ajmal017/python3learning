
import asyncio
import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def tailf():
    """
        use yield (coroutin) to implement tailf function which is in shell script
    :return:
    """

    def init_coroutine(func):
        def wrapper(*args, **kwargs):
            rs = func(*args, **kwargs)
            next(rs)
            return rs

        return wrapper

    def follow(f, target):
        f.seek(0, 1)
        while True:
            last_line = f.readline()
            if last_line is not None:
                target.send(last_line)

    @init_coroutine
    def printer():
        while True:
            line = yield
            print(line, end='')

    @init_coroutine
    def filter(key_string, target):
        while True:
            line = yield
            if key_string in line:
                target.send(line)

    f = open('access-log', 'r')
    follow(f, filter("python", printer()))

def coroutineBlock():
    """
        the logical of coroutine is when one coroutine function blocks, even loop will find the next coroutine function

        in the case below, when print_sum wait for compute and compute blocks one second, there is no other coroutine.
        So, print "1 + 2 = 3" after print" compute 1 + 2 " one second.
    :return:
    """

    async def compute(x, y):
        print("compute %s + %s" % (x, y))
        await asyncio.sleep(1.0)
        return x + y

    async def print_sum(x, y):
        result = await compute(x, y)
        print("%s + %s = %s" % (x, y, result))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(print_sum(1, 2))

def differentTask():
    """
        use asyncio.wait to implement different task parallelly
        coroutine compute that block one second, wont affect coroutine printer
    :return:
    """
    async def compute(x, y):
        print("computing x + y")
        await asyncio.sleep(1)
        print("result is %s" % (x + y))

    async def printer(msg):
        print(msg)

    def coroutineParralel():
        task = [compute(1, 2), printer("hello world"), ]
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.wait(task))


def sameTask():
    """
        use ThreadPoolExecutor to run tasks in a separate thread and to monitor their progress using futures.
    :return:
    """

    def fetch_urls(urls):
        return asyncio.gather(*[loop.run_in_executor(executor, requests.get, url)
                                for url in urls])

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=3)

    result = loop.run_until_complete(fetch_urls(['http://www.baidu.com',
                                            'http://www.taobao.com',
                                            'http://www.weibo.com',
                                                 'http://www.163.com']))
    print(result)



def cpuFullTask():
    """
        Test whether it will speed up cpu full task in coroutine
        The answer is NO.
        coroutine is best suitable for i/o tasks
    :return:
    """

    def normal():
        res = 0
        for _ in range(2):
            for i in range(100):
                # res += i + i ** 2 + i ** 3
                res += np.dot(np.random.random((100,100)), np.random.random((100, 100)))

    async def asyjob(q):
        res = 0
        for i in range(100):
            res += np.dot(np.random.random((100, 100)), np.random.random((100, 100)))
            # res += i + i ** 2 + i ** 3
        q.append(res)  # queue

    def coroutine():
        q = []
        futures = [asyjob(q) for _ in range(2)]

        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.wait(futures))

    st = time.time()
    normal()
    st2 = time.time()
    print("normal is %.4f" % (st2-st))
    coroutine()
    st3 = time.time()
    print("coroutine is %.4f" % (st3 - st2))

def gpuFullTask():
    """

    :return:
    """
    import tensorflow as tf

    def normal():
        a = tf.placeholder(dtype=tf.float32)
        b = tf.matmul(a, tf.transpose(a))

        with tf.Session() as sess:
            res = 0
            sess.run(tf.global_variables_initializer())
            for _ in range(200):
                res += b.eval(session=sess, feed_dict={a:np.random.random((1000,100))})

    async def asyjob(q, a, b, sess):
        res = 0
        for _ in range(100):
            res += b.eval(session=sess, feed_dict={a: np.random.random((1000, 100))})
        q.append(res)

    def coroutine():
        q = []
        a = tf.placeholder(dtype=tf.float32)
        b = tf.matmul(a, tf.transpose(a))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.global_variables_initializer())
            futures = [asyjob(q, a, b, sess) for _ in range(2)]

            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.wait(futures))

    st = time.time()
    normal()
    st2 = time.time()
    print("normal is %.4f" % (st2 - st))
    coroutine()
    st3 = time.time()
    print("coroutine is %.4f" % (st3 - st2))


if __name__ == '__main__':
    # cpuFullTask()
    gpuFullTask()