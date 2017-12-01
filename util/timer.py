
import time

def timer(func):
    def wrapper(*args, **kwargs):
        st = time.time()
        func(*args, **kwargs)
        print(" %s consume %.8f" % (func.__name__, time.time()-st))
    return wrapper