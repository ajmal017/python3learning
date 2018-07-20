

class Content(object):
    def __init__(self, a):
        self.a = a

class Single(object):
    CONT = Content(1)


class SingleExtend(Single):
    pass