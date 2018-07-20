
from singleton.base import SingleExtend


def main():
    s = SingleExtend()
    s.CONT.a = 2
    s1 = SingleExtend()
    print(s.CONT.a)


class A():

    @classmethod
    def get(cls):
        return cls.test()

    @classmethod
    def test(cls):
        return 1

if __name__ == '__main__':
    print(A.get())