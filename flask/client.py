
import requests


def getIndex():
    api = "http://127.0.0.1:5000/getindex"
    result = eval(requests.get(api).text)
    print(result['index'])


if __name__ == '__main__':
    getIndex()