import json
import asyncio
from flask import Flask, url_for, request
from flask import render_template

app = Flask(__name__)

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


@app.route('/hello/<name>', methods=['GET'])
def hello(name):
    return json.dumps({"hello": "world", "name": name})

@app.route('/hello/', methods=['GET'])
def hello2():
    return json.dumps({"hello": "world", })

@app.route('/cortest', methods=['GET'])
def cortest():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    coroutineBlock()
    return json.dumps({"code": 1})

if __name__ == '__main__':
    app.run(debug=True)