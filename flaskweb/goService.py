
import random
import json
import asyncio
from flask import Flask, url_for, request
from flask import render_template
from flaskweb.thirdparty.agent import SelfPlayAgent
from flask_cache import Cache

# def create_app():
#     app = Flask(__name__)
#     def run_on_start(*args, **argv):
#         import requests
#         url = "http://localhost:5000/start/"
#         requests.get(url)
#     run_on_start()
#     return app
app = Flask(__name__)
MODEL_DICT = {}


cache = Cache(app, config={'CACHE_TYPE': 'simple'})


@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/go')
def go():
    return render_template('go/index.html')

@app.route('/nextMove/<name>', methods=['GET'])
def netmove(name):
    global MODEL_DICT
    if name not in MODEL_DICT:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        MODEL_DICT[name] = SelfPlayAgent()
    isEnd, x, y = MODEL_DICT[name].getMove()
    # isEnd, x, y = False, 1, 1
    return json.dumps({'end':isEnd, 'row': str(x), 'col': str(y)})

@app.route('/start/', methods=['GET'])
def startAgent():

    return json.dumps({'code': "1"})

if __name__ == '__main__':
    with app.app_context():
        cache.clear()
    app.run()

