
import random
import json
import os
from flask import Flask, url_for, request
from flask import render_template
from flaskweb.thirdparty.agent import SelfPlayAgent

app = Flask(__name__)


@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/go')
def go():
    return render_template('go/index.html')

@app.route('/nextMove/', methods=['GET'])
def netmove():
    isEnd, x, y = SelfPlayAgent().getMove()
    return json.dumps({'isEnd':isEnd, 'row': x, 'col': y})

if __name__ == '__main__':
    app.run(debug=True)