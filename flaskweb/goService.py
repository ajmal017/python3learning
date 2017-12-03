
import random
import json
from flask import Flask, url_for, request
from flask import render_template


app = Flask(__name__)

@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/go')
def go():
    return render_template('go/index.html')

@app.route('/nextMove/', methods=['GET'])
def netmove():
    x = request.args.get('x', type=int)
    y = request.args.get('y', type=int)
    # print(request.args.get('x'))
    return json.dumps({'row': random.randint(0, 19), 'col': random.randint(0, 19)})


if __name__ == '__main__':
    app.run(debug=True)