
from flask import Flask
from flask.ext import restful

app = Flask(__name__)
api = restful.Api(app)

COUNT = 0

class HelloWorld(restful.Resource):
    def get(self):
        return {'hello': 'world'}

class GetIndex(restful.Resource):
    def get(self):
        global COUNT
        if COUNT % 2 == 1:
            COUNT += 1
            return {'index': '1'}
        else:
            COUNT += 1
            return {'index': '0'}

api.add_resource(GetIndex, '/getindex')

if __name__ == '__main__':
    app.run(debug=True)
