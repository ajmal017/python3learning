
from flask import Flask
import flask_restful

app = Flask(__name__)
api = flask_restful.Api(app)

COUNT = 0

class HelloWorld(flask_restful.Resource):
    def get(self):
        return {'hello': 'world'}

class GetIndex(flask_restful.Resource):
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
