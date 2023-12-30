from flask import Flask, request
from flask_restful import Resource, reqparse
from gevent.pywsgi import WSGIServer

from main import checkTheFile

app = Flask(__name__)


class Users(Resource):
    def post(self):
        parser = reqparse.RequestParser()  # initialize
        print('parser ==> ', parser)
        return {'data': parser}, 200  # return data and 200 OK code


@app.route('/check_file', methods=['POST'])
def form_example():
    requestJson = request.get_json()  # initialize
    Path = requestJson.get('Path')
    answer = checkTheFile(Path)
    return {"data": {"Answer": answer}}, 200  # return data and 200 OK code


if __name__ == '__main__':
    # run app in debug mode on port 5000
    ip = input("Enter the IP, 'default is 127.0.0.1'")
    port = input("Enter the Prot, 'default is 5000'")
    if ip == '':
        ip = "127.0.0.1"
    if port == '':
        port = "5000"
    print("running on ip :" + ip + " port : " + port)
    http_server = WSGIServer((ip, int(port)), app)
    http_server.serve_forever()
