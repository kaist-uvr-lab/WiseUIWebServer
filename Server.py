import threading
import ujson
import time
import numpy as np
from flask import Flask, request
import requests
import cv2
from module.Message import Message
import argparse
import torch

#WSGI
from gevent.pywsgi import WSGIServer

##################################################
# API part
##################################################

app = Flask(__name__)
#cors = CORS(app)
#CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

#work에서 호출하는 cv가 필요함.
def work(cv,  queue):
    print("Start Message Processing Thread")
    while True:
        cv.acquire()
        cv.wait()
        message = queue.pop()
        cv.release()
        # 처리 시작

        # processing end


@app.route("/Receive", methods=['POST'])
def Receive():
    user = request.args.get('user')
    map = request.args.get('map')
    id = request.args.get('id')
    message = Message(user, map, id, request.data)
    queue.append(message)
    ConditionVariable.acquire()
    ConditionVariable.notify()
    ConditionVariable.release()
    return ""

##################################################
# END API part
##################################################

if __name__ == "__main__":

    ##################################################
    ##arguments
    parser = argparse.ArgumentParser(
        description='WISE UI Web Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ip', type=str,default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--port', type=int, default=35006,
        help='port number')
    parser.add_argument(
        '--use_gpu', type=str, default='0',
        help='port number')
    parser.add_argument(
        '--FACADE_SERVER_ADDR', type=str,
        help='facade server address')
    parser.add_argument(
        '--PROCESS_SERVER_ADDR', type=str,
        help='process server address')

    opt = parser.parse_args()
    device = torch.device("cuda:"+opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")

    queue = []
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    PROCESS_SERVER_ADDR = opt.PROCESS_SERVER_ADDR
    ConditionVariable = threading.Condition()

    th1 = threading.Thread(target=work, args=(ConditionVariable, queue))
    th1.start()

    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)

    keyword = 'bdepth'
    requests.post(FACADE_SERVER_ADDR + "/ConnectServer", ujson.dumps({
        'port':opt.port,'key': keyword
    }))

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



