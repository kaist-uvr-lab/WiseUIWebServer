import threading
import ujson
import time
import numpy as np
from flask import Flask, request
import requests
import cv2
from socket import *
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

import os
def processingthread():
    print("Start Message Processing Thread")
    while True:
        ConditionVariable2.acquire()
        ConditionVariable2.wait()
        message = processqueue.pop()
        ConditionVariable2.release()
        # 처리 시작

        # processing end
bufferSize = 1024
def datathread():

    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        data = ujson.loads(message.decode())
        id = data['id']
        res =sess.post(FACADE_SERVER_ADDR + "/Load", ujson.dumps({
            'keyword':data['keyword'],'id':id
        }))
        print(res.request.body)
        """
        ConditionVariable.acquire()
        ConditionVariable.wait()
        message = dataqueue.pop()
        ConditionVariable.release()
        # processing start
        response = requests.post(FACADE_SERVER_ADDR + "/SendData?map=" + message.map + "&id=" + message.id + "&key=bimage","")
        message.data = response.content
        processqueue.append(message)
        # processing end
        ConditionVariable2.acquire()
        ConditionVariable2.notify()
        ConditionVariable2.release()
        """
@app.route("/Receive", methods=['POST'])
def Receive():
    user = request.args.get('user')
    map = request.args.get('map')
    id = request.args.get('id')
    message = Message(user, map, id)
    dataqueue.append(message)
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
        '--prior', type=str, default='0',
        help='port number')
    parser.add_argument(
        '--ratio', type=str, default='1',
        help='port number')
    parser.add_argument(
        '--FACADE_SERVER_ADDR', type=str,
        help='facade server address')
    parser.add_argument(
        '--PROCESS_SERVER_ADDR', type=str,
        help='process server address')
    parser.add_argument(
        '--ECHO_SERVER_IP', type=str, default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--ECHO_SERVER_PORT', type=int, default=35001,
        help='port number')
    opt = parser.parse_args()
    device = torch.device("cuda:"+opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")



    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)

    ##Echo server
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    PROCESS_SERVER_ADDR = opt.PROCESS_SERVER_ADDR
    ReceivedKeywords=['Image','Matching']
    SendKeywords = 'Keypoints, Descriptors, Matches'
    sess = requests.Session()
    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        #'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'id':'FeatureServer', 'type1':'Server', 'type2':'test','keyword': SendKeywords, 'Additional':None
    }))
    ECHO_SERVER_ADDR = (opt.ECHO_SERVER_IP, opt.ECHO_SERVER_PORT)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    for keyword in ReceivedKeywords:
        temp = ujson.dumps({'type1':'connect', 'keyword':keyword})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
    #Echo server connect

    #thread
    dataqueue = []
    processqueue = []

    ConditionVariable = threading.Condition()
    ConditionVariable2 = threading.Condition()

    th1 = threading.Thread(target=datathread)
    th2 = threading.Thread(target=processingthread)
    th1.start()
    th2.start()
    # thread

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



