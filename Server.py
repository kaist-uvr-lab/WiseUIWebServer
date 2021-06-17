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

import os
def processingthread():
    print("Start Message Processing Thread")
    while True:
        ConditionVariable.acquire()
        ConditionVariable.wait()
        message = msgqueue.pop()
        ConditionVariable.release()
        # 처리 시작

        data = ujson.loads(message.decode())
        id = data['id']
        keyword = data['keyword']
        src = data['src']

        res =sess.post(FACADE_SERVER_ADDR + "/Load?keyword="+keyword+"&id="+str(id),"")

        img_array = np.frombuffer(res.content, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        Data[keyword][id] ={}
        Data[keyword][id]['data'] = img_cv
        Data[keyword][id]['src']  = src

        cv2.imshow('img', img_cv)
        cv2.waitKey(1)
        # processing end

bufferSize = 1024
def udpthread():

    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        #address = bytesAddressPair[1]
        print(message)
        msgqueue.append(message)
        ConditionVariable.acquire()
        ConditionVariable.notify()
        ConditionVariable.release()

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
        '--ECHO_SERVER_IP', type=str, default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--ECHO_SERVER_PORT', type=int, default=35001,
        help='port number')
    opt = parser.parse_args()
    device = torch.device("cuda:"+opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")

    print('Starting the API')

    Data = {}
    msgqueue = []

    ##Echo server
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
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
        temp = ujson.dumps({'type1':'connect', 'keyword':keyword, 'src':'FeatureServer', 'type2':'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[keyword]={}
    #Echo server connect

    ConditionVariable = threading.Condition()

    th1 = threading.Thread(target=udpthread)
    th2 = threading.Thread(target=processingthread)
    th1.start()
    th2.start()
    # thread

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



