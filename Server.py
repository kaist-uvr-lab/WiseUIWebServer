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

import os
def processingthread():
    print("Start Message Processing Thread")
    while True:
        ConditionVariable2.acquire()
        ConditionVariable2.wait()
        message = processqueue.pop()
        ConditionVariable2.release()
        # 처리 시작
        start = time.time()
        img_array = np.frombuffer(message.data, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        input_batch = transform(img_cv).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_cv.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        requests.post(FACADE_SERVER_ADDR + "/ReceiveData?map=" + message.map + "&id=" + message.id + "&key=bdepth",bytes(prediction))
        requests.post(PROCESS_SERVER_ADDR + "/notify", ujson.dumps({'user':message.user, 'map':message.map, 'id':int(message.id),'key': 'bdepth'}))
        end = time.time()
        print("Depth Estimation Processing = %s : %f : %d" % (message.id, end - start, len(processqueue)))

        # processing end

def datathread():

    while True:
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
        '--port', type=int, default=35007,
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

    dataqueue = []
    processqueue = []
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    PROCESS_SERVER_ADDR = opt.PROCESS_SERVER_ADDR
    ConditionVariable = threading.Condition()
    ConditionVariable2 = threading.Condition()

    ###LOAD MIDAS
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform


    th1 = threading.Thread(target=datathread)
    th2 = threading.Thread(target=processingthread)

    th1.start()
    th2.start()

    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)

    keyword = 'depth'
    requests.post(FACADE_SERVER_ADDR + "/ConnectServer", ujson.dumps({
        'port':opt.port,'key': keyword
    }))

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



