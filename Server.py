import threading
import ujson
import time
import numpy as np
from PIL import Image
from flask import Flask, request
import requests
from flask_cors import CORS
import base64
import cv2

import argparse
import torch

####WSGI
from gevent.pywsgi import WSGIServer
from gevent import monkey

##################################################
# API part
##################################################


app = Flask(__name__)
#cors = CORS(app)
#CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

##work에서 호출하는 cv가 필요함.

def work(cv,  mapQueue, frameQueue, dataQueue, addr):
    print("Start Message Processing Thread")
    while True:
        cv.acquire()
        cv.wait()
        map = mapQueue.pop()
        id = frameQueue.pop()
        data = dataQueue.pop()
        cv.release()
        ##### 처리 시작
        start = time.time()
        img_array = np.frombuffer(data, dtype=np.uint8)
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

        h = prediction.shape[0]
        w = prediction.shape[1]

        #res = base64.b64encode(prediction).decode('ascii')
        requests.post(addr + "?map=" + map + "&id="+id+"&key=bdepth", bytes(prediction))
        end = time.time()
        print("Depth Processing = %s : %f : %d"%(id, end-start, len(dataQueue)))
    print("End Message Processing Thread")

@app.route("/depthestimate", methods=['POST'])
def depthestimate():
    maps.append(request.args.get('map'))
    ids.append(request.args.get('id'))
    datas.append(request.data)
    ConditionVariable.acquire()
    ConditionVariable.notify()
    ConditionVariable.release()
    return "" #ujson.dumps({'id':0})

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
        '--FACADE_SERVER', type=str,
        help='ip address')
    parser.add_argument(
        '--use_gpu', type=str, default='0',
        help='port number')
    parser.add_argument(
        '--FACADE_SERVER_ADDR', type=str,
        help='port number')

    opt = parser.parse_args()
    device = torch.device("cuda:"+opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")

    ###LOAD MIDAS
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    maps = []
    datas = []
    ids = []
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    facadeserver_addr = FACADE_SERVER_ADDR + '/ReceiveData'
    ConditionVariable = threading.Condition()

    th1 = threading.Thread(target=work, args=(ConditionVariable,maps, ids, datas, facadeserver_addr))
    th1.start()

    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)
    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



