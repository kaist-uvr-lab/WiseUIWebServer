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

##model method
from lib.utils.net_tools import load_ckpt
from lib.utils.logging import setup_logging
import torchvision.transforms as transforms
from tools.parse_arg_test import TestOptions
from lib.models.metric_depth_model import MetricDepthModel
from lib.core.config import cfg, merge_cfg_from_file
from lib.models.image_transfer import bins_to_depth

def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img

import os
def processingthread():
    print("Start Message Processing Thread")
    while True:
        ConditionVariable2.acquire()
        ConditionVariable2.wait()
        message = processqueue.pop()
        ConditionVariable2.release()
        # 처리 시작
        with torch.no_grad():
            img_array = np.frombuffer(message.data, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_resize = cv2.resize(img_cv, (int(img_cv.shape[1]), int(img_cv.shape[0])), interpolation=cv2.INTER_LINEAR)
            img_torch = scale_torch(img_resize, 255)
            img_torch = img_torch[None, :, :, :].cuda()

            _, pred_depth_softmax = model.module.depth_model(img_torch)
            pred_depth = bins_to_depth(pred_depth_softmax)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            #print(pred_depth)
            #pred_depth_scale = (pred_depth / pred_depth.max() * 60000).astype(
            #    np.uint16)  # scale 60000 for visualization
            #print(pred_depth.dtype)
            #cv2.imwrite('./test.jpg', pred_depth)
        requests.post(FACADE_SERVER_ADDR + "/ReceiveData?map=" + message.map + "&id=" + message.id + "&key=bdepth",
                      bytes(pred_depth))
        requests.post(PROCESS_SERVER_ADDR + "/notify", ujson.dumps(
                {'user': message.user, 'map': message.map, 'id': int(message.id), 'key': 'bdepth'}))
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
    parser =  TestOptions().initialize(parser)
    #parser = TestOptions().parse()

    parser.thread = 1
    parser.batchsize = 1
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

    ##module args
    opt = parser.parse_args()
    merge_cfg_from_file(opt)
    device = torch.device("cuda:"+opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")
    #load model


    # load model
    model = MetricDepthModel()
    # load checkpoint
    if opt.load_ckpt:
        load_ckpt(opt, model)
    #model.cuda()
    model.eval().to(device)
    model = torch.nn.DataParallel(model)

    dataqueue = []
    processqueue = []
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    PROCESS_SERVER_ADDR = opt.PROCESS_SERVER_ADDR
    ConditionVariable = threading.Condition()
    ConditionVariable2 = threading.Condition()

    th1 = threading.Thread(target=datathread)
    th2 = threading.Thread(target=processingthread)
    th1.start()
    th2.start()

    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)

    keyword = 'depth'
    requests.post(FACADE_SERVER_ADDR + "/ConnectServer", ujson.dumps({
        'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
    }))

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



