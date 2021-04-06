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

from superglue.matching import Matching
from superglue.utils import (frame2tensor, keyframe2tensor)

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
        img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        frame_tensor = frame2tensor(img, device)

        last_data = matching.superpoint({'image': frame_tensor})
        desc = last_data['descriptors'][0].cpu().detach().numpy()
        kpts = last_data['keypoints'][0].cpu().detach().numpy()
        # scores = last_data['scores'][0].cpu().detach().numpy()

        requests.post(FACADE_SERVER_ADDR + "/ReceiveData?map=" + message.map + "&id=" + message.id + "&key=bkpts",kpts.tobytes())
        requests.post(FACADE_SERVER_ADDR + "/ReceiveData?map=" + message.map + "&id=" + message.id + "&key=bdesc",desc.transpose().tobytes())
        requests.post(PROCESS_SERVER_ADDR + "/notify",
                      ujson.dumps({'user': message.user, 'map': message.map, 'id': int(message.id), 'key': 'bkpts'}))
        requests.post(PROCESS_SERVER_ADDR + "/notify",
                      ujson.dumps({'user': message.user, 'map': message.map, 'id': int(message.id), 'key': 'bdesc'}))

        end = time.time()
        print("Super Point Processing = %s : %f : %d"%(message.id, end-start, len(processqueue)))

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

    # super glue and point
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=500,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')


    opt = parser.parse_args()
    device = torch.device("cuda:"+opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")

    ##LOAD SuperGlue & SuperPoint
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

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

    keyword = 'superpointandglue'
    requests.post(FACADE_SERVER_ADDR + "/ConnectServer", ujson.dumps({
        'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
    }))

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



