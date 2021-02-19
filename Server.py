import threading
##################################################
##import super glue and super point
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

from superglue.matching import Matching
from superglue.utils import (frame2tensor, keyframe2tensor)
##import super glue and super point
##################################################
"""
####segmentataion
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
"""
####WSGI
from gevent.pywsgi import WSGIServer
from gevent import monkey

##################################################
# API part
##################################################

global device0, matching
global MatchData, prevID1, prevID2, data1, data2
global KnnMatchData

NUM_MAX_MATCH = 20
FrameData = {}
MatchData = {}
pointserver_addr = "http://143.248.96.81:35005/ReceiveDepth"
mappingserver_addr = "http://143.248.96.81:35006/NotifyNewFrame"
mappingserver_addr2 = "http://143.248.96.81:35006/ReceiveMapData"
nReferenceFrameID = -1;

ConditionVariable = threading.Condition()
ConditionVariable2 = threading.Condition()
message = []
ids = []

#matchGlobalIDX = 0
prevID1 = -1
prevID2 = -1

app = Flask(__name__)
#cors = CORS(app)
#CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

##work에서 호출하는 cv가 필요함.

def work(cv, queue, queue2):
    print("Start Message Processing Thread")
    global pointserver_addr
    while True:
        cv.acquire()
        cv.wait()
        message = queue.pop()
        id = queue2.pop()
        queue.clear()
        queue2.clear()
        cv.release()
        ##### 처리 시작
        start = time.time()
        img_array = np.frombuffer(message, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        input_batch = transform(img_cv).to(device0)
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
        #requests.post(pointserver_addr, ujson.dumps({'id':id,'depth':res}))
        requests.post(pointserver_addr+"?id="+id, bytes(prediction))
        end = time.time()
        print("Depth Processing = %s : %f : %d"%(id, end-start, len(queue)))
    print("End Message Processing Thread")

@app.route("/depthestimate", methods=['POST'])
def depthestimate():
    #params = ujson.loads(request.data)
    global message, ids
    ids.append(request.args.get('id'))
    message.append(request.data)
    global ConditionVariable
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
        '--ip', type=str,
        help='ip address')
    parser.add_argument(
        '--port', type=int, default=35005,
        help='port number')

    #super glue and point
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
        '--max_keypoints', type=int, default=300,
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

    global device0, matching
    opt = parser.parse_args()
    device0 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #device3 = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

    ###LOAD MIDAS
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device0)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    th1 = threading.Thread(target=work, args=(ConditionVariable,message, ids))
    th1.start()

    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)
    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



