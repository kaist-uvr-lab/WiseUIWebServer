
##################################################
##import super glue and super point
import ujson
import time
import numpy as np
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
import base64
import cv2

import argparse
import torch

from superglue.matching import Matching
from superglue.utils import (frame2tensor, keyframe2tensor)
##import super glue and super point
##################################################

####WSGI
from gevent.pywsgi import WSGIServer
from gevent import monkey

##################################################
# API part
##################################################

global device0, matching
global FrameData, MatchData, prevID1, prevID2, data1, data2

NUM_MAX_MATCH = 20
FrameData = {}
MatchData = {}
#matchGlobalIDX = 0
prevID1 = -1
prevID2 = -1

app = Flask(__name__)
#cors = CORS(app)
#CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

@app.route("/receiveimage", methods=['POST'])
def receiveimage():

    global FrameData
    start = time.time()
    params = ujson.loads(request.data)
    img_encoded = base64.b64decode(params['img'])

    width = int(params['w'])
    height = int(params['h'])
    channel = int(params['c'])
    id = int(params['id'])

    # Convert PIL Image
    ######
    img_array = np.frombuffer(img_encoded, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    #img_resized = cv2.resize(img_cv, dsize=(int(width/2), int(height/2)))
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    temp = {}
    temp['image'] = img_gray
    temp['rgb'] = img_cv
    FrameData[id] = temp

    json_data = ujson.dumps({'res': 0})
    print("Data %d, time : %f" % (len(FrameData), time.time() - start))
    return json_data;

@app.route("/depthestimate", methods=['POST'])
def depthestimate():
    global FrameData
    start = time.time()
    params = ujson.loads(request.data)
    id = int(params['id'])

    Frame = FrameData.get(id)
    if Frame == None :
        json_data = ujson.dumps({'res': (), 'w': 0, 'h': 0, 'b':False})
        return json_data
    img = Frame['rgb']
    input_batch = transform(img).to(device0)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

    h = prediction.shape[0]
    w = prediction.shape[1]
    #print(prediction.tolist())
    #prediction = np.reshape(prediction, w*h)
    #res_str = ' '.join(str(i) for i in prediction)

    json_data = ujson.dumps({'res':prediction.tolist() , 'w':w, 'h':h, 'b':True})
    print("Depth Estimation: %f" % (time.time() - start))
    return json_data

@app.route("/reset", methods=['POST'])
def reset():
    global FrameData, prevID1, prevID2
    FrameData = {}
    prevID1 = -1
    prevID2 = -1
    json_data = ujson.dumps({'res': 0})
    print("Reset FrameData")
    return json_data;

@app.route("/detect", methods=['POST'])
def detect():
    global device0, matching
    global FrameData#, matchGlobalIDX
    #matchIDX = matchGlobalIDX
    #matchGlobalIDX = (matchGlobalIDX+1)%NUM_MAX_MATCH
    #print("Detect=Start=%d"% (matchIDX))

    start = time.time()
    params = ujson.loads(request.data)
    id = int(params['id'])

    Frame = FrameData[id]
    img = Frame['image']
    frame_tensor = frame2tensor(img, device0)
    last_data = matching.superpoint({'image': frame_tensor})

    ####data 수정
    kpts = last_data['keypoints'][0].cpu().detach().numpy()
    desc = last_data['descriptors'][0].cpu().detach().numpy()
    Frame['keypoints'] = kpts#last_data['keypoints'][0].cpu().detach().numpy() #에러가능성
    Frame['descriptors'] = desc
    Frame['scores'] = last_data['scores'][0].cpu().detach().numpy()
    FrameData[id] = Frame
    ####data 수정
    n = len(kpts)

    json_data = ujson.dumps({'res': kpts.tolist(), 'desc' : desc.tolist(), 'n':n})
    #json_data = ujson.dumps({'res': kpts.tolist(), 'n': n})
    print("Detect=End: %f, %d" % (time.time() - start, n))
    return json_data

@app.route("/match", methods=['POST'])
def match():
    global device0, matching
    global FrameData, MatchData#, matchGlobalIDX#, prevID1, prevID2, data1, data2
    start = time.time()
    print("Match=Start")
    #matchIDX = matchGlobalIDX
    #matchGlobalIDX = (matchGlobalIDX + 1) % NUM_MAX_MATCH
    #print("Match=Start=%d" % (matchIDX))
    ##data 처리
    params = ujson.loads(request.data)
    id1 = int(params['id1'])
    id2 = int(params['id2'])
    ####data 불러오기
    #if id1 != prevID1 :
    data1 = keyframe2tensor(FrameData[id1], device0, '0')
    #    prevID1 = id1
    #if id2 != prevID2:
    data2 = keyframe2tensor(FrameData[id2], device0, '1')
    #    prevID2 = id2

    pred = matching({**data1, **data2})
    matches0 = pred['matches0'][0].cpu().numpy()
    matches1 = pred['matches1'][0].cpu().numpy()

    MatchData[id1] = {}
    MatchData[id1][id2] = matches0

    MatchData[id2] = {}
    MatchData[id2][id1] = matches1

    # 딕셔너리 키 traverse
    #for key in FrameData.keys():
    #    print(key)

    #json_data = ujson.dumps({'res': 0})
    json_data = ujson.dumps({'res': matches0.tolist(), 'n': len(matches0)})
    print("Match=End : id1 = %d, id2 = %d time = %f %d" % (id1, id2, time.time() - start, len(matches0)))
    return json_data

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
    matching = Matching(config).eval().to(device0)
    #matching=[]
    #for i in range(NUM_MAX_MATCH):
    #    matching.append(Matching(config).eval().to(device0))

    keys = ['keypoints', 'scores', 'descriptors']

    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)
    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()