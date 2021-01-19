import ujson
import time
import numpy as np
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
import base64
import cv2

from pathlib import Path
import argparse
import matplotlib.cm as cm
import torch

from superglue.matching import Matching
from superglue.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

from matplotlib import gridspec
from matplotlib import pyplot as plt

##################################################
# API part
##################################################
#cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('detect', (640,480))

global data1, data2
data1 = None
data2 = None

app = Flask(__name__)
cors = CORS(app)
@app.route("/api/detect", methods=['POST'])
def detect():
    global data1, data2
    start = time.time()
    params = ujson.loads(request.data)
    img_encoded = base64.b64decode(params['img'])
    itype = int(params['t'])
    type = str(params['t'])
    width = int(params['w'])
    height = int(params['h'])
    channel = int(params['c'])

    # Convert PIL Image
    ######
    img_array = np.frombuffer(img_encoded, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, 1)
    if channel == 3:
        print('Receive Color Image')
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    frame_tensor = frame2tensor(img_cv, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k + type: last_data[k] for k in keys}
    last_data['image'+type] = frame_tensor

    kpts0 = last_data['keypoints'+type][0].cpu().numpy()
    n = len(kpts0)

    #print("detect : %d" %(n))

    #cv2.imshow('detect', img_cv)
    #cv2.waitKey(1)

    #img = Image.fromarray(img_cv)
    json_data = ujson.dumps({'res': kpts0.tolist(), 'n':n})

    if itype == 0:
        data1 = last_data
    else:
        data2 = last_data

    print("Time spent handling the request: %f, %d" % (time.time() - start, itype))
    return json_data;

@app.route("/api/match", methods=['POST'])
def match():
    global data1, data2
    start = time.time()
    #if data2 == None:
    #    json_data = ujson.dumps({'res': kpts0.tolist(), 'n': len(kpts0)})
    #    print('??????????????????????')
    #    return json_data
    pred = matching({**data1, **data2})
    kpts0 = data1['keypoints0'][0].cpu().numpy()
    kpts1 = data2['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    #confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    n1 = len(mkpts0)
    n2 = len(mkpts1)
    print("Time spent handling the request: %f, %d" % (time.time() - start, n1))
    json_data = ujson.dumps({'res': matches.tolist(), 'n': n1})
    return json_data
##################################################
# END API part
##################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        '--max_keypoints', type=int, default=-1,
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
    #print(opt)

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

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

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
    keys = ['keypoints', 'scores', 'descriptors']

    print('Starting the API')
    app.run(host='143.248.96.81', port = 35005)