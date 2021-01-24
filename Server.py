
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

from pathlib import Path
import argparse
import matplotlib.cm as cm
import torch

from superglue.matching import Matching
from superglue.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor, keyframe2tensor)
##import super glue and super point
##################################################

##################################################
##import MiDaS
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
##import MiDaS
##################################################

from matplotlib import gridspec
from matplotlib import pyplot as plt

##################################################
# API part
##################################################

global FrameData
FrameData = {}

app = Flask(__name__)
#cors = CORS(app)
#CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

@app.route("/api/receiveImage", methods=['POST'])
def receiveImage():
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
    img_cv = cv2.cvtColor(img_cv, cv2.BGR2RGB)
    img_resized = cv2.resize(img_cv, dsize=(width/2, height/2))
    img_gray = cv2.cvtColor(img_cv, cv2.RGB2GRAY)

    temp = {}
    temp['image'] = img_gray
    temp['resize'] = img_resized
    FrameData[id] = temp

    json_data = ujson.dumps({'res': 0})
    print("Data %d, time : %f" % (len(FrameData), time.time() - start))
    return json_data;
@app.route("/api/depthestimate", methods=['POST'])
def depthestimate():
    start = time.time()
    params = ujson.loads(request.data)
    img_encoded = base64.b64decode(params['img'])

    width = int(params['w'])
    height = int(params['h'])
    channel = int(params['c'])
    id = int(params['id'])

    # Convert PIL Image
    img_array = np.frombuffer(img_encoded, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

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
    prediction = np.reshape(prediction, w*h)
    res_str = ' '.join(str(i) for i in prediction)

    json_data = ujson.dumps({'res':res_str , 'w':w, 'h':h})
    print("Depth Time spent handling the request: %f, %d" % (time.time() - start, 0))
    return json_data;

@app.route("/api/reset", methods=['POST'])
def reset():
    global FrameData
    FrameData = {}
    json_data = ujson.dumps({'res': 0})
    print("Reset FrameData")
    return json_data;

@app.route("/api/detect", methods=['POST'])
def detect():
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
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    #if channel == 3:
    #    print('Receive Color Image')
    #img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    frame_tensor = frame2tensor(img_cv, device)
    last_data = matching.superpoint({'image': frame_tensor})

    ####data 저장
    temp = {}
    temp['image'] = img_cv
    kpts0 = last_data['keypoints'][0].cpu().detach().numpy()
    temp['keypoints'] = kpts0#last_data['keypoints'][0].cpu().detach().numpy() #에러가능성
    temp['descriptors'] = last_data['descriptors'][0].cpu().detach().numpy()
    temp['scores'] = last_data['scores'][0].cpu().detach().numpy()
    FrameData[id] = temp
    print("Data %d" %(len(FrameData)))
    ####data 저장

    n = len(kpts0)
    json_data = ujson.dumps({'res': kpts0.tolist(), 'n':n})

    print("Time spent handling the request: %f, %d" % (time.time() - start, n))
    return json_data;

@app.route("/api/match", methods=['POST'])
def match():
    global FrameData
    start = time.time()

    ##data 처리
    params = ujson.loads(request.data)
    id1 = int(params['id1'])
    id2 = int(params['id2'])
    ####data 불러오기
    data1 = keyframe2tensor(FrameData[id1], device, '0')
    data2 = keyframe2tensor(FrameData[id2], device, '1')

    pred = matching({**data1, **data2})
    matches = pred['matches0'][0].cpu().numpy()
    print("Time spent handling the request: %f %d" % (time.time() - start, len(matches)))
    json_data = ujson.dumps({'res': matches.tolist(), 'n': len(matches)})
    return json_data
##################################################
# END API part
##################################################




if __name__ == "__main__":

    ##################################################
    ##arguments
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #midas
    """
    parser.add_argument('--model_path',
                        default='model-f6b98070.pt',
                        help='path to the trained weights of model'
                        )

    parser.add_argument('--model_type',
                        default='large',
                        help='model type: large or small'
                        )
    """
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ###LOAD MIDAS

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    """
    if opt.model_type == "large":
        midas = MidasNet(opt.model_path, non_negative=True)
        net_w, net_h = 384, 384
    elif opt.model_type == "small":
        midas = MidasNet_small(opt.model_path, features=64, backbone="efficientnet_lite3", exportable=True,
                               non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    midas.eval()
    midas.to(device)
    """
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

    #device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
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
    ##Super Glue and Super Point
    ##################################################

    print('Starting the API')
    app.run(host='127.0.0.1', port = 35005)