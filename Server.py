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
mappingserver_addr = "http://143.248.96.81:35006/NotifyNewFrame"
mappingserver_addr2 = "http://143.248.96.81:35006/ReceiveMapData"
nReferenceFrameID = -1;

ConditionVariable = threading.Condition()
ConditionVariable2 = threading.Condition()
message = []

#matchGlobalIDX = 0
prevID1 = -1
prevID2 = -1

app = Flask(__name__)
#cors = CORS(app)
#CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

##work에서 호출하는 cv가 필요함.
def work2(conv2, matcher):
    print("Start Matching Thread")
    while True:
        conv2.acquire()
        conv2.wait()
        start = time.time()
        ids = list(FrameData.keys())
        conv2.release()
        lastID = ids.pop()
        ids.reverse()

        nID = 0
        for id in ids:
            desc1 = FrameData[lastID]['descriptors'].transpose()
            desc2 = FrameData[id]['descriptors'].transpose()
            matches = matcher.knnMatch(desc1, desc2, k=2)
            good = np.empty((len(matches)))
            success = 0
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    good[i] = m.trainIdx
                    success = success + 1
                else:
                    good[i] = 10000
            nID = nID+1
            if nID == 6:
                break
            if success < 30:
                break
        end = time.time()
        print("Matching Thread ID= %d, %f = %d"%(lastID,end-start, nID))


def work(cv, condition2, SuperPointAndGlue, queue):
    print("Start Message Processing Thread")
    global mappingserver_addr, FrameData
    while True:
        cv.acquire()
        cv.wait()
        message = queue.pop()
        queue.clear()
        cv.release()
        ##### 처리 시작
        start = time.time()
        img_encoded = base64.b64decode(message['img'])
        id = int(message['id'])
        width = int(message['w'])
        height = int(message['h'])
        channel = int(message['c'])

        img_array = np.frombuffer(img_encoded, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        frame_tensor = frame2tensor(img, device0)
        last_data = SuperPointAndGlue.superpoint({'image': frame_tensor})

        kpts = last_data['keypoints'][0].cpu().detach().numpy()
        desc = last_data['descriptors'][0].cpu().detach().numpy()
        scores = last_data['scores'][0].cpu().detach().numpy()
        end1 = time.time()
        Frame = {}
        Frame['image'] = img
        Frame['keypoints'] = kpts  # last_data['keypoints'][0].cpu().detach().numpy() #에러가능성
        Frame['descriptors'] = desc  # descriptor를 미리 트랜스 포즈하고나서 수퍼글루를 더 적게 쓰면 그 때만 트랜스 포즈 하도록 하게 하자.
        Frame['scores'] = scores
        Frame['rgb'] = img_cv
        n = len(kpts)
        ##mapping server에 전달

        FrameData[id] = Frame
        requests.post(mappingserver_addr, ujson.dumps({'id': id, 'n': n}))
        end2 = time.time()

        condition2.acquire()
        condition2.notify()
        condition2.release()

        #thtemp =  threading.Thread(target=supergluematch2, args=(SuperPointAndGlue, id))
        #thtemp.start()
        print("Message Processing = %d : %d : %f %f %d"%(id, len(FrameData), end2-start, end2-end1, len(queue)))
    print("End Message Processing Thread")

def supergluematch2(SuperGlue, id):
    global FrameData, nReferenceFrameID
    start = time.time()
    data1 = keyframe2tensor(FrameData[id], device0, '0')
    data2 = keyframe2tensor(FrameData[nReferenceFrameID], device0, '1')
    pred = matching({**data1, **data2})
    matches0 = pred['matches0'][0].cpu().numpy()
    end = time.time()
    print("SuperGlue = %f %d"%(end-start, len(matches0)))

#######################################################
@app.route("/SaveMap", methods=['POST'])
def SaveMap():
    global FrameData
    idx = 0
    total = 0
    keys = []
    StoringData = {};
    for id, Frame in FrameData.items():
        NewFrame = {}
        idx = idx+1
        if idx % 6 !=0:
            continue
        total = total+1
        img = Frame['rgb']
        _, img = cv2.imencode('.jpg', img)
        NewFrame['image'] = str(base64.b64encode(img))
        NewFrame['keypoints'] = Frame['keypoints'].tolist()#str(base64.b64encode(Frame['keypoints']))#Frame['keypoints'].tolist()
        NewFrame['descriptors'] =Frame['descriptors'].tolist()
        StoringData[str(id)] = NewFrame;
        keys.append(str(id))

    StoringData['total']=total
    StoringData['keys'] = keys
    a = ujson.dumps(StoringData)
    f = open('./map/map.bin', 'wb')
    f.write(a.encode())
    f.close()
    return ujson.dumps({'id': 0})

@app.route("/LoadMap", methods=['POST'])
def LoadMap():
    global FrameData
    f = open('./map/map.bin', 'rb')
    data = f.read().decode()
    f.close()
    DATA = ujson.loads(data)
    n = DATA['total']
    print("LoadMap %d"%(n))
    requests.post(mappingserver_addr2, ujson.dumps(DATA))


    """
    DATA = ujson.loads(data)
    for id, Frame in DATA.items():
        if id=='total':
            break
        print("ID = %d"%(int(id)))
        print(Frame['keypoints'])
    n = DATA['total']
    """
    return ujson.dumps({'id': 0})

@app.route("/SetReferenceFrameID", methods=['POST'])
def SetReferenceFrameID():
    global nReferenceFrameID
    params = ujson.loads(request.data)
    id = int(params['id'])
    nReferenceFrameID = id
    #print("Set Reference Frame ID = %d"%(nReferenceFrameID))
    return ujson.dumps({'id': id})

@app.route("/GetReferenceFrameID", methods=['POST'])
def GetReferenceFrameID():
    global nReferenceFrameID
    #print("Get Reference Frame ID = %d" % (nReferenceFrameID))
    return ujson.dumps({'n': nReferenceFrameID})

@app.route("/ReceiveAndDetect", methods=['POST'])
def ReceiveAndDetect():
    start = time.time()
    params = ujson.loads(request.data)
    id = int(params['id'])
    end = time.time()
    print("Receive Time : %f = %d"%(end-start, id))
    global message
    message.append(params)
    global ConditionVariable
    ConditionVariable.acquire()
    ConditionVariable.notify()
    ConditionVariable.release()

    """
    img_encoded = base64.b64decode(params['img'])
    width = int(params['w'])
    height = int(params['h'])
    channel = int(params['c'])
    
    img_array = np.frombuffer(img_encoded, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    frame_tensor = frame2tensor(img, device0)
    last_data = matching.superpoint({'image': frame_tensor})

    kpts   = last_data['keypoints'][0].cpu().detach().numpy()
    desc   = last_data['descriptors'][0].cpu().detach().numpy()
    scores = last_data['scores'][0].cpu().detach().numpy()

    #rgb인지 bgr인지 확인해야 함
    Frame = {}
    Frame['image'] = img
    Frame['keypoints'] = kpts #last_data['keypoints'][0].cpu().detach().numpy() #에러가능성
    Frame['descriptors'] = desc #descriptor를 미리 트랜스 포즈하고나서 수퍼글루를 더 적게 쓰면 그 때만 트랜스 포즈 하도록 하게 하자.
    Frame['scores'] = scores
    Frame['rgb'] = img_cv
    global FrameData
    FrameData[id] = Frame
    n = len(kpts)
    
    ##mapping server에 전달
    global mappingserver_addr
    requests.post(mappingserver_addr,  ujson.dumps({'id':id,'n':n}))
    return ujson.dumps({'id':id,'n':n})
    """
    return ujson.dumps({'id':id})

########################################################
@app.route("/sendimage", methods=['POST'])
def sendimage():
    params = ujson.loads(request.data)
    id = int(params['id'])
    global FrameData
    if id not in FrameData:
        print("Frame Error::id=%d" % (id))
    img = FrameData[id]['rgb']
    _, img = cv2.imencode('.jpg', img)
    img_encoded = str(base64.b64encode(img))
    json_data = ujson.dumps({'img': img_encoded})
    return json_data

@app.route("/receiveimage", methods=['POST'])
def receiveimage():
    params = ujson.loads(request.data)
    global FrameData
    img_encoded = base64.b64decode(params['img'])
    width = int(params['w'])
    height = int(params['h'])
    channel = int(params['c'])
    id = int(params['id'])

    # Convert PIL Image
    ######
    img_array = np.frombuffer(img_encoded, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    """
    if channel == 4:
        print(params)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    """
    """
    #cv2.imwrite("./img/test" + str(id) + ".jpg", img_cv)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    #img_resized = cv2.resize(img_cv, dsize=(int(width/2), int(height/2)))
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    """
    temp = {}
    #temp['image'] = img_gray
    temp['rgb'] = img_cv
    FrameData[id] = temp
    json_data = ujson.dumps({'id': id})
    return json_data

@app.route("/detect", methods=['POST'])
def detect():

    params = ujson.loads(request.data)
    id = int(params['id'])

    global device0
    global FrameData
    if id not in FrameData:
        print("Frame Error::id=%d"%(id))
    Frame = FrameData[id]
    img = cv2.cvtColor(Frame['rgb'], cv2.COLOR_RGB2GRAY)
    frame_tensor = frame2tensor(img, device0)
    last_data = matching.superpoint({'image': frame_tensor})
    Frame['image'] = img
    ####data 수정
    kpts = last_data['keypoints'][0].cpu().detach().numpy()

    desc = last_data['descriptors'][0].cpu().detach().numpy()
    Frame['keypoints'] = kpts #last_data['keypoints'][0].cpu().detach().numpy() #에러가능성
    Frame['descriptors'] = desc #descriptor를 미리 트랜스 포즈하고나서 수퍼글루를 더 적게 쓰면 그 때만 트랜스 포즈 하도록 하게 하자.
    Frame['scores'] = last_data['scores'][0].cpu().detach().numpy()
    FrameData[id] = Frame
    ####data 수정
    n = len(kpts)
    #print(res2)
    json_data = ujson.dumps({'id':id,'n':n})
    return json_data

"""
@app.route("/segment", methods=['POST'])
def segment():
    global FrameData
    params = ujson.loads(request.data)
    id = int(params['id'])
    Frame = FrameData.get(id)
    img = Frame['rgb']
    #img.shape
    #img.channel()

    r,g,b = cv2.split(img)
    input = np.array([r,g,b])
    input = torch.from_numpy(input).float().to(device0).unsqueeze(0)
    #print(len(input[0]))
    #input = preprocess_input(img)
    res = segmentation(input)[0][0].cpu().detach().numpy()
    #cv2.normalize(res,res,0,255,cv2.NORM)
    ret, res = cv2.threshold(res, 0.5, 255.0, cv2.THRESH_BINARY)
    print(res)
    cv2.imwrite("seg.jpg", res)
    json_data = ujson.dumps({'res': 0})
    return json_data
"""

@app.route("/depthestimate", methods=['POST'])
def depthestimate():
    global FrameData
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

    res1 = str(base64.b64encode(prediction))
    #print(res1)

    json_data = ujson.dumps({'res':res1 , 'w':w, 'h':h, 'b':True})
    return json_data

@app.route("/reset", methods=['POST'])
def reset():
    global FrameData, prevID1, prevID2
    FrameData = {}
    nReferenceFrameID = -1
    prevID1 = -1
    prevID2 = -1
    json_data = ujson.dumps({'res': 0})
    print("Reset FrameData")
    return json_data

@app.route("/detectWithDesc", methods=['POST'])
def detectWithDesc():
    global device0
    global FrameData#, matchGlobalIDX
    #matchIDX = matchGlobalIDX
    #matchGlobalIDX = (matchGlobalIDX+1)%NUM_MAX_MATCH
    #print("Detect=Start=%d"% (matchIDX))

    #start = time.time()
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
    res1 = str(base64.b64encode(kpts))
    res2 = str(base64.b64encode(desc))
    #print("Detect=End: %f %d" % (time.time() - start, n))
    #print(res2)
    json_data = ujson.dumps({'pts': res1, 'desc' : res2, 'n':n}) #desc.tolist()
    #json_data = ujson.dumps({'res': kpts.tolist(), 'n': n})
    return json_data

@app.route("/detectOnlyPts", methods=['POST'])
def detectOnlyPts():
    global device0
    global FrameData

    #start = time.time()
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
    res1 = str(base64.b64encode(kpts))
    #print("Detect=End: %f %d" % (time.time() - start, n))
    #print(res2)
    json_data = ujson.dumps({'pts': res1, 'n':n})
    #json_data = ujson.dumps({'res': kpts.tolist(), 'n': n})
    return json_data



@app.route("/getPts", methods=['POST'])
def getPts():
    global device0
    global FrameData
    ##이것도 사용할 경우 n은 없애기
    ##이것만 요청하는 경우에는 속ㄷ고가 상당히 빠름. 0.006초 정도, 그렇다면 포인트도 분리한다면?
    params = ujson.loads(request.data)
    id = int(params['id'])
    pts = FrameData[id]['keypoints'] #256x300 으로 이게 아마 row 부터 전송이 될 수 있음. 받는 곳에서 이것도 다시 확인이 필요함. 수퍼 글루가 아니면 트랜스포즈 해서 넣는것도 방법일 듯.
    #res = str(base64.b64encode(pts))
    json_data = ujson.dumps({'pts': pts.tolist(), 'n':len(pts)})
    return json_data

@app.route("/getDesc", methods=['POST'])
def getDesc():
    global device0
    global FrameData
    ##이것만 요청하는 경우에는 속ㄷ고가 상당히 빠름. 0.006초 정도, 그렇다면 포인트도 분리한다면?
    ##이것은 받는쪽이나 보내는쪽에서 인코딩, 디코딩 하는 시간도 고려해야 한다는 것을 의미하는듯
    params = ujson.loads(request.data)
    id = int(params['id'])
    desc1 = FrameData[id]['descriptors'] #256x300 으로 이게 아마 row 부터 전송이 될 수 있음. 받는 곳에서 이것도 다시 확인이 필요함. 수퍼 글루가 아니면 트랜스포즈 해서 넣는것도 방법일 듯.
    res = str(base64.b64encode(desc1))
    json_data = ujson.dumps({'desc': res})
    return json_data

@app.route("/supergluematch", methods=['POST'])
def supergluematch():
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
    #print("Match=End : id1 = %d, id2 = %d time = %f %d" % (id1, id2, time.time() - start, len(matches0)))
    return json_data

@app.route("/featurematch", methods=['POST'])
def featurematch():
    params = ujson.loads(request.data)
    id1 = int(params['id1'])
    id2 = int(params['id2'])

    desc1 = FrameData[id1]['descriptors'].transpose()
    desc2 = FrameData[id2]['descriptors'].transpose()
    matches = bf.knnMatch(desc1, desc2, k=2)
    #matches  = flann.knnMatch(desc1, desc2, k=2)
    good = np.empty((len(matches)), np.int32)
    success = 0
    for i, (m, n) in enumerate(matches):
        #print("%d %d : %f %f"%(m.queryIdx, n.trainIdx, m.distance, n.distance))
        if m.distance < 0.7 * n.distance:
            good[i] = m.trainIdx
            success = success+1
        else:
            good[i] = 10000
    #print("match : id = %d, %d, res %d"%(id1, id2, success))
    #print("KnnMatch time = %f , %d %d" % (time.time() - start, len(matches), nres))
    # print("featurematch %d : %d %d"%(len(good), len(desc1), len(desc2)))
    #res = str(base64.b64encode(good))
    return ujson.dumps({'matches': good.tolist()})
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

    #flann based matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher()#
    """
    segmentation = smp.DeepLabV3Plus(
        encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet', encoder_output_stride=16,
        decoder_channels=256, decoder_atrous_rates=(12, 24, 36),
        in_channels=3, classes=1, activation=None, upsampling=4, aux_params=None).eval().to(device0)
    """
    keys = ['keypoints', 'scores', 'descriptors']

    th1 = threading.Thread(target=work, args=(ConditionVariable, ConditionVariable2, matching, message))
    th1.start()
    th2 = threading.Thread(target=work2, args=(ConditionVariable2, bf))
    th2.start()
    #th1.join()

    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)
    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



