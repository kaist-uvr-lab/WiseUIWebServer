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

from module.User    import User
from module.Map     import Map
from module.Message import Message

####WSGI
from gevent.pywsgi import WSGIServer
from gevent import monkey

##################################################
# API part
##################################################
#LABEL_NAMES = np.array(['wall' ,'building' ,'sky' ,'floor' ,'tree' ,'ceiling' ,'road' ,'bed' ,'windowpane' ,'grass' ,'cabinet' ,'sidewalk' ,'person' ,'earth' ,'door' ,'table' ,'mountain' ,'plant' ,'curtain' ,'chair' ,'car' ,'water' ,'painting' ,'sofa' ,'shelf' ,'house' ,'sea' ,'mirror' ,'rug' ,'field' ,'armchair' ,'seat' ,'fence' ,'desk' ,'rock' ,'wardrobe' ,'lamp' ,'bathtub' ,'railing' ,'cushion' ,'base' ,'box' ,'column' ,'signboard' ,'chest of drawers' ,'counter' ,'sand' ,'sink' ,'skyscraper' ,'fireplace' ,'refrigerator' ,'grandstand' ,'path' ,'stairs' ,'runway' ,'case' ,'pool table' ,'pillow' ,'screen door' ,'stairway' ,'river' ,'bridge' ,'bookcase' ,'blind' ,'coffee table' ,'toilet' ,'flower' ,'book' ,'hill' ,'bench' ,'countertop' ,'stove' ,'palm' ,'kitchen island' ,'computer' ,'swivel chair' ,'boat' ,'bar' ,'arcade machine' ,'hovel' ,'bus' ,'towel' ,'light' ,'truck' ,'tower' ,'chandelier' ,'awning' ,'streetlight' ,'booth' ,'television' ,'airplane' ,'dirt track' ,'apparel' ,'pole' ,'land' ,'bannister' ,'escalator' ,'ottoman' ,'bottle' ,'buffet' ,'poster' ,'stage' ,'van' ,'ship' ,'fountain' ,'conveyer belt' ,'canopy' ,'washer' ,'plaything' ,'swimming pool' ,'stool' ,'barrel' ,'basket' ,'waterfall' ,'tent' ,'bag' ,'minibike' ,'cradle' ,'oven' ,'ball' ,'food' ,'step' ,'tank' ,'trade name' ,'microwave' ,'pot' ,'animal' ,'bicycle' ,'lake' ,'dishwasher' ,'screen' ,'blanket' ,'sculpture' ,'hood' ,'sconce' ,'vase' ,'traffic light' ,'tray' ,'ashcan' ,'fan' ,'pier' ,'crt screen' ,'plate' ,'monitor' ,'bulletin board' ,'shower' ,'radiator' ,'glass' ,'clock' ,'flag'])

app = Flask(__name__)
# cors = CORS(app)
# CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

##work에서 호출하는 cv가 필요함.
def work2(conv2, queue):
    print("Start Matching Thread")
    while True:
        conv2.acquire()
        conv2.wait()
        start = time.time()
        message = queue.pop()
        map = message.map
        conv2.release()
        lastID = message.id
        print("Depth %d" % (lastID))
        requests.post(depthserver_addr+"?map="+map+"&id="+str(lastID), message.data)
        requests.post(semanticserver_addr+"?map="+map+"&id="+str(lastID), message.data)

def work(condition1, condition2, SuperPointAndGlue, queue, queue2):
    print("Start Message Processing Thread")
    while True:
        condition1.acquire()
        condition1.wait()
        message = queue[-1]
        condition1.release()
        ##### 처리 시작
        start = time.time()
        img_array = np.frombuffer(message.data, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        frame_tensor = frame2tensor(img, device0)

        time1 = time.time()
        last_data = SuperPointAndGlue.superpoint({'image': frame_tensor})
        time2 = time.time()

        kpts = last_data['keypoints'][0].cpu().detach().numpy()
        desc = last_data['descriptors'][0].cpu().detach().numpy()
        scores = last_data['scores'][0].cpu().detach().numpy()
        end1 = time.time()

        Frame = {}
        Frame['image'] = img
        Frame['keypoints'] = kpts  # last_data['keypoints'][0].cpu().detach().numpy() #에러가능성
        Frame['descriptors'] = desc  # descriptor를 미리 트랜스 포즈하고나서 수퍼글루를 더 적게 쓰면 그 때만 트랜스 포즈 하도록 하게 하자.
        Frame['bkpts'] = kpts.tobytes()
        Frame['bdesc'] = desc.transpose().tobytes()

        Frame['scores'] = scores
        Frame['rgb'] = message.data #인코딩된 바이트 형태
        n = len(kpts)
        ##mapping server에 전달
        MapData[message.map].Frames[message.id] = Frame
        end2 = time.time()
        requests.post(mappingserver_addr, ujson.dumps({'id': message.id, 'map':message.map, 'n': n}))
        end3 = time.time()

        queue2.append(message)
        condition2.acquire()
        condition2.notify()
        condition2.release()

        # thtemp =  threading.Thread(target=supergluematch2, args=(SuperPointAndGlue, id))
        # thtemp.start()
        print("Message Processing = %s %d : %d : %f : " % (message.map, message.id, len(MapData[message.map].Frames), end3 - start))
        #print("Message Processing = %d : %d : %f : %f : %f %f %f %d" % (id, len(FrameData), time1-start, time2-time1, end1 - time2, end2 - end1, end3-end2, len(queue)))
    print("End Message Processing Thread")

#######################################################
@app.route("/Connect", methods=['POST'])
def Connect():
    data = ujson.loads(request.data)
    id = data['userID']
    map = data['mapName']
    bMapping = data['bMapping']
    fx = data['fx']
    fy = data['fy']
    cx = data['cx']
    cy = data['cy']
    w = data['w']
    h = data['h']
    cameraParam = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0,0.0,1.0]], dtype = np.float32)
    imgSize = np.array([h, w])
    user = User(id, map, bMapping, cameraParam, imgSize)
    requests.post(mappingserver_connect, ujson.dumps({
        'fx':fx,
        'fy':fy,
        'cx':cx,
        'cy':cy,
        'w':w,
        'h':h
    }))

    """
    user.id
    user['map'] = map
    user['mapping'] = bMapping
    user['camera'] = cameraParam
    user['image'] = imgSize
    """
    UserData[user.id] = user
    if MapData.get(map) is None:
        MapData[map] = Map(map)

    #print('Connect %s'%(user))
    print('Connect Num = %d'%(len(UserData)))
    print('Connect Map = %s'%(MapData[map].name))
    return ""


@app.route("/reset", methods=['POST'])
def reset():
    map = MapData[request.args.get('map')]
    map.reset()
    json_data = ujson.dumps({'res': 0})
    return json_data

@app.route("/SaveMap", methods=['POST'])
def SaveMap():
    """
    idx = 0
    total = 0
    keys = []
    StoringData = {};
    for id, Frame in FrameData.items():
        NewFrame = {}
        idx = idx + 1
        if idx % 6 != 0:
            continue
        total = total + 1
        img = Frame['rgb']
        _, img = cv2.imencode('.jpg', img)
        NewFrame['image'] = str(base64.b64encode(img))
        NewFrame['keypoints'] = Frame[
            'keypoints'].tolist()  # str(base64.b64encode(Frame['keypoints']))#Frame['keypoints'].tolist()
        NewFrame['descriptors'] = Frame['descriptors'].tolist()
        StoringData[str(id)] = NewFrame;
        keys.append(str(id))

    StoringData['total'] = total
    StoringData['keys'] = keys
    a = ujson.dumps(StoringData)
    f = open('./map/map.bin', 'wb')
    f.write(a.encode())
    f.close()
    """
    return ujson.dumps({'id': 0})

@app.route("/LoadMap", methods=['POST'])
def LoadMap():
    f = open('./map/map.bin', 'rb')
    data = f.read().decode()
    f.close()
    DATA = ujson.loads(data)
    n = DATA['total']
    print("LoadMap %d" % (n))
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

@app.route("/GetLastFrameID", methods=['POST'])
def GetLastFrameID():
    map = MapData[request.args.get('map')]
    return ujson.dumps({'n': map.UpdateIDs[request.args.get('key')]})

@app.route("/SetLastFrameID", methods=['POST'])
def SetLastFrameID():
    map = MapData[request.args.get('map')]
    id = int(request.values.get('id'))
    map.UpdateIDs[request.values.get('key')] = id
    return "a"

@app.route("/ReceiveAndDetect", methods=['POST'])
def ReceiveAndDetect():
    start = time.time()
    map = MapData[request.args.get('map')]
    id = int(request.args.get('id'))

    message = Message(map.name, id, request.data)
    end = time.time()
    print("Receive Time : %f = %d" % (end - start, id))

    messages.append(message)

    ConditionVariable.acquire()
    ConditionVariable.notify()
    ConditionVariable.release()
    return ujson.dumps({'id': id})

@app.route("/ReceiveData", methods=['POST'])
def ReceiveData():
    map = MapData[request.args.get('map')]
    id = int(request.args.get('id'))
    key = request.args.get('key')

    map.Frames[id][key] = request.data
    map.UpdateIDs[key] = id
    return "a"

@app.route("/SendData", methods=['POST'])
def SendData():
    map = MapData[request.args.get('map')]
    id = int(request.args.get('id'))
    key = request.args.get('key')
    print("senddata test %d"%(len(map.Frames)))
    data = map.Frames[id][key] #이걸 전부 바이트로 변환???
    return data


@app.route("/featurematch", methods=['POST'])
def featurematch():
    map = MapData[request.args.get('map')]
    id1 = int(request.args.get('id1'))
    id2 = int(request.args.get('id2'))

    desc1 = map.Frames[id1]['descriptors'].transpose()
    desc2 = map.Frames[id2]['descriptors'].transpose()
    matches = bf.knnMatch(desc1, desc2, k=2)
    # matches  = flann.knnMatch(desc1, desc2, k=2)
    good = np.empty((len(matches)), np.int32)
    success = 0
    for i, (m, n) in enumerate(matches):
        # print("%d %d : %f %f"%(m.queryIdx, n.trainIdx, m.distance, n.distance))
        if m.distance < 0.7 * n.distance:
            good[i] = m.trainIdx
            success = success + 1
        else:
            good[i] = 10000
    # print("match : id = %d, %d, res %d"%(id1, id2, success))
    # print("KnnMatch time = %f , %d %d" % (time.time() - start, len(matches), nres))
    # print("featurematch %d : %d %d"%(len(good), len(desc1), len(desc2)))
    # res = str(base64.b64encode(good))
    return ujson.dumps({'matches': good.tolist()})
########################################################



@app.route("/ReceiveDepth", methods=['POST'])
def ReceiveDepth():
    """
    id = int(request.args.get('id'))
    FrameData[id]['bdepth'] = request.data

    #depth image 저장
    ##base64인코딩 된 결과임. 즉 문자열.
    ##뎁스 이미지로 변환하는 과정. 참고용
    #bdepth = base64.b64decode(params['depth'])
    darray = np.frombuffer(request.data, dtype=np.float32)
    depth = darray.reshape((480,640))
    cv2.normalize(depth, depth, 0, 255, cv2.NORM_MINMAX)
    depth = np.array(depth, dtype=np.uint8)

    cv2.imwrite("depth.jpg", depth)
    FrameData[id]['depth'] = depth
    FrameData[id]['bdepth'] = bytes(depth)
    #print("Depth Map Update = %d"%(nLastDepthID))
    """
    nLastDepthID = id
    return ""
"""
@app.route("/getPts", methods=['POST'])
def getPts():
    ##이것도 사용할 경우 n은 없애기
    ##이것만 요청하는 경우에는 속ㄷ고가 상당히 빠름. 0.006초 정도, 그렇다면 포인트도 분리한다면?

    map = MapData[request.args.get('map')]
    params = ujson.loads(request.data)
    id = int(params['id'])
    pts = FrameData[id]['keypoints']  # 256x300 으로 이게 아마 row 부터 전송이 될 수 있음. 받는 곳에서 이것도 다시 확인이 필요함. 수퍼 글루가 아니면 트랜스포즈 해서 넣는것도 방법일 듯.
    # res = str(base64.b64encode(pts))
    json_data = ujson.dumps({'pts': pts.tolist(), 'n': len(pts)})
    return json_data

@app.route("/getDesc", methods=['POST'])
def getDesc():
    ##이것만 요청하는 경우에는 속ㄷ고가 상당히 빠름. 0.006초 정도, 그렇다면 포인트도 분리한다면?
    ##이것은 받는쪽이나 보내는쪽에서 인코딩, 디코딩 하는 시간도 고려해야 한다는 것을 의미하는듯
    params = ujson.loads(request.data)
    id = int(params['id'])
    desc1 = FrameData[id]['descriptors']  # 256x300 으로 이게 아마 row 부터 전송이 될 수 있음. 받는 곳에서 이것도 다시 확인이 필요함. 수퍼 글루가 아니면 트랜스포즈 해서 넣는것도 방법일 듯.
    #res = str(base64.b64encode(desc1))
    #json_data = ujson.dumps({'desc': res})
    return bytes(desc1)
"""

@app.route("/supergluematch", methods=['POST'])
def supergluematch():
    """
    start = time.time()
    print("Match=Start")
    # matchIDX = matchGlobalIDX
    # matchGlobalIDX = (matchGlobalIDX + 1) % NUM_MAX_MATCH
    # print("Match=Start=%d" % (matchIDX))
    ##data 처리
    params = ujson.loads(request.data)
    id1 = int(params['id1'])
    id2 = int(params['id2'])
    ####data 불러오기
    # if id1 != prevID1 :
    data1 = keyframe2tensor(FrameData[id1], device0, '0')
    #    prevID1 = id1
    # if id2 != prevID2:
    data2 = keyframe2tensor(FrameData[id2], device0, '1')
    #    prevID2 = id2

    pred = matching({**data1, **data2})
    matches0 = pred['matches0'][0].cpu().numpy()
    matches1 = pred['matches1'][0].cpu().numpy()
    """
    """
    #match 정보 저장
    MatchData[id1] = {}
    MatchData[id1][id2] = matches0

    MatchData[id2] = {}
    MatchData[id2][id1] = matches1
    """
    # 딕셔너리 키 traverse
    # for key in FrameData.keys():
    #    print(key)

    # json_data = ujson.dumps({'res': 0})
    # print("Match=End : id1 = %d, id2 = %d time = %f %d" % (id1, id2, time.time() - start, len(matches0)))

    #json_data = ujson.dumps({'res': matches0.tolist(), 'n': len(matches0)})
    return ''  #json_data





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

    opt = parser.parse_args()
    device0 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device3 = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

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
    # matching=[]
    # for i in range(NUM_MAX_MATCH):
    #    matching.append(Matching(config).eval().to(device0))

    # flann based matcher

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher()  #

    UserData = {}
    MapData = {}

    mappingserver_addr = "http://143.248.96.81:35006/NotifyNewFrame"
    mappingserver_addr2 = "http://143.248.96.81:35006/ReceiveMapData"
    mappingserver_connect = "http://143.248.96.81:35006/connect"
    depthserver_addr = "http://143.248.95.112:35005/depthestimate"
    semanticserver_addr = "http://143.248.95.112:35006/segment"

    ConditionVariable = threading.Condition()
    ConditionVariable2 = threading.Condition()

    messages = []
    messages2 = []

    th1 = threading.Thread(target=work, args=(ConditionVariable, ConditionVariable2, matching, messages, messages2))
    th1.start()
    th2 = threading.Thread(target=work2, args=(ConditionVariable2, messages2))
    th2.start()
    # th1.join()

    print('Starting the API')
    # app.run(host=opt.ip, port=opt.port)
    # app.run(host=opt.ip, port = opt.port, threaded = True)
    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()
