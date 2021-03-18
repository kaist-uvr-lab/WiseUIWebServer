import threading
##################################################
##import super glue and super point
import ujson
import time
import numpy as np
from flask import Flask, request
import requests
from flask_cors import CORS
import base64
import cv2

import argparse
import torch
import os

from superglue.matching import Matching
from superglue.utils import (frame2tensor, keyframe2tensor)
##import super glue and super point
##################################################
import pickle
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

def work3(conv2, queue):
    while True:
        conv2.acquire()
        conv2.wait()
        start = time.time()
        message = queue[-1]
        conv2.release()
        mapname = message.map
        lastID = message.id

        Frames = MapData[mapname].Frames
        #Matches = MapData[mapname].Matches
        #Matches[lastId] = {}
        Frame = Frames[lastID]

        #matchIDs =Frame['ids']
        ids = Frames["ids"] #전체 프레임 리스트를 의미함
        for id in ids:
            desc1 = Frames[lastID]['descriptors'].transpose()
            desc2 = Frames[id]['descriptors'].transpose()
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
            # match 정보 저장
            #Matches[id2] = {}
            #Matches[id2][id1] = matches1

            # print("match %d %d = %d" % (lastId, id, success))
            if success < 20:
                break
            #matchIDs.append(id)
            strid = 'b'+str(id)
            Frame[strid] = good
        print("Match Frames %d, %d"%(lastID, len(ids)))
        #Frame['bids'] = bytes(matchIDs)
        #Frames["ids"].insert(0, message.id)


##work에서 호출하는 cv가 필요함.
##다른 서버에 뿌리는 정보
def work2(conv2, queue):
    print("Start Matching Thread")
    while True:
        conv2.acquire()
        conv2.wait()
        start = time.time()
        message = queue[-1]
        map = message.map
        conv2.release()
        lastID = message.id

        requests.post(depthserver_addr+"?map="+map+"&id="+str(lastID), message.data)
        #requests.post(semanticserver_addr+"?map="+map+"&id="+str(lastID), message.data)

def work1(condition1, condition2, SuperPointAndGlue, queue, queue2):
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
        desc = last_data['descriptors'][0].cpu().detach().numpy()
        kpts = last_data['keypoints'][0].cpu().detach().numpy()
        #scores = last_data['scores'][0].cpu().detach().numpy()
        end1 = time.time()

        user = UserData[message.user]
        Frame = {}
        Frame['descriptors'] = desc  # descriptor를 미리 트랜스 포즈하고나서 수퍼글루를 더 적게 쓰면 그 때만 트랜스 포즈 하도록 하게 하자.
        Frame['bkpts'] = kpts.tobytes()
        Frame['bdesc'] = desc.transpose().tobytes()
        Frame['bimage'] = message.data #인코딩된 바이트 형태
        Frame['binfo'] = user.info
        #Frame['image'] = img
        #Frame['scores'] = scores
        #Frame['keypoints'] = kpts  # last_data['keypoints'][0].cpu().detach().numpy() #에러가능성
        ##id 관리 및 user 정보에 추가
        lastid = MapData[message.map].AddFrame(Frame)
        user.AddData(lastid, message.timestamp)
        message.id = lastid
        n = len(kpts)
        ##mapping server에 전달
        #MapData[message.map].Frames[message.id] = Frame
        end2 = time.time()
        requests.post(mappingserver_addr, ujson.dumps({'id': message.id, 'user': message.user}))
        end3 = time.time()

        queue2.append(message)
        condition2.acquire()
        condition2.notifyAll()
        condition2.release()

        # thtemp =  threading.Thread(target=supergluematch2, args=(SuperPointAndGlue, id))
        # thtemp.start()
        #print("Message Processing = %s %d : %d : %f : " % (message.map, message.id, len(MapData[message.map].Frames), end3 - start))
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
    info = np.array([fx, fy, cx, cy, w, h], dtype=np.float32).tobytes()
    requests.post(mappingserver_connect, ujson.dumps({
        'fx':fx,
        'fy':fy,
        'cx':cx,
        'cy':cy,
        'w':w,
        'h':h,
        'b':bMapping,
        'n':map,
        'u':id
    }))

    """
    user.id
    user['map'] = map
    user['mapping'] = bMapping
    user['camera'] = cameraParam
    user['image'] = imgSize
    """
    if UserData.get(id) is None:
        user = User(id, map, bMapping, fx, fy, cx, cy, w, h, info)#cameraParam, imgSize)
        UserData[user.id] = user
        print('Connect Num = %d' % (len(UserData)))
        #print('Connect Map = %s %s' % (UserData[user.id].id, MapData[map].name))
    else:
        user = UserData[id]
        print('Connect Num = %d' % (len(UserData)))
    if MapData.get(map) is None:
        MapData[map] = Map(map)

    #print('Connect %s'%(user))

    return ""
@app.route("/Disconnect", methods=['POST'])
def Disconnect():
    user = request.args.get('userID')
    print("Disconnect : "+UserData[user].id)
    requests.post(SLAM_SERVER_ADDR+"/Disconnect", ujson.dumps({
        'u': user
    }))
    return ""
@app.route("/reset", methods=['POST'])
def reset():
    print("Reset")
    mapname = request.args.get('map')
    map = MapData[request.args.get('map')]
    map.reset()
    json_data = ujson.dumps({'res': 0})
    requests.post(SLAM_SERVER_ADDR + "/reset", ujson.dumps({'map': mapname}))
    return json_data

@app.route("/RequestSaveMap", methods=['POST'])
def RequestSaveMap():
    mapname = request.args.get('map')
    requests.post(mappingserver_addr3, ujson.dumps({'map': mapname}))
    return ""

@app.route("/SaveMap", methods=['POST'])
def SaveMap():
    data = ujson.loads(request.data)
    mpids = np.frombuffer(base64.b64decode(data['ids']), dtype=np.int32)
    x3ds = np.frombuffer(base64.b64decode(data['x3ds']), dtype=np.float32)
    kfids = np.frombuffer(base64.b64decode(data['kfids']), dtype=np.int32)
    poses = np.frombuffer(base64.b64decode(data['poses']), dtype=np.float32)
    mpidxs = np.frombuffer(base64.b64decode(data['idxs']), dtype=np.int32)

    bmpids = mpids.tobytes()
    bx3ds = x3ds.tobytes()
    bkfids = kfids.tobytes()
    bposes = poses.tobytes()
    bmpidxs = mpidxs.tobytes()

    mapname = request.args.get('map')
    map = MapData[mapname]
    map.MapPoints['bmpids'] = bmpids
    map.MapPoints['bx3ds'] = bx3ds
    map.Frames['bkfids'] = bkfids

    #map.Frames['bposes'] = bposes
    #map.Frames['bmpidxs'] = bmpidxs
    """
    for i in range(len(mpids)):
        id = mpids[i]
        mp = {}
        mp['X3D']= x3ds[i*3:i*3+3]
        map.MapPoints[id] = mp
    """
    sIDX = 0
    for i in range(len(kfids)):
        id = kfids[i]
        map.Frames[id]["bpose"] = poses[i*12:i*12+12].tobytes()
        n = len(map.Frames[id]["descriptors"][0])
        eIDX = sIDX+n
        map.Frames[id]["bmpidx"] = mpidxs[sIDX:eIDX].tobytes()

        print("save %d %d"%(n, len(mpidxs[sIDX:eIDX])))
        sIDX = eIDX


    #for id in kfids:
    #    map.Frames[id]["pose"]

    fname = os.path.dirname(os.path.realpath(__file__))+'/map/'+mapname+'.bin'
    #f = open(fname, 'w')
    #f.close()
    f = open(fname, "wb+")
    pickle.dump(map, f)
    f.close()
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
    mapname = request.args.get('map')
    fname = os.path.dirname(os.path.realpath(__file__)) + '/map/' + mapname + '.bin'
    f = open(fname, 'rb')
    map = pickle.load(f)
    f.close()
    MapData[mapname] = map
    print(len(MapData[mapname].Frames))
    requests.post(mappingserver_addr2, ujson.dumps({'map': mapname}))
    """
    data = f.read().decode()
    f.close()
    DATA = ujson.loads(data)
    n = DATA['total']
    print("LoadMap %d" % (n))
    requests.post(mappingserver_addr2, ujson.dumps(DATA))
    """

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

def StartMap(mapname):
    f = open('./map/'+mapname+'.bin', 'rb')
    map = pickle.load(f)
    f.close()
    MapData[mapname] = map
    print(len(MapData[mapname].Frames))
    requests.post(mappingserver_addr2, "a")

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

@app.route("/AddKeyFrame", methods=['POST'])
def AddKeyFrame():
    map = MapData[request.args.get('map')]
    id = int(request.values.get('id'))
    attr = request.args.get('attr', 'Frames')
    getattr(map, attr)["ids"].insert(0, id)    #.append(id)
    return "a"

@app.route("/ReceiveTrackingData", methods=['POST'])
def ReceiveTrackingData():
    id = int(request.values.get('id'))
    #map = MapData[request.args.get('map')]
    #attr = request.args.get('attr', 'Frames')
    #getattr(map, attr)["ids"].insert(0, id)    #.append(id)
    print("Tracking %d"%(id))
    return "a"


@app.route("/ReceiveAndDetect", methods=['POST'])
def ReceiveAndDetect():
    start = time.time()
    user = request.args.get('user')
    map = MapData[request.args.get('map')]
    id = request.args.get('id') #string

    message = Message(user, map.name, id, request.data)
    end = time.time()
    #print("Receive Time : %f = %d" % (end - start, id))

    messages.append(message)

    ConditionVariable.acquire()
    ConditionVariable.notify()
    ConditionVariable.release()
    return ujson.dumps({'id': id})

####단말에 전송할 데이터 관련 함수
@app.route("/ReceiveFrameID", methods=['POST'])
def ReceiveFrameID():
    userID = request.args.get('user')
    id = int(request.args.get('id'))
    user = UserData[userID]
    user.frameIDs.insert(0, id)
    return ''
@app.route("/SendFrameID", methods=['POST'])
def SendFrameID():
    userID = request.args.get('user')
    #map = MapData[request.args.get('map')]
    user = UserData[userID]
    id = user.frameIDs[0]
    ts = user.TimeStamps[id]
    """
    if id is not -1:
        ts = map.TimeStamps[id]
    else:
        ts = "invalid"
    """
    return ujson.dumps({'id':id, 'ts':ts})

"""
##Send와 Receive를 자유롭게 이용하기 위해서는
1) map 정보, 2)attr 정보(프레임인지, 맵포인트인지) 3)id 4)key : 해당 데이터의 attr
다만, 맵포인트의 경우 이 함수로 정보를 변경하면 호출이 너무 많아짐..
테스트가 필요함.
"""
@app.route("/ReceiveData", methods=['POST'])
def ReceiveData():
    map = MapData[request.args.get('map')]
    id = int(request.args.get('id'))
    key = request.args.get('key')
    attr = request.args.get('attr', 'Frames')
    getattr(map, attr)[id][key] = request.data
    #map.Frames[id][key] = request.data
    map.UpdateIDs[key] = id
    return "a"

@app.route("/SendData", methods=['POST'])
def SendData():
    map = MapData[request.args.get('map')]
    attr = request.args.get('attr', 'Frames')
    id = int(request.args.get('id', -1))
    key = request.args.get('key')

    if id == -1:
        data = getattr(map, attr)[key]
    else:
        data = getattr(map, attr)[id][key]
    #data = map.Frames[id][key] #이걸 전부 바이트로 변환???
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
    print("%d %d : %d"%(desc1.shape[0], desc2.shape[0], success))
    # print("match : id = %d, %d, res %d"%(id1, id2, success))
    # print("KnnMatch time = %f , %d %d" % (time.time() - start, len(matches), nres))
    # print("featurematch %d : %d %d"%(len(good), len(desc1), len(desc2)))
    # res = str(base64.b64encode(good))
    return good.tobytes()#ujson.dumps({'matches': good.tolist()})
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
        '--ip', type=str, default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--port', type=int, default=35005,
        help='port number')

    parser.add_argument(
        '--SLAM_SERVER', type=str,
        help='http://xxx.xxx.xxx.xxx:xxxx')

    parser.add_argument(
        '--DEPTH_SERVER', type=str,
        help='http://xxx.xxx.xxx.xxx:xxxx')

    parser.add_argument(
    '--SEGMENTATION_SERVER', type=str,
        help='http://xxx.xxx.xxx.xxx:xxxx')

    parser.add_argument(
        '--MAP', type=str,
        help='load map name')

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
    device0 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device3 = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

    ###LOAD MIDAS
    """
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.to(device0)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform
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

    SLAM_SERVER_ADDR = opt.SLAM_SERVER
    DEPTH_SERVER_ADDR = opt.DEPTH_SERVER
    SEGMENTATION_SERVER_ADDR = opt.SEGMENTATION_SERVER
    mappingserver_addr = SLAM_SERVER_ADDR+'/NotifyNewFrame'
    mappingserver_addr2 = SLAM_SERVER_ADDR+'/ReceiveMapData'
    mappingserver_addr3 = SLAM_SERVER_ADDR + '/SaveMapData'
    mappingserver_connect = SLAM_SERVER_ADDR+'/connect'
    depthserver_addr = DEPTH_SERVER_ADDR+'/depthestimate'#95.112, #6.143
    semanticserver_addr = SEGMENTATION_SERVER_ADDR+'/segment'

    ConditionVariable = threading.Condition()
    ConditionVariable2 = threading.Condition()

    messages = []
    messages2 = []

    if opt.MAP:
        StartMap(opt.MAP)

    th1 = threading.Thread(target=work1, args=(ConditionVariable, ConditionVariable2, matching, messages, messages2))
    th1.start()
    th2 = threading.Thread(target=work2, args=(ConditionVariable2, messages2))
    th2.start()
    #th3 = threading.Thread(target=work3, args=(ConditionVariable2, messages2))
    #th3.start()

    print('Starting the API')
    # app.run(host=opt.ip, port=opt.port)
    # app.run(host=opt.ip, port = opt.port, threaded = True)
    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()
