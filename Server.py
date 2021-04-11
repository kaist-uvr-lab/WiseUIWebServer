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


##import super glue and super point
##################################################
import pickle
import struct
from module.User    import User
from module.Map     import Map
from module.Message import Message

##multicast
from socket import *

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
            desc1 = Frames[lastID]['descriptors']#.transpose()
            desc2 = Frames[id]['descriptors']#.transpose()
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


class DataMessenger:#(threading.Thread):
    def __init__(self, addr, prior, ratio):
        #threading.Thread.__init__(self)
        self.addr = addr
        self.bdata = False
        self.prior = prior
        self.ratio = ratio
    def setData(self, addr2, data):
        self.addr2 = addr2
        self.data = data
        #self.bdata = True
    def run(self):
        #if self.bdata:
        requests.post(self.addr + self.addr2, self.data)
        #self.bdata = False

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
        lastID = int(message.id)

        addr2 = "?user="+message.user+"&map="+map+"&id="+message.id
        for messenger in PreprocessingServerList:
            #messenger.setData(addr2, '')
            #messenger.run()
            if lastID % messenger.ratio == 0:
                requests.post(messenger.addr + addr2,'')#message.data
        end = time.time()
        print("data send id = %d %f"%(lastID, end-start))
        #requests.post(depthserver_addr+addr2, message.data)
        #requests.post(semanticserver_addr+"?map="+map+"&id="+str(lastID), message.data)

def work1(condition1, condition2, queue, queue2):
    print("Start Message Processing Thread")
    while True:
        condition1.acquire()
        condition1.wait()
        message = queue[-1]
        condition1.release()
        ##### 처리 시작
        ##message id = time stamp -> string(id)
        start = time.time()
        map = MapData[message.map]
        user = UserData[message.user]
        id = map.IncreaseID()
        user.AddData(id, message.id)
        Frame = {}
        Frame['bimage'] = message.data #인코딩된 바이트 형태
        Frame['binfo'] = user.info
        map.Frames[id] = Frame
        message.id = id
        ##id 관리 및 user 정보에 추가
        #lastid = MapData[message.map].AddFrame(Frame)

        end2 = time.time()
        #requests.post(mappingserver_addr, ujson.dumps({'id': message.id, 'user': message.user}))
        end3 = time.time()

        queue2.append(message)
        condition2.acquire()
        condition2.notifyAll()
        condition2.release()

    print("End Message Processing Thread")

#######################################################
@app.route("/ConnectServer", methods=['POST'])
def ConnectServer():
    #print(request.remote_addr)
    #print(request.environ['REMOTE_PORT'])
    data = ujson.loads(request.data)
    port = data['port']
    datakey = data['key']
    prior = int(data['prior'])
    ratio = int(data['ratio'])
    addr = 'http://'+request.remote_addr+':'+str(port)+'/Receive'

    if datakey not in PreSererKeys:
        t = DataMessenger(addr, prior, ratio)
        #t.start()
        PreprocessingServerList.append(t)
        PreSererKeys.append(datakey)
        sorted(PreprocessingServerList, key=lambda messenger:messenger.prior)
        print("Connect %s %d %d"%(addr, prior, ratio))
    return ''

@app.route("/Connect", methods=['POST'])
def Connect():
    #print(request.remote_user)
    #print(request.remote_addr)
    #print(request.environ['REMOTE_PORT'])

    data = ujson.loads(request.data)
    id = data['userID']
    map = data['mapName']
    bMapping = data['bMapping']
    bManager = data['bManager']
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
        if bManager:
            print("Connect Map Manager")
        user = User(id, map, bMapping, bManager, fx, fy, cx, cy, w, h, info)#cameraParam, imgSize)
        UserData[user.id] = user
        #print('Connect Map = %s %s' % (UserData[user.id].id, MapData[map].name))
    else:
        user = UserData[id]
    if MapData.get(map) is None:
        MapData[map] = Map(map)

    TempMap = MapData[map]
    TempMap.Connect(id, UserData[user.id])

    print("connect : %s, %d" % (user.id, len(UserData)))
    print(TempMap.Users[user.id]['id'])

    #multi cast code
    #byte
    #connect 1 id, disconnect 2 id, pose 3 id + data
    msg = np.array([1, (TempMap.Users[user.id]['id'])], dtype=np.float32)
    #print(msg)
    sented = mcast_manage_soc.sendto(msg.tobytes(), (MCAST_MANAGE_IP, MCAST_MANAGAE_PORT))
    print("senteed %d , %d"%(sented, len(msg.tobytes())))
    #print('Connect %s'%(user))

    return ""
@app.route("/Disconnect", methods=['POST'])
def Disconnect():
    user = request.args.get('userID')
    print("Disconnect : "+UserData[user].id)
    mapName = request.args.get('mapName')
    tempMap = MapData[mapName]
    tid = tempMap.Users[user]['id']
    tempMap.Disconnect(user)

    #multicast
    msg = np.array([2, tid], dtype=np.float32)
    msg.tobytes()
    mcast_manage_soc.sendto(msg.tobytes(), (MCAST_MANAGE_IP, MCAST_MANAGAE_PORT))
    # multicast

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

    ##delete data
    del map.Users

    fname = os.path.dirname(os.path.realpath(__file__))+'/map/'+mapname+'.bin'
    #f = open(fname, 'w')
    #f.close()
    f = open(fname, "wb+")
    pickle.dump(map, f)
    f.close()

    ##add manager id



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
    ts = request.args.get('id') #string

    message = Message(user, map.name, ts, request.data)
    end = time.time()
    #print("Receive Time : %f = %d" % (end - start, id))
    messages.append(message)
    ConditionVariable.acquire()
    ConditionVariable.notify()
    ConditionVariable.release()
    return ''#struct.pack('>i',5)#int(id).to_bytes(4, 'little')

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
    id = request.args.get('id')
    key = request.args.get('key')
    attr = request.args.get('attr', 'Frames')

    #map.Frames[id][key] = request.data
    if key == 'bdesc':
        darray = np.frombuffer(request.data, dtype=np.float32)
        da = darray.reshape((-1, 256))
        map.Frames[id]['descriptors'] = da
    elif attr == 'Models':
        model = getattr(map, attr)
        if model.get(id) is None:
            model[id] = {}
    #if key == 'refid':
    #    print(type(request.data))
    getattr(map, attr)[id][key] = request.data

    return "a"

@app.route("/SendData", methods=['POST'])
def SendData():
    map = MapData[request.args.get('map')]
    attr = request.args.get('attr', 'Frames')
    id = request.args.get('id', -1)
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

    desc1 = map.Frames[id1]['descriptors']#.transpose()
    desc2 = map.Frames[id2]['descriptors']#.transpose()
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
    print("%d %d : %d"%(id1, id2, success))
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

    opt = parser.parse_args()
    device0 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device3 = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")


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

    ##Preprocessing Server
    PreprocessingServerList = []
    PreSererKeys = []

    if opt.MAP:
        StartMap(opt.MAP)

    th1 = threading.Thread(target=work1, args=(ConditionVariable, ConditionVariable2, messages, messages2))
    th1.start()
    th2 = threading.Thread(target=work2, args=(ConditionVariable2, messages2))
    th2.start()
    #th3 = threading.Thread(target=work3, args=(ConditionVariable2, messages2))
    #th3.start()

    #run echo server

    ##mutli cast
    mcast_manage_soc = socket(AF_INET, SOCK_DGRAM)
    mcast_manage_soc.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, 4)
    MCAST_MANAGE_IP = '235.26.17.10'
    MCAST_MANAGAE_PORT = 37000
    #multi_soc.sendto('Multicasting',('235.26.17.10',37000))


    print('Starting the API')
    # app.run(host=opt.ip, port=opt.port)
    # app.run(host=opt.ip, port = opt.port, threaded = True)
    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()
