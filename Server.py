import threading
import ujson
import time
import numpy as np
from flask import Flask, request
import requests
import cv2
from socket import *
from module.Message import Message
import argparse
import torch

from superglue.SuperPointFeature2D import SuperPointFeature2D
from superglue.matching import Matching
from superglue.utils import (frame2tensor, keyframe2tensor)
from superglue.superpoint2 import SuperPoint2

#WSGI
from gevent.pywsgi import WSGIServer

#Thread Pool
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
#from multiprocessing import Pool

##################################################
# API part
##################################################

from module.ProcessingTime import ProcessingTime
import pickle
import keyboard

def printProcessingTime():
    Data["Time"].update()
    print(Data["Time"].print())
def saveProcessingTime():
    Data["Time"].update()
    pickle.dump(Data["Time"], open('./evaluation/processing_time.bin', "wb"))
#keyboard.add_hotkey("ctrl+p",lambda: printProcessingTime())
#keyboard.add_hotkey("ctrl+s",lambda: saveProcessingTime())

#app = Flask(__name__)
#cors = CORS(app)
#CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

import os
def matchingthread():
    while True:
        if len(msgqueue2) == 0:
            continue
        data = msgqueue2.pop(0)
        id1 = data['id']
        keyword = data['keyword']
        src = data['src']
        res = sess.post(FACADE_SERVER_ADDR + "/Load?keyword=" + keyword + "&id=" + str(id1), "")
        id2 = int(res.content)
        print("%s %d %d"%(keyword, id1, id2))
"""
list_quality = [100,90,80,70,60,50,40,30,20,10,1]
desc_dst = {}
for q in list_quality:
    desc_dst[q] = []
"""
def predict(frame_tensor):
    with torch.no_grad():
        last_data = module({'image': frame_tensor})
        ori_kp = last_data['keypoints'][0].cpu().detach().numpy()
        return ori_kp

def processingthread2(data):
    t1 = time.time()
    id = data['id']
    strid = str(id)
    keyword = data['keyword']
    src = data['src']
    type2 = data['type2']

    Data[keyword][id] = True
    if Data['ReqSuperPoint'][id] and Data['Image'][id]:

        res = sess.post(FACADE_SERVER_ADDR + "/Load?keyword=" + "Image" + "&id=" + str(id)+"&src="+src, "")
        img_array = np.frombuffer(res.content, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

        ori_data = []

        frame_tensor = frame2tensor(gray, device)
        with torch.no_grad():
            """
            ta1 = time.time()
            last_data = matching.superpoint({'image': frame_tensor})
            tb1 = time.time()
            last_data['image'] = frame_tensor
            ori_kp = last_data['keypoints'][0].cpu().detach().numpy()
            ori_desc = last_data['descriptors'][0].cpu().detach().numpy().transpose()
            #ori_data = list(zip(ori_kp, ori_desc))
            #ori_data = sorted(ori_data, key = lambda x: (x[0][0], x[0][1]))
            """
            #ta2 = time.time()
            #last_data = SuperPoint({'image': frame_tensor})
            #ori_kp = last_data['keypoints'][0].cpu().detach().numpy()
            #tb2 = time.time()
            #print("TEST = ", tb1-ta1, tb2-ta2)
            #res = module_pool.apply_async(predict, frame_tensor)
            kpts = predict(frame_tensor)
            #print(res.get(timeput=1))
            sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Keypoints&id=" + strid + "&src=" + src, kpts.tobytes())

        t2 = time.time()
        print("processing time", t2-t1)

    #Data["Time"].add(t2-t1, len(ori_kp))

    """
    print("IMAGE ", id)
    for q in list_quality:
        encoded_param = [int(cv2.IMWRITE_JPEG_QUALITY),q]
        res, buffer = cv2.imencode('.jpg', img_cv, encoded_param)
        decoded_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2GRAY)

        ##super point
        frame_tensor = frame2tensor(img, device)
        with torch.no_grad():
            #jpeg_data = matching.superpoint({'image': frame_tensor})
            jpeg_data = matching.superpoint({'image': frame_tensor, 'keypoints':last_data['keypoints']})
            #jpeg_data['image'] = jpeg_tensor
            t1 = jpeg_data['keypoints'][0].cpu().detach().numpy()
            t2 = jpeg_data['descriptors'][0].cpu().detach().numpy().transpose()

            #jpeg_data = list(zip(t1,t2))
            #jpeg_data = sorted(jpeg_data, key=lambda x: (x[0][0], x[0][1]))
            #for idx in range(len(t1)):
                #print(idx, ori_kp[idx],t1[idx])

            for idx in range(len(t2)):
                desc_dst[q].append(cv2.norm(ori_desc[idx], t2[idx], cv2.NORM_L2))

        print(q, np.mean(desc_dst[q]))
            #kpts.append(t1)
            #kpts.append(t2)
    """


def processingthread(Data):

    ##optical flow
    old_gray = None
    mask = None
    p0 = None
    bOpt = False

    print("Start Message Processing Thread")
    while True:
        ConditionVariable.acquire()
        datas = []
        while len(msgqueue1) > 0:
            datas.append(msgqueue1.pop(0))
        ConditionVariable.wait()

        for data in datas:
            id = data['id']
            strid = str(id)
            keyword = data['keyword']
            src = data['src']
            type2 = data['type2']

            #res = sess.post(FACADE_SERVER_ADDR + "/Load?keyword=" + keyword + "&id=" + str(id), "")
            if keyword == "Matching":
                id2 = data['id2']
                desc1 = Data["Image"][id]['Descriptors']
                desc2 = Data["Image"][id2]['Descriptors']
                print("%s %d %d" % (keyword, id, id2))
                matches = desc_matcher.knnMatch(desc1, desc2, k=2)
                good = np.zeros([(len(matches)), 1], dtype=np.int32)

                for i, (m, n) in enumerate(matches):
                    if m.distance < 0.7 * n.distance:
                        good[i] = m.trainIdx
                    else:
                        good[i] = -1
                sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Matches&type2="+type2+"&id=" + strid +"&id2="+str(id2)+ "&src=" + src, good.tobytes())


            elif keyword == 'Image':
                print("Superpoint")
                res = sess.post(FACADE_SERVER_ADDR + "/Load?keyword=" + keyword + "&id=" + str(id), "")
                img_array = np.frombuffer(res.content, dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

                #fids = list(Data[keyword].keys())
                fdata = {}
                Data[keyword][id] = fdata
                Data[keyword][id]['rgb'] = img_cv
                Data[keyword][id]['gray'] = img

                a1 = time.time()
                detector.detectAndCompute(img)
                a2 = time.time()
                print("Superpoint = %f"%(a2-a1))


                ##super point
                frame_tensor = frame2tensor(img, device)
                with torch.no_grad():
                    last_data = matching.superpoint({'image': frame_tensor})
                    last_data['image'] = frame_tensor
                    kpts = last_data['keypoints'][0].cpu().detach().numpy()
                    desc = last_data['descriptors'][0].cpu().detach().numpy()
                    Data[keyword][id]['Descriptors'] = desc.transpose()
                    Data[keyword][id]['Keypoints'] = kpts

                sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Keypoints&id=" + strid + "&src=" + src, kpts.tobytes())
                sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Descriptors&id=" + strid + "&src=" + src,desc.transpose().tobytes())
                ##super point


        #print("queue %d"%(len(msgqueue1)))
        continue

        if len(msgqueue1) == 0:
            continue
        data = msgqueue1.pop(0)
        #ConditionVariable.release()

        start = time.time()
        # 처리 시작
        id = data['id']
        keyword = data['keyword']
        src = data['src']
        res =sess.post(FACADE_SERVER_ADDR + "/Load?keyword="+keyword+"&id="+str(id),"")

        if src not in Data[keyword]['src']:
            Data[keyword]['src'].add(src)
            Data[keyword][src] = {}

        if keyword == 'Image':
            img_array = np.frombuffer(res.content, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

            fids = list(Data[keyword][src].keys())
            Data[keyword][src][id] = {}
            Data[keyword][src][id]['rgb'] = img_cv
            Data[keyword][src][id]['gray'] = img

            ##super point
            frame_tensor = frame2tensor(img, device)
            last_data = matching.superpoint({'image': frame_tensor})
            last_data['image'] = frame_tensor
            kpts = last_data['keypoints'][0].cpu().detach().numpy()
            desc = last_data['descriptors'][0].cpu().detach().numpy()
            Data[keyword][src][id]['Descriptors'] = desc.transpose()
            Data[keyword][src][id]['Keypoints'] = kpts
            strid = str(id)
            sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Keypoints&id=" + strid + "&src=" + src, kpts.tobytes())
            sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Descriptors&id=" + strid + "&src=" + src, desc.transpose().tobytes())
            ##super point

            if len(fids) % 3 == 0:

                ##optical flow
                """
                old_gray = img.copy()
                mask = np.zeros_like(img_cv)
                p0 = kpts.reshape(-1, 1, 2)
                i0 = np.arange(len(p0)).reshape(-1, 1, 1)
                """
                ##optical flow


                """
                ##orb
                kp1, des1 = orb.detectAndCompute(img, None)
                ##orb
                frame = img_cv.copy()
                for i in range(0, len(kp1)):
                    x1, y1 = kp1[i].pt
                    frame = cv2.circle(frame, (int(x1), int(y1)), 3, (255,0,0), -1)
                for i in range(0, len(kpts)):
                    frame = cv2.circle(frame, (kpts[i][0], kpts[i][1]), 2, color[i].tolist(), -1)
                cv2.imshow("feature="+src,frame)
                cv2.waitKey(1)
                """

                """
                ##desc matching
                if len(fids) > 3:
                    id1 = id
                    id2 = fids[-3]
                    desc1 = Data[keyword][src][id1]['Descriptors']
                    desc2 = Data[keyword][src][id2]['Descriptors']
                    pts1 = Data[keyword][src][id1]['Keypoints']
                    pts2 = Data[keyword][src][id2]['Keypoints']
                    pts_new = np.zeros(pts1.shape, dtype = pts1.dtype)
                    matches = desc_matcher.knnMatch(desc1, desc2, k=2)
                    good = np.zeros([(len(matches)),1], dtype=np.int8)

                    for i, (m, n) in enumerate(matches):
                        if m.distance < 0.7 * n.distance:
                            pts_new[i] = pts2[m.trainIdx]
                            good[i] = 1
                        else:
                            good[i] = 0
                    pts1 = pts1.reshape(-1,1,2)
                    pts_new = pts_new.reshape(-1, 1, 2)
                    pts_old = pts1[good==1]
                    pts_new = pts_new[good == 1]

                    # draw the tracks
                    frame = img_cv.copy()
                    for i, (new, old) in enumerate(zip(pts_new, pts_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)

                    cv2.imshow('desc=' + src, frame)
                    cv2.waitKey(1)
                ##desc matching
                """
            """
            ##Suplerglue test
            #Data['Frame'][id] = last_data
            if len(fids) > 20:
                del Data['Frame'][fids[0]]
            tmatch = 0.0
            if len(fids) > 4:
                data1 = Data['Frame'][fids[-2]]
                data2 = Data['Frame'][fids[-1]]
                data1 = {key + '0': data1[key] for key in SuperPointKeywords}
                data2 = {key + '1': data2[key] for key in SuperPointKeywords}

                t5 = time.time()
                pred = matching({**data1, **data2})
                t6 = time.time()
                tmatch = t6-t5
            ##Suplerglue test
            """

            topt_s = time.time()
            """
            Data[keyword][id] = {}
            Data[keyword][id]['rgb'] = img_cv
            Data[keyword][id]['gray'] = img
            Data[keyword][id]['src'] = src
            """

            #Data[keyword][id]['src'] = src

            """
            if bOpt is not True or id % 3 == 0:
                old_gray = img.copy()
                mask = np.zeros_like(img_cv)
                bOpt = True
                p0 = kpts.reshape(-1, 1, 2)
                i0 = np.arange(len(p0)).reshape(-1,1,1)
            """
            """
            ##Optical flow matching
            if len(fids) % 3 is not 0:
                frame = img_cv.copy()
                frame_gray = img.copy()
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                good_new = p1[st == 1]
                good_old = p0[st == 1]
                good_idx = i0[st == 1]
                print("%d %d"%(id, len(good_new)))
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    idx = int(good_idx[i])
                    mask = cv2.line(mask, (a, b), (c, d), color[idx].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 3, color[idx].tolist(), -1)

                oimg = cv2.add(frame, mask)
                cv2.imshow('opticalflow='+src, oimg)
                cv2.waitKey(1)
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                i0 = good_idx.reshape(-1, 1, 1)
            ##Optical flow matching
            """
            """
            if len(fids) > 2:

                old_gray = Data[keyword][fids[-2]]['gray']
                frame_gray = Data[keyword][fids[-1]]['gray']
                mask = np.zeros_like(img_cv)
                frame = img_cv.copy()
                n1 = kpts.shape[0]
                n2 = kpts.shape[1]
                p1 = kpts.reshape(n1, 1, n2)

                p0, st, err = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)
                print(p1[30])
                print(kpts[30])

                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                oimg = cv2.add(frame, mask)
                cv2.imshow('frame', oimg)
                cv2.waitKey(1)
            """

            topt_e = time.time()
            ##Opticalflow test

            ##Opticalflow test

            """
            desc = last_data['descriptors'][0].cpu().detach().numpy()
            kpts = last_data['keypoints'][0].cpu().detach().numpy()
            Data[keyword][id]['Descriptors'] = desc
            Data[keyword][id]['Keypoints'] = kpts
            strid = str(id)
            sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Keypoints&id="+strid+"&src="+src, kpts.tobytes())
            sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Descriptors&id=" +strid+ "&src=" + src, desc.transpose().tobytes())
            """

        elif keyword == 'Matching':
            id2 = int(res.content)
            desc1 = Data[keyword][src][id]['Descriptors']
            desc2 = Data[keyword][src][id2]['Descriptors']

            matches = desc_matcher.knnMatch(desc1, desc2, k=2)
            good = np.zeros([(len(matches)), 1], dtype=np.int32)

            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    good[i] = m.trainIdx
                else:
                    good[i] = -1
            sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Matches&id=" + strid + "&src=" + src, good.tobytes())

        end = time.time()
        print("Super Point Processing = %s = %d : %f %f" % (id, len(msgqueue1), end - start, topt_e-topt_s))
        # processing end


def udpthread():
    bufferSize = 1024
    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        data = ujson.loads(message.decode())

        with ThreadPoolExecutor() as executor:
            executor.submit(processingthread2, data)

        """
        ConditionVariable.acquire()
        msgqueue1.append(data)
        ConditionVariable.notify()
        ConditionVariable.release()
        """
        """
        if data['keyword'] == 'Image':
            msgqueue1.append(data)
        elif data['keyword'] == 'Matching':
            msgqueue2.append(data)
        """
        #ConditionVariable.acquire()
        #ConditionVariable.notify()
        #ConditionVariable.release()


import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    ##################################################
    ##basic arguments
    parser = argparse.ArgumentParser(
        description='WISE UI Web Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--RKeywords', type=str,
        help='Received keyword lists')
    parser.add_argument(
        '--SKeywords', type=str,
        help='Sendeded keyword lists')
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
        '--ECHO_SERVER_IP', type=str, default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--ECHO_SERVER_PORT', type=int, default=35001,
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
        '--max_keypoints', type=int, default=1000,
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

    """
    modules = []
    for i in range(1,9):
        matching = Matching(config).eval().to(device)
        modules.append(matching)
    print(len(modules))
    """

    mp.set_start_method('spawn')
    module = SuperPoint2(config.get('superpoint', {})).eval().to(device)
    #module.share_memory()
    #module_pool = mp.Pool(processes=1)

    SuperPointKeywords = ['keypoints', 'scores', 'descriptors','image']
    #detector = SuperPointFeature2D(True)
    ##test
    print("init start")

    rgb = np.random.randint(255, size=(640, 480), dtype=np.uint8)
    test_tesnsor = frame2tensor(rgb, device)
    #detector.compute(rgb)
    #for module in modules:
    module({'image': test_tesnsor})
    #matching.superpoint({'image': test_tesnsor})

    print("init end")

    ##LOAD SuperGlue & SuperPoint

    ##ORB
    orb = cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )
    ##ORB

    ##Descriptor-based matching
    #flann-based
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #bf-based
    bf = cv2.BFMatcher()
    desc_matcher = bf
    ##Descriptor-based matching

    ##Opticalflow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (500, 3))
    ##Opticalflow

    Data = {}
    try:
        path = os.path.dirname(os.path.realpath(__file__))
        f = open(path + '/evaluation/processing_time.bin', 'rb')
        Data["Time"] = pickle.load(f)
        f.close()
    except FileNotFoundError:
        Data["Time"] = ProcessingTime()

    msgqueue1 = []
    msgqueue2 = []

    ##Echo server
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    ReceivedKeywords= opt.RKeywords.split(',')
    SendKeywords = opt.SKeywords
    sess = requests.Session()
    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        #'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'id':'FeatureServer', 'type1':'Server', 'type2':'test','keyword': SendKeywords, 'Additional':None
    }))
    ECHO_SERVER_ADDR = (opt.ECHO_SERVER_IP, opt.ECHO_SERVER_PORT)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    for keyword in ReceivedKeywords:
        temp = ujson.dumps({'type1':'connect', 'keyword':keyword, 'src':'FeatureServer', 'type2':'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[keyword]={}
        Data[keyword]['src'] = set()
    Data['Frame'] = {}
    #Echo server connect

    ConditionVariable = threading.Condition()

    th1 = threading.Thread(target=udpthread)
    #th2 = threading.Thread(target=processingthread, args=(Data,))
    #th3 = threading.Thread(target=matchingthread)
    th1.start()
    #th2.start()
    #th3.start()
    # thread

    #http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    #http.serve_forever()



