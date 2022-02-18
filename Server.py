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

import datetime
import argparse
import torch
import os
import keyboard

##import super glue and super point
##################################################
import pickle
import struct
from module.ProcessingTime import ProcessingTime
from module.Device import Device
from module.Scheduler import Scheduler
from module.User import User
from module.Map import Map
from module.Message import Message

##multicast
from socket import *

####WSGI
from multiprocessing.pool import Pool
import gevent
from gevent.pywsgi import WSGIServer
from gevent import monkey

def printProcessingTime():
    keys = Data["TS"].keys()
    for key in keys:
        t1 = Data["TS"][key]["IN"]
        t1.update()
        t2 = Data["TS"][key]["OUT"]
        t2.update()
        t3 = Data["TS"][key]["NOTIFICATION"]
        t3.update()
        print("%s==IN==%s" % (key, t1.print()))
        print("%s==OUT==%s" % (key, t2.print()))
        print("%s==NOTI==%s" % (key, t3.print()))
def saveProcessingTime():
    keys = Data["TS"].keys()
    for key in keys:
        t1 = Data["TS"][key]["IN"]
        t1.update()
        t2 = Data["TS"][key]["OUT"]
        t2.update()
        t3 = Data["TS"][key]["NOTIFICATION"]
        t3.update()
    pickle.dump(Data["TS"], open('./evaluation/processing_time.bin', "wb"))

keyboard.add_hotkey("ctrl+p",lambda: printProcessingTime())
keyboard.add_hotkey("ctrl+s",lambda: saveProcessingTime())

def SendNotification(keyword, id, src, type2, ts):
    try:
        if keyword in KeywordAddrLists:
            ts1 = time.time()
            json_data = ujson.dumps({'keyword': keyword, 'type1': 'notification', 'type2': type2, 'id': id, 'ts':ts, 'src': src})
            for addr in KeywordAddrLists[keyword]['all']:
                UDPServerSocket.sendto(json_data.encode(), addr)

            ###Check Scheduling:
            #SchedulerData[keyword]
            #DeviceData[src]
            #if

            addr = KeywordAddrLists[keyword].get(src)
            if addr is not None:
                UDPServerSocket.sendto(json_data.encode(), addr)
            ts2 = time.time()
            Data["TS"][keyword]["NOTIFICATION"].add(ts2-ts1,len(json_data))
    except KeyError:
        a = 0
    except ConnectionResetError:
        a = 0
    except UnicodeDecodeError:
        a = 0

####UDP for notification
def udpthread():
    while True:
        try:
            bytesAddressPair = UDPServerSocket.recvfrom(udpbufferSize)
            t1 = time.time()

            message = bytesAddressPair[0]
            address = bytesAddressPair[1]

            data = ujson.loads(message.decode())
            method = data['type1']
            keyword = data['keyword']
            src = data['src']

            if not data.get('ts'):
                sts = 0.0
            else:
                sts = data['ts']

            if method == 'connect':
                if keyword not in KeywordAddrLists:
                    KeywordAddrLists[keyword] = {}  # set()  # = {}
                    KeywordAddrLists[keyword]['all'] = set()
                if keyword not in SchedulerData:
                    SchedulerData[keyword]=Scheduler(keyword)
                multi = data['type2']
                if multi == 'single':
                    KeywordAddrLists[keyword][src] = address
                else:
                    KeywordAddrLists[keyword]['all'].add(address)
                # print('%s %s %s' % (method, keyword, multi))
                if src not in DeviceData:
                    print("????????????????????????????")
                    DeviceData[src] = Device(src)
                DeviceData[src].receive_keyword.add(keyword)
                SchedulerData[keyword].add_receive_list(DeviceData[src])
            elif method == 'disconnect':
                multi = data['type2']
                if multi == 'single':
                    KeywordAddrLists[keyword].pop(src)
                else:
                    KeywordAddrLists[keyword]['all'].remove(address)
                if src in DeviceData and keyword in SchedulerData:
                    SchedulerData[keyword].remove_receive_list(DeviceData[src])
            #print("S2 = ", keyword, len(SchedulerData[keyword].send_list), len(SchedulerData[keyword].receive_list))
        except KeyError:
            # print("Key Error")
            pass
        except ConnectionResetError:
            # print("connection error")
            pass
        except UnicodeDecodeError:
            # print("unicode error")
            pass
        continue

####UDP for notification

##################################################
# API part
##################################################
app = Flask(__name__)

@app.route("/Disconnect", methods=['POST'])
def Disconnect():
    src = request.args.get('src')
    type = request.args.get('type')
    if src in DeviceData:
        for key in list(DeviceData[src].send_keyword):
            if key in SchedulerData:
                SchedulerData[key].remove_send_list(DeviceData[src])
        #print(DeviceData[src].send_keyword)

    #if type == 'device' and src in Data['device']:
    #    Data['device_capacity'] = Data['device_capacity'] - Data['device'][src]['capacity']
    return ''

@app.route("/Connect", methods=['POST'])
def Connect():

    data = ujson.loads(request.data)
    Tempkeywords = data['keyword'].split(',')
    type1 = data['type1'] #server, device
    type2 = data['type2']
    src = data['src']

    if src not in DeviceData:
        DeviceData[src] = Device(src)
    try:
        capacity = data['capacity']
    except:
        capacity = 0
    #finally:
    #    print('id = ', src, ', capacity= ', capacity, ', type = ', type1)
    DeviceData[src].capacity = capacity
    DeviceData[src].type = type1

    ####Update Sending Keyword
    for keyword in Tempkeywords:
        DeviceData[src].send_keyword.add(keyword)
        DeviceData[src].Skipping[keyword] = 0
        DeviceData[src].Sending[keyword] = 0

        if keyword not in Keywords:
            Keywords.add(keyword)
            Data[keyword] = {}
            Data[keyword]["pair"] = type2
        if keyword not in SchedulerData:
            SchedulerData[keyword] = Scheduler(keyword)
        SchedulerData[keyword].add_send_list(DeviceData[src])

        if Data["TS"].get(keyword) is None:
            Data["TS"][keyword] = {}
            Data["TS"][keyword]["IN"] = ProcessingTime()
            Data["TS"][keyword]["OUT"] = ProcessingTime()
            Data["TS"][keyword]["NOTIFICATION"] = ProcessingTime()

        #print("S1 = ", keyword, len(SchedulerData[keyword].send_list), len(SchedulerData[keyword].receive_list))
    ####Update Sending Keyword

    ####Update Capacity
    """
    temp = {}
    temp['capacity'] = capacity
    if type1 == 'server':
        if capacity > 0 and Data['device_capacity'] >= capacity:
            temp['schedule'] = True
        else:
            temp['schedule'] = False
        for device in Data['device'].keys():
            print(device)
    if type1 == 'device':
        temp['queue'] = {}
        Data['device_capacity'] = Data['device_capacity'] + capacity
        for server in Data['server'].keys():
            print(server)
    Data[type1][src] = temp
    print("Device = ", len(Data['device']), ",Server = ", len(Data['server']))
    """
    ### update server capacity

    return 'a'

@app.route("/Store", methods=['POST'])
def Store():
    # ts
    ts1 = time.time()
    keyword = request.args.get('keyword')
    id = int(request.args.get('id'))
    src = request.args.get('src')
    ts = request.args.get('ts', '0.0')
    type2 = request.args.get('type2','None')

    if keyword in Keywords:
        if Data[keyword].get(src) is None:
            Data[keyword][src] = {}
        Data[keyword][src][id] = bytes(request.data)
        ts2 = time.time()
        SendNotification(keyword, id, src, type2, ts)
        Data["TS"][keyword]["IN"].add(ts2-ts1, len(Data[keyword][src][id]))
    return 'a'#str(id1).encode()

@app.route("/Load", methods=['POST'])
def Load():
    ts1 = time.time()
    keyword = request.args.get('keyword')
    id = int(request.args.get('id'))
    src = request.args.get('src')
    if keyword in Keywords:
        #while Data[keyword][src].get(id) is None:
        #    print("empty key!!")
        ts2 = time.time()
        Data["TS"][keyword]["OUT"].add(ts2 - ts1, len(Data[keyword][src][id]))
        return (Data[keyword][src][id]) #bytes
    return ''
###########################################################################################################################

##################################################
# END API part
##################################################
import signal

def handler(signum, frame):
    a = 0

signal.signal(signal.SIGINT, handler)

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
        '--CONTENT_IP', type=str, default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--CONTENT_PORT', type=int, default=35001,
        help='port number')
    """
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
    """
    opt = parser.parse_args()

    """
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
    """

    # UserData = {}
    # MapData = {}

    DeviceData={}
    SchedulerData={}
    Data = {}
    try:
        path = os.path.dirname(os.path.realpath(__file__))
        f = open(path+'/evaluation/processing_time.bin', 'rb')
        Data["TS"] = pickle.load(f)
        f.close()
    except FileNotFoundError:
        Data["TS"] = {}
    #SERVER, DEVICE CAPACITY for scheduler
    #Data['server']={}
    #Data['device']={}
    #Data['device_capacity'] = 0

    Keywords = set()
    nKeywordID = 0

    ##mutli cast
    mcast_manage_soc = socket(AF_INET, SOCK_DGRAM)
    mcast_manage_soc.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, 32)
    MCAST_MANAGE_IP = '235.26.17.10'
    MCAST_MANAGAE_PORT = 37000

    ##udp socket
    udpbufferSize = 1024
    KeywordAddrLists = {}
    UDPServerSocket = socket(family=AF_INET, type=SOCK_DGRAM)
    UDPServerSocket.bind((opt.ip, 35001))

    th1 = threading.Thread(target=udpthread)
    th1.start()

    print('Starting the API')
    # app.run(host=opt.ip, port=opt.port)
    # app.run(host=opt.ip, port = opt.port, threaded = True)

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app, environ={'wsgi.multithread':False, 'wsgi.multiprocess':True})
    http.serve_forever()
    # gevent.spawn(http.serve_forever())

