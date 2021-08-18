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
app = Flask(__name__)
@app.route("/Disconnect", methods=['POST'])
def Disconnect():

    return ''
@app.route("/Connect", methods=['POST'])
def Connect():

    data = ujson.loads(request.data)
    Tempkeywords = data['keyword'].split(',')
    method = data['type1'] #server, device
    type2 = data['type2']

    for keyword in Tempkeywords:
        print("%s %s %s"%(keyword, method, type2))
        if keyword not in Keywords:
            Keywords.add(keyword)
            Data[keyword] = {}
            """
            if type2 == "raw":
                Data[keyword]['id'] = int(0)
            else:
                Data[keyword]['id'] = int(-1)
            """
    return 'a'

@app.route("/Store", methods=['POST'])
def Store():

    keyword = request.args.get('keyword')
    id1 = int(request.args.get('id'))
    id2 = int(request.args.get('id2',-1))
    src = request.args.get('src')
    type2 = request.args.get('type2','None')
    """
    if keyword == 'Matching':
        print(type2)
    if keyword == "Matches":
        print("1 Matches id %d %d"%(id1, id2))
    """
    if keyword in Keywords:
        if Data[keyword].get(src) is None:
            Data[keyword][src] = {}
        """
        tid = int(Data[keyword]['id']+1)
        if tid == 0:
            tid = id1
        else:
            Data[keyword]['id'] = tid
        id1 = tid
        """
        """
        Data[keyword]['src'] = src  # data['src']
        if id2 >= 0:
            id = str(tid)+","+str(id2)
            Data[keyword][id] = request.data
            json_str = {'keyword': keyword, 'type1': 'notification', 'id': tid, 'id2':id2, 'type2':type2, 'src': src}
        else:
       """
        Data[keyword][src][id1] = request.data
        json_str = {'keyword': keyword, 'type1': 'notification', 'type2':type2, 'id': id1, 'src': src}
        json_data = ujson.dumps(json_str)
        udp_manage_soc.sendto(json_data.encode(), CONTENT_ECHO_SERVER_ADDR)
        #print(data['data'])
    return 'a'#str(id1).encode()

@app.route("/Load", methods=['POST'])
def Load():
    keyword = request.args.get('keyword')
    id1 = int(request.args.get('id'))
    id2 = int(request.args.get('id2', -1))
    src = request.args.get('src')
    if keyword in Keywords:
        """
        if id2 >= 0:
            id = str(id1) + "," + str(id2)
            return bytes(Data[keyword][id])
        else:
        """
        return bytes(Data[keyword][src][id1])
    return ''
###########################################################################################################################

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

    #UserData = {}
    #MapData = {}
    Data={}
    Keywords = set()
    nKeywordID = 0

    ##mutli cast
    mcast_manage_soc = socket(AF_INET, SOCK_DGRAM)
    mcast_manage_soc.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, 32)
    MCAST_MANAGE_IP = '235.26.17.10'
    MCAST_MANAGAE_PORT = 37000

    ##udp socket
    CONTENT_ECHO_SERVER_ADDR = (opt.CONTENT_IP, opt.CONTENT_PORT)
    udp_manage_soc = socket(AF_INET, SOCK_DGRAM)

    print('Starting the API')
    # app.run(host=opt.ip, port=opt.port)
    # app.run(host=opt.ip, port = opt.port, threaded = True)
    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()
