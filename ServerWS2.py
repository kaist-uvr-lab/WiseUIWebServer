import threading
import ujson
import time
import numpy as np

import socketio
from flask import Flask, request, app
from gevent.pywsgi import WSGIServer, WSGIHandler
from geventwebsocket.handler import WebSocketHandler

import argparse
import os

import pickle
import struct
from module.ProcessingTime import ProcessingTime
from module.Device import Device
from module.Scheduler import Scheduler
from module.Keyword import Keyword as KeywordData
from module.User import User
from module.Map import Map
from module.Message import Message

from socket import *
import base64

def SendNotification(keyword, id, src, type2, length,ts,ts2):
    try:
        if keyword in SchedulerData:

            #if 'client_' in src and keyword == 'ReferenceFrame':
            #    print("test = ",src,datetime.now())

            ts1 = time.time()
            json_data = ujson.dumps({'keyword': keyword, 'type1': 'notification', 'type2': type2, 'id': id, 'ts': ts, 'ts2':ts2, 'length':length , 'src': src})
            for key in SchedulerData[keyword].broadcast_list.keys():

                bSend = True
                desc = SchedulerData[keyword].broadcast_list[key]

                if desc.bSchedule:
                    #print('error case???')
                    #print('skiping',DeviceData[src].Skipping[desc.name])
                    #print('sending',DeviceData[src].Sending[keyword])
                    DeviceData[src].Sending[keyword] += 1
                    #if DeviceData[src].Sending[keyword] % DeviceData[src].Skipping[desc.name] != 0:
                    #    bSend = False

                    #print("Scheduling = ", src, key, DeviceData[src].Sending[keyword],DeviceData[src].Skipping[desc.name], bSend)

                if bSend:
                    #print(json_data)
                    UDPServerSocket.sendto(json_data.encode(), desc.addr)

            if src in SchedulerData[keyword].unicast_list.keys():
                #print(json_data)
                nres = UDPServerSocket.sendto(json_data.encode(), SchedulerData[keyword].unicast_list[src].addr)
            tsend = time.time()
            Data["TS"][keyword]["NOTIFICATION"].add(tsend - ts1, len(json_data))
    except KeyError:
        pass
    except ConnectionResetError:
        pass
    except UnicodeDecodeError:
        pass
    except Exception as e:
        print(e)

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

            print(data)

            if not data.get('ts'):
                sts = 0.0
            else:
                sts = data['ts']

            if method == 'connect':
                if src not in DeviceData:
                    print("Connect new device ", src)
                    DeviceData[src] = Device(src)
                DeviceData[src].addr = address
                DeviceData[src].receive_keyword.add(keyword)

                if keyword not in SchedulerData:
                    SchedulerData[keyword]=Scheduler(keyword)
                print('udpthread',src,keyword,address)
                multi = data['type2']
                SchedulerData[keyword].add_receive_list(DeviceData[src], multi)

            elif method == 'disconnect':
                multi = data['type2']
                SchedulerData[keyword].remove_receive_list(DeviceData[src], multi)

        except KeyError:
            # print("Key Error")
            pass
        except ConnectionResetError:
            # print("connection error")
            pass
        except UnicodeDecodeError:
            # print("unicode error")
            pass
        except Exception as e:
            print(e)

sio = socketio.Server(async_mode='gevent')
app = socketio.WSGIApp(sio, app)

@sio.on('my event2')
def test_data(sid):
    print("asdfasdfasdfasdfasdf")

import sys
@sio.on('download')
def test_download(sid, keyword, src, id, tdata):
    if id == 0:
        path = os.path.dirname(os.path.realpath(__file__))
        f = open(path + '/db/' + keyword + '_' + src + '.bin', 'rb')
        if Data.get(keyword) is None:
            Data[keyword] = {}
        Data[keyword][src] = pickle.load(f)
        f.close()

    try:
        #a = np.random.randint(1, 200, size=5, dtype=np.uint8)
        #a = base64.b64encode(a).decode("utf-8")
        #a = np.frombuffer(Data[keyword][src][id], dtype=np.uint8)
        #print("asdfasdfasdf", keyword, src, id, len(a), type(a), len(tdata), sys.getsizeof(a.tobytes()), sys.getsizeof(a.tolist()))
        a = base64.b64encode(Data[keyword][src][id]).decode("utf-8")
        # return Data[keyword][src][id]
        sio.emit('download data',a)
        #print(len(a), len(Data[keyword][src][id]), len(tdata), a)
    except Exception as e:
        print(e)
    #return Data[keyword][src][id]
@sio.event
def connect(sid,environ, auths):
    print("connected",sid)
    sio.emit('my response', {'data': 'Connected'})

@sio.event
def disconnect(sid):
    print('disconnected')

if __name__ == "__main__":
    ##################################################
    #c = ntplib.NTPClient()
    #response = c.request('europe.pool.ntp.org', version=3)
    ServerTime = time.time()#-response.tx_time
    print('diff', ServerTime)
    ##arguments
    parser = argparse.ArgumentParser(
        description='WISE UI Web Server',
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ip', type=str, default='143.248.6.25',
        help='ip address')
    parser.add_argument(
        '--port', type=int, default=35005,
        help='port number')

    opt = parser.parse_args()

    DeviceData={}
    SchedulerData={}
    KeywordDatas = {}
    Data = {}
    try:
        path = os.path.dirname(os.path.realpath(__file__))
        f = open(path+'/evaluation/processing_time.bin', 'rb')
        Data["TS"] = pickle.load(f)
        f.close()
    except FileNotFoundError:
        Data["TS"] = {}

    Keywords = set()
    nKeywordID = 0

    ##udp socket
    #udpbufferSize = 1024
    #UDPServerSocket = socket(family=AF_INET, type=SOCK_DGRAM)
    #UDPServerSocket.bind((opt.ip, 35001))

    server = WSGIServer((opt.ip,opt.port), app, handler_class=WebSocketHandler)
    server.serve_forever()
    #socketio.run(app, host=opt.ip, port=opt.port)

    print("asdf")