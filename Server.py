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

###############################
#Async Echo Server
###############################
import socket

"""
async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    while True:
        # 클라이언트가 보낸 내용을 받기
        data: bytes = await reader.read(1024)
        
        # 받은 내용을 출력하고,
        # 가공한 내용을 다시 내보내기
        peername = writer.get_extra_info('peername')
        print(f"[S] received: {len(data)} bytes from {peername}")
        mes = data.decode()
        print(f"[S] message: {mes}")
        res = mes.upper()[::–1]
        await asyncio.sleep(random() * 2)
        writer.write(res.encode())
        await writer.drain()

async def test(ip, port):
    await asyncio.wait([run_server(ip, port)])

async def run_server(ip, port):
    # 서버를 생성하고 실행
    server = await asyncio.start_server(handler, host=ip, port=port)

    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    async with server:
        # serve_forever()를 호출해야 클라이언트와 연결을 수락합니다.
        await server.serve_forever()

async def udp_echo_server():
    stream = await asyncio_dgram.bind(("127.0.0.1", 8888))

    print(f"Serving on {stream.sockname}")

    data, remote_addr = await stream.recv()
    print(f"Echoing {data.decode()!r}")
    await stream.send(data, remote_addr)

    await asyncio.sleep(0.5)
    print(f"Shutting down server")
"""
bufferSize = 1024
msgFromServer = "Hello UDP Client"
bytesToSend = str.encode(msgFromServer)
def work1():
    print("Start Message Processing Thread = %d"%(len(ConnectedAddrList)))
    while True:
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]
        if len(message)==4:
            code, = struct.unpack('<f', message)
            #code = float.from_bytes(message, byteorder='little', signed=True)
            #print(code)
            if code == 10000.0:
                #ConnectedAddrList.append(address)
                print("aaaa Connect %d"%(len(ConnectedAddrList)))
            else:
                #ConnectedAddrList.remove(address)
                print("Disconnect %d" % (len(ConnectedAddrList)))
        else:
            print("%s=%s"%(address,message))
            data = np.frombuffer(message, dtype = np.float32)
            if data[0] == 1.0:
                id = int(data[2])
                if data[1] ==1.0:
                    tempip = ".".join([str(int(data[4])), str(int(data[5])), str(int(data[6])), str(int(data[7]))])
                    port = int(data[8])
                    ConnectedAddrList[id] = (tempip, port)
                    if data[3] == 1.0:
                        ManagerAddr = (tempip, port)
                    if ManagerAddr is not None:
                        UDPServerSocket.sendto(message, ManagerAddr)
                    print("Connect = %d"%(len(ConnectedAddrList)))
                elif data[1] == 2.0:
                    if ConnectedAddrList.get(id) is not None:
                        del ConnectedAddrList[id]
                        if ManagerAddr is not None:
                            UDPServerSocket.sendto(message, ManagerAddr)
                        print("Disonnect = %d" % (len(ConnectedAddrList)))
                elif data[1] == 3.0:
                    print("Send?? %s, %d"%(ConnectedAddrList[id][0], ConnectedAddrList[id][1]))
                    UDPServerSocket.sendto(message, ConnectedAddrList[id])
                    if ManagerAddr is not None and ConnectedAddrList[id] is not ManagerAddr:
                        UDPServerSocket.sendto(message, ManagerAddr)
            elif data[0] >= 2.0:
                print("%d = %f %f %f"%(len(message), data[0], data[1], data[2]))
                for addr in ConnectedAddrList.values():
                    print("send to %s:%d"%(addr[0], addr[1]))
                    UDPServerSocket.sendto(message, addr)

if __name__ == "__main__":

    ##################################################
    ##arguments
    parser = argparse.ArgumentParser(
        description='WISE UI Echo Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ip', type=str, default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--port', type=int, default=35001,
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
    # run echo server
    print("Run Echo Server")

    ConnectedAddrList = {}
    ManagerAddr = None
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPServerSocket.bind((opt.ip, opt.port))
    #asyncio.run(test(opt.ip, opt.port), debug=True)

    th1 = threading.Thread(target=work1)
    th1.start()
    #th2 = threading.Thread(target=work2, args=(ConditionVariable2, messages2))
    #th2.start()
    #th3 = threading.Thread(target=work3, args=(ConditionVariable2, messages2))
    #th3.start()



