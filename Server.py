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
        try:
            bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
            message = bytesAddressPair[0]
            address = bytesAddressPair[1]
            print(address, message)
            data = ujson.loads(message.decode())
            method = data['type1']
            keyword = data['keyword']
            src = data['src']
            if method == 'connect':
                if keyword not in KeywordAddrLists:
                    KeywordAddrLists[keyword] = {}  # set()  # = {}
                    KeywordAddrLists[keyword]['all'] = set()
                multi = data['type2']
                if multi == 'single':
                    KeywordAddrLists[keyword][src] = address
                else:
                    KeywordAddrLists[keyword]['all'].add(address)
                # print('%s %s %s' % (method, keyword, multi))
            elif method == 'disconnect':
                multi = data['type2']
                if multi == 'single':
                    KeywordAddrLists[keyword].pop(src)
                else:
                    KeywordAddrLists[keyword]['all'].remove(address)
            elif method == 'notification':
                if keyword in KeywordAddrLists:
                    id = data['id']
                    # src = data['src']
                    type2 = data['type2']
                    if 'id2' in data:
                        id2 = data['id2']
                        json_data = ujson.dumps(
                            {'keyword': keyword, 'type1': 'notification', 'type2': type2, 'id': id, 'id2': id2,
                             'src': src})
                    else:
                        json_data = ujson.dumps(
                            {'keyword': keyword, 'type1': 'notification', 'type2': type2, 'id': id, 'src': src})
                    for addr in KeywordAddrLists[keyword]['all']:
                        UDPServerSocket.sendto(json_data.encode(), addr)
                    addr = KeywordAddrLists[keyword].get(src)
                    if addr is not None:
                        UDPServerSocket.sendto(json_data.encode(), addr)
        except ConnectionResetError:
            print("connection error")
        except UnicodeDecodeError:
            print("unicode error")
        continue


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

    opt = parser.parse_args()
    # run echo server
    print("Run Echo Server")

    KeywordAddrLists = {}

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



