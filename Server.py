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
import torch, torchvision

#WSGI
from gevent.pywsgi import WSGIServer

#Thread Pool
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

##CSAILVision
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from mit_semseg.config import cfg

##################################################
# API part
##################################################

app = Flask(__name__)
#cors = CORS(app)
#CORS(app, resources={r'*': {'origins': ['143.248.96.81', 'http://localhost:35005']}})

import os

#https://snowdeer.github.io/python/2017/11/13/python-producer-consumer-example/
#https://docs.python.org/ko/3.7/library/concurrent.futures.html
def processingthread2(message):
    t1 = time.time()
    data = ujson.loads(message.decode())
    id = data['id']
    src = data['src']
    res = sess.post(FACADE_SERVER_ADDR + "/Load?keyword=Image"+"&id=" + str(id)+"&src="+src, "")
    img_array = np.frombuffer(res.content, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    segSize = (img_cv.shape[0], img_cv.shape[1])
    img_data = pil_to_tensor(img_cv)
    singleton_batch = {'img_data': img_data[None].cuda()}
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=segSize)
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy().astype('int8')
    t2 = time.time()
    print("Segmentation Processing = %s : %f" % (id, t2-t1))
    sess.post(FACADE_SERVER_ADDR + "/Store?keyword=Segmentation&id=" + str(id) + "&src=" + src, pred.tobytes())
    cv2.imshow('seg', pred)
    cv2.waitKey(1)

def processingthread():
    print("Start Message Processing Thread")
    while True:
        ConditionVariable.acquire()
        ConditionVariable.wait()
        message = msgqueue.pop()
        ConditionVariable.release()
        # 처리 시작

        data = ujson.loads(message.decode())
        id = data['id']
        keyword = data['keyword']
        src = data['src']

        res =sess.post(FACADE_SERVER_ADDR + "/Load?keyword="+keyword+"&id="+str(id),"")

        t1 = time.time()
        img_array = np.frombuffer(res.content, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        segSize = (img_cv.shape[0], img_cv.shape[1])

        img_data = pil_to_tensor(img_cv)
        singleton_batch = {'img_data': img_data[None].cuda()}
        with torch.no_grad():
            scores = segmentation_module(singleton_batch, segSize=segSize)
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy().astype('int8')
        t2 = time.time()
        print("Segmentation Processing = %s : %f" % (id, t2-t1))

        cv2.imshow('seg', pred)
        cv2.waitKey(1)
        # processing end

bufferSize = 1024
def udpthread():

    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        #address = bytesAddressPair[1]
        print(message)

        with ThreadPoolExecutor() as executor:
            executor.submit(processingthread2, message)
        """
        ConditionVariable.acquire()
        msgqueue.append(message)
        ConditionVariable.notify()
        ConditionVariable.release()
        """
if __name__ == "__main__":

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

    ##segmentation arguments
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    ##segmentation arguments

    opt = parser.parse_args()
    device = torch.device("cuda:"+opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")

    print('Starting the API')

    ####segmentation module configuration
    cfg.merge_from_file(opt.cfg)
    cfg.merge_from_list(opt.opts)

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.dirname(os.path.realpath(__file__)) + '/model/encoder_' + cfg.TEST.checkpoint
    cfg.MODEL.weights_decoder = os.path.dirname(os.path.realpath(__file__)) + '/model/decoder_' + cfg.TEST.checkpoint

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    modules = []
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).eval().to(device)
    segmentation_module2 = SegmentationModule(net_encoder, net_decoder, crit).eval().to(device)
    modules.append(segmentation_module);
    modules.append(segmentation_module2);

    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

    ##init
    rgb = np.random.randint(255, size=(640, 480, 3), dtype=np.uint8)
    tempsize = (rgb.shape[0], rgb.shape[1])
    init_data = pil_to_tensor(rgb)
    init_singleton_batch = {'img_data': init_data[None].cuda()}
    with torch.no_grad():
        segmentation_module(init_singleton_batch, segSize=tempsize)
        segmentation_module2(init_singleton_batch, segSize=tempsize)
    print("initialization!!")
    ##init
    ####segmentation module configuration

    Data = {}
    msgqueue = []

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
    #Echo server connect

    ConditionVariable = threading.Condition()

    th1 = threading.Thread(target=udpthread)
    #th2 = threading.Thread(target=processingthread)
    th1.start()
    #th2.start()
    # thread

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



