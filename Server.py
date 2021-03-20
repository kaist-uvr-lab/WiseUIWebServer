import os
import threading
import ujson
import time
import numpy as np
from flask import Flask, request
import requests
import cv2
from module.Message import Message
import argparse
import torch, torchvision

#WSGI
from gevent.pywsgi import WSGIServer

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

#work에서 호출하는 cv가 필요함.
def work(cv,  queue):
    print("Start Message Processing Thread")
    while True:
        cv.acquire()
        cv.wait()
        message = queue.pop()
        cv.release()
        # 처리 시작
        start = time.time()
        img_array = np.frombuffer(message.data, dtype=np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        segSize = (img_cv.shape[0], img_cv.shape[1])

        pil_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])
        img_data = pil_to_tensor(img_cv)
        singleton_batch = {'img_data': img_data[None].cuda()}
        with torch.no_grad():
            scores = segmentation_module(singleton_batch, segSize=segSize)
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy().astype('int8')
        requests.post(FACADE_SERVER_ADDR + "/ReceiveData?map=" + message.map + "&id=" + message.id + "&key=bseg",
                      bytes(pred))
        requests.post(PROCESS_SERVER_ADDR + "/notify",
                      ujson.dumps({'user': message.user, 'map': message.map, 'id': int(message.id), 'key': 'bseg'}))
        end = time.time()
        print("segmentation Processing = %s : %f : %d" % (message.id, end - start, len(queue)))

    # processing end


@app.route("/Receive", methods=['POST'])
def Receive():
    user = request.args.get('user')
    map = request.args.get('map')
    id = request.args.get('id')
    message = Message(user, map, id, request.data)
    queue.append(message)
    ConditionVariable.acquire()
    ConditionVariable.notify()
    ConditionVariable.release()
    return ""

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
        '--ip', type=str,default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--port', type=int, default=35008,
        help='port number')
    parser.add_argument(
        '--use_gpu', type=str, default='0',
        help='port number')
    parser.add_argument(
        '--FACADE_SERVER_ADDR', type=str,
        help='facade server address')
    parser.add_argument(
        '--PROCESS_SERVER_ADDR', type=str,
        help='process server address')

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

    opt = parser.parse_args()
    device = torch.device("cuda:"+opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")

    ##segmentation module configuration
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

    crit = torch.nn.NLLLoss(ignore_index=-1)

    """
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)
    for batch_data in loader_test:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
    """
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).eval().to(device)

    ##Data queue
    queue = []
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    PROCESS_SERVER_ADDR = opt.PROCESS_SERVER_ADDR
    ConditionVariable = threading.Condition()

    th1 = threading.Thread(target=work, args=(ConditionVariable, queue))
    th1.start()

    print('Starting the API')
    #app.run(host=opt.ip, port=opt.port)
    #app.run(host=opt.ip, port = opt.port, threaded = True)

    keyword = 'bsegmentation'
    requests.post(FACADE_SERVER_ADDR + "/ConnectServer", ujson.dumps({
        'port':opt.port,'key': keyword
    }))

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()



