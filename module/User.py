class User:
    def __init__(self, _id, _map, _mapping, _manager, _addr, fx, fy, cx, cy, d1, d2, d3, d4, w, h, info):
        self.id = _id
        self.map = _map
        self.mapping = _mapping
        self.manage  = _manager
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4
        self.w = w
        self.h = h
        self.addr = _addr
        self.info = info

        #cameraParam = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        #imgSize = np.array([h, w])
        #self.camera = _cam
        #self.imgSize = _img

        self.frameIDs=[] #slam에서 넣어줄 것
        self.frameIDs.append(-1)

        self.IDs = {}
        self.TimeStamps = {}
        self.TimeStamps[-1] = "invalid case"

    def AddData(self, id, ts):
        self.IDs[ts] = id
        self.TimeStamps[id] = ts