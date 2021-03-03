class User:
    def __init__(self, _id, _map, _mapping, _cam, _img):
        self.id = _id
        self.map = _map
        self.mapping = _mapping
        self.camera = _cam
        self.imgSize = _img
        self.frameIDs=[] #slam에서 넣어줄 것
        self.frameIDs.append(-1)
