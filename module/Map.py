class Map:
    def __init__(self, name):
        self.name = name
        self.reset()

    def IncreaseID(self):
        id = str(self.fid)
        #self.Frames[id] = Frame
        self.fid = self.fid + 1
        return id

    def reset(self):
        self.fid = 0
        self.uid = 0
        self.Users = {}
        self.Frames = {}
        self.MapPoints = {}
        self.Models = {}
        self.Matches = {}

    def Connect(self, user, User):
        if self.Users.get(user) is None:
            self.uid = self.uid + 1
            self.Users[user] = {}
            self.Users[user]['id'] = self.uid
            self.Users[user]['data'] = User
            self.Users[user]['refid'] = (-1).to_bytes(4, 'little', signed=True)
            return True
        return False

    def Disconnect(self, user):
        if self.Users.get(user) is not None:
            del self.Users[user]
            return True
        return False
