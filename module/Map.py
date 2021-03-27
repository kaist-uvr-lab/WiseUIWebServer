class Map:
    def __init__(self, name):
        self.name = name
        self.reset()

    def IncreaseID(self):
        id = str(self.id)
        #self.Frames[id] = Frame
        self.id = self.id + 1
        return id

    def reset(self):
        self.id = 0
        self.Users = {}
        self.Frames = {}
        self.MapPoints = {}
        self.Models = {}
        self.Matches = {}

    def Connect(self, user, User):
        self.Users[user] = {}
        self.Users[user]['data'] = User
        self.Users[user]['refid'] = (-1).to_bytes(4, 'little', signed=True)

    def Disconnect(self, user):
        if self.Users.get(user) is not None:
            del self.Users[user]
