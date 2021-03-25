class Map:
    def __init__(self, name):
        self.name = name
        self.reset()

    def AddFrame(self, Frame):
        id = self.id
        self.Frames[id] = Frame
        self.id = self.id + 1
        return id

    def reset(self):
        self.id = 0
        self.Frames = {}
        self.MapPoints = {}
        self.Matches = {}
        self.Users = {}

    def Connect(self, user, User):
        self.Users[user] = User

    def Disconnect(self, user):
        if self.Users.get(user) is not None:
            del self.Users[user]
