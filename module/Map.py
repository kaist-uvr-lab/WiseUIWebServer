class Map:
    def __init__(self, name):
        self.name = name
        self.keys_updated_id = ['bsegmentation', 'bdepth', 'rdepth', 'reference']
        self.reset()

    def AddFrame(self, Frame):
        id = self.id
        self.Frames[id] = Frame
        self.id = self.id + 1
        return id

    def reset(self):
        self.UpdateIDs = {}
        for key in self.keys_updated_id:
            self.UpdateIDs[key] = -1
        self.id = 0
        self.Frames = {}
        self.MapPoints = {}
        self.Matches = {}
        #self.Frames["ids"] = []
        #self.MapPoints["ids"] = []
        #self.Matches["ids"] = []
