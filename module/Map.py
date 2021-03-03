class Map:
    def __init__(self, name):
        self.name = name
        self.keys_updated_id = ['bsegmentation', 'bdepth', 'rdepth', 'reference']
        self.reset()

    def AddFrame(self, timestamp, Frame):
        id = self.id
        self.id = self.id+1
        self.IDs[timestamp] = id
        self.TimeStamps[id] = timestamp
        self.Frames[id] = Frame

    def reset(self):
        self.UpdateIDs = {}
        for key in self.keys_updated_id:
            self.UpdateIDs[key] = -1
        self.id = 0
        self.IDs = {}
        self.TimeStamps = {}
        self.TimeStamps[-1]="invalid case"
        self.Frames = {}
        self.MapPoints = {}
        self.Matches = {}
        self.Frames["ids"] = []
        self.MapPoints["ids"] = []
        self.Matches["ids"] = []
