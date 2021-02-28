class Map:
    def __init__(self, name):
        self.name = name
        self.keys_updated_id = ['bsegmentation', 'bdepth', 'reference']
        self.reset()

    def reset(self):
        self.UpdateIDs = {}
        for key in self.keys_updated_id:
            self.UpdateIDs[key] = -1
        self.Frames = {}
        self.MapPoints = {}
        self.Matches = {}
        self.Frames["ids"] = []
        self.MapPoints["ids"] = []
        self.Matches["ids"] = []
