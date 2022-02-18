class Device:
    def __init__(self, src):
        self.send_keyword=set()
        self.receive_keyword=set()
        self.capacity=0
        self.name = src
        self.bSchedule = False
        self.Skipping = {}
        self.Sending = {}
        self.type = 'device'

    def SetScheduling(self, keyword, Total):
        if self.type == 'device':
            return 0
        nSkip = int(Total / self.capacity) + 1
        if nSkip == 1:
            self.bSchedule = False
            nSkip = 0
        else:
            print("Scheduling = ", self.name, keyword, nSkip)
            self.bSchedule = True
        return nSkip
    #def SetScheduling(self, keyword, skip):
    #    self.Skipping[keyword] = skip

    def __eq__(self, other):
        if type(other) is type(self):
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)