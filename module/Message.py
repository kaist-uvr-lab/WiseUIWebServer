class Message:
    def __init__(self, user, map, timestamp, data):
        self.user = user
        self.map = map
        self.timestamp  = timestamp
        self.data = data
        self.id = 0