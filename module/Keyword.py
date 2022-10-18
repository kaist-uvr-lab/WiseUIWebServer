class Keyword:
    def __init__(self, key, pair=None, atype=None):
        self.keyword = key
        if pair == 'NONE':
            self.pair = None
        else:
            self.pair = pair
        self.type = atype