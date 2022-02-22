class Scheduler:
    def __init__(self, keyword):
        self.keyword = keyword
        self.send_list= {}
        self.broadcast_list={} # all
        self.unicast_list={}   # single
        self.total_send_capacity = 0

    def update_send(self, n, name):
        for s in self.send_list.keys():
            self.send_list[s].Skipping[name] = n
    def update_receive(self):
        for r in self.broadcast_list.keys():
            n = self.broadcast_list[r].SetScheduling(self.keyword, self.total_send_capacity)
            self.update_send(n, self.broadcast_list[r].name)

    def add_send_list(self, schedule):
        if schedule.name not in self.send_list:
            self.send_list[schedule.name] = schedule
            self.total_send_capacity = self.total_send_capacity+schedule.capacity
            #print('send = ',self.keyword, schedule.name, self.total_send_capacity, len(self.send_list), len(self.receive_list))
            ##update receive
            self.update_receive()
    def add_unicast(self, schedule):
        if schedule.name not in self.unicast_list:
            self.unicast_list[schedule.name] = schedule
    def add_broadcast(self, schedule):
        if schedule.name not in self.broadcast_list:
            self.broadcast_list[schedule.name] = schedule
            ##update send
            n = schedule.SetScheduling(self.keyword, self.total_send_capacity)
            self.update_send(n, schedule.name)
    def add_receive_list(self,schedule, type):
        if type == 'all':
            self.add_broadcast(schedule)
        else:
            self.add_unicast(schedule)
    def remove_broadcast(self, schedule):
        if schedule.name in self.broadcast_list:
            del self.broadcast_list[schedule.name]
            self.update_send(0, schedule.name)
        pass
    def remove_unicast(self, schedule):
        if schedule.name in self.unicast_list:
            del self.unicast_list[schedule.name]
        pass
    def remove_send_list(self, schedule):
        if schedule.name in self.send_list:
            del self.send_list[schedule.name]
            self.total_send_capacity = self.total_send_capacity - schedule.capacity
            ##update receive
            self.update_receive()

    def remove_receive_list(self, schedule, type):
        if type == 'all':
            self.remove_broadcast(schedule)
        else:
            self.remove_unicast(schedule)
