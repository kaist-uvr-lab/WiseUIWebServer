class Scheduler:
    def __init__(self, keyword):
        self.keyword = keyword
        self.send_list=set()
        self.receive_list = set()
        self.total_send_capacity = 0

    def update_send(self, n):
        for s in list(self.send_list):
            s.Skipping[self.keyword] = n
    def update_receive(self):
        for r in list(self.receive_list):
            n = r.SetScheduling(self.keyword, self.total_send_capacity)
            self.update_send(n)
    def add_send_list(self, schedule):
        if schedule not in self.send_list:
            self.send_list.add(schedule)
            self.total_send_capacity = self.total_send_capacity+schedule.capacity
            #print('send = ',self.keyword, schedule.name, self.total_send_capacity, len(self.send_list), len(self.receive_list))
            ##update receive
            self.update_receive()
    def add_receive_list(self, schedule):
        if schedule not in self.receive_list:
            self.receive_list.add(schedule)
            ##update send
            n = schedule.SetScheduling(self.keyword, self.total_send_capacity)
            self.update_send(n)

    def remove_send_list(self, schedule):
        if schedule in self.send_list:
            self.send_list.remove(schedule)
            self.total_send_capacity = self.total_send_capacity - schedule.capacity
            ##update receive
            self.update_receive()

    def remove_receive_list(self, schedule):
        if schedule in self.receive_list:
            self.receive_list.remove(schedule)
            self.update_send(0)
