class Note:
    
    def __init__(self, duration=None, name=None, idx=-1, time_step=-1):

        self.duration = duration
        self.name = name
        self.piano_idx = idx
        self.time_step = time_step

    def get_duration(self):
        return self.duration

    def get_name(self):
        return self.name

    def get_idx(self):
        return self.piano_idx

    def get_time(self):
        return self.time_step

    def set_duration(self, duration):
        self.duration = duration

    def set_name(self, name):
        self.name = name

    def set_idx(self, idx):
        self.piano_idx = idx

    def set_time_step(self, time_step):
        self.time_step = time_step
