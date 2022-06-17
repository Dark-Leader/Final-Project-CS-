import json

class Note:
    
    def __init__(self, duration=None, name=None, pitch=-1, time_step=-1):

        self.duration = duration
        self.name = name
        self.pitch = pitch
        self.time_step = time_step

    def get_duration(self):
        return self.duration

    def get_name(self):
        return self.name

    def get_pitch(self):
        return self.pitch

    def get_time(self):
        return self.time_step

    def set_duration(self, duration):
        self.duration = duration

    def set_name(self, name):
        self.name = name

    def set_pitch(self, pitch):
        self.pitch = pitch

    def set_time_step(self, time_step):
        self.time_step = time_step

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def from_json(json_str):
        obj_dict = json.loads(json_str)
        note = Note(obj_dict.get('duration', None), obj_dict.get('name', None),
                    obj_dict.get('pitch', None), obj_dict.get('time_step', None))
        return note
