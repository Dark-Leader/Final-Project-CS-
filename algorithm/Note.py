import json

class Note:
    '''
    represents a note to played on the piano.
    '''
    
    def __init__(self, duration=None, name=None, pitch=-1, time_step=-1):
        '''
        constructor
        @param duration: (float) duration in seconds the note needs to be played.
        @param name: (str) name of the note -> E.G C4.
        @param pitch: (int) pitch value of the note.
        @param time_step: (int) timeStep the note needs to be played.
        '''
        self.duration = duration
        self.name = name
        self.pitch = pitch
        self.time_step = time_step

    def get_duration(self):
        '''
        getter
        @return: (float) duration.
        '''
        return self.duration

    def get_name(self):
        '''
        getter
        @return: (str) name.
        '''
        return self.name

    def get_pitch(self):
        '''
        getter
        @return: (int) pyrch.
        '''
        return self.pitch

    def get_time(self):
        '''
        getter
        @return: (float) timeStep.
        '''
        return self.time_step

    def set_duration(self, duration):
        '''
        setter
        @return: (float) duration.
        '''
        self.duration = duration

    def set_name(self, name):
        '''
        setter
        @return: (str) name.
        '''
        self.name = name

    def set_pitch(self, pitch):
        '''
        setter
        @return: (int) pitch.
        '''
        self.pitch = pitch

    def set_time_step(self, time_step):
        '''
        setter
        @return: (float) timeStep.
        '''
        self.time_step = time_step

    def to_json(self):
        '''
        convert Note object to json inorder to be JSON serializable.
        @return: (JSON) JSON string representation of the note.
        '''
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def from_json(json_str):
        '''
        create Note of json string.
        @param json_str: (str) json string
        @return: (Note) output note.
        '''
        obj_dict = json.loads(json_str)
        note = Note(obj_dict.get('duration', None), obj_dict.get('name', None),
                    obj_dict.get('pitch', None), obj_dict.get('time_step', None))
        return note
