# simulator/client.py
# For this simplified project, client logic lives inside env but we keep a helper for future extension.
class Client:
    def __init__(self, id, state):
        self.id = id
        self.battery, self.data_q, self.data_qual, self.past, self.uplink = state
    def to_array(self):
        return [self.battery, self.data_q, self.data_qual, self.past, self.uplink]
