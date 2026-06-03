# Stub: future collector (no-op for open-source)
class FutureWrapper:
    def __init__(self, future=None):
        self.future = future
    def result(self):
        return None

def get_future_collector(*args, **kwargs):
    return None
