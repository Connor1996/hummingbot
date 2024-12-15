from enum import Enum


class MsgSource(Enum):
    CLI = 0
    TELEGRAM = 1
    STRATEGY = 2


class NotifierBase:
    def __init__(self):
        self._started = False

    def add_msg_to_queue(self, msg: str, msg_source: MsgSource):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError
