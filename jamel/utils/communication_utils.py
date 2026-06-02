from typing import Tuple
import queue
import multiprocessing

channels = {}

def get_channel(channel_name, channel_id: int, queue_cls) -> Tuple[queue.Queue, queue.Queue]:
    '''
    channel id: 0 or 1.
    '''
    if channel_name not in channels:
        reader = queue_cls()
        writer = queue_cls()
        channels[channel_name] = (reader, writer)
    return channels[channel_name][channel_id], channels[channel_name][1-channel_id]

def get_channel_queue(channel_name, channel_id: int):
    return get_channel(channel_name, channel_id, queue.Queue)

def get_channel_multiprocessing(channel_name, channel_id: int):
    return get_channel(channel_name, channel_id, multiprocessing.Queue)
