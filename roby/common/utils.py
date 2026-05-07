from collections import deque


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WindowAverageMeter:
    def __init__(self, window_size=10):
        self.queue = deque(maxlen=window_size)
        self.reset()

    def reset(self):
        self.queue.clear()

    def update(self, value):
        self.queue.append(value)

    @property
    def avg(self):
        if not self.queue or len(self.queue) == 0:
            return None
        return sum(self.queue) / len(self.queue)
    
    @property
    def max(self):
        if not self.queue or len(self.queue) == 0:
            return None
        return max(self.queue)
    
    @property
    def min(self):
        if not self.queue or len(self.queue) == 0:
            return None
        return min(self.queue)