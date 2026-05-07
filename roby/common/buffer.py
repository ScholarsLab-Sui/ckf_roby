from queue import Queue


class FIFOBuffer(Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)

    def put(self, item, block=True, timeout=None):
        if self.full():
            self.get()  # Remove the oldest item if the buffer is full
        super().put(item, block, timeout)


if __name__ == "__main__":
    buffer = FIFOBuffer(3)

    buffer.put(0)
    buffer.put(1)
    buffer.put(2)
    buffer.put(3)
    print(list(buffer.queue))