import pandas as pd
import numpy as np
import threading
import cv2 as cv
import queue
import os

class OTB100Loader:

    _END = object()
    
    def __init__(self, dir, data, queue_sz=8, exts=(".jpg", ".jpeg", ".png", ".bmp")):
        data_path = os.path.join(dir, data)

        self.img_path = os.path.join(data_path, 'img')
        self.images = sorted(
            f for f in os.listdir(self.img_path)
            if f.lower().endswith(exts)
        )
        self.gt = pd.read_csv(os.path.join(data_path, 'groundtruth_rect.txt'),
                   sep=r'[,\s]+', header=None, engine='python',
                   dtype=np.int32).to_numpy()

        self.queue_sz = queue_sz

    def set_queue_size(self, sz):
        self.queue_sz = sz

    def get_frame(self, idx):
        if 0 <= idx < len(self.images):
            frame = cv.imread(
                os.path.join(self.img_path, self.images[idx])
            )
            return frame
            
        return None

    def _reader(self):

        def _safe_put(item, timeout=0.1):
            while not self.stop_event.is_set():
                try:
                    self.q.put(item, timeout=timeout)
                    return True

                except queue.Full:
                    pass
            return False

        try:
            for img_name, bbox in zip(self.images, self.gt):
                if self.stop_event.is_set():
                    break

                frame = cv.imread(
                    os.path.join(self.img_path, img_name)
                )

                if not _safe_put((frame, bbox)):
                    break
                
        finally:
            _safe_put(self._END)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()

        if item is self._END:
            raise StopIteration
        
        return item
        
    def __enter__(self):
        self.q = queue.Queue(self.queue_sz)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

        return self
    
    def __exit__(self, *arg):
        self.stop_event.set()
        self.thread.join()