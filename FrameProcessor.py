from collections import deque
import cv2
import numpy as np
import bettercam

class FrameProcessor:
    def __init__(self):
        """
        Constructor for frame capturing and processing (grayscale binary).
        """
        self.camera = bettercam.create(output_color="BGR")
        self.queue = deque(maxlen=4)
        self.window = (600, 380, 1320, 860)
        self.target_size = (64, 64)

    def capture(self):
        """
        @returns: a frame of the screen and the shape of the image, returns
        None if there is no snapshot.
        """
        snapshot = self.camera.grab(self.window)
        if snapshot is None:
            return None, None
        return snapshot, snapshot.shape

    def process(self, frame):
        """
        @returns: binary_frame, compressed_bin (shape: 64x64)
        """
        converting_factor = 127

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_scale, converting_factor, 255, cv2.THRESH_BINARY)

        compressed_bin = cv2.resize(binary_frame, self.target_size, interpolation=cv2.INTER_AREA)

        self.queue.append(compressed_bin) 
        return binary_frame, compressed_bin

    def reset_stack(self, initial_frame):
        """
        Fills the queue with 4 copies of the initial frame on environment reset.
        Prevents get_state() from returning None at episode start.
        """
        self.queue.clear()
        _, compressed_bin = self.process(initial_frame)
        for _ in range(3):
            self.queue.append(compressed_bin)

    def get_state(self):
        """
        @returns a numpy array of shape (4, 64, 64) normalized float32 [0.0, 1.0],
        or None if queue isn't full yet.
        """
        if len(self.queue) < 4:
            return None
   
        stacked = np.stack(self.queue, axis=0).astype(np.float32) / 255.0
        return stacked

    def clear_queue(self):
        """
        Empties the queue.
        """
        self.queue.clear()