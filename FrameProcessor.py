from collections import deque
import cv2
import numpy as np
import bettercam
class FrameProcessor:
    def __init__(self):
        """
        constructor for frame capturing and processing (grayscale binary)
        """
        self.camera = bettercam.create(output_color = "BGR")
        self.queue = deque(maxlen = 4)
        self.window = (600, 380, 1320, 860)
    
    def capture(self):
        """
        @returns: a frame of the screen and the shape of the image, returns
        none if there is no snapshot
        """
        snapshot = self.camera.grab(self.window)
        if snapshot is None:
            return None, None
        else:
            return snapshot, snapshot.shape
    def process(self, frame):
        """
        @returns: threshold, and the binary grayscale image in 
        respective order, to feed to model
        """
        compressed_size = (100, 100)
        converting_factor = 127

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_scale, converting_factor, 255, cv2.THRESH_BINARY)
        compressed_bin = cv2.resize(binary_frame, compressed_size, interpolation = cv2.INTER_AREA)

        self.queue.append(compressed_bin) 
        return binary_frame, compressed_bin
    
    def get_state(self):
        """
        @returns a numpy array of shape 4 X 100 X 100, to pass into model
        """
        if len(self.queue) < 4:
            return None
        else:
            return np.stack(self.queue, axis = 0)
            

    def clear_queue(self):
        """
        empties the queue returns None
        """
        self.queue.clear()

        



