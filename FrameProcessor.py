from collections import deque
import cv2
import numpy as np
import bettercam

class FrameProcessor:
    def __init__(self):
        """
        Constructor for frame capturing and dual-output processing 
        (Canny for agent state, Binary for UI/Death metrics).
        """
        self.camera = bettercam.create(output_color="BGR")
        self.queue = deque(maxlen=4)
        self.window = (600, 380, 1320, 860)
        self.target_size = (64, 64)

    def capture(self):
        """
        @returns: raw snapshot frame and shape, or (None, None)
        """
        snapshot = self.camera.grab(self.window)
        if snapshot is None:
            return None, None
        return snapshot, snapshot.shape

    def process(self, frame):
        """Pure function: transforms frame without altering state queue."""
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.resize(gray_scale, self.target_size, interpolation=cv2.INTER_AREA)
        normalized_gray = compressed_gray.astype(np.float32) / 255.0

        # 2. Binary
        _, binary_img = cv2.threshold(gray_scale, 127, 255, cv2.THRESH_BINARY)
        
        return normalized_gray, binary_img

    def push_frame(self, canny_frame):
        """Explicitly add a processed frame to the observation queue."""
        self.queue.append(canny_frame)

    def reset_stack(self, initial_frame):
        """Fills stack with 4 identical copies of the initial frame."""
        self.queue.clear()
        canny, binary = self.process(initial_frame)
        for _ in range(4):
            self.queue.append(canny)
        return binary

    def get_state(self):
        """
        @returns: numpy array of shape (4, 64, 64) float32 [0.0, 1.0],
                  or None if queue length < 4.
        """
        if len(self.queue) < 4:
            return None
   
        return np.stack(self.queue, axis=0)

    def clear_queue(self):
        self.queue.clear()