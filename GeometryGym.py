import FrameProcessor
import numpy as np
import pydirectinput as pdi
import cv2
import time 

class GeometryGym:
    def __init__(self):
        self.BUFF = 0.05
        pdi.PAUSE = 0.0
        pdi.FAILSAFE = False
        self.fp = FrameProcessor.FrameProcessor()
        self.max_progress = 0
        self.all_time_high = 0.0
        self.current_steps = 0
        self.zero_progress_counter = 0
        self.stuck_counter = 0

    def reset(self):
        pdi.keyUp('space')
        self.max_progress = 0
        self.current_steps = 0
        self.zero_progress_counter = 0
        
        # Clear residual death frames
        self.fp.clear_queue()

        img = None 
        while img is None:
            img, _ = self.fp.capture()

        self.fp.reset_stack(img)

        return self.fp.get_state()

    def step(self, action, frame_skip = 1):
        self.current_steps += 1

        if action == 1:
            pdi.keyDown('space')
        else:
            pdi.keyUp('space')

        binary_frame = None
        for _ in range(frame_skip):
            img = None
            while img is None:
                img, _ = self.fp.capture()
            # process() pushes Canny to queue and returns uint8 binary_frame
            last_canny, binary_frame = self.fp.process(img)

        if action == 1:
            pdi.keyUp('space')
        self.fp.push_frame(last_canny)

        # Extract progress bar pixels from uint8 binary frame
        if binary_frame is not None:
            progress_sprite = binary_frame[8:10, 208:512]
            current_progress = cv2.countNonZero(progress_sprite)
        else:
            current_progress = 0

        dead = self.is_dead(binary_frame, current_progress)
        next_state = self.fp.get_state()
        reward = 0.0

        if dead:
            pdi.keyUp('space')
            reward -= 1.0
            
            if self.all_time_high > 0:
                progress_ratio = current_progress / float(self.all_time_high)
            else:
                progress_ratio = 1.0

            if progress_ratio < 0.6:
                reward = -0.5 - 0.5 * (1.0 - progress_ratio)
            else:
                reward = -0.5

        else:
            if current_progress > self.max_progress:
                progress_delta = (current_progress - self.max_progress) / (320 * 1.5)
                reward += progress_delta * 25.0

                if current_progress > self.all_time_high:
                    reward += 2.0 + (progress_delta * 80.0)
                    self.all_time_high = float(current_progress)

                self.max_progress = current_progress

        if action == 1:
            reward -= 0.01

        info = {
            "action": action,
            "reward": reward,
            "progress": current_progress,
            "all_time_high": self.all_time_high
        }

        return next_state, reward, dead, info

    def is_dead(self, frame, current_progress):
        if self.max_progress > 2 and current_progress == 0 and self.current_steps > 10:
            return True

        if current_progress < 1:
            self.zero_progress_counter += 1
        else:
            self.zero_progress_counter = 0 

        if self.zero_progress_counter > 60 and self.current_steps > 5:
            return True

        return False

    def update_stuck_state(self, rollout_pb):
        if rollout_pb < self.all_time_high:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter >= 10 and self.all_time_high > 10:
            self.all_time_high *= 0.85
            self.stuck_counter = 0