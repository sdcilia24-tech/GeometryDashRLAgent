import FrameProcessor
import numpy as np
import time
import cv2
import pydirectinput as pdi

class GeometryGym:
    def __init__(self):
        self.BUFF = 0.05
        pdi.PAUSE = 0.0
        self.fp = FrameProcessor.FrameProcessor()
        self.max_progress = 0
        self.current_steps = 0
        self.zero_progress_counter = 0

    def reset(self):
        """
        detects if the player has crashed, and will reset the level accordingly given that the
        menu pops up whenever the bot has died 
        @returns None
        """
        pdi.keyUp('space')
        self.max_progress = 0 
        self.current_steps = 0
        self.fp.clear_queue()
        self.zero_progress_counter = 0

        time.sleep(.3)

        self.fp.clear_queue()

        while self.fp.get_state() is None:
            img = None 
            while img is None:
                img, _ = self.fp.capture()
            self.fp.process(img)
        return self.fp.get_state()

    def step(self, action):
        """
        @param: action: a binary action to take, 0 indicating no press, and 1 indicating press
        returns: None
        """
        self.current_steps += 1
        if action == 1:
            pdi.keyDown('space')
        else:
            pdi.keyUp('space')
        img = None
        while img is None:
            img, _ = self.fp.capture()
        binary_frame, compressed_bin = self.fp.process(img)
        next_state = self.fp.get_state()
        dead = self.is_dead(binary_frame)

        if dead:
            reward = -100
            pdi.keyUp('space')
        else:
            reward = .01

        info = {
            "action": action,
            "max_progress": self.max_progress
        }

        return next_state, reward, dead, info
    
    def is_dead(self, frame):
        """
        this will determine whether or not the player has died, it will do two 
        checks to ensure no false flags occur, one will check if the cube has become a
        bright flash of white, and the second will detect if the "attempt x" has apppeard on the 
        screen
        """
        progress_sprite = frame[8:10, 208:512]
        current_progress = cv2.countNonZero(progress_sprite)
        if self.max_progress > 2 and current_progress < 2:
            return True
        if current_progress > self.max_progress:
            self.max_progress = current_progress
        if current_progress < 1:
            self.zero_progress_counter += 1
        else:
            self.zero_progress_counter = 0 
        if self.zero_progress_counter > 100 and self.current_steps > 5:
            return True

        return False

        

    

        
