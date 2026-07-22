import bettercam
import time
import numpy as np
import cv2
import pydirectinput as pdi 

"""Test"""
def ss_testing():
    #creates a bettercame instant, using Blue, green Red colorscale
    camera = bettercam.create(output_color = "BGR")
    #defines pixel region on the monitor
    region = (0, 0, 100, 200)
    i = 0
    while i < 20:
        flag_1 = time.time()
        #grabs the current frame in given region creates np array
        image = camera.grab(region = region)
        #Buffer for GPU
        if image is None:
            continue
        flag_2 = time.time()
        #convert from bgr to grayscale (as the param says)
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #returns threshold, and frame as binary np array
        thresh, binary_frame = cv2.threshold(gray_scale, 127, 255, cv2.THRESH_BINARY)
        flag_3 = time.time()
        cap = (flag_2 - flag_1) * 1000
        pro = (flag_3 - flag_2) * 1000
        tot = (flag_3 - flag_1) * 1000
        print(f"Capture: {cap:.1f}ms | Process: {pro:.1f}ms | Total: {tot:.1f}ms")
        i += 1

def input_testing():
    pdi.PAUSE = 0.0
    while True:
        print("pressing the space bar")
        pdi.keyDown('space')
        time.sleep(.05)
        pdi.keyUp('space')
        print("Lifting the space bar")

def getting_coor():
    img = cv2.imread("progress_bar_cropped.png")
    x, y, w, h = cv2.selectROI("Select Attempt Text", img, showCrosshair=True)
    print(f"x_min: {x}, y_min: {y}, x_max: {x + w}, y_max: {y + h}")
    print(f"NumPy Slice: frame[{y}:{y+h}, {x}:{x+w}]")
    cv2.destroyAllWindows()

    
import cv2
import pydirectinput as pdi
from FrameProcessor import FrameProcessor


def sprite_cal():
    pdi.PAUSE = 0.0
    fp = FrameProcessor()

    print("🎯 CROP COORDINATE CALIBRATOR")
    print("Press 'q' in the window to quit.\n")
    while True:
        img = None
        while img is None:
            img, _ = fp.capture()

        binary_frame, _ = fp.process(img)
        
        # Get total dimensions of processed frame
        h, w = binary_frame.shape[:2]

        # Convert binary frame to BGR so we can draw colored bounding boxes on it
        debug_frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)

        # --- CURRENT PROGRESSBAR ROI ---
        # y1:y2 = 347:388, x1:x2 = 227:271
        cv2.rectangle(debug_frame, (210, 8), (512, 10), (0, 0, 255), 2)  # RED Box (Progress Bar ROI)  ### updated 

        cv2.putText(debug_frame, "Cube ROI", (227, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) ### updated

        # --- CURRENT ATTEMPT ROI ---
        # y1:y2 = 136:172, x1:x2 = 263:521
        cv2.rectangle(debug_frame, (100, 86), (375, 130), (255, 0, 0), 2)  # BLUE Box
        cv2.putText(debug_frame, "Attempt ROI", (263, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Show total image shape on screen
        cv2.putText(debug_frame, f"Frame Size: {w}x{h}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Coordinate Calibrator (Red=Cube, Blue=Attempt)", debug_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
sprite_cal()
#input_testing()  
#2ss_testing()                            
#getting_coor()