
import time
import timeit
import mss
import numpy as np
import cv2
import queue
from collections import deque

from consts import FPS_LIMIT,\
                   LOG_EVERY_N_FRAME

cv2.setNumThreads(0)

BUF_SIZE = 4

class Grabber():

    def __init__(self, window_xy):
        self.mon = window_xy
        self.scale_factor = 1

    def capture(self, end, in_queue, out_queue):

        sct = mss.mss()
        t_start = time.time()
        t_per_frame = int(1000 / FPS_LIMIT)-1 # estimated time per frame
        ct = 0 # frames counter

        buf = deque(maxlen=BUF_SIZE)

        global_counter = 0
        while True:
            
            t0 = timeit.default_timer()

            if end.is_set():
                break

            for _ in range(BUF_SIZE):
                img = sct.grab(self.mon)
                img = np.array(img)
                buf.append(img)
                
            img = None
            obs = np.max(np.stack(buf), axis=0)

            ct += 1                                

            try:
                out_queue.put_nowait(obs)
            except queue.Full:
                t_delay = (t_per_frame - ((timeit.default_timer() - t0)))*0.001
                if t_delay > 0:
                    time.sleep(t_delay)
                continue
            except Exception as err:
                continue

            if ct % LOG_EVERY_N_FRAME == 0:
                print(f'Capturing at {(ct / (time.time() - t_start)):.2f} fps') 
                ct = 0
                t_start =  time.time()
            
            t_delay = (t_per_frame - ((timeit.default_timer() - t0)))*0.001
            if t_delay > 0:
                time.sleep(t_delay) # delay (fps limit)
