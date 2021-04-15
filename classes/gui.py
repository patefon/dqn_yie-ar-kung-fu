import logging
import time
import cv2
import queue
import numpy as np
from PIL import Image

from classes.utils import get_roi3d, resize_img

class Screen():

    def __init__(self):
        pass

    def __group_img(self, a, b):
        a = resize_img(a.transpose(2,1,0), scale_factor=2)
        b = get_roi3d(b, 16, 217, 480, 148)
        extended_img = Image.fromarray(a)
        extended_img = cv2.cvtColor(np.array(extended_img.resize(size=b.shape[:2])).transpose(1,0,2), cv2.COLOR_RGB2RGBA)
        return np.vstack((b, extended_img))

    def show(self, end, in_queue, out_queue):

        fontscale = 0.5
        color = (54,247,247)
        fontface = cv2.FONT_HERSHEY_TRIPLEX
        rectangle_bgr = (0, 0, 0)

        while True:

            if end.is_set():
                cv2.destroyAllWindows()
                break
            
            try:
                data, obs = in_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue

            try:

                res = self.__group_img(data, obs.get('original'))

                if obs.get('player_health') is not None:

                    text = f'player: {obs["player_health"]}, enemy: {obs["enemy_health"]}, score: {obs["highscore"]}'

                    (text_width, text_height) = cv2.getTextSize(text, fontface, fontscale, thickness=1)[0]

                    text_offset_x = 5
                    text_offset_y = 15

                    box_coords = ((5, 5), (text_offset_x + text_width + 5, text_offset_y + text_height - 10))

                    cv2.rectangle(res, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
                    cv2.putText(res, text, (text_offset_x, text_offset_y), fontface, fontscale, color, thickness=1)

                cv2.imshow('game', res)

                if cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    break

            except Exception as err:
                print(f'Gui error: {err}')


