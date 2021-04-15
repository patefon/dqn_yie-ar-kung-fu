import logging
import time
import cv2
import timeit
import queue
import traceback
import numpy as np

from copy import deepcopy

from classes.utils import load_pickled_dict, read_text_stat, get_roi, resize_img
from consts import HEALTH_COLOR_CODE, LOG_EVERY_N_FRAME, FONT_ENCODED, FPS_LIMIT

class Environment():

    def __init__(self):
        self.__font_chars = load_pickled_dict(FONT_ENCODED)
        self.current_state = {
            'player_health': 0,
            'enemy_health': 0,
            'reward': 0,
            'highscore': 0,
            'stage': 0,
            'is_gameover': 0
        }

    def __is_level_split_screen(self, data):
        return True if np.count_nonzero(data) / (data.size / 100) < 5 else False

    def __is_gameover(self, roi):
        return int(np.count_nonzero(roi) == 934)

    def __get_game_stats(self, img=None):
        
        stats = {}
        
        p_health_roi = cv2.cvtColor(get_roi(img, 80, 400, 144, 1), cv2.COLOR_BGR2GRAY)
        e_health_roi = cv2.cvtColor(get_roi(img, 288, 400, 144, 1), cv2.COLOR_BGR2GRAY)

        highscore_roi = cv2.cvtColor(get_roi(img, 48, 74, 95, 14), cv2.COLOR_BGR2GRAY)
        stage_roi = cv2.cvtColor(get_roi(img, 432, 58, 30, 14), cv2.COLOR_BGR2GRAY)
        gameover_roi = cv2.cvtColor(get_roi(img, 183, 228, 143, 14), cv2.COLOR_BGR2GRAY)
        
        stats['highscore'] = read_text_stat(highscore_roi, self.__font_chars)
        stats['stage'] = read_text_stat(stage_roi, self.__font_chars)

        stats['player_health'] = np.round(np.count_nonzero(np.logical_or(p_health_roi == HEALTH_COLOR_CODE, p_health_roi == 98)) / p_health_roi.size, 2)
        
        stats['enemy_health'] = np.round(np.count_nonzero(np.logical_or(e_health_roi == HEALTH_COLOR_CODE, e_health_roi == 98)) / e_health_roi.size, 2)

        # stats['out_of_lives'] = np.round(np.count_nonzero(lives_roi) / (lives_roi.size / 100)) < 10
        # stats['lives'] = np.round(np.count_nonzero(lives_roi) / (lives_roi.size / 100))

        stats['is_gameover'] = self.__is_gameover(gameover_roi)

        return stats

    def __get_reward(self, state):

        reward = 0
        previous_state = self.current_state

        if state.get('is_level_upgrade_screen', None):
            return 0

        if state['highscore'] > previous_state['highscore']:
           reward += (state['highscore'] - previous_state['highscore']) * 0.01

        if state['highscore'] == previous_state['highscore'] and \
           state['player_health'] < previous_state['player_health']:
           reward -= 5

        if state['enemy_health'] < previous_state['enemy_health']:
            reward += 1

        return reward

    def get_data_stack(self, in_queue, buf_size=3):
        buf = []
        while len(buf) < buf_size:
            try:
                data = in_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue
            if data is None:
                continue
            buf.append(data)
        return buf

    def get_obs(self, imgs_stack):

        last_img = imgs_stack[-1]

        # health level, highscore
        state = self.__get_game_stats(last_img)

        state['reward'] = self.__get_reward(state)
        state['is_done'] = state['is_gameover']
        state['img'] = last_img
        state['original'] = deepcopy(last_img)

        last_img_frame = cv2.cvtColor(get_roi(last_img, 16, 217, 480, 148), cv2.COLOR_BGR2GRAY)

        # 1) crop game frame
        # 2) resize / 2
        # 3) prepare 4 NN
        data = np.stack([resize_img(cv2.cvtColor(get_roi(img, 16, 217, 480, 148),\
                                                 cv2.COLOR_BGR2GRAY)) for img in imgs_stack],\
                                                 axis=0)

        state['is_level_upgrade_screen'] = self.__is_level_split_screen(last_img_frame)

        self.current_state = state
        
        return data, state

    def loop(self, end, in_queue, out_queue):

        ct = 0

        t_start = timeit.default_timer()
        t_per_frame = int(1000 / FPS_LIMIT)-1 # estimated time per frame
    
        time.sleep(8)

        while True:

            if end.is_set():
                break

            t0 = timeit.default_timer()

            try:

                imgs_stack = self.get_data_stack(in_queue)
                data, obs = self.get_obs(imgs_stack)

            except Exception as err:
                print(''.join(traceback.format_exception(etype=type(err), value=err, tb=err.__traceback__)))

            try:
                out_queue.put_nowait((data, obs))
            except queue.Full:
                pass

            ct += 1
            
            if ct % LOG_EVERY_N_FRAME == 0:
                print(f'Detection working at {(ct / (timeit.default_timer() - t_start)):.2f} fps') 
                ct = 0
                t_start = timeit.default_timer()
            
            t1 = timeit.default_timer() - t0 # op. time

            t_delay = (t_per_frame - t1)*0.001
            if t_delay > 0:
                time.sleep(t_delay) # delay (fps limit)
