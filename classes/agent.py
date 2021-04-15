import random
import time
import timeit
import traceback
import queue
import numpy as np
import torch as T

from functools import partial
from collections import namedtuple

from classes.control import GameController
from classes.network import SimpleNet
from classes.replaybuffer import ReplayBuffer

from consts import LOG_EVERY_N_FRAME, FPS_LIMIT

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

# CONTROLLER
ctrl = GameController()

class Agent():

    def __init__(self):
        # learn parameters
        self.gamma=.99
        self.batch_size=16
        self.lr=.001
        self.eps=.999
        self.eps_dec=.0001
        self.eps_min=.01
        self.burn_in=5_000
        self.mem_size=200_000
        self.sync_target_frames=5_000
        self.learn_every=4

        # possible actions
        self.action_space_named = ['stop','sit','kick-right','kick-left','left','right','up']
        self.action_space = [idx for idx, a in enumerate(self.action_space_named)]
        self.n_actions = len(self.action_space)
        
        self.num_of_steps = 0

        self.q_eval = SimpleNet(lr=self.lr, n_actions=self.n_actions, chkpnt_dir='./weights', name='dqn-eval-v1')
        self.q_target = SimpleNet(lr=self.lr, n_actions=self.n_actions, chkpnt_dir='./weights', name='dqn-tgt-v1')
        
        self.device = self.q_eval.device

        self.memory = ReplayBuffer(capacity=self.mem_size)

    def __act(self, action):

        fn = {
            'confirm': ctrl.confirm,
            'stop': ctrl.stop,
            'sit': ctrl.sit,
            'kick-right': partial(ctrl.kick, 'right'),
            'kick-left': partial(ctrl.kick, 'left'),
            'sit-kick-right': partial(ctrl.sit_kick, 'right'),
            'sit-kick-left': partial(ctrl.sit_kick, 'left'),
            'right': partial(ctrl.move, 'right'), 
            'left': partial(ctrl.move, 'left'),
            'up': partial(ctrl.move, 'up'),
            'reset': ctrl.reset
        }.get(action, 0)

        if fn != 0:
            fn()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_target.load_checkpoint()

    def reset(self):
        self.__act('reset')
        time.sleep(1)
        self.__act('confirm')

    def store_transition(self, state, action, reward, done, state_):
        transition = Experience(state, action, reward, done, state_)
        self.memory.append(transition)

    def choose_action(self, obs):
        if np.random.random() > self.eps:
            action = self.q_eval.forward(obs).argmax().item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrement_epsilon(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):

        if self.num_of_steps < self.burn_in:
            return

        self.q_eval.optimizer.zero_grad()

        states, actions, rewards, dones, states_ = self.memory.sample(self.batch_size)

        rewards = T.tensor(rewards).reshape(1,-1).to(self.device)
        dones = T.tensor(dones).reshape(1,-1).to(self.device)
        actions = T.tensor(actions).reshape(-1,1).to(self.device)

        # [batch_size, 1, n_actions] --> [batch_size, n_actions]
        q_pred = T.gather(self.q_eval.forward(states).squeeze(1), 1, actions).float()

        # [batch_size, 1, n_actions] --> max [batch_size, n_actions]
        with T.no_grad():
            q_next = self.q_target(states_).squeeze(1).max(dim=1)[0].detach()
            mask = 1 - dones
            target = (rewards + self.gamma * q_next * mask).float().to(self.device)
        
        loss = self.q_eval.loss(q_pred.view(-1), target.view(-1)).to(self.device)
        loss.backward()

        self.q_eval.optimizer.step()

        self.decrement_epsilon()

    def replace_target_net(self):
        if self.num_of_steps % self.sync_target_frames == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

    def loop(self, end, in_queue, out_queue):

        ct = 0
        t_start = timeit.default_timer()
        t_per_frame = int(1000 / FPS_LIMIT)-1 # estimated time per frame

        scores = []
        episode_score = 0
        avg_score = best_score = -9999

        num_of_episodes = 0
        current_obs = previous_obs = reward = action = None

        self.reset()

        while True:
            
            t0 = timeit.default_timer()

            if end.is_set():
                break

            try:
                data, obs = in_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.001)
                continue

            # immediately send to gui
            try:
                out_queue.put_nowait((data, obs))
            except queue.Full:
                pass

            try:

                current_obs, reward, is_done = data, obs.get('reward'), obs.get('is_done')

                if not current_obs is None and \
                   not previous_obs is None and \
                   (self.num_of_steps % self.learn_every) == 0:
                    self.learn(previous_obs, action, reward, current_obs)

                if (not current_obs is None) and (not previous_obs is None):
                    self.store_transition(previous_obs, action, reward, is_done, current_obs)

                if reward:
                    episode_score += reward

                if is_done:

                    current_obs = previous_obs = reward = action = is_done = None
                                        
                    num_of_episodes += 1

                    print('... ep: %s, average score: %.2f; last score: %.2f, best: %.2f ...' % (num_of_episodes, avg_score, episode_score, best_score))
                    print('... eps: %.3f' % (self.eps))

                    if episode_score > 0 and episode_score > avg_score:
                        scores.append(episode_score)
                        avg_score = np.mean(scores)
                        self.save_models()

                    if episode_score > best_score:
                        best_score = episode_score

                    episode_score = 0
                    self.reset()

                    continue
                    
                action = self.choose_action(current_obs)
                
                self.__act(self.action_space_named[action])

                previous_obs = current_obs
                self.num_of_steps += 1

                self.replace_target_net()

            except Exception as err:
                print(''.join(traceback.format_exception(etype=type(err), value=err, tb=err.__traceback__)))

            ct += 1
            if ct % LOG_EVERY_N_FRAME == 0:
                print(f'Agent working at {(ct / (timeit.default_timer() - t_start)):.2f} fps') 
                ct = 0
                t_start = timeit.default_timer()
            
            t_delay = (t_per_frame - ((timeit.default_timer() - t0)))*0.001
            if t_delay > 0:
                time.sleep(t_delay) # delay (fps limit)