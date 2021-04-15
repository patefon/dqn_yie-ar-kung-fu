from pynput.keyboard import Key, Controller
import datetime, time

KEYS = {
    'reset': Key.f12,
    'qsave': Key.f5,
    'qload': Key.f6,
    'start': Key.enter,
    'select': Key.shift_r,
    'A': 'z',
    'B': 'a',
    'T_A': 'x',
    'T_B': 's',
    'up': Key.up,
    'down': Key.down,
    'left': Key.left,
    'right': Key.right
}

class GameController():

    def __init__(self):
        self.kb = Controller()
        self.current_move = None
    
    def confirm(self):
        self.stop()
        k = KEYS.get('start')
        self.kb.press(key=k)
        time.sleep(0.5)
        self.kb.release(key=k)

    def reset(self):
        self.stop()
        print(f'Game resetted at {datetime.datetime.now()}')
        k = KEYS.get('reset')
        self.kb.press(key=k)
        time.sleep(0.5)
        self.kb.release(key=k)

    def stop(self):
        if self.current_move:
            self.kb.release(key=self.current_move)

    def move(self, direction):
        assert direction in set(['left','right','up','down'])
        k = KEYS.get(direction)
        if self.current_move == k:
            pass
        self.stop()
        self.kb.press(key=k)
        self.current_move = k

    def kick(self, direction):
        assert direction in set(['right','left'])
        k = KEYS.get(direction)
        kick_k = KEYS.get('B')
        if self.current_move == kick_k:
            pass
        self.stop()
        self.kb.tap(key=k) # switch direction
        # kick
        self.kb.press(key=kick_k)
        self.current_move = kick_k

    def sit_kick(self, direction):
        assert direction in set(['right','left'])
        k = KEYS.get(direction)
        kick_k = KEYS.get('B')
        if self.current_move == kick_k:
            pass
        self.stop()
        self.kb.tap(key=k) # switch direction
        # kick
        with self.kb.pressed(KEYS.get('down')):
            self.kb.press(kick_k)
            self.current_move = kick_k
        
    def sit(self):
        k = KEYS.get('down')
        if self.current_move == k:
            pass
        self.stop()
        self.kb.press(key=k)
        self.current_move = k


