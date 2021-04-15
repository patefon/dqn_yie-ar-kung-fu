import logging
import os
import sys
import time
from subprocess import Popen

from consts import EMULATOR_WINDOW_NAME

class Game():

    def __init__(self, emulator_path, rom_path, rom_name, pid=None):
        self.pid = pid
        self.emulator_path = emulator_path
        self.rom_path = rom_path
        self.rom_name = rom_name        
        self.process = None

    def run(self, end):
        
        if self.process:
            return
        
        # args = f'{self.emulator_path} -L ./emu/nestopia_libretro.so {self.rom_path}'
        args = f'{self.emulator_path} {self.rom_path}'
        self.process = Popen(args, shell=True, preexec_fn=os.setpgrp)
        
        time.sleep(5)
        self.__always_on_top()

        self.pid.value = int(self.process.pid)
        print(chr(27) + "[2J") 
        
        try:
            while True:
                if end.is_set():
                    self.stop(pid=self.pid.value)
                    return
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop(pid=self.pid.value)

    def stop(self, pid=None):
        if not pid:
            return
        try:
            os.killpg(pid, 9)
            self.process = None
        except ProcessLookupError:
            pass

    def __always_on_top(self):
        cmd = f'/usr/bin/wmctrl wmctrl -r {EMULATOR_WINDOW_NAME} -e 0,5,5,512,496 && /usr/bin/wmctrl wmctrl -r {EMULATOR_WINDOW_NAME} -b add,above'
        Popen(cmd, shell=True, preexec_fn=os.setpgrp)