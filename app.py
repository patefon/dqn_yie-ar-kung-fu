import logging, multiprocessing, signal, time, timeit
from multiprocessing import Pool, Manager

from classes.game import Game
from classes.reader import Grabber
from classes.gui import Screen
from classes.environment import Environment
from classes.agent import Agent

from consts import EMULATOR_PATH, ROM_PATH, ROM_NAME,\
                   GAME_WINDOW_XY

class App():

    def __init__(self):
        self.__init_logging()

    def __init_logging(self):
        self.log = logging.getLogger(__name__)
        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

    def init_process(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def __init_mp_pool(self, pool_size=4):

        if pool_size > 16:
            raise ValueError('num of child processes must be lt 16')
        elif pool_size <= 0:
            raise ValueError('num of child processes must be gt 0')
        elif not isinstance(pool_size, int):
            raise ValueError('num of child processes must integer')

        pool = Pool(processes=pool_size, initializer=self.init_process)

        return pool

    def init_environment(self, pool, end, q_source=None, q_target=None):
        return [pool.apply_async(Environment().loop, args=(end, q_source, q_target,))]

    def init_agent(self, pool, end, q_source=None, q_target=None):
        return [pool.apply_async(Agent().loop, args=(end, q_source, q_target,))]

    def init_grabber(self, pool, end, q_source=None, q_target=None):
        return [pool.apply_async(Grabber(window_xy=GAME_WINDOW_XY).capture, args=(end, q_source, q_target,))]

    def init_gui(self, pool, end, q_source=None, q_target=None):
        return [pool.apply_async(Screen().show, args=(end, q_source, q_target,))]

    def main(self):

        t0 = timeit.default_timer()

        # shared resources manager
        m = Manager()
        # original screenshots of the game window
        raw_frames = m.Queue(maxsize=3)
        # environment data
        env_data = m.Queue(maxsize=2)
        # output images
        output_frames = m.Queue(maxsize=3)

        # end event
        end = m.Event()
        
        # proc's pool
        pool = self.__init_mp_pool(pool_size=16)

        # need to get sub-process (external game) pid to
        # for graceful shutdown
        game_pid = m.Value('pid', None)
        
        game = Game(rom_path=ROM_PATH,
                    emulator_path=EMULATOR_PATH,
                    rom_name=ROM_NAME,
                    pid=game_pid)

        processes = [pool.apply_async(game.run, args=(end,))] + \
                     self.init_environment(pool, end, q_source=raw_frames, q_target=env_data) + \
                     self.init_agent(pool, end, q_source=env_data, q_target=output_frames) + \
                     self.init_grabber(pool, end, q_target=raw_frames) + \
                     self.init_gui(pool, end, q_source=output_frames)

        fin_processes = []

        try:
          while True:
              for proc in processes:
                if (proc.ready() and proc not in fin_processes):
                    fin_processes.append(proc)
              if len(fin_processes) == len(processes):
                  break
        except KeyboardInterrupt:
            self.log.info('\nCaught Ctrl+C, terminating workers.')
            game.stop(game_pid.value)
            pool.terminate()
            pool.join()
        except Exception as err:
            self.log.error('\nMain process err: %s ' % err)
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()
        finally:
            m.shutdown()

        self.log.info('Finished processing.\n'+
                      'Main process worked for %.2f seconds'
                      % (timeit.default_timer() - t0))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    app = App()
    app.main()