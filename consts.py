# NES EMULATOR
#
# We use Nestopia to emulate a game console
# 
# Installation (Ubuntu/Debian/Mint):
# sudo apt update
# sudo apt install nestopia
# 
EMULATOR_PATH = '/usr/games/nestopia'
EMULATOR_WINDOW_NAME = 'yie_ar_kung-fu'
ROM_PATH = './roms/yie_ar_kung-fu.nes'
ROM_NAME = 'yie_ar_kung-fu'

# GAME WINDOW LOCATION
# define location of the emulator window
GAME_WINDOW_XY = {
    "top": 60,
    "left": 5,
    "width": 510,
    "height": 435
}

# CAPTURE
# limits capturing frame rate
FPS_LIMIT = 100 
LOG_EVERY_N_FRAME = 1000

# DETECTION
HEALTH_COLOR_CODE = 161

GAME_X0 = 15
GAME_W = 480
GAME_Y0 = 204
GAME_H = 140

# NES FONT ENCODED
FONT_ENCODED = './resources/font.pickle'
