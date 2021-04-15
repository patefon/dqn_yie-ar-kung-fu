## AI Kung-Fu
----
This is a demo project built with a goal to explore in more details computer vision (CV) and reincorcement learning (RL) and has no any commercial use.

![alt](resources/preview.gif)

[agent gameplay / youtube](https://www.youtube.com/watch?v=fHK_tgkpKNg "agent gameplay / youtube")

#### Requirements:
- Linux Debian / Ubuntu / Mint
- python 3.8 / pytorch / openCV
- nestopia (as a NES emulator)

#### Goals
- __Ultimate goal:__ beat 50+ levels.
- __Current result:__ no more than 5 levels.

#### An approach:

- Q-learning / Deep Q Network
- do not use frameworks like OpenAI Gym, just plain python and torch as dep.

#### Run an app:

    # 1) install Nestopia (NES emulator):
	~ sudo apt-get update && apt-get install Nestopia
	# 2) Place a rom with a game inside [./roms] dir.
	# 3) Init python venv and install deps:
	~ python3 -m venv [path_to_venv_dir]
	~ source [path_to_venv_dir]/bin/activate
	~ pip install -r requirements.txt
	# 4) Check [./consts.py] and make changes if it necessary
	# 5) Run an app
	~ (venv) python app.py

#### Overview of dependencies:
| Name | Version | License |
| ---------|---------------|---------------|
| Cython | 0.29.23 | Apache Software License |
| cycler | 0.10.0  | BSD |
| dataclasses | 0.6 | Apache Software License |
| evdev            |  1.4.0 |   BSD License |  
| future      | 0.18.2 | MIT License |
| imutils     | 0.5.3  | MIT License |
| kiwisolver  | 1.3.1  | BSD License |
| matplotlib  | 3.4.1  | Python Software Foundation License |
| mss         | 6.1.0  | MIT License |
| numpy       | 1.19.4 | BSD |
| opencv-python | 4.5.1.48  | MIT License |
| Pillow | 8.2.0   | Historical Permission Notice and Disclaimer (HPND) |
| pyparsing     | 2.4.7     | MIT License |
| pynput          |   1.7.3   |  LGPLv3 |
| python-dateutil | 2.8.1   | BSD License |
| six             | 1.15.0  | MIT License |
| torch           | 1.7.0   | BSD License |
| torchvision     | 0.8.0   | BSD |
| typing-extensions | 3.7.4.3 | Python Software Foundation License|
                

#### References and thanks to:
-  "Deep Reinforcment Learning Hands-on" book,  Max Lapan
