# For The King Oracle

A real-time damage probability calculator for the RPG game [For The King](https://ironoakgames.com/).

The goal of this simple program is to help the player in doing what she likes the most - playing efficiently. During fights, many times you will be confused on what attack to choose: maybe this precise strike will do... but what about this powerful, yet inaccurate blow?
This program calculates for you the probability of dealing a certain amount of damage - or, more conveniently, the probability of dealing _at least_ that damage you need.

## Downloads

You can check out the current version in the [releases](https://github.com/Pentracchiano/for-the-king-oracle/releases) page. Just download the .zip file, extract it, and execute the .exe inside.


## How it works

The concept is quite simple: a Qt GUI is always on top of your fullscreen app, and periodically reads the screen.
It processes the image using simple heuristics and OpenCV2 computer vision algorithms in `screen_reader.py` and passes the parsed results to `fight.py`. Here the calculations are done using SciPy's modelization of the binomial distribution, which is what you see in the GUI.

--- 
### License

This software is distributed according to the GNUv3 license: for more information, check the [LICENSE](https://github.com/Pentracchiano/for-the-king-oracle/blob/master/LICENSE) file.
