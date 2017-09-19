PyGenSound
==========

Read an audio file or generate sound, compute it, and write to file.

ファイルから音を読み込んだり周波数から生成して、重ねたり繋げたりして、ファイルに保存するやつ。

[![Build Status](https://travis-ci.org/macrat/PyGenSound.svg?branch=master)](https://travis-ci.org/macrat/PyGenSound)
[![Code Climate](https://codeclimate.com/github/macrat/PyGenSound/badges/gpa.svg)](https://codeclimate.com/github/macrat/PyGenSound)
[![Coverage Status](https://coveralls.io/repos/github/macrat/PyGenSound/badge.svg?branch=master)](https://coveralls.io/github/macrat/PyGenSound?branch=master)

## install
``` shell
$ git clone http://github.com/macrat/PyGenSound
$ cd PyGenSound
$ pip3 install .
```

## uninstall
``` shelll
$ pip3 uninstall PyGenSound
```

## example
This is code for generating time signal sound of NHK.

NHKの時報を作るやつ。

``` python
import gensound


wait = gensound.Sound.silence().repeat(0.9)  # Generate 0.9 seconds silence
a = gensound.Sound.from_sinwave(440, duration=0.1, volume=1.0).concat(wait)  # Generate 440Hz sin wave 0.1 seconds, and 0.9 seconds silence
b = gensound.Sound.from_sinwave(880, duration=1.0, volume=1.0)  # Generate 880Hz sin wave 1 seconds

time_signal = gensound.concat(a, a, a, b)  # Concatenate those

time_signal.write('test.wav')  # Save to test.wav
time_signal.play()  # Play sound
```

## requirements
- python3
- numpy
- scipy
- PySoundFile
- PyAudio

## License
MIT License
