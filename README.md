PyGenSound
==========

Read an audio file or generate sound, compute it, and write to file.

ファイルから音を読み込んだり周波数から生成して、重ねたり繋げたりして、ファイルに保存するやつ。

## example
This is code for generating time signal sound of NHK.

NHKの時報を作るやつ。

``` python
import gensound


wait = gensound.Sound.silence().repeat(0.9)  # Generate 0.9 seconds silence
a = gensound.Sound.from_sinwave(440, volume=1.0).repeat(0.1).concat(wait)  # Generate 440Hz sin wave 0.1 seconds, and 0.9 seconds silence
b = gensound.Sound.from_sinwave(880, volume=1.0).repeat(1.5)  # Generate 880Hz sin wave 1.5 seconds

concat(a, a, a, b).write('test.wav')  # Concatenate those and save to test.wav
```

``` shell
$ python3 gensound.py
$ mplayer test.wav
```

## requirements
- python3
- numpy
- PySoundFile

## License
MIT License
