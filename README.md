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
a = gensound.Sound.from_sinwave(440, duration=0.1, volume=1.0).concat(wait)  # Generate 440Hz sin wave 0.1 seconds, and 0.9 seconds silence
b = gensound.Sound.from_sinwave(880, duration=1.0, volume=1.0)  # Generate 880Hz sin wave 1 seconds

time_signal = gensound.concat(a, a, a, b)  # Concatenate those

time_signal.write('test.wav')  # Save to test.wav
time_signal.play()  # Play sound
```

``` shell
$ python3 gensound.py
$ mplayer test.wav
```

## requirements
- python3
- numpy
- scipy
- PySoundFile
- PyAudio

## License
MIT License
