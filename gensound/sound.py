""" The Sound class for generating sound


>>> from gensound.effect import LinearFadeOut
>>> fade_out = LinearFadeOut()
>>> a = Sound.from_sinwave(440, duration=0.1, volume=1.0)
>>> a = fade_out.apply(a)
>>> b = Sound.from_sinwave(880, duration=2.0, volume=1.0)
>>> b = fade_out.apply(b)
>>> wait = Sound.silence(duration=0.9)
>>> #concat(a, wait, a, wait, a, wait, b).write('test.wav')
"""

import time
import typing

import numpy
import pyaudio
import soundfile


def _repeat_array(sound: numpy.array, want_length: int) -> numpy.array:
    """ Repeat numpy.array for enlarging a sound duration

    >>> array = numpy.array([1, 2, 3])
    >>> numpy.allclose(_repeat_array(array, 6),
    ...                numpy.array([1, 2, 3, 1, 2, 3]))
    True
    >>> numpy.allclose(_repeat_array(array, 8),
    ...                numpy.array([1, 2, 3, 1, 2, 3, 1, 2]))
    True
    >>> numpy.allclose(_repeat_array(array, 2), numpy.array([1, 2]))
    True


    sound       -- Sound data for repeat.
    want_length -- The length of an output array.

    return -- Tepeated sound data.
    """

    if want_length <= 0:
        raise ValueError(
            'want_length must be greater than 0 but got {}'.format(want_length)
        )

    if len(sound) <= 0:
        raise ValueError('sound should have least one element')

    if len(sound.shape) != 1:
        raise ValueError('sound should single dimension')

    need_length = int(numpy.ceil(want_length / len(sound)))
    drop_length = int(need_length * len(sound) - want_length)

    repeated = numpy.repeat(sound.reshape([1, -1]),
                            need_length,
                            axis=0).flatten()

    if drop_length > 0:
        return repeated[:-drop_length]
    else:
        return repeated


class InvalidFrequencyError(ValueError):
    """ The exception that raises when passed invalid frequency """

    def __init__(self, freq: float) -> None:
        super().__init__(
            'frequency must be greater than 0 but got {}'.format(freq)
        )
        self.frequency = freq


class InvalidSamplerateError(InvalidFrequencyError):
    """ The exception that raises when passed invalid samplerate """

    def __init__(self, freq: float) -> None:
        ValueError.__init__(
            self,
            'samplerate must be greater than 0 but got {}'.format(freq)
        )
        self.frequency = freq


class DifferentSamplerateError(InvalidSamplerateError):
    """ The exception that raises when different samplerates of sounds to joint
    """

    def __init__(self, frequencies: typing.Tuple[float]) -> None:
        ValueError.__init__(
            self,
            'all samplerates must be the same value but got {}'.format(
                ' and '.join(str(x) for x in set(frequencies))
            )
        )
        self.frequency = frequencies


class InvalidDurationError(ValueError):
    """ The exception that raises when passed sound was invalid duration """

    def __init__(self, duration: float) -> None:
        super().__init__('duration of sound must not 0 or short but got {}'
                         .format(duration))
        self.duration = duration


class OutOfDurationError(IndexError):
    """ The exception that raises when passed index that out of duration """

    def __init__(self, duration: float, min_: float, max_: float) -> None:
        super().__init__('index must between {} to {} but got {}'
                         .format(min_, max_, duration))
        self.duration = duration
        self.min = min_
        self.max = max_


class InvalidVolumeError(ValueError):
    """ The exception that raises when passed invalid volume """

    def __init__(self, volume: float) -> None:
        super().__init__(
            'volume must be between 0.0 and 1.0 but got {}'.format(volume)
        )
        self.volume = volume


def _assertFrequency(frequency: float) -> None:
    """ Check if the frequency is greater than 0 """

    if frequency <= 0:
        raise InvalidFrequencyError(frequency)


def _assertSamplerate(samplerate: float) -> None:
    """ Check if the samplerate is greater than 0 """

    if samplerate <= 0:
        raise InvalidSamplerateError(samplerate)


def _assertSameSamplerate(samplerates: typing.Sequence[float]) -> None:
    """ Check if all samplerates is same value """

    if any(samplerates[0] != s for s in samplerates[1:]):
        raise DifferentSamplerateError(tuple(samplerates))


def _assertDuration(duration: float) -> None:
    """ Check if the duration is longer than 0 """

    if duration <= 0:
        raise InvalidDurationError(duration)


def _assertVolume(volume: float) -> None:
    """ Check if the volume in between 0.0 to 1.0 """

    if volume < 0.0 or 1.0 < volume:
        raise InvalidVolumeError(volume)


class Sound:
    """ The class for handling sound

    >>> alpha = Sound.from_sinwave(440, duration=1.0, volume=0.5)  # gen 440Hz
    >>> beta = Sound.from_sinwave(880, duration=1.0, volume=0.5)
    >>> double = alpha.overlay(beta)
    >>> concated = concat(alpha, beta, double)  # concatenate sounds
    >>> # concated.write('out.wav')  # save into file
    """

    def __init__(self, data: numpy.array, samplerate: float) -> None:
        """
        data       -- Sound data array. must be single channel.
                      Will clipping if value were out of -1.0 to 1.0.
        samplerate -- Sampling rate of sound data.
        """

        _assertDuration(len(data))
        _assertSamplerate(samplerate)

        if len(data.shape) != 1:
            raise ValueError('multiple channel sound is not supported')

        self.data = data.clip(-1.0, 1.0)
        self._samplerate = samplerate

    @classmethod
    def from_sinwave(cls,
                     frequency: float,
                     duration: float = 1.0,
                     volume: float = 1.0,
                     samplerate: float = 44100,
                     smooth_end: bool = True) -> 'Sound':
        """ Generate sine wave sound

        This example makes 440Hz sine wave sound.

        >>> sound = Sound.from_sinwave(440)

        Can set duration and volume with arguments.
        This example makes 2 seconds, 50% volume.

        >>> sound = Sound.from_sinwave(440, duration=2.0, volume=0.5)

        Can make 2 seconds sine wave with repeat too. But, this way may make
        noise at the joint point of sounds.
        Recommend using from_sinwave() as possible.


        Make a smooth end if smooth_end is true. but duration will inaccuracy.
        This error is not critical most cases so smooth_end is true by default.

        >>> sound = Sound.from_sinwave(880, duration=1.0, smooth_end=True)
        >>> sound.duration
        1.017687074829932

        Please pass false to smooth_end if want accurate duration. But please
        be careful, may make noise in end of sound if disable smooth_end.

        >>> sound = Sound.from_sinwave(880, duration=1.0, smooth_end=False)
        >>> sound.duration
        1.0


        frequency  -- Frequency of new sound.
        duration   -- Duration in seconds of new sound.
        volume     -- The volume of new sound.
        samplerate -- Sampling rate of new sound.
        smooth_end -- Do make smooth end or not. Please see above.

        return -- A new Sound instance.
        """

        _assertDuration(duration)
        _assertFrequency(frequency)
        _assertSamplerate(samplerate)
        _assertVolume(volume)

        wavelength = samplerate / frequency

        one_wave = numpy.sin(
            numpy.arange(wavelength) / wavelength * 2 * numpy.pi
        ) * volume

        repeat_count = int(numpy.round(duration * samplerate / wavelength))
        repeated = numpy.repeat(one_wave.reshape([1, -1]),
                                repeat_count,
                                axis=0).flatten()

        if smooth_end is False:
            repeated = repeated[:int(numpy.round(duration * samplerate))]

        return cls(repeated, samplerate)

    @classmethod
    def from_sawtoothwave(cls,
                          frequency: float,
                          duration: float = 1.0,
                          volume: float = 1.0,
                          samplerate: float = 44100) -> 'Sound':
        """ Generate sawtooth wave sound


        frequency  -- Frequency of new sound.
        duration   -- Duration in seconds of new sound.
        volume     -- The volume of new sound.
        samplerate -- Sampling rate of new sound.

        return -- A new Sound instance.
        """

        _assertDuration(duration)
        _assertFrequency(frequency)
        _assertSamplerate(samplerate)
        _assertVolume(volume)

        count = numpy.arange(0, duration, 1 / samplerate) * frequency
        data = count % 1
        data /= data.max()

        return cls((data * 2 - 1) * volume, samplerate)

    @classmethod
    def silence(cls,
                duration: float = 1.0,
                samplerate: float = 44100) -> 'Sound':
        """ Generate silent sound

        duration   -- Duration of new sound.
        samplerate -- Sampling rate of new sound.

        return -- A new Sound instance.
        """

        _assertDuration(duration)
        _assertSamplerate(samplerate)

        length = int(numpy.round(duration * samplerate))
        return cls(numpy.array([0] * length), samplerate)

    @classmethod
    def from_whitenoise(cls,
                        duration: float = 1.0,
                        volume: float = 1.0,
                        samplerate: float = 44100) -> 'Sound':
        """ Generate white noise

        Making white noise.
        >>> noise = Sound.from_whitenoise()


        duration   -- Duration in seconds of new sound.
        volume     -- The volume of new sound.
        samplerate -- Sampling rate of new sound.

        return -- A new Sound instance.
        """

        _assertDuration(duration)
        _assertSamplerate(samplerate)
        _assertVolume(volume)

        length = int(numpy.round(duration * samplerate))
        return cls(numpy.random.rand(length) * volume, samplerate)

    @classmethod
    def from_file(cls, file_: typing.Union[str, typing.BinaryIO]) -> 'Sound':
        """ Read sound from file or file-like

        file_   -- File name or file-like object.

        return -- A new Sound instance.
        """

        data, samplerate = soundfile.read(file_)
        if len(data.shape) != 1:
            data = numpy.average(data, axis=1)
        data /= numpy.max([-data.min(), data.max()])
        return cls(data, samplerate)

    @classmethod
    def from_array(cls,
                   array: typing.Sequence[float],
                   samplerate: float) -> 'Sound':
        """ Make new sound from float array

        This method is same as passing numpy.array to the Sound constructor.

        >>> (Sound.from_array([-0.1, 0.0, 1.0], 3)
        ...  == Sound(numpy.array([-0.1, 0.0, 1.0]), 3))
        True


        array      -- Sound data. Elements must in between -1.0 to 1.0.
        samplerate -- Sampling rate of new sound.

        return -- A new Sound instance.
        """

        return Sound(numpy.array(array), samplerate)

    @classmethod
    def from_fft(cls,
                 spectrum: numpy.array,
                 samplerate: float = None) -> 'Sound':
        """ Make new sound from spectrum data like a result from fft() method.


        spectrum   -- A spectrum data. Please see fft() method about a format.
        samplerate -- Sampling rate of new sound. Use spectrum data if None.

        return -- A new Sound instance.
        """

        if samplerate is None:
            samplerate = spectrum[-1, 0].real * 2

        _assertSamplerate(samplerate)

        return Sound(numpy.fft.irfft(spectrum[:, 1]), samplerate)

    @property
    def duration(self) -> float:
        """ Duration in seconds of this sound """

        return len(self.data) / self.samplerate

    @property
    def samplerate(self) -> float:
        """ Sampling rate of this sound """

        return self._samplerate

    @property
    def volume(self) -> float:
        """ Volume of this dound

        This volume means the maximum value of the wave.
        Please be careful that is not gain.
        """

        return max(self.data.max(), -self.data.min())

    def __eq__(self, another: typing.Any) -> bool:
        """ Compare with another Sound instance """

        if not isinstance(another, Sound):
            return False

        return (self.samplerate == another.samplerate
                and numpy.allclose(self.data, another.data))

    def __ne__(self, another: typing.Any) -> bool:
        return not (self == another)

    def __getitem__(self, position: typing.Union[float, slice]) -> 'Sound':
        """ Slice a sound

        If passed float a position, returns very short sound that only has
        1/samplerate seconds.

        >>> sound = Sound.from_sinwave(440)

        >>> short_sound = sound[0.5]
        >>> short_sound.duration == 1 / sound.samplerate
        True


        The step of the slice is not supported. Will raise ValueError if passed
        slice that has a step.


        position -- A position in seconds in float or slice.

        return -- A new Sound instance that sliced.
        """

        if isinstance(position, (int, float)):
            if position < 0 or self.duration < position:
                raise OutOfDurationError(position, 0.0, self.duration)

            index = int(numpy.round(position * self.samplerate))
            if index >= len(self.data):
                index = len(self.data) - 1
            data = numpy.array([self.data[index]])
        else:
            if position.step is not None:
                raise ValueError('step is not supported')

            start = position.start
            if start is not None:
                start = int(numpy.round(start * self.samplerate))

            stop = position.stop
            if stop is not None:
                stop = int(numpy.round(stop * self.samplerate))

            data = self.data[start:stop]

        return Sound(data, self.samplerate)

    def fft(self) -> numpy.array:
        """ Calculate fft

        return -- An array that pair of frequency and value.
        """

        return numpy.hstack([
            (
                numpy.fft.rfftfreq(len(self.data)) * self.samplerate
            ).reshape([-1, 1]),
            numpy.fft.rfft(self.data).reshape([-1, 1]),
        ])

    def change_volume(self, volume: float) -> 'Sound':
        """ Create a new instance that changed volume

        This volume means the maximum value of the wave.
        Please be careful that is not gain.


        >>> sound = Sound.from_sinwave(440, volume=1.0)

        >>> 0.999 <= sound.data.max() <= 1.0
        True
        >>> -0.999 >= sound.data.min() >= -1.0
        True

        >>> half = sound.change_volume(0.5)
        >>> 0.499 <= half.volume <= 0.501
        True


        volume -- New volume.

        return -- A new Sound instance that changed volume.
        """

        _assertVolume(volume)

        return Sound(
            self.data * (volume / self.volume),
            self.samplerate
        )

    def repeat(self, duration: float) -> 'Sound':
        """ Create a new instance that repeated same sound

        >>> sound = Sound.from_sinwave(440)
        >>> sound.repeat(5).duration
        5.0

        This function can not only repeat but trimming.
        But recommend use slice because become hard to understand if using it
        for trimming.

        >>> sound.repeat(0.5).duration
        0.5
        >>> sound.repeat(0.5) == sound[:0.5]
        True


        duration -- Duration in seconds to repeat.

        return -- A new Sound instance that repeated same sound.
        """

        _assertDuration(duration)

        return Sound(
            _repeat_array(self.data,
                          int(numpy.round(duration * self.samplerate))),
            self.samplerate,
        )

    def concat(self, other: 'Sound') -> 'Sound':
        """ Create a new instance that concatenated another sound

        >>> sound = Sound.from_sinwave(440, duration=3)
        >>> a, b = sound[:1], sound[1:]
        >>> a.concat(b) == sound
        True

        Recommend using gensound.concat if concatenate many sounds. Because
        gensound.concat is optimized for many sounds.

        >>> concat(a, b) == a.concat(b)
        True


        other -- The sound that concatenates after of self.
                 Must it has same sampling rate.

        return -- A new Sound that concatenated self and other.
        """

        _assertSameSamplerate([self.samplerate, other.samplerate])

        return Sound(numpy.hstack([self.data, other.data]), self.samplerate)

    def overlay(self, other: 'Sound') -> 'Sound':
        """ Create a new instance that was overlay another sound

        >>> a = Sound.from_array([0.1, 0.2], 1)
        >>> b = Sound.from_array([0.1, 0.2, 0.3], 1)
        >>> a.overlay(b) == Sound.from_array([0.2, 0.4, 0.3], 1)
        True

        Recommend using gensound.overlay if overlay many sounds. Because
        gensound.overlay is optimized for many sounds.

        >>> overlay(a, b) == a.overlay(b)
        True


        other -- The sound that overlay.

        return -- A new Sound that overlay another sound.
        """

        _assertSameSamplerate([self.samplerate, other.samplerate])

        x = self.data
        y = other.data

        if len(x) > len(y):
            y = numpy.hstack([y, [0] * (len(x) - len(y))])
        if len(y) > len(x):
            x = numpy.hstack([x, [0] * (len(y) - len(x))])

        return Sound(x + y, self.samplerate)

    def write(self,
              file_: typing.Union[str, typing.BinaryIO],
              format_: typing.Optional[str] = None) -> None:
        """ Write sound into file or file-like

        file_   -- A file name or file-like object to write sound.
        format_ -- Format type of output like a 'wav'. Automatically detect
                   from file name if None.
        """

        soundfile.write(file_, self.data, self.samplerate, format=format_)

    def play(self) -> None:
        """ Play sound """

        class Callback:
            def __init__(self, data: numpy.array) -> None:
                self.idx = 0
                self.data = data

            def next(self,
                     in_: None,
                     frame_count: int,
                     time_info: dict,
                     status: int) -> typing.Tuple[numpy.array, int]:

                d = self.data[self.idx:self.idx + frame_count]
                self.idx += frame_count

                flag = pyaudio.paContinue
                if len(d) <= 0:
                    flag = pyaudio.paComplete

                return d.astype(numpy.float32), flag

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=self.samplerate,
                         output=True,
                         stream_callback=Callback(self.data).next)

        stream.start_stream()
        while stream.is_active():
            time.sleep(0.1)
        stream.stop_stream()

        stream.close()
        pa.terminate()


def concat(*sounds: Sound) -> Sound:
    """ Concatenate multiple sounds

    >>> a = Sound.from_array([0.1, 0.2], 1)
    >>> b = Sound.from_array([0.3, 0.4], 1)
    >>> c = Sound.from_array([0.5, 0.6], 1)
    >>> concat(a, b, c) == Sound.from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 1)
    True


    sounds -- Sound instances to concatenate. Must they has some sampling rate.

    return -- A concatenated Sound instance.
    """

    _assertSameSamplerate([s.samplerate for s in sounds])

    return Sound(numpy.hstack([x.data for x in sounds]), sounds[0].samplerate)


def overlay(*sounds: Sound) -> Sound:
    """ Overlay multiple sounds

    BE CAREFUL: This function doesn't care about clipping. Perhaps, need to
                change volume before use this if overlay many sounds.

    >>> a = Sound.from_array([0.1, 0.2], 1)
    >>> b = Sound.from_array([0.3, 0.4], 1)
    >>> c = Sound.from_array([0.5, 0.6], 1)
    >>> overlay(a, b, c) == Sound.from_array([0.9, 1.0], 1)
    True

    The second element of this sample isn't 1.2 but 1.0 because of clipping was
    an occurrence.



    sounds -- Sound instances to overlay. Must they has some sampling rate and
              same duration.

    return -- A Sound instance that overlay all sounds.
    """

    _assertSameSamplerate([s.samplerate for s in sounds])

    longest = max(len(x.data) for x in sounds)
    padded = numpy.array([numpy.hstack([x.data, [0] * (longest - len(x.data))])
                          for x in sounds])

    return Sound(padded.sum(axis=0), sounds[0].samplerate)
