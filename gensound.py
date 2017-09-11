""" Generate sound like a chiptune

Read an audio file or generate sound, compute it, and write to file.
"""

import typing

import numpy
import soundfile


Number = typing.Union[int, float]


def _repeat_array(sound: numpy.array, want_length: int) -> numpy.array:
    """ Repeat numpy.array for enlarging a sound duration

    sound       -- Sound data for repeat.
    want_length -- The length of an output array.

    return -- Tepeated sound data.


    >>> array = numpy.array([1, 2, 3])
    >>> numpy.allclose(_repeat_array(array, 6),
    ...                numpy.array([1, 2, 3, 1, 2, 3]))
    True
    >>> numpy.allclose(_repeat_array(array, 8),
    ...                numpy.array([1, 2, 3, 1, 2, 3, 1, 2]))
    True
    >>> numpy.allclose(_repeat_array(array, 2), numpy.array([1, 2]))
    True
    """
    assert 0 <= want_length

    need_length = int(numpy.ceil(want_length / len(sound)))
    drop_length = int(need_length * len(sound) - want_length)

    repeated = numpy.repeat(sound.reshape([1, -1]),
                            need_length,
                            axis=0).flatten()

    if drop_length > 0:
        return repeated[:-drop_length]
    else:
        return repeated


class Sound:
    """ The class for handling sound

    >>> alpha = Sound.from_sinwave(440, duration=1.0, volume=0.5)  # gen 440Hz
    >>> beta = Sound.from_sinwave(880, duration=1.0, volume=0.5)
    >>> double = alpha.overlay(beta)
    >>> concated = concat(alpha, beta, double)  # concatenate sounds
    >>> # concated.write('out.wav')  # save into file
    """

    def __init__(self, data: numpy.array, samplerate: Number) -> None:
        """
        data       -- Sound data array. must be single channel.
                      Will clipping if value were out of -1.0 to 1.0.
        samplerate -- Sampling rate of sound data.
        """
        assert 0 <= samplerate
        assert len(data.shape) == 1
        assert 0 < len(data)

        self.data = data.clip(-1.0, 1.0)
        self.samplerate = samplerate

    @classmethod
    def from_sinwave(cls,
                     frequency: Number,
                     duration: float = 1.0,
                     volume: float = 1.0,
                     samplerate: Number = 44100) -> 'Sound':
        """ Generate sin wave sound

        frequency  -- Frequency of new sound.
        duration   -- Duration in seconds of new sound.
        volume     -- The volume of new sound.
        samplerate -- Sampling rate of new sound.

        return -- A new Sound instance.
        """
        assert 0 < frequency
        assert 0 < duration
        assert 0.0 <= volume <= 1.0

        wavelength = samplerate / frequency

        one_wave = numpy.sin(numpy.arange(wavelength)/wavelength * 2*numpy.pi)

        repeat_count = int(numpy.round(duration * samplerate / wavelength))
        repeated = numpy.repeat(one_wave.reshape([1, -1]),
                                repeat_count,
                                axis=0)

        return cls(repeated.flatten() * volume, samplerate)

    @classmethod
    def silence(cls, samplerate: Number = 44100) -> 'Sound':
        """ Generate silent sound

        This function returns VERY VERY short sound.
        Please use repeat function.

        samplerate -- Sampling rate of new sound.

        return -- A new Sound instance.
        """

        return cls(numpy.array([0]), samplerate)

    @classmethod
    def from_file(cls, file_: typing.Union[str, typing.BinaryIO]) -> 'Sound':
        """ Read sound from file or file-like

        file_ -- File name or file-like object.

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
                   samplerate: Number) -> 'Sound':
        """ Make new sound from float array

        array      -- Sound data. Elements must in between -1.0 to 1.0.
        samplerate -- Sampling rate of new sound.

        return -- A new Sound instance.


        >>> s = Sound.from_array([-0.1, 0.0, 0.1], 3)
        >>> s.samplerate
        3
        >>> s == Sound.from_array([-0.1, 0.0, 0.1], 3)
        True
        """

        return Sound(numpy.array(array), samplerate)

    @property
    def duration(self) -> float:
        """ Duration in seconds of this sound


        >>> s = Sound.from_array([0.1, 0.2, 0.3], 1)
        >>> s.duration
        3.0
        >>> s.samplerate = 2
        >>> s.duration
        1.5
        """

        return len(self.data) / self.samplerate

    def __eq__(self, another: typing.Any) -> bool:
        """ Compare with another Sound instance.

        >>> a = Sound.from_array([0.1, 0.5], 1)
        >>> b = Sound.from_array([0.1, 0.5], 1)
        >>> c = Sound.from_array([-0.1, -0.5], 1)
        >>> d = Sound.from_array([-0.1, -0.5], 2)
        >>> a == b
        True
        >>> a == c
        False
        >>> b == c
        False
        >>> b == d
        False
        >>> c == d
        False
        """

        if not isinstance(another, Sound):
            return False

        return (self.samplerate == another.samplerate
                and numpy.allclose(self.data, another.data))

    def volume(self, vol: float) -> 'Sound':
        """ Create a new instance that changed volume

        This volume means the maximum value of the wave.
        Please be careful that is not gain.

        vol -- New volume.

        return -- A new Sound instance that changed volume.


        >>> s = Sound.from_array([0.1, 0.5], 1)
        >>> s == Sound.from_array([0.1, 0.5], 1)
        True
        >>> s.volume(1) == Sound.from_array([0.2, 1.0], 1)
        True
        >>> s.volume(0.25) == Sound.from_array([0.05, 0.25], 1)
        True
        >>> s.volume(2) == Sound.from_array([0.4, 1.0], 1)
        True
        """

        return Sound(
            self.data * (vol / numpy.max([-self.data.min(), self.data.max()])),
            self.samplerate
        )

    def repeat(self, duration: Number) -> 'Sound':
        """ Create a new instance that repeated same sound

        This function can not only repeat but trimming.
        But recommend use trim function because become hard to understand
        if using it for trimming.

        duration -- Duration in seconds to repeat.

        return -- A new Sound instance that repeated same sound.


        >>> s = Sound.from_array([0.1, 0.2, 0.3], 1)
        >>> s.repeat(6) == Sound.from_array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], 1)
        True
        >>> s.repeat(4) == Sound.from_array([0.1, 0.2, 0.3, 0.1], 1)
        True
        >>> s.repeat(2) == Sound.from_array([0.1, 0.2], 1)
        True
        """
        assert 0 <= duration

        return Sound(
            _repeat_array(self.data,
                          int(numpy.round(duration * self.samplerate))),
            self.samplerate
        )

    def trim(self, duration: Number) -> 'Sound':
        """ Create a new instance that trimmed

        duration -- Duration in seconds of new sound.
                    Must be equals or shorter than original duration.

        return -- A new Sound instance that trimmed.


        >>> s = Sound.from_array([0.1, 0.2, 0.3], 1)
        >>> s.trim(2) == Sound.from_array([0.1, 0.2], 1)
        True
        >>> s.trim(3) == Sound.from_array([0.1, 0.2, 0.3], 1)
        True
        """
        assert 0 <= duration <= self.duration

        return Sound(
            self.data[:int(numpy.round(duration * self.samplerate))],
            self.samplerate
        )

    def split(self, duration: Number) -> typing.Tuple['Sound', 'Sound']:
        """ Spit sound

        duration -- The pivot of sound splitting.

        return -- Splitted sounds.


        >>> s = Sound.from_array([0.1, 0.2, 0.3], 1)
        >>> a, b = s.split(1)
        >>> a == Sound.from_array([0.1], 1)
        True
        >>> b == Sound.from_array([0.2, 0.3], 1)
        True
        """
        assert 0 < duration < self.duration

        pivot = int(numpy.round(duration * self.samplerate))

        return (Sound(self.data[:pivot], self.samplerate),
                Sound(self.data[pivot:], self.samplerate))

    def concat(self, other: 'Sound') -> 'Sound':
        """ Create a new instance that concatenated another sound

        Recommend using gensound.concat if concatenate many sounds. Because
        gensound.concat is optimized for many sounds.

        other -- The sound that concatenates after of self.
                 Must it has same sampling rate.

        return -- A new Sound that concatenated self and other.


        >>> a = Sound.from_array([0.1, 0.2], 1)
        >>> b = Sound.from_array([0.3, 0.4], 1)
        >>> a.concat(b) == Sound.from_array([0.1, 0.2, 0.3, 0.4], 1)
        True
        """
        assert self.samplerate == other.samplerate

        return Sound(numpy.hstack([self.data, other.data]), self.samplerate)

    def overlay(self, other: 'Sound') -> 'Sound':
        """ Create a new instance that was overlay another sound

        Recommend using gensound.overlay if overlay many sounds. Because
        gensound.overlay is optimized for many sounds.

        other -- The sound that overlay.

        return -- A new Sound that overlay another sound.


        >>> a = Sound.from_array([0.1, 0.2], 1)
        >>> b = Sound.from_array([0.1, 0.2, 0.3], 1)
        >>> a.overlay(a) == Sound.from_array([0.2, 0.4], 1)
        True
        >>> a.overlay(b) == Sound.from_array([0.2, 0.4, 0.3], 1)
        True
        >>> b.overlay(a) == Sound.from_array([0.2, 0.4, 0.3], 1)
        True
        """
        assert self.samplerate == other.samplerate

        x = self.data
        y = other.data

        if len(x) > len(y):
            y = numpy.hstack([y, [0] * (len(x) - len(y))])
        if len(y) > len(x):
            x = numpy.hstack([x, [0] * (len(y) - len(x))])

        return Sound(x + y, self.samplerate)

    def write(self, file_: typing.Union[str, typing.BinaryIO]) -> None:
        """ Write sound into file or file-like

        file_ -- A file name or file-like object to write sound
        """

        soundfile.write(file_, self.data, self.samplerate)


def concat(*sounds: Sound) -> Sound:
    """ Concatenate multiple sounds

    sounds -- Sound instances to concatenate. Must they has some sampling rate.

    return -- A concatenated Sound instance.


    >>> a = Sound.from_array([0.1, 0.2], 1)
    >>> b = Sound.from_array([0.3, 0.4], 1)
    >>> c = Sound.from_array([0.5, 0.6], 1)
    >>> concat(a, b, c) == Sound.from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 1)
    True
    """
    assert all(sounds[0].samplerate == s.samplerate for s in sounds[1:])

    return Sound(numpy.hstack([x.data for x in sounds]), sounds[0].samplerate)


def overlay(*sounds: Sound) -> Sound:
    """ Overlay multiple sounds

    BE CAREFUL: This function doesn't care about clipping. Perhaps, need to
                change volume before use this if overlay many sounds.

    sounds -- Sound instances to overlay. Must they has some sampling rate and
              same duration.

    return -- A Sound instance that overlay all sounds.


    >>> a = Sound.from_array([0.1, 0.2], 1)
    >>> b = Sound.from_array([0.3, 0.4], 1)
    >>> c = Sound.from_array([0.5, 0.6], 1)
    >>> overlay(a, b, c) == Sound.from_array([0.9, 1.0], 1)
    True

    The second element of this sample isn't 1.2 but 1.0 because of clipping was
    an occurrence.
    """
    assert all(sounds[0].samplerate == s.samplerate for s in sounds[1:])

    longest = max(len(x.data) for x in sounds)
    padded = numpy.array([numpy.hstack([x.data, [0] * (longest - len(x.data))])
                          for x in sounds])

    return Sound(padded.sum(axis=0), sounds[0].samplerate)


class Effect:
    """ Base class of rsound effect """

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        sound -- Sound instance to appling effect.

        return -- A new Sound instance that applied effect.
        """

        raise NotImplementedError()

    def then(self, effect: 'Effect') -> 'Effect':
        """ Join effect

        effect -- Effect that will apply after this effect.

        return -- Joined effect.
        """

        return JoinedEffect(self, effect)


class JoinedEffect(Effect):
    """ Joined multiple effect """

    def __init__(self, *effects: Effect) -> None:
        self.effects = effects

    def apply(self, sound: Sound) -> Sound:
        """ Apply all effects

        sound -- Sound instance to appling effect.

        return -- A new Sound instance that applied all effects.
        """

        for e in self.effects:
            sound = e.apply(sound)

        return sound


class MaskEffect(Effect):
    """ Masking effect """

    def __init__(self, duration: typing.Optional[Number] = None) -> None:
        """ Initialize

        duration -- Duration in seconds of mask.
        """

        self.duration = duration

    def gen_mask(self, length: int) -> numpy.array:
        """ Generate mask

        length -- Length of mask array.

        return -- Mask value.
        """

        raise NotImplementedError()


class MaskStartEffect(MaskEffect):
    """ Effect that masking start of sound """

    def apply(self, sound: Sound) -> Sound:
        length = len(sound.data)
        if self.duration is not None:
            length = int(numpy.round(self.duration * sound.samplerate))

        mask = self.gen_mask(length)

        return Sound(numpy.hstack([
                        sound.data[:length] * mask[:len(sound.data)],
                        sound.data[length:],
                     ]),
                     sound.samplerate)


class MaskEndEffect(MaskEffect):
    """ Effect that masking end of sound """

    def apply(self, sound: Sound) -> Sound:
        length = len(sound.data)
        if self.duration is not None:
            length = int(numpy.round(self.duration * sound.samplerate))

        offset = max(0, length - len(sound.data))
        mask = self.gen_mask(length)[offset:]

        return Sound(numpy.hstack([
                        sound.data[:-length],
                        sound.data[-length:] * mask,
                     ]),
                     sound.samplerate)


class LinearFadeIn(MaskStartEffect):
    """ Linear fade-in effect

    >>> s = Sound.from_array([1, 1, 1, 1, 1], 1)
    >>> (LinearFadeIn().apply(s)
    ...  == Sound.from_array([0.0, 0.25, 0.5, 0.75, 1.0], 1))
    True
    >>> (LinearFadeIn(duration=3).apply(s)
    ...  == Sound.from_array([0.0, 0.5, 1.0, 1.0, 1.0], 1))
    True
    """

    def gen_mask(self, length: int) -> numpy.array:
        return numpy.arange(length) / (length - 1)


class LinearFadeOut(MaskEndEffect):
    """ Linear fade-out effect

    >>> s = Sound.from_array([1, 1, 1, 1, 1], 1)
    >>> (LinearFadeOut().apply(s)
    ...  == Sound.from_array([1.0, 0.75, 0.5, 0.25, 0.0], 1))
    True
    >>> (LinearFadeOut(duration=3).apply(s)
    ...  == Sound.from_array([1.0, 1.0, 1.0, 0.5, 0.0], 1))
    True
    """

    def gen_mask(self, length: int) -> numpy.array:
        return 1.0 - numpy.arange(length) / (length - 1)


class LowPassFilter(Effect):
    """ Low pass filter

    >>> a = Sound.from_sinwave(100, duration=0.1, volume=1/3)
    >>> b = Sound.from_sinwave(200, duration=0.1, volume=1/3)
    >>> c = Sound.from_sinwave(300, duration=0.1, volume=1/3)
    >>> filtered = LowPassFilter(210).apply(overlay(a, b, c))
    >>> sum((filtered.data - overlay(a, b).data) ** 2) < 0.1
    True
    """

    def __init__(self, freq: Number) -> None:
        """
        freq -- A threshold frequency.
        """

        self.freq = freq

    def apply(self, sound: Sound) -> Sound:
        f = numpy.fft.rfft(sound.data)
        freq = numpy.fft.rfftfreq(len(sound.data))

        f[freq > self.freq / sound.samplerate] = 0

        return Sound(numpy.fft.irfft(f), sound.samplerate)


class HighPassFilter(Effect):
    """ High pass filter

    >>> a = Sound.from_sinwave(100, duration=0.1, volume=1/3)
    >>> b = Sound.from_sinwave(200, duration=0.1, volume=1/3)
    >>> c = Sound.from_sinwave(300, duration=0.1, volume=1/3)
    >>> filtered = HighPassFilter(190).apply(overlay(a, b, c))
    >>> sum((filtered.data - overlay(b, c).data) ** 2) < 0.1
    True
    """

    def __init__(self, freq: Number) -> None:
        """
        freq -- A threshold frequency.
        """

        self.freq = freq

    def apply(self, sound: Sound) -> Sound:
        f = numpy.fft.rfft(sound.data)
        freq = numpy.fft.rfftfreq(len(sound.data))

        f[freq < self.freq / sound.samplerate] = 0

        return Sound(numpy.fft.irfft(f), sound.samplerate)


if __name__ == '__main__':
    fade_out = LinearFadeOut()

    a = Sound.from_sinwave(440, duration=0.1, volume=1.0)
    a = fade_out.apply(a)

    b = Sound.from_sinwave(880, duration=2.0, volume=1.0)
    b = fade_out.apply(b)

    wait = Sound.silence().repeat(0.9)

    concat(a, wait, a, wait, a, wait, b).write('test.wav')
