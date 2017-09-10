""" Generate sound like a chiptune

Read an audio file or generate sound, compute it, and write to file.
"""

import functools
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
    >>> numpy.allclose(_repeat_array(array, 6), numpy.array([1, 2, 3, 1, 2, 3]))
    True
    >>> numpy.allclose(_repeat_array(array, 8), numpy.array([1, 2, 3, 1, 2, 3, 1, 2]))
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

    >>> alpha = Sound.from_sinwave(440, volume=0.5).repeat(1.0)  # make 440 1sec
    >>> beta = Sound.from_sinwave(880, volume=0.5).repeat(1.0)
    >>> double = alpha.overlay(beta)
    >>> concat = alpha.concat(beta).concat(double)  # concatenate sounds
    >>> # concat.write('out.wav')  # save into file
    """

    def __init__(self, data: numpy.array, samplerate: Number) -> None:
        """
        data       -- Sound data array. must be single channel.
                      Will clipping if value were out of -1.0 to 1.0.
        samplerate -- Sampling rate of sound data.
        """
        assert samplerate > 0
        assert len(data.shape) == 1

        self.data = data.clip(-1.0, 1.0)
        self.samplerate = samplerate

    @classmethod
    def from_sinwave(cls, frequency: Number,
                          volume: float = 1.0,
                          samplerate: Number = 44100) -> 'Sound':
        """ Generate sin wave sound

        This function returns very very short sound. Please use repeat function.

        frequency  -- Frequency of new sound.
        volume     -- The volume of new sound.
        samplerate -- Sampling rate of new sound.

        return -- A new Sound instance.
        """

        wavelength = samplerate / frequency
        one_wave = numpy.sin(numpy.arange(wavelength) * 2*numpy.pi / wavelength)
        return cls(one_wave * volume, samplerate)

    @classmethod
    def silence(cls, samplerate: Number = 44100) -> 'Sound':
        """ Generate silent sound

        This function returns VERY VERY short sound. Please use repeat function.

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

    @property
    def duration(self) -> float:
        """ Duration in seconds of this sound


        >>> s = Sound(numpy.array([0.1, 0.2, 0.3]), 1)
        >>> s.duration
        3.0
        >>> s.samplerate = 2
        >>> s.duration
        1.5
        """

        return len(self.data) / self.samplerate

    def __eq__(self, another: typing.Any) -> bool:
        """ Compare with another Sound instance.

        >>> a = Sound(numpy.array([0.1, 0.5]), 1)
        >>> b = Sound(numpy.array([0.1, 0.5]), 1)
        >>> c = Sound(numpy.array([-0.1, -0.5]), 1)
        >>> d = Sound(numpy.array([-0.1, -0.5]), 2)
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

        vol -- New volume. must be between -1.0 and 1.0.

        return -- A new Sound instance that changed volume.


        >>> s = Sound(numpy.array([0.1, 0.5]), 1)
        >>> s == Sound(numpy.array([0.1, 0.5]), 1)
        True
        >>> s.volume(1) == Sound(numpy.array([0.2, 1.0]), 1)
        True
        >>> s.volume(0.25) == Sound(numpy.array([0.05, 0.25]), 1)
        True
        >>> s.volume(2) == Sound(numpy.array([0.4, 1.0]), 1)
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


        >>> s = Sound(numpy.array([0.1, 0.2, 0.3]), 1)
        >>> s.repeat(6) == Sound(numpy.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]), 1)
        True
        >>> s.repeat(4) == Sound(numpy.array([0.1, 0.2, 0.3, 0.1]), 1)
        True
        >>> s.repeat(2) == Sound(numpy.array([0.1, 0.2]), 1)
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


        >>> s = Sound(numpy.array([0.1, 0.2, 0.3]), 1)
        >>> s.trim(2) == Sound(numpy.array([0.1, 0.2]), 1)
        True
        >>> s.trim(3) == Sound(numpy.array([0.1, 0.2, 0.3]), 1)
        True
        """
        assert 0 <= duration <= self.duration

        return Sound(
            self.data[:int(numpy.round(duration * self.samplerate))],
            self.samplerate
        )

    def concat(self, other: 'Sound') -> 'Sound':
        """ Create a new instance that concatenated another sound

        other -- The sound that concatenates after of self.
                 Must it has same sampling rate.

        return -- A new Sound that concatenated self and other.


        >>> a = Sound(numpy.array([0.1, 0.2]), 1)
        >>> b = Sound(numpy.array([0.3, 0.4]), 1)
        >>> a.concat(b) == Sound(numpy.array([0.1, 0.2, 0.3, 0.4]), 1)
        True
        """
        assert self.samplerate == other.samplerate

        return Sound(numpy.hstack([self.data, other.data]), self.samplerate)

    def overlay(self, other: 'Sound') -> 'Sound':
        """ Create a new instance that was overlay another sound

        other -- The sound that overlay.

        return -- A new Sound that overlay another sound.


        >>> s = Sound(numpy.array([0.1, 0.2]), 1)
        >>> s.overlay(s) == Sound(numpy.array([0.2, 0.4]), 1)
        True
        """
        assert self.samplerate == other.samplerate

        return Sound(self.data + other.data, self.samplerate)

    def write(self, file_: typing.Union[str, typing.BinaryIO]) -> None:
        """ Write sound into file or file-like

        file_ -- A file name or file-like object to write sound
        """

        soundfile.write(file_, self.data, self.samplerate)


def concat(*sounds: Sound) -> Sound:
    """ Concatenate multiple sounds

    sounds -- Sound instances to concatenate. Must they has some sampling rate.

    return -- A concatenated Sound instance.


    >>> a = Sound(numpy.array([0.1, 0.2]), 1)
    >>> b = Sound(numpy.array([0.3, 0.4]), 1)
    >>> c = Sound(numpy.array([0.5, 0.6]), 1)
    >>> concat(a, b, c) == Sound(numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 1)
    True
    """

    return functools.reduce(lambda x, y: x.concat(y), sounds)


def overlay(*sounds: Sound) -> Sound:
    """ Overlay multiple sounds

    BE CAREFUL: This function doesn't care about clipping. Perhaps, need to
                change volume before use this if overlay many sounds.

    sounds -- Sound instances to overlay. Must they has some sampling rate and
              same duration.

    return -- A Sound instance that overlay all sounds.


    >>> a = Sound(numpy.array([0.1, 0.2]), 1)
    >>> b = Sound(numpy.array([0.3, 0.4]), 1)
    >>> c = Sound(numpy.array([0.5, 0.6]), 1)
    >>> overlay(a, b, c) == Sound(numpy.array([0.9, 1.0]), 1)
    True

    The second element of this sample isn't 1.2 but 1.0 because of clipping was
    an occurrence.
    """

    return functools.reduce(lambda x, y: x.overlay(y), sounds)


if __name__ == '__main__':
    wait = Sound.silence().repeat(0.9)
    a = Sound.from_sinwave(440, volume=1.0).repeat(0.1).concat(wait)
    b = Sound.from_sinwave(880, volume=1.0).repeat(1.5)
    a.concat(a).concat(a).concat(b).write('test.wav')
