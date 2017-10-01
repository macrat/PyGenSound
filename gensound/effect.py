""" The module of audio effects """

import typing

import numpy
import scipy.interpolate

from gensound.sound import Sound
from gensound.exceptions import *


class Effect:
    """ Base class of sound effect """

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        raise NotImplementedError()

    def then(self, effect: 'Effect') -> 'Effect':
        """ Join effect

        :effect: Effect that will apply after this effect.

        :return: Joined effect.


        >>> in_ = LinearFadeIn()
        >>> out = LinearFadeOut()
        >>> sound = Sound.from_sinwave(440)

        >>> out.apply(in_.apply(sound)) == in_.then(out).apply(sound)
        True
        """

        return JoinedEffect(self, effect)


class JoinedEffect(Effect):
    """ Joined multiple effects

    :param effects: Effect instances to joint.


    >>> in_ = LinearFadeIn()
    >>> out = LinearFadeOut()
    >>> sound = Sound.from_sinwave(440)

    >>> out.apply(in_.apply(sound)) == JoinedEffect(in_, out).apply(sound)
    True

    >>> out.apply(in_.apply(sound)) == in_.then(out).apply(sound)
    True
    """

    def __init__(self, *effects: Effect) -> None:
        self.effects = effects

    def apply(self, sound: Sound) -> Sound:
        """ Apply all effects

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied all effects.
        """

        for e in self.effects:
            sound = e.apply(sound)

        return sound


class MaskEffect(Effect):
    """ Masking effect

    :param duration: Duration in seconds of mask. Mathing to sound duration if
                     None.
    """

    def __init__(self, duration: typing.Optional[float] = None) -> None:
        self.duration = duration

    def gen_mask(self, length: int) -> numpy.array:
        """ Generate mask

        :param length: Length of mask array.

        :return: Mask value.
        """

        raise NotImplementedError()


class MaskStartEffect(MaskEffect):
    """ Effect that masking start of sound """

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        length = len(sound.data)
        if self.duration is not None:
            length = int(numpy.round(self.duration * sound.samplerate))

        mask = self.gen_mask(length)

        if len(mask.shape) == 1:
            mask = mask.reshape([-1, 1]).repeat(sound.n_channels, axis=1)

        return Sound(numpy.vstack([sound.data[:length] * mask[:length],
                                   sound.data[length:]]),
                     sound.samplerate)


class MaskEndEffect(MaskEffect):
    """ Effect that masking end of sound """

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        length = sound.data.shape[0]
        if self.duration is not None:
            length = int(numpy.round(self.duration * sound.samplerate))

        offset = max(0, length - sound.data.shape[0])
        mask = self.gen_mask(length)[offset:]

        if len(mask.shape) == 1:
            mask = mask.reshape([-1, 1]).repeat(sound.n_channels, axis=1)

        return Sound(numpy.vstack([sound.data[:-length],
                                   sound.data[-length:] * mask]),
                     sound.samplerate)


class LinearFadeIn(MaskStartEffect):
    """ Linear fade-in effect

    :param duration: Duration in seconds of mask. Mathing to sound duration if
                     None.


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

    :param duration: Duration in seconds of mask. Mathing to sound duration if
                     None.


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

    :param freq: A threshold frequency.
    """

    def __init__(self, freq: float) -> None:
        self.freq = freq

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        f = sound.fft()
        f[f[:, :, 0] > self.freq, 1] = 0
        return Sound.from_fft(f, sound.samplerate)


class HighPassFilter(Effect):
    """ High pass filter

    :param freq: A threshold frequency.
    """

    def __init__(self, freq: float) -> None:
        self.freq = freq

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        f = sound.fft()
        f[f[:, :, 0] < self.freq, 1] = 0
        return Sound.from_fft(f, sound.samplerate)


class Resampling(Effect):
    """ Resampling effect

    :param samplerate: New sampling rate.
    :param kind:       The way to interpolating data. Please see document of
                       scipy.interpolate.interp1d.


    Change sampling rate without changes sound duration.

    If the sampling rate of passed sound is same as target sampling rate, will
    return the same instance without re-sampling process.


    This example does resampling from 44100 Hz to 88200 Hz.

    >>> original = Sound.from_sinwave(440, duration=1, samplerate=44100)
    >>> original.samplerate
    44100
    >>> abs(original.duration - 1) < 0.01
    True

    >>> resampled = Resampling(88200).apply(original)
    >>> resampled.samplerate
    88200
    >>> abs(resampled.duration - 1) < 0.01
    True
    """

    def __init__(self, samplerate: float, kind: str = 'cubic') -> None:
        assert 0 < samplerate

        self.samplerate = samplerate
        self.kind = kind

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        if sound.samplerate == self.samplerate:
            return sound

        length = sound.data.shape[0]
        in_space = numpy.linspace(0, 1, length)
        out_space = numpy.linspace(
            0,
            1,
            int(numpy.round(length * self.samplerate / sound.samplerate)),
        )

        result = numpy.array([
            scipy.interpolate.interp1d(numpy.linspace(0, 1, length),
                                       sound.data[:, channel],
                                       kind=self.kind)(out_space)
            for channel in range(sound.n_channels)
        ]).T

        return Sound(result, self.samplerate)


class ChangeSpeed(Effect):
    """ Change sound speed effect

    :param speed_rate: Speed rate of new sound. 1.0 means don't change speed.
    :param kind:       The way to interpolating data. Please see document of
                       scipy.interpolate.interp1d.


    Change sound duration without changes sampling rate.


    >>> original = Sound.from_sinwave(440, duration=1, smooth_end=False)
    >>> original.duration == 1.0
    True

    This example changes duration from 1sec to 2sec.

    >>> slow = ChangeSpeed(2).apply(original)
    >>> slow.duration == 0.5
    True

    And, changes duration to 0.5sec.

    >>> fast = ChangeSpeed(0.5).apply(original)
    >>> fast.duration == 2.0
    True

    Automatically use ReversePlay if speed_rate was lower than 0.
    >>> ChangeSpeed(-1).apply(original) == ReversePlay().apply(original)
    True

    Sampling rate will not be changed.

    >>> original.samplerate == slow.samplerate
    True
    >>> original.samplerate == fast.samplerate
    True
    """

    def __init__(self, speed_rate: float, kind: str = 'cubic') -> None:
        if speed_rate == 0:
            raise ValueError('speed_rate must not 0')

        self.speed_rate = speed_rate
        self.kind = kind

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        if self.speed_rate < 0:
            sound = ReversePlay().apply(sound)

        resampler = Resampling(sound.samplerate / abs(self.speed_rate))

        return Sound(resampler.apply(sound).data, sound.samplerate)


class ChangeVolume(Effect):
    """ Change volume effect

    :param new_volume: New target volume.

    :exception InvalidVolumeError: Volume was lower than 0 or higher than 1.


    This volume means the maximum value of the wave.
    Please be careful that is not gain.

    >>> sound = Sound.from_sinwave(440, volume=1.0)

    >>> 0.999 <= sound.data.max() <= 1.0
    True
    >>> -0.999 >= sound.data.min() >= -1.0
    True

    >>> half = ChangeVolume(0.5).apply(sound)
    >>> 0.499 <= half.volume <= 0.501
    True


    This effect will return the same instance if given sound had the same
    volume as the target volume.
    """

    def __init__(self, new_volume: float) -> Sound:
        if new_volume < 0.0 or 1.0 < new_volume:
            raise InvalidVolumeError(new_volume)

        self.volume = new_volume

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A :class:`Sound` instance that applied effect.
        """

        if sound.volume == self.volume:
            return sound

        return Sound(sound.data * (self.volume / sound.volume),
                     sound.samplerate)


class ReversePlay(Effect):
    """ Reverse play effect """

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        return Sound(sound.data[::-1], sound.samplerate)


class Trim(Effect):
    """ Trim sound

    :param start: The start position of trimming in seconds.
                  If None, won't trim start side. Default is None.
    :param end:   The end position of trimming in seconds.
                  If None, won't trim end side. Default is None.

    :exception InvalidDurationError: If start was same or greater than end.


    This is alias of
    :func:`Sound.__getitem__<gensound.sound.Sound.__getitem__>`.

    >>> sound = Sound.from_sinwave(440)

    >>> Trim(end=0.5).apply(sound) == sound[:0.5]
    True
    >>> Trim(start=0.5).apply(sound) == sound[0.5:]
    True
    >>> Trim(start=0.3, end=0.7).apply(sound) == sound[0.3: 0.7]
    True
    """

    def __init__(self,
                 start: typing.Optional[float] = None,
                 end: typing.Optional[float] = None) -> None:

        self.start = start
        self.end = end

        if start is not None and end is not None and start >= end:
            raise InvalidDurationError(end - start)

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        :param sound: :class:`Sound` instance to appling effect.

        :return: A new :class:`Sound` instance that applied effect.
        """

        return sound[self.start: self.end]
