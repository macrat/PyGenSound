""" Audio effects """

import typing

import numpy
import scipy.interpolate

from gensound.sound import Sound


class Effect:
    """ Base class of sound effect """

    def apply(self, sound: Sound) -> Sound:
        """ Apply effect to sound

        sound -- Sound instance to appling effect.

        return -- A new Sound instance that applied effect.
        """

        raise NotImplementedError()

    def then(self, effect: 'Effect') -> 'Effect':
        """ Join effect

        >>> in_ = LinearFadeIn()
        >>> out = LinearFadeOut()
        >>> sound = Sound.from_sinwave(440)

        >>> out.apply(in_.apply(sound)) == in_.then(out).apply(sound)
        True


        effect -- Effect that will apply after this effect.

        return -- Joined effect.
        """

        return JoinedEffect(self, effect)


class JoinedEffect(Effect):
    """ Joined multiple effects

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

        sound -- Sound instance to appling effect.

        return -- A new Sound instance that applied all effects.
        """

        for e in self.effects:
            sound = e.apply(sound)

        return sound


class MaskEffect(Effect):
    """ Masking effect """

    def __init__(self, duration: typing.Optional[float] = None) -> None:
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

        return Sound(numpy.hstack([sound.data[:length] * mask[:length],
                                   sound.data[length:]]),
                     sound.samplerate)


class MaskEndEffect(MaskEffect):
    """ Effect that masking end of sound """

    def apply(self, sound: Sound) -> Sound:
        length = len(sound.data)
        if self.duration is not None:
            length = int(numpy.round(self.duration * sound.samplerate))

        offset = max(0, length - len(sound.data))
        mask = self.gen_mask(length)[offset:]

        return Sound(numpy.hstack([sound.data[:-length],
                                   sound.data[-length:] * mask]),
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
    """ Low pass filter """

    def __init__(self, freq: float) -> None:
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
    """ High pass filter """

    def __init__(self, freq: float) -> None:
        """
        freq -- A threshold frequency.
        """

        self.freq = freq

    def apply(self, sound: Sound) -> Sound:
        f = numpy.fft.rfft(sound.data)
        freq = numpy.fft.rfftfreq(len(sound.data))

        f[freq < self.freq / sound.samplerate] = 0

        return Sound(numpy.fft.irfft(f), sound.samplerate)


class Resampling(Effect):
    """ Resampling effect

    Change sampling rate without changes sound duration.


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
        """
        samplerate -- New sampling rate.
        kind       -- The way to interpolating data.
                      Please see document of scipy.interpolate.interp1d.
        """
        assert 0 < samplerate

        self.samplerate = samplerate
        self.kind = kind

    def apply(self, sound: Sound) -> Sound:
        length = len(sound.data)
        f = scipy.interpolate.interp1d(numpy.linspace(0, 1, length),
                                       sound.data,
                                       kind=self.kind)
        new_x = numpy.round(length * self.samplerate / sound.samplerate)
        return Sound(f(numpy.linspace(0, 1, int(new_x))), self.samplerate)


class ChangeSpeed(Effect):
    """ Change sound speed effect

    Change sound duration without changes sampling rate.


    >>> original = Sound.from_sinwave(440, duration=1)
    >>> abs(original.duration - 1) < 0.01
    True

    This example changes duration from 1sec to 2sec.

    >>> slow = ChangeSpeed(2).apply(original)
    >>> abs(slow.duration - 2) < 0.01
    True

    And, changes duration to 0.5sec.

    >>> fast = ChangeSpeed(0.5).apply(original)
    >>> abs(fast.duration - 0.5) < 0.01
    True

    Sampling rate will not be changed.

    >>> original.samplerate == slow.samplerate
    True
    >>> original.samplerate == fast.samplerate
    True
    """

    def __init__(self, speed_rate: float, kind: str = 'cubic') -> None:
        """
        speed_rate -- Speed rate of new sound. 1.0 means don't change speed.
        kind       -- The way to interpolating data.
                      Please see document of scipy.interpolate.interp1d.
        """

        self.speed_rate = speed_rate
        self.kind = kind

    def apply(self, sound: Sound) -> Sound:
        resampler = Resampling(sound.samplerate / self.speed_rate)

        return Sound(resampler.apply(sound).data, sound.samplerate)
