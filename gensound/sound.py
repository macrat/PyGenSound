""" The module of Sound class for handling sound """

import time
import typing

import numpy
import pyaudio
import soundfile

from gensound.exceptions import *


def _repeat_array(sound: numpy.array, want_length: int) -> numpy.array:
    """ Repeat numpy.array for enlarging a sound duration

    :param sound:       Sound data for repeat.
    :param want_length: The length of an output array.

    :return: Repeated sound data.

    :exception ValueError:


    >>> array = numpy.array([[1, 2, 3]]).T
    >>> numpy.allclose(_repeat_array(array, 6),
    ...                numpy.array([[1, 2, 3, 1, 2, 3]]).T)
    True
    >>> numpy.allclose(_repeat_array(array, 8),
    ...                numpy.array([[1, 2, 3, 1, 2, 3, 1, 2]]).T)
    True
    >>> numpy.allclose(_repeat_array(array, 2), numpy.array([[1, 2]]).T)
    True
    """

    if want_length <= 0:
        raise ValueError(
            'want_length must be greater than 0 but got {}'.format(want_length)
        )

    if len(sound.shape) != 2:
        raise ValueError('sound should two dimensions')

    if sound.shape[0] <= 0 or sound.shape[1] <= 0:
        raise ValueError('sound should have least one element')

    need_length = int(numpy.ceil(want_length / sound.shape[0]))
    drop_length = int(need_length * sound.shape[0] - want_length)

    repeated = (sound.T
                .reshape([sound.shape[1], sound.shape[0], 1])
                .repeat(need_length, axis=0)
                .reshape([sound.shape[1], -1]).T)

    if drop_length > 0:
        return repeated[:-drop_length]
    else:
        return repeated


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

    :param data:       Sound data array. First dimension is time, second
                       dimension is channel. Will clipping if value were out of
                       -1.0 to 1.0.
    :param samplerate: Sampling rate of sound data.

    :exception ValueError:             Data was invalid dimensions.
    :exception InvalidSamplerateError: Samplerate was 0 or less.
    :exception InvalidDurationError:   Data was empty.
    """

    def __init__(self, data: numpy.array, samplerate: float) -> None:
        if len(data.shape) > 2:
            raise ValueError('data dimensions must be 1 or 2 but got {}'
                             .format(len(data.shape)))

        if len(data.shape) == 1:
            data = data.reshape([-1, 1])

        _assertSamplerate(samplerate)
        _assertDuration(data.shape[0])

        self._data = data.clip(-1.0, 1.0)
        self._samplerate = samplerate

    @classmethod
    def from_sinwave(cls,
                     frequency: float,
                     duration: float = 1.0,
                     volume: float = 1.0,
                     samplerate: float = 44100,
                     smooth_end: bool = True) -> 'Sound':
        """ Generate sine wave sound

        :param frequency:  Frequency of new sound.
        :param duration:   Duration in seconds of new sound.
        :param volume:     The volume of new sound.
        :param samplerate: Sampling rate of new sound.
        :param smooth_end: Do make smooth end or not. Please see above.

        :return: A new :class:`Sound` instance.

        :exception InvalidDurationError:   Duration was 0 or less.
        :exception InvalidFrequencyError:  Frequency was 0 or less.
        :exception InvalidSamplerateError: Samplerate was 0 or less.
        :exception InvalidVolumeError:     Volume was lower than 0.0 or higher
                                           than 1.0.


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

        :param frequency:  Frequency of new sound.
        :param duration:   Duration in seconds of new sound.
        :param volume:     The volume of new sound.
        :param samplerate: Sampling rate of new sound.

        :return: A new :class:`Sound` instance.

        :exception InvalidDurationError:   Duration was 0 or less.
        :exception InvalidFrequencyError:  Frequency was 0 or less.
        :exception InvalidSamplerateError: Samplerate was 0 or less.
        :exception InvalidVolumeError:     Volume was lower than 0.0 or higher
                                           than 1.0.
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

        :duration:   Duration of new sound.
        :samplerate: Sampling rate of new sound.

        :return: A new :class:`Sound` instance.

        :exception InvalidDurationError:   Duration was 0 or less.
        :exception InvalidSamplerateError: Samplerate was 0 or less.
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

        :param duration:   Duration in seconds of new sound.
        :param volume:     The volume of new sound.
        :param samplerate: Sampling rate of new sound.

        :return: A new Sound instance.

        :exception InvalidDurationError:   Duration was 0 or less.
        :exception InvalidSamplerateError: Samplerate was 0 or less.
        :exception InvalidVolumeError:     Volume was lower than 0.0 or higher
                                           than 1.0.
        """

        _assertDuration(duration)
        _assertSamplerate(samplerate)
        _assertVolume(volume)

        length = int(numpy.round(duration * samplerate))
        return cls(numpy.random.rand(length) * volume, samplerate)

    @classmethod
    def from_file(cls, file_: typing.Union[str, typing.BinaryIO]) -> 'Sound':
        """ Read sound from file or file-like

        :param file_: File name or file-like object.

        :return: A new :class:`Sound` instance.
        """

        data, samplerate = soundfile.read(file_)
        data /= numpy.max([-data.min(), data.max()])
        return cls(data, samplerate)

    @classmethod
    def from_array(cls,
                   array: typing.Sequence[float],
                   samplerate: float) -> 'Sound':
        """ Make new sound from float array

        :param array:      Sound data. Elements must in between -1.0 to 1.0.
        :param samplerate: Sampling rate of new sound.

        :return: A new :class:`Sound` instance.

        :exception InvalidDurationError:   The array was empty.
        :exception InvalidSamplerateError: Samplerate was 0 or less.


        This method is same as passing numpy.array to the Sound constructor.

        >>> (Sound.from_array([-0.1, 0.0, 1.0], 3)
        ...  == Sound(numpy.array([-0.1, 0.0, 1.0]), 3))
        True
        """

        return Sound(numpy.array(array), samplerate)

    @classmethod
    def from_fft(cls,
                 spectrum: numpy.array,
                 samplerate: float = None) -> 'Sound':
        """ Make new sound from spectrum data like a result from fft() method.

        :param spectrum:   A spectrum data. Please see fft() method about a
                           format.
        :param samplerate: Sampling rate of new sound. Use spectrum data if
                           None.

        :return: A new :class:`Sound` instance.

        :exception InvalidDurationError:   Duration was 0 or less.
        :exception InvalidSamplerateError: Samplerate was 0 or less.
        """

        if samplerate is None:
            samplerate = spectrum[0, -1, 0].real * 2

        _assertSamplerate(samplerate)

        return Sound(
            numpy.array([numpy.fft.irfft(x[:, 1]) for x in spectrum]).T,
            samplerate,
        )

    @property
    def duration(self) -> float:
        """ Duration in seconds of this sound """

        return self.data.shape[0] / self.samplerate

    @property
    def samplerate(self) -> float:
        """ Sampling rate of this sound """

        return self._samplerate

    @property
    def n_channels(self) -> int:
        """ Number of channels """

        return self.data.shape[1]

    @property
    def volume(self) -> float:
        """ Volume of this dound

        This volume means the maximum value of the wave.
        Please be careful that is not gain.
        """

        return max(self.data.max(), -self.data.min())

    @property
    def data(self) -> numpy.array:
        """ Raw data of sound

        This array is two dimensions. The first dimension means time, and the
        second dimension means channels.
        """

        return self._data

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

        :param position: A position in seconds in float or slice.

        :return: A new :class:`Sound` instance that sliced.

        :exception ValueError: Passed slice had step value.


        If passed float a position, returns very short sound that only has
        1/samplerate seconds.

        >>> sound = Sound.from_sinwave(440)

        >>> short_sound = sound[0.5]
        >>> short_sound.duration == 1 / sound.samplerate
        True


        The step of the slice is not supported. Will raise ValueError if passed
        slice that has a step.

        >>> sound[::1]
        Traceback (most recent call last):
            ...
        ValueError: step is not supported
        """

        if isinstance(position, (int, float)):
            if position < 0 or self.duration < position:
                raise OutOfDurationError(position, 0.0, self.duration)

            index = int(numpy.round(position * self.samplerate))
            if index >= self.data.shape[0]:
                index = self.data.shape[0] - 1
            data = self.data[index, :].reshape([1, -1])
        else:
            if position.step is not None:
                raise ValueError('step is not supported')

            start = position.start
            if start is not None:
                start = int(numpy.round(start * self.samplerate))

            stop = position.stop
            if stop is not None:
                stop = int(numpy.round(stop * self.samplerate))

            data = self.data[start:stop, :]

        return Sound(data, self.samplerate)

    def split_channels(self) -> typing.Sequence[numpy.array]:
        """ Split channels into Sound

        :return: A list of :class:`Sound` instances.
        """

        return [
            Sound(self.data[:, i], self.samplerate)
            for i in range(self.n_channels)
        ]

    def as_monaural(self) -> 'Sound':
        """ Create a new instance that converted to monaural sound

        :return: A new :class:`Sound` instance that monaural.


        If an instance already monaural sound, may returns the same instance.
        """

        if self.n_channels == 1:
            return self

        return Sound(numpy.average(self.data, axis=1), self.samplerate)

    def as_stereo(self) -> 'Sound':
        """ Create a new instance that converted to stereo sound

        :return: A new :class:`Sound` instance that stereo.

        :exception ValueError: The sound wasn't monaural sound.


        Must be used for monaural sound. If used for multiple channel sound,
        raises ValueError.
        """

        if self.n_channels != 1:
            raise ValueError('Sound must be monaural')

        return Sound(self.data.reshape([-1, 1]).repeat(2, axis=1),
                     self.samplerate)

    def fft(self) -> numpy.array:
        """ Calculate fft

        :return: An array that pair of frequency and value.
        """

        freqs = (numpy.fft.rfftfreq(self.data.shape[0]) * self.samplerate)
        freqs = freqs.reshape([-1, 1])

        return numpy.array([
            numpy.hstack([
                freqs,
                numpy.fft.rfft(self.data[:, channel]).reshape([-1, 1])
            ])
            for channel in range(self.n_channels)
        ])

    def change_volume(self, volume: float) -> 'Sound':
        """ Create a new instance that changed volume

        :param volume: New volume.

        :return: A new :class:`Sound` instance that changed volume.

        :exception InvalidVolumeError: Volume was lower than 0.0 or higher than
                                       1.0.


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
        """

        _assertVolume(volume)

        return Sound(
            self.data * (volume / self.volume),
            self.samplerate
        )

    def repeat(self, duration: float) -> 'Sound':
        """ Create a new instance that repeated same sound

        :param duration: Duration in seconds to repeat.

        :return: A new :class:`Sound` instance that repeated same sound.

        :exception InvalidDurationError: Duration was shorter than 0.


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
        """

        _assertDuration(duration)

        return Sound(
            _repeat_array(self.data,
                          int(numpy.round(duration * self.samplerate))),
            self.samplerate,
        )

    def concat(self, another: 'Sound') -> 'Sound':
        """ Create a new instance that concatenated another sound

        :param another: The sound that concatenates after of self.
                        Must it has same sampling rate.

        :return: A new :class:`Sound` that concatenated self and other.


        >>> sound = Sound.from_sinwave(440, duration=3)
        >>> a, b = sound[:1], sound[1:]
        >>> a.concat(b) == sound
        True

        Recommend using :func:`concat<gensound.sound.concat>` function instead
        of this method if concatenate many sounds. Because
        :func:`concat<gensound.sound.concat>` function is optimized for many
        sounds.

        >>> concat(a, b) == a.concat(b)
        True
        """

        return concat(self, another)

    def overlay(self, another: 'Sound') -> 'Sound':
        """ Create a new instance that was overlay another sound

        :param another: The sound that overlay.

        :return: A new :class:`Sound` that overlay another sound.


        >>> a = Sound.from_array([0.1, 0.2], 1)
        >>> b = Sound.from_array([0.1, 0.2, 0.3], 1)
        >>> a.overlay(b) == Sound.from_array([0.2, 0.4, 0.3], 1)
        True

        Recommend using :func:`overlay<gensound.sound.overlay>` function
        instead of this method if overlay many sounds. Because
        :func:`overlay<gensound.sound.overlay>` function is optimized for many
        sounds.

        >>> overlay(a, b) == a.overlay(b)
        True
        """

        return overlay(self, another)

    def write(self,
              file_: typing.Union[str, typing.BinaryIO],
              format_: typing.Optional[str] = None) -> None:
        """ Write sound into file or file-like

        :param file_:   A file name or file-like object to write sound.
        :param format_: Format type of output like a 'wav'. Automatically
                        detect from file name if None.
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

    :param sounds: Sound instances to concatenate. Must they has some sampling
                   rate.

    :return: A concatenated :class:`Sound` instance.

    :exception DifferentSamplerateError: The samplerate of sounds was
                                         different.


    >>> a = Sound.from_array([0.1, 0.2], 1)
    >>> b = Sound.from_array([0.3, 0.4], 1)
    >>> c = Sound.from_array([0.5, 0.6], 1)
    >>> concat(a, b, c) == Sound.from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 1)
    True
    """

    _assertSameSamplerate([s.samplerate for s in sounds])

    return Sound(numpy.vstack([x.data for x in sounds]), sounds[0].samplerate)


def overlay(*sounds: Sound) -> Sound:
    """ Overlay multiple sounds

    :param sounds: Sound instances to overlay.
                   Must they has some sampling rate.

    :return: A new :class:`Sound` instance that overlay all sounds.

    :exception DifferentSamplerateError: The samplerate of sounds was
                                         different.


    BE CAREFUL: This function doesn't care about clipping. Perhaps, need to
                change volume before use this if overlay many sounds.

    >>> a = Sound.from_array([0.1, 0.2], 1)
    >>> b = Sound.from_array([0.3, 0.4], 1)
    >>> c = Sound.from_array([0.5, 0.6], 1)
    >>> overlay(a, b, c) == Sound.from_array([0.9, 1.0], 1)
    True

    The second element of this sample isn't 1.2 but 1.0 because of clipping was
    an occurrence.
    """

    _assertSameSamplerate([s.samplerate for s in sounds])

    longest = max(x.data.shape[0] for x in sounds)
    padded = numpy.array([
        numpy.vstack([x.data,
                      numpy.zeros([longest - x.data.shape[0], x.n_channels])])
        for x in sounds
    ])

    return Sound(padded.sum(axis=0), sounds[0].samplerate)


def merge_channels(*sounds: Sound) -> Sound:
    """ Merge multiple sounds as a sound that has multiple channels

    :param sounds: Sound instances to merge. Must they has some sampling rate.

    :return: A new :class:`Sound` that merged sounds as channels.

    :exception DifferentSamplerateError: The samplerate of sounds was
                                         different.


    BE CAREFUL: all sounds calculate as monaural sound.
    """

    _assertSameSamplerate([s.samplerate for s in sounds])

    longest = max(x.data.shape[0] for x in sounds)
    padded = numpy.hstack([
        numpy.vstack([x.as_monaural().data,
                      numpy.zeros([longest - x.data.shape[0], 1])])
        for x in sounds
    ])

    return Sound(padded, sounds[0].samplerate)
