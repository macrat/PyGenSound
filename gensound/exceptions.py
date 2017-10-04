import typing


class InvalidFrequencyError(ValueError):
    """ The exception that raises when passed invalid frequency

    :var frequency: Passed frequency.
    """

    def __init__(self, freq: float) -> None:
        super().__init__(
            'frequency must be greater than 0 but got {}'.format(freq)
        )
        self.frequency = freq


class InvalidSamplerateError(InvalidFrequencyError):
    """ The exception that raises when passed invalid samplerate

    :var frequency: Passed samplerate.
    """

    def __init__(self, freq: float) -> None:
        ValueError.__init__(
            self,
            'samplerate must be greater than 0 but got {}'.format(freq)
        )
        self.frequency = freq


class DifferentSamplerateError(InvalidSamplerateError):
    """ The exception that raises when different samplerates of sounds to joint

    :var frequency: List of passed frequencies.
    """

    def __init__(self, frequencies: typing.Tuple[float, ...]) -> None:
        ValueError.__init__(
            self,
            'all samplerates must be the same value but got {}'.format(
                ' and '.join(str(x) for x in set(frequencies))
            )
        )
        self.frequency = frequencies


class DifferentChannelsError(ValueError):
    """ The exception that raises when different number of channels of sounds
    to joint

    :var channels: List of n_channels of sounds.
    """

    def __init__(self, channels: typing.Tuple[int, ...]) -> None:
        ValueError.__init__(
            self,
            'all sounds must has a same number of channels but got {}'.format(
                str(x) for x in set(channels)
            ),
        )
        self.channels = channels


class InvalidDurationError(ValueError):
    """ The exception that raises when passed sound was invalid duration

    :var duration: Passed duration.
    """

    def __init__(self, duration: float) -> None:
        super().__init__('duration of sound must not 0 or short but got {}'
                         .format(duration))
        self.duration = duration


class OutOfDurationError(IndexError):
    """ The exception that raises when passed index that out of duration

    :var duration: Passed duration.
    :var min:      Minimal acceptable value.
    :var max:      Maximum acceptable value.
    """

    def __init__(self, duration: float, min_: float, max_: float) -> None:
        super().__init__('index must between {} to {} but got {}'
                         .format(min_, max_, duration))
        self.duration = duration
        self.min = min_
        self.max = max_


class InvalidVolumeError(ValueError):
    """ The exception that raises when passed invalid volume

    :var volume: Passed volume.
    """

    def __init__(self, volume: float) -> None:
        super().__init__(
            'volume must be between 0.0 and 1.0 but got {}'.format(volume)
        )
        self.volume = volume
