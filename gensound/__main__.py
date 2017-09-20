#!/usr/bin/python3

import argparse
import io
import sys
import typing

import gensound


def save(sound: gensound.Sound, args: argparse.Namespace):
    if args.output == sys.stdout:
        with io.BytesIO() as buf:
            sound.write(buf, format_='wav')
            args.output.buffer.write(buf.getvalue())
    else:
        sound.write(args.output)


def load(file_: typing.BinaryIO) -> gensound.Sound:
    if file_ == sys.stdin:
        with io.BytesIO(file_.buffer.read()) as buf:
            return gensound.Sound.from_file(buf)
    else:
        return gensound.Sound.from_file(file_)


def load_files(files: typing.Iterable[typing.BinaryIO]) \
        -> typing.Iterator[gensound.Sound]:

    stdin = None

    for f in files:
        if f == sys.stdin:
            if stdin is not None:
                yield stdin
            else:
                stdin = load(f)
                yield stdin
        else:
            yield load(f)


def command_sine(args: argparse.Namespace):
    save(gensound.Sound.from_sinwave(
        args.frequency,
        duration=args.duration,
        volume=args.volume / 100,
        samplerate=args.samplerate,
    ), args)


def command_sawtooth(args: argparse.Namespace):
    save(gensound.Sound.from_sawtoothwave(
        args.frequency,
        duration=args.duration,
        volume=args.volume / 100,
        samplerate=args.samplerate,
    ), args)


def command_silence(args: argparse.Namespace):
    save(gensound.Sound.silence(args.samplerate).repeat(args.duration), args)


def command_noise(args: argparse.Namespace):
    save(gensound.Sound.from_whitenoise(duration=args.duration,
                                        volume=args.volume,
                                        samplerate=args.samplerate), args)


def command_overlay(args: argparse.Namespace):
    save(gensound.overlay(*load_files(args.file)), args)


def command_concat(args: argparse.Namespace):
    save(gensound.concat(*load_files(args.file)), args)


def command_fadein(args: argparse.Namespace):
    save(gensound.LinearFadeIn(args.duration).apply(load(args.input)), args)


def command_fadeout(args: argparse.Namespace):
    save(gensound.LinearFadeOut(args.duration).apply(load(args.input)), args)


def command_highpass(args: argparse.Namespace):
    save(gensound.HighPassFilter(args.frequency).apply(load(args.input)), args)


def command_lowpass(args: argparse.Namespace):
    save(gensound.LowPassFilter(args.frequency).apply(load(args.input)), args)


def command_resample(args: argparse.Namespace):
    save(gensound.Resampling(args.frequency).apply(load(args.input)), args)


def command_speed(args: argparse.Namespace):
    sound = load(args.input)
    save(gensound.ChangeSpeed(sound.duration * args.rate).apply(sound), args)


def _setup_input(parser: argparse.ArgumentParser):
    parser.add_argument('-i',
                        '--input',
                        metavar='FILE',
                        type=argparse.FileType('rb'),
                        default=sys.stdin,
                        help='File to input.')


def _setup_output(parser: argparse.ArgumentParser):
    parser.add_argument('-o',
                        '--output',
                        metavar='FILE',
                        type=argparse.FileType('wb'),
                        default=sys.stdout,
                        help='File to output.')


def _setup_gencommand(parser: argparse.ArgumentParser,
                      has_frequency: bool = True,
                      has_volume: bool = True):

    if has_frequency:
        parser.add_argument('frequency', type=float, help='Frequency')

    parser.add_argument('-d',
                        '--duration',
                        default=1.0,
                        metavar='SECOND',
                        type=float,
                        help='Duration in seconds. default is 1.0 seconds.')

    if has_volume:
        parser.add_argument('-v',
                            '--volume',
                            type=float,
                            default=100,
                            help='Percentage of volume. default is 100.')

    parser.add_argument('-s',
                        '--samplerate',
                        type=float,
                        default=44100,
                        help='Sampling rate. Default is 44100.')

    _setup_output(parser)


def _setup_joincommand(parser: argparse.ArgumentParser, work_name: str):
    parser.add_argument('file',
                        type=argparse.FileType('rb'),
                        default=sys.stdin,
                        nargs='+',
                        help='File to {}.'.format(work_name))

    _setup_output(parser)


def _setup_freq_filtercommand(parser: argparse.ArgumentParser):
    parser.add_argument('frequency', type=float, nargs=1, help='Frequency.')

    _setup_input(parser)
    _setup_output(parser)


def _setup_duration_filtercommand(parser: argparse.ArgumentParser):
    parser.add_argument('-d',
                        '--duration',
                        metavar='SECOND',
                        type=float,
                        help='Duration in seconds to apply the effect.')

    _setup_input(parser)
    _setup_output(parser)


def make_parser(prog: str = 'gensound'):
    parser = argparse.ArgumentParser(
        prog=prog,
        usage='{} COMMAND [options...]'.format(prog),
        description='Command line interface of PyGenSound.',
        epilog='\n'.join([
            'EXAMPLE: ',
            '  $ gensound sine 440 -d 0.1 | gensound fadeout -o nhk.wav',
            '  $ gensound silence -d 0.9 | gensound concat nhk.wav -',
            ('  $ gensound sine 880 -d 2.0 | gensound fadeout | gensound'
             + 'concat nhk.wav -'),
        ]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(title='commands', metavar='COMMAND')

    sine = subparsers.add_parser('sine',
                                 description='Generate sine wave sound.')
    _setup_gencommand(sine)
    sine.set_defaults(handler=command_sine)

    sawtooth = subparsers.add_parser(
        'sawtooth',
        description='Generate sawtooth wave sound.',
    )
    _setup_gencommand(sawtooth)
    sawtooth.set_defaults(handler=command_sawtooth)

    silence = subparsers.add_parser('silence',
                                    help='Generate cilence sound.',
                                    description='Generate cilence sound.')
    _setup_gencommand(silence, has_frequency=False, has_volume=False)
    silence.set_defaults(handler=command_silence)

    noise = subparsers.add_parser('noise',
                                  help='Generate white noise sound.',
                                  description='Generate white noise sound.')
    _setup_gencommand(noise, has_frequency=False)
    noise.set_defaults(handler=command_noise)

    overlay = subparsers.add_parser('overlay',
                                    help='Overlay some sounds.',
                                    description='Overlay some sounds.')
    _setup_joincommand(overlay, 'overlay')
    overlay.set_defaults(handler=command_overlay)

    concat = subparsers.add_parser('concat',
                                   help='Concatnate some sounds.',
                                   description='Concatnate some sounds.')
    _setup_joincommand(concat, 'concatenate')
    concat.set_defaults(handler=command_concat)

    fadeIn = subparsers.add_parser(
        'fadein',
        help='Apply fade-in effect to sound.',
        description='Apply fade-in effect to sound.',
    )
    _setup_duration_filtercommand(fadeIn)
    fadeIn.set_defaults(handler=command_fadein)

    fadeOut = subparsers.add_parser(
        'fadeout',
        help='Apply fade-out effect to sound.',
        description='Apply fade-out effect to sound.',
    )
    _setup_duration_filtercommand(fadeOut)
    fadeOut.set_defaults(handler=command_fadeout)

    highpass = subparsers.add_parser(
        'highpass',
        help='Apply high pass filter to sound.',
        description='Apply high pass filter to sound.',
    )
    _setup_freq_filtercommand(highpass)
    highpass.set_defaults(handler=command_highpass)

    lowpass = subparsers.add_parser(
        'lowpass',
        help='Apply low pass filter to sound.',
        description='Apply low pass filter to sound.',
    )
    _setup_freq_filtercommand(lowpass)
    lowpass.set_defaults(handler=command_lowpass)

    resample = subparsers.add_parser('resample',
                                     help='Resampling sound.',
                                     description='Resampling sound.')
    _setup_freq_filtercommand(resample)
    resample.set_defaults(handler=command_resample)

    speed = subparsers.add_parser('speed',
                                  help='Change speed of sound.',
                                  description='Change speed of sound.')
    speed.add_argument('rate',
                       type=float,
                       nargs=1,
                       help="Speed rate. Doesn't change speed if 1.0.")
    _setup_input(speed)
    _setup_output(speed)
    speed.set_defaults(handler=command_speed)

    duration = subparsers.add_parser('duration',
                                     help='Change duration of sound.',
                                     description='Change duration of sound.')
    duration.add_argument('duration',
                          type=float,
                          nargs=1,
                          help='New duration in seconds.')
    _setup_input(duration)
    _setup_output(duration)
    duration.set_defaults(handler=command_speed)

    return parser


def main(args: typing.Optional[typing.Iterable[str]] = None) -> None:
    parser = make_parser()

    if args is None:
        args = sys.argv[1:]

    opts = parser.parse_args(args)

    if hasattr(opts, 'handler'):
        opts.handler(opts)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[1:])
