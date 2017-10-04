import io
import tempfile
import unittest

import numpy

from gensound.sound import *
from gensound.sound import _repeat_array


class SoundUtilsTest(unittest.TestCase):
    def test_repeat_array_single_channel(self):
        array = numpy.array([[1, 2, 3]]).T
        repeated = [1, 2, 3] * 3

        self.assertEqual(tuple(_repeat_array(array, 6).flatten()),
                         tuple(repeated[:6]))

        self.assertEqual(tuple(_repeat_array(array, 8).flatten()),
                         tuple(repeated[:8]))

        self.assertEqual(tuple(_repeat_array(array, 2).flatten()),
                         tuple(repeated[:2]))

    def test_repeat_array_two_channel(self):
        array = numpy.array([[1, 2, 3], [4, 5, 6]]).T
        repeated = numpy.array([[1, 2, 3] * 3, [4, 5, 6] * 3]).T

        six = _repeat_array(array, 6)
        self.assertEqual(six.shape, (6, 2))
        self.assertEqual(six.tolist(), repeated[:6].tolist())

        eight = _repeat_array(array, 8)
        self.assertEqual(eight.shape, (8, 2))
        self.assertEqual(eight.tolist(), repeated[:8].tolist())

        two = _repeat_array(array, 2)
        self.assertEqual(two.shape, (2, 2))
        self.assertEqual(two.tolist(), repeated[:2].tolist())

    def test_repeat_array_invalid_input(self):
        array = numpy.array([[1, 2, 3]])
        null = numpy.array([[]])
        lessdimen = numpy.array([1, 2, 3])
        overdimen = numpy.array([[[1, 2]], [[3, 4]]])

        with self.assertRaises(ValueError) as cm:
            _repeat_array(array, 0)

        self.assertEqual(str(cm.exception),
                         'want_length must be greater than 0 but got 0')

        with self.assertRaises(ValueError) as cm:
            _repeat_array(array, -1)

        self.assertEqual(str(cm.exception),
                         'want_length must be greater than 0 but got -1')

        with self.assertRaises(ValueError) as cm:
            _repeat_array(null, 1)

        self.assertEqual(str(cm.exception),
                         'sound should have least one element')

        with self.assertRaises(ValueError) as cm:
            _repeat_array(lessdimen, 1)

        self.assertEqual(str(cm.exception), 'sound should two dimensions')

        with self.assertRaises(ValueError) as cm:
            _repeat_array(overdimen, 1)

        self.assertEqual(str(cm.exception), 'sound should two dimensions')

    def test_overlay(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5], 2)

        self.assertEqual(overlay(a, b, c).samplerate, 2)
        self.assertEqual(overlay(a, b, c).duration, 1)

        self.assertEqual(overlay(a, b, c),
                         Sound.from_array([0.6, 0.9], 2))

        self.assertEqual(overlay(overlay(a, b), c),
                         Sound.from_array([0.6, 0.9], 2))

        self.assertEqual(overlay(a, overlay(b, c)),
                         Sound.from_array([0.6, 0.9], 2))

        self.assertEqual(overlay(overlay(a, c), b),
                         Sound.from_array([0.6, 0.9], 2))

    def test_overlay_different_duration(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5, 0.6, 0.7], 2)

        self.assertEqual(overlay(a, b, c).samplerate, 2)
        self.assertEqual(overlay(a, b, c).duration, 2)

        self.assertEqual(overlay(a, b, c),
                         Sound.from_array([0.6, 0.9, 0.6, 0.7], 2))

    def test_overlay_different_samplerate(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5], 3)

        with self.assertRaises(DifferentSamplerateError) as cm:
            overlay(a, b, c)

        self.assertEqual(cm.exception.frequency, (2, 2, 3))

    def test_overlay_different_channels(self):
        a = Sound.from_sinwave(220)
        b = Sound.from_sinwave(440)
        c = Sound.from_sinwave(880).as_stereo()

        with self.assertRaises(DifferentChannelsError) as cm:
            overlay(a, b, c)

        self.assertEqual(cm.exception.channels, (1, 1, 2))

    def test_concat(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5], 2)

        self.assertEqual(concat(a, b, c).samplerate, 2)
        self.assertEqual(concat(a, b, c).duration, 3)

        self.assertEqual(concat(a, b, c),
                         Sound.from_array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 2))

        self.assertEqual(concat(concat(a, b), c),
                         Sound.from_array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 2))

        self.assertEqual(concat(a, concat(b, c)),
                         Sound.from_array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 2))

        self.assertEqual(concat(c, b, a),
                         Sound.from_array([0.4, 0.5, 0.2, 0.3, 0.0, 0.1], 2))

    def test_concat_different_duration(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5, 0.6, 0.7], 2)

        self.assertEqual(concat(a, b, c).samplerate, 2)
        self.assertEqual(concat(a, b, c).duration, 4)

        self.assertEqual(concat(a, b, c), Sound.from_array([
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
        ], 2))

    def test_concat_different_samplerate(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5], 3)

        with self.assertRaises(DifferentSamplerateError) as cm:
            concat(a, b, c)

        self.assertEqual(cm.exception.frequency, (2, 2, 3))

    def test_concat_different_channels(self):
        a = Sound.from_sinwave(220)
        b = Sound.from_sinwave(440)
        c = Sound.from_sinwave(880).as_stereo()

        with self.assertRaises(DifferentChannelsError) as cm:
            concat(a, b, c)

        self.assertEqual(cm.exception.channels, (1, 1, 2))

    def test_merge_channels(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5], 2)
        abc = Sound.from_array([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]], 2)

        self.assertEqual(merge_channels(a, b, c), abc)

        c_stereo = merge_channels(Sound.from_array([0.3, 0.6], 2),
                                  Sound.from_array([0.5, 0.4], 2))
        self.assertEqual(merge_channels(a, b, c_stereo), abc)


class SoundTest(unittest.TestCase):
    def test_constructor(self):
        sound = Sound(numpy.array([-1.0, 0.5, 1.0]), 1)

        self.assertEqual(sound.samplerate, 1)
        self.assertEqual(sound.duration, 3)
        self.assertEqual(tuple(sound.data), (-1.0, 0.5, 1.0))

    def test_constructor_clip(self):
        sound = Sound(numpy.array([-1.1, -1.0, 1.0, 1.1]), 2)

        self.assertEqual(sound.samplerate, 2)
        self.assertEqual(sound.duration, 2)
        self.assertEqual(tuple(sound.data), (-1.0, -1.0, 1.0, 1.0))

    def test_constructor_invalid(self):
        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound(numpy.array([0]), 0)

        self.assertEqual(cm.exception.frequency, 0)

        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound(numpy.array([0]), -1)

        self.assertEqual(cm.exception.frequency, -1)

        with self.assertRaises(InvalidDurationError) as cm:
            Sound(numpy.array([]), 1)

        self.assertEqual(cm.exception.duration, 0)

        with self.assertRaises(ValueError) as cm:
            Sound(numpy.array([[[1], [2]], [[3], [4]]]), 1)

        self.assertEqual(str(cm.exception),
                         'data dimensions must be 1 or 2 but got 3')

    def test_equals(self):
        a = Sound.from_array([0.1, 0.2, 0.3], 1)
        b = Sound.from_array([0.4, 0.5, 0.6], 1)

        self.assertEqual(a, a)
        self.assertEqual(b, b)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, None)
        self.assertNotEqual(a, 1)
        self.assertNotEqual(a, 'a')

    def test_from_array(self):
        self.assertEqual(Sound(numpy.array([-0.5, 0.5]), 44100),
                         Sound.from_array([-0.5, 0.5], 44100))

        self.assertEqual(Sound(numpy.array([0.1, -0.1]), 100),
                         Sound.from_array([0.1, -0.1], 100))

    def test_from_array_invalid(self):
        with self.assertRaises(InvalidDurationError) as cm:
            Sound.from_array([], 44100)

        self.assertEqual(cm.exception.duration, 0)

        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound.from_array([0], 0)

        self.assertEqual(cm.exception.frequency, 0)

    def test_from_sinwave_with_smooth_end(self):
        sound = Sound.from_sinwave(440,
                                   duration=1,
                                   volume=0.5,
                                   samplerate=44100,
                                   smooth_end=True)

        self.assertEqual(sound.samplerate, 44100)
        self.assertAlmostEqual(sound.duration, 1, places=1)
        self.assertAlmostEqual(sound.volume, 0.5, places=4)

        sound = Sound.from_sinwave(880,
                                   duration=2,
                                   volume=0.8,
                                   samplerate=88200,
                                   smooth_end=True)

        self.assertEqual(sound.samplerate, 88200)
        self.assertAlmostEqual(sound.duration, 2, places=1)
        self.assertAlmostEqual(sound.volume, 0.8, places=4)

    def test_from_sinwave_without_smooth_end(self):
        sound = Sound.from_sinwave(440,
                                   duration=1,
                                   volume=0.5,
                                   samplerate=44100,
                                   smooth_end=False)

        self.assertEqual(sound.samplerate, 44100)
        self.assertEqual(sound.duration, 1.0)
        self.assertAlmostEqual(sound.volume, 0.5, places=4)

        sound = Sound.from_sinwave(880,
                                   duration=2,
                                   volume=0.8,
                                   samplerate=88200,
                                   smooth_end=False)

        self.assertEqual(sound.samplerate, 88200)
        self.assertEqual(sound.duration, 2.0)
        self.assertAlmostEqual(sound.volume, 0.8, places=4)

    def test_from_sinwave_invalid(self):
        with self.assertRaises(InvalidFrequencyError) as cm:
            Sound.from_sinwave(0)

        self.assertEqual(cm.exception.frequency, 0)

        with self.assertRaises(InvalidDurationError) as cm:
            Sound.from_sinwave(440, duration=0)

        self.assertEqual(cm.exception.duration, 0)

        with self.assertRaises(InvalidVolumeError) as cm:
            Sound.from_sinwave(440, volume=-0.1)

        self.assertEqual(cm.exception.volume, -0.1)

        with self.assertRaises(InvalidVolumeError) as cm:
            Sound.from_sinwave(440, volume=1.1)

        self.assertEqual(cm.exception.volume, 1.1)

        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound.from_sinwave(440, samplerate=0)

        self.assertEqual(cm.exception.frequency, 0)

    def test_from_sawtoothwave(self):
        sound = Sound.from_sawtoothwave(440,
                                        duration=1,
                                        volume=0.5,
                                        samplerate=44100)

        self.assertEqual(sound.samplerate, 44100)
        self.assertEqual(sound.duration, 1.0)
        self.assertTrue((-0.5 <= sound.data).all())
        self.assertTrue((sound.data <= 0.5).all())
        self.assertEqual(sound.volume, 0.5)

        sound = Sound.from_sawtoothwave(880,
                                        duration=2,
                                        volume=0.8,
                                        samplerate=88200)

        self.assertEqual(sound.samplerate, 88200)
        self.assertTrue(sound.duration, 2.0)
        self.assertTrue((-0.8 <= sound.data).all())
        self.assertTrue((sound.data <= 0.8).all())
        self.assertTrue(sound.volume, 0.8)

        sound = Sound.from_sawtoothwave(1, duration=2, samplerate=3)
        self.assertTrue(numpy.allclose(sound.data[:, 0],
                                       (-1.0, 0.0, 1.0, -1.0, 0.0, 1.0)))

    def test_from_sawtoothwave_invalid(self):
        with self.assertRaises(InvalidFrequencyError) as cm:
            Sound.from_sawtoothwave(0)

        self.assertEqual(cm.exception.frequency, 0)

        with self.assertRaises(InvalidDurationError) as cm:
            Sound.from_sawtoothwave(440, duration=0)

        self.assertEqual(cm.exception.duration, 0)

        with self.assertRaises(InvalidVolumeError) as cm:
            Sound.from_sawtoothwave(440, volume=-0.1)

        self.assertEqual(cm.exception.volume, -0.1)

        with self.assertRaises(InvalidVolumeError) as cm:
            Sound.from_sawtoothwave(440, volume=1.1)

        self.assertEqual(cm.exception.volume, 1.1)

        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound.from_sawtoothwave(440, samplerate=0)

        self.assertEqual(cm.exception.frequency, 0)

    def test_from_squarewave(self):
        sound = Sound.from_squarewave(440,
                                      duration=1,
                                      volume=0.5,
                                      samplerate=44100)

        self.assertEqual(sound.samplerate, 44100)
        self.assertEqual(sound.duration, 1.0)
        self.assertTrue((-0.5 <= sound.data).all())
        self.assertTrue((sound.data <= 0.5).all())
        self.assertEqual(sound.volume, 0.5)

        sound = Sound.from_squarewave(880,
                                      duration=2,
                                      volume=0.8,
                                      samplerate=88200)

        self.assertEqual(sound.samplerate, 88200)
        self.assertTrue(sound.duration, 2.0)
        self.assertTrue((-0.8 <= sound.data).all())
        self.assertTrue((sound.data <= 0.8).all())
        self.assertTrue(sound.volume, 0.8)

        sound = Sound.from_squarewave(1, duration=2, samplerate=4)
        self.assertTrue(numpy.allclose(sound.data[:, 0],
                                       (1, 1, -1, -1, 1, 1, -1, -1)))

    def test_from_squarewave_invalid(self):
        with self.assertRaises(InvalidFrequencyError) as cm:
            Sound.from_squarewave(0)

        self.assertEqual(cm.exception.frequency, 0)

        with self.assertRaises(InvalidDurationError) as cm:
            Sound.from_squarewave(440, duration=0)

        self.assertEqual(cm.exception.duration, 0)

        with self.assertRaises(InvalidVolumeError) as cm:
            Sound.from_squarewave(440, volume=-0.1)

        self.assertEqual(cm.exception.volume, -0.1)

        with self.assertRaises(InvalidVolumeError) as cm:
            Sound.from_squarewave(440, volume=1.1)

        self.assertEqual(cm.exception.volume, 1.1)

        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound.from_squarewave(440, samplerate=0)

        self.assertEqual(cm.exception.frequency, 0)

    def test_silence(self):
        sound = Sound.silence(duration=1.0, samplerate=100)

        self.assertEqual(sound.samplerate, 100)
        self.assertEqual(sound.duration, 1.0)
        self.assertTrue((sound.data == 0).all())

        sound = Sound.silence(duration=2.0, samplerate=20)

        self.assertEqual(sound.samplerate, 20)
        self.assertEqual(sound.duration, 2.0)
        self.assertTrue((sound.data == 0).all())

    def test_silence_invlid(self):
        with self.assertRaises(InvalidDurationError) as cm:
            Sound.silence(duration=0)

        self.assertEqual(cm.exception.duration, 0)

        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound.silence(samplerate=0)

        self.assertEqual(cm.exception.frequency, 0)

    def test_whitenoise(self):
        sound = Sound.from_whitenoise(duration=2, volume=0.1, samplerate=100)

        self.assertEqual(sound.samplerate, 100)
        self.assertEqual(sound.duration, 2)
        self.assertTrue((-0.1 <= sound.data).all())
        self.assertTrue((sound.data <= 0.1).all())

    def test_whitenoise_invalid(self):
        with self.assertRaises(InvalidDurationError) as cm:
            Sound.from_whitenoise(duration=0)

        self.assertEqual(cm.exception.duration, 0)

        with self.assertRaises(InvalidVolumeError) as cm:
            Sound.from_whitenoise(volume=-0.1)

        self.assertEqual(cm.exception.volume, -0.1)

        with self.assertRaises(InvalidVolumeError) as cm:
            Sound.from_whitenoise(volume=1.1)

        self.assertEqual(cm.exception.volume, 1.1)

        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound.from_whitenoise(samplerate=0)

        self.assertEqual(cm.exception.frequency, 0)

    def test_from_fft(self):
        f = numpy.zeros([2, 1024 // 2 + 1, 2], numpy.complex)
        f[:, -1, 0] = 1024 // 2
        f[0, 128, 1] = numpy.complex(0, -numpy.pi)
        f[1, 256, 1] = numpy.complex(0, -numpy.pi)

        s = Sound.from_fft(f)

        self.assertEqual(s.samplerate, 1024)
        self.assertEqual(s.n_channels, 2)
        self.assertEqual(s.duration, 1.0)

        from_sin = merge_channels(Sound.from_sinwave(128, samplerate=1024),
                                  Sound.from_sinwave(256, samplerate=1024))
        self.assertTrue(numpy.allclose(s.data / s.volume,
                                       from_sin.data / from_sin.volume))

    def test_from_fft_invalid(self):
        f = numpy.zeros([1024 // 2 + 1, 2], numpy.complex)

        with self.assertRaises(InvalidSamplerateError) as cm:
            Sound.from_fft(f, samplerate=0)

        self.assertEqual(cm.exception.frequency, 0)

    def test_fft_single_channel(self):
        sound = Sound.from_sinwave(440, duration=0.1)
        f = sound.fft()

        self.assertTrue((0 <= f[:, :, 0]).all())
        self.assertEqual(f.shape[0], 1)
        self.assertEqual(f.shape[2], 2)
        self.assertGreaterEqual(f[:, :, 0].min(), 0)
        self.assertLessEqual(f[:, :, 0].max(), sound.samplerate)
        self.assertEqual(f[0, :, 1].argmax(), abs(f[0, :, 0] - 440).argmin())

    def test_fft_two_channel(self):
        sound = merge_channels(Sound.from_sinwave(440, duration=0.1),
                               Sound.from_sinwave(220, duration=0.1))
        f = sound.fft()

        self.assertTrue((0 <= f[:, :, 0]).all())
        self.assertEqual(f.shape[0], 2)
        self.assertEqual(f.shape[2], 2)
        self.assertGreaterEqual(f[:, :, 0].min(), 0)
        self.assertLessEqual(f[:, :, 0].max(), sound.samplerate)
        self.assertEqual(f[0, :, 1].argmax(), abs(f[0, :, 0] - 440).argmin())
        self.assertEqual(f[1, :, 1].argmax(), abs(f[1, :, 0] - 220).argmin())

    def test_repeat(self):
        sound = Sound.from_array([0.0, 0.1, 0.2], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(tuple(sound.data), (0.0, 0.1, 0.2))

        sound = sound.repeat(2)

        self.assertEqual(sound.duration, 2)
        self.assertEqual(tuple(sound.data), (0.0, 0.1, 0.2, 0.0, 0.1, 0.2))

        sound = sound.repeat(2 / 3)

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(tuple(sound.data), (0.0, 0.1))

    def test_repeat_invalid(self):
        sound = Sound.from_sinwave(440)

        with self.assertRaises(InvalidDurationError) as cm:
            sound.repeat(0)

        self.assertEqual(cm.exception.duration, 0)

    def test_trim_just(self):
        sound = Sound.from_array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], 1)

        self.assertEqual(sound[0.0].n_channels, 2)
        self.assertEqual(tuple(sound[0.0].data[0, :]), (0.0, 0.3))

        self.assertEqual(sound[0.4999999999].n_channels, 2)
        self.assertEqual(tuple(sound[0.4999999999].data[0, :]), (0.0, 0.3))

        self.assertEqual(sound[0.5000000001].n_channels, 2)
        self.assertEqual(tuple(sound[0.5000000001].data[0, :]), (0.1, 0.4))

        self.assertEqual(sound[1.0].n_channels, 2)
        self.assertEqual(tuple(sound[1.0].data[0, :]), (0.1, 0.4))

        self.assertEqual(sound[2.0].n_channels, 2)
        self.assertEqual(tuple(sound[2.0].data[0, :]), (0.2, 0.5))

        self.assertEqual(sound[3.0].n_channels, 2)
        self.assertEqual(tuple(sound[3.0].data[0, :]), (0.2, 0.5))

    def test_trim_just_invalid(self):
        sound = Sound.from_array([0.0, 0.1, 0.2], 3)

        with self.assertRaises(OutOfDurationError) as cm:
            sound[-0.001]

        self.assertEqual(cm.exception.duration, -0.001)
        self.assertEqual(cm.exception.min, 0)
        self.assertEqual(cm.exception.max, 1)

        with self.assertRaises(OutOfDurationError) as cm:
            sound[3.001]

        self.assertEqual(cm.exception.duration, 3.001)
        self.assertEqual(cm.exception.min, 0)
        self.assertEqual(cm.exception.max, 1)

    def test_trim_head(self):
        sound = Sound.from_array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4, 0.5))

        sound = sound[:2 / 3]

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4))

        sound = sound[:1 / 3]

        self.assertEqual(sound.duration, 1 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, ))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, ))

    def test_trim_head_by_end(self):
        sound = Sound.from_array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4, 0.5))

        sound = sound[:-1 / 3]

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4))

        sound = sound[:-1 / 3]

        self.assertEqual(sound.duration, 1 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, ))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, ))

    def test_trim_tail(self):
        sound = Sound.from_array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4, 0.5))

        sound = sound[1 / 3:]

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.4, 0.5))

        sound = sound[1 / 3:]

        self.assertEqual(sound.duration, 1 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.2, ))
        self.assertEqual(tuple(sound.data[:, 1]), (0.5, ))

    def test_trim_tail_by_end(self):
        sound = Sound.from_array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4, 0.5))

        sound = sound[-2 / 3:]

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.4, 0.5))

        sound = sound[-1 / 3:]

        self.assertEqual(sound.duration, 1 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.2, ))
        self.assertEqual(tuple(sound.data[:, 1]), (0.5, ))

    def test_trim_between(self):
        sound = Sound.from_array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4, 0.5))

        sound = sound[1 / 3: 2 / 3]

        self.assertEqual(sound.duration, 1 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.1, ))
        self.assertEqual(tuple(sound.data[:, 1]), (0.4, ))

    def test_trim_between_by_end(self):
        sound = Sound.from_array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4, 0.5))

        sound = sound[-2 / 3: -1 / 3]

        self.assertEqual(sound.duration, 1 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.1, ))
        self.assertEqual(tuple(sound.data[:, 1]), (0.4, ))

    def test_trim_between_invalid(self):
        sound = Sound.from_sinwave(440)

        with self.assertRaises(InvalidDurationError):
            sound[0.6:0.5]

        with self.assertRaises(InvalidDurationError):
            sound[0.5:0.5]

    def test_trim_step(self):
        sound = Sound.from_sinwave(440)

        with self.assertRaises(ValueError) as cm:
            sound[::1]

        self.assertEqual(str(cm.exception), 'step is not supported')

    def test_split_channels(self):
        a = Sound.from_array([0.1, 0.2], 2)
        b = Sound.from_array([0.3, 0.4], 2)
        ab = Sound.from_array([[0.1, 0.3], [0.2, 0.4]], 2)

        a_, b_ = ab.split_channels()
        self.assertEqual(a_, a)
        self.assertEqual(b_, b)

    def test_as_monaural(self):
        stereo = Sound.from_array([[0.1, 0.3], [0.2, 0.4]], 2)
        monaural = Sound.from_array([0.2, 0.3], 2)

        self.assertEqual(stereo.as_monaural(), monaural)

        self.assertIs(monaural.as_monaural(), monaural)

    def test_as_stereo(self):
        monaural = Sound.from_array([0.1, 0.2], 2)
        stereo = Sound.from_array([[0.1, 0.1], [0.2, 0.2]], 2)
        triple = Sound.from_array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], 2)

        self.assertEqual(monaural.as_stereo(), stereo)
        self.assertEqual(monaural.as_stereo(channels=3), triple)

        self.assertIs(stereo.as_stereo(), stereo)

    def test_as_stereo_invalid(self):
        stereo = Sound.from_array([[0.1, 0.1], [0.2, 0.2]], 2)

        with self.assertRaises(ValueError) as cm:
            self.assertEqual(stereo.as_stereo(1), stereo)

        self.assertEqual(str(cm.exception),
                         'channels must be 2 or greater but got 1')

    def test_concat(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)

        self.assertEqual(a.concat(b),
                         Sound.from_array([0.0, 0.1, 0.2, 0.3], 2))

        self.assertEqual(b.concat(a),
                         Sound.from_array([0.2, 0.3, 0.0, 0.1], 2))

    def test_concat_different_samplerate(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 3)

        with self.assertRaises(DifferentSamplerateError) as cm:
            a.concat(b)

        self.assertEqual(cm.exception.frequency, (2, 3))

        with self.assertRaises(DifferentSamplerateError) as cm:
            b.concat(a)

        self.assertEqual(cm.exception.frequency, (3, 2))

    def test_concat_different_channels(self):
        a = Sound.from_sinwave(440)
        b = Sound.from_sinwave(880).as_stereo()

        with self.assertRaises(DifferentChannelsError) as cm:
            a.concat(b)

        self.assertEqual(cm.exception.channels, (1, 2))

        with self.assertRaises(DifferentChannelsError) as cm:
            b.concat(a)

        self.assertEqual(cm.exception.channels, (2, 1))

    def test_overlay(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3, 0.4], 2)

        self.assertEqual(a.overlay(b),
                         Sound.from_array([0.2, 0.4, 0.4], 2))

        self.assertEqual(b.overlay(a),
                         Sound.from_array([0.2, 0.4, 0.4], 2))

    def test_overlay_different_samplerate(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 3)

        with self.assertRaises(DifferentSamplerateError) as cm:
            a.overlay(b)

        self.assertEqual(cm.exception.frequency, (2, 3))

        with self.assertRaises(DifferentSamplerateError) as cm:
            b.overlay(a)

        self.assertEqual(cm.exception.frequency, (3, 2))

    def test_overlay_different_channels(self):
        a = Sound.from_sinwave(440)
        b = Sound.from_sinwave(880).as_stereo()

        with self.assertRaises(DifferentChannelsError) as cm:
            a.overlay(b)

        self.assertEqual(cm.exception.channels, (1, 2))

        with self.assertRaises(DifferentChannelsError) as cm:
            b.overlay(a)

        self.assertEqual(cm.exception.channels, (2, 1))

    def test_save_and_load_file(self):
        original = Sound.from_sinwave(440).concat(Sound.from_sinwave(880))

        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            original.write(f.name)

            loaded = Sound.from_file(f.name)

        self.assertEqual(original.samplerate, loaded.samplerate)
        self.assertEqual(original.duration, loaded.duration)
        self.assertTrue(numpy.allclose(original.data, loaded.data, rtol=0.01))

    def test_save_and_load_buffer(self):
        original = Sound.from_sinwave(440).concat(Sound.from_sinwave(880))

        with io.BytesIO() as f:
            original.write(f, format_='wav')

            f.seek(0)

            loaded = Sound.from_file(f)

        self.assertEqual(original.samplerate, loaded.samplerate)
        self.assertEqual(original.duration, loaded.duration)
        self.assertTrue(numpy.allclose(original.data, loaded.data, rtol=0.01))
