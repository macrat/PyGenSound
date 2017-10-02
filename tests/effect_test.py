import unittest

from gensound.sound import Sound, overlay
from gensound.effect import *


class JoinedEffectTest(unittest.TestCase):
    def test_then_method(self):
        sound = Sound.from_sinwave(440)

        in_ = LinearFadeIn()
        out = LinearFadeOut()

        self.assertEqual(in_.then(out).apply(sound),
                         in_.apply(out.apply(sound)))

    def test_join_class(self):
        sound = Sound.from_sinwave(440)

        in_ = LinearFadeIn()
        out = LinearFadeOut()

        self.assertEqual(in_.then(out).apply(sound),
                         JoinedEffect(in_, out).apply(sound))

    def test_join_with_operator(self):
        sound = Sound.from_sinwave(440)
        a = Resampling(88200)
        b = Resampling(44100)

        resampled = b << a << sound
        self.assertEqual(resampled.samplerate, sound.samplerate)

        resampled = sound >> a >> b
        self.assertEqual(resampled.samplerate, sound.samplerate)

        resampled = sound >> a >> b
        self.assertEqual(resampled.samplerate, sound.samplerate)

        resampled = (a >> b) << sound
        self.assertEqual(resampled.samplerate, sound.samplerate)

        resampled = (b << a) << sound
        self.assertEqual(resampled.samplerate, sound.samplerate)

        resampled = sound >> (a >> b)
        self.assertEqual(resampled.samplerate, sound.samplerate)

        resampled = sound >> (b << a)
        self.assertEqual(resampled.samplerate, sound.samplerate)

        resampled = a << sound >> b
        self.assertEqual(resampled.samplerate, sound.samplerate)


class LinearFadeTest(unittest.TestCase):
    def setUp(self):
        self.sound = Sound.from_array([1, -1, 1, -1, 1], 1)

    def test_linear_fade_in(self):
        self.assertEqual(LinearFadeIn().apply(self.sound),
                         Sound.from_array([0.0, -0.25, 0.5, -0.75, 1.0], 1))

        self.assertEqual(LinearFadeIn(duration=2).apply(self.sound),
                         Sound.from_array([0.0, -1.0, 1.0, -1.0, 1.0], 1))

        self.assertEqual(LinearFadeIn(duration=3).apply(self.sound),
                         Sound.from_array([0.0, -0.5, 1.0, -1.0, 1.0], 1))

    def test_linear_fade_out(self):
        self.assertEqual(LinearFadeOut().apply(self.sound),
                         Sound.from_array([1.0, -0.75, 0.5, -0.25, 0.0], 1))

        self.assertEqual(LinearFadeOut(duration=2).apply(self.sound),
                         Sound.from_array([1.0, -1.0, 1.0, -1.0, 0.0], 1))

        self.assertEqual(LinearFadeOut(duration=3).apply(self.sound),
                         Sound.from_array([1.0, -1.0, 1.0, -0.5, 0.0], 1))


class PassFilterTest(unittest.TestCase):
    def setUp(self):
        self.a = Sound.from_sinwave(100,
                                    duration=0.1,
                                    volume=1 / 3,
                                    samplerate=60000)

        self.b = Sound.from_sinwave(200,
                                    duration=0.1,
                                    volume=1 / 3,
                                    samplerate=60000)

        self.c = Sound.from_sinwave(300,
                                    duration=0.1,
                                    volume=1 / 3,
                                    samplerate=60000)

        self.ab = overlay(self.a, self.b)
        self.bc = overlay(self.b, self.c)
        self.abc = overlay(self.a, self.b, self.c)

    def test_low_pass_filter(self):
        filtered = LowPassFilter(210).apply(self.abc)
        self.assertEqual(filtered.samplerate, self.abc.samplerate)
        self.assertEqual(filtered.duration, self.abc.duration)
        self.assertTrue(numpy.allclose(filtered.data, self.ab.data))

        filtered = LowPassFilter(110).apply(self.abc)
        self.assertEqual(filtered.samplerate, self.abc.samplerate)
        self.assertEqual(filtered.duration, self.abc.duration)
        self.assertTrue(numpy.allclose(filtered.data, self.a.data))

    def test_high_pass_filter(self):
        filtered = HighPassFilter(190).apply(self.abc)
        self.assertEqual(filtered.samplerate, self.abc.samplerate)
        self.assertEqual(filtered.duration, self.abc.duration)
        self.assertTrue(numpy.allclose(filtered.data, self.bc.data))

        filtered = HighPassFilter(290).apply(self.abc)
        self.assertEqual(filtered.samplerate, self.abc.samplerate)
        self.assertEqual(filtered.duration, self.abc.duration)
        self.assertTrue(numpy.allclose(filtered.data, self.c.data))


class ResamplingTest(unittest.TestCase):
    def test_resampling(self):
        sound = Sound.from_array([-0.4, -0.2, 0.0, 0.2, 0.4], 5)

        self.assertEqual(Resampling(9).apply(sound),
                         Sound.from_array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1,
                                           0.2, 0.3, 0.4], 9))

        self.assertEqual(Resampling(3).apply(sound),
                         Sound.from_array([-0.4, 0.0, 0.4], 3))

    def test_same_samplerate(self):
        sound = Sound.from_sinwave(440, samplerate=44100)

        resample_to_same_rate = Resampling(sound.samplerate).apply(sound)

        self.assertIs(resample_to_same_rate, sound)
        self.assertTrue((resample_to_same_rate.data == sound.data).all())

    def test_twice_resampling(self):
        sound = Sound.from_sinwave(440, samplerate=44100)

        twice_resampler = Resampling(48000).then(Resampling(sound.samplerate))
        twice_resampled = twice_resampler.apply(sound)

        self.assertEqual(twice_resampled.samplerate, sound.samplerate)
        self.assertIsNot(twice_resampled, sound)
        self.assertFalse((twice_resampled.data == sound.data).all())
        self.assertTrue(numpy.allclose(twice_resampled.data,
                                       sound.data,
                                       atol=0.01))


class ChangeSpeedTest(unittest.TestCase):
    def test_change_speed(self):
        sound = Sound.from_array([-0.4, -0.2, 0.0, 0.2, 0.4], 5)

        self.assertEqual(ChangeSpeed(5 / 9).apply(sound),
                         Sound.from_array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1,
                                           0.2, 0.3, 0.4], 5))

        self.assertEqual(ChangeSpeed(5 / 3).apply(sound),
                         Sound.from_array([-0.4, 0.0, 0.4], 5))

    def test_change_speed_minus(self):
        sound = Sound.from_array([-0.4, -0.2, 0.0, 0.2, 0.4], 5)

        self.assertEqual(ChangeSpeed(-5 / 9).apply(sound),
                         Sound.from_array([0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2,
                                           -0.3, -0.4], 5))

        self.assertEqual(ChangeSpeed(-5 / 3).apply(sound),
                         Sound.from_array([0.4, 0.0, -0.4], 5))

    def test_change_speed_invalid(self):
        with self.assertRaises(ValueError, msg='speed_rate must not 0'):
            ChangeSpeed(0)


class ChangeVolumeTest(unittest.TestCase):
    def test_volume(self):
        sound = Sound.from_sinwave(440)

        self.assertAlmostEqual(sound.volume, 1, places=4)

        sound = ChangeVolume(0.8).apply(sound)

        self.assertAlmostEqual(sound.volume, 0.8, places=4)

        sound = ChangeVolume(0.3).apply(sound)

        self.assertAlmostEqual(sound.volume, 0.3, places=4)

        sound = ChangeVolume(1.0).apply(sound)

        self.assertAlmostEqual(sound.volume, 1, places=4)

    def test_volume_invalid(self):
        sound = Sound.from_sinwave(440)

        with self.assertRaises(InvalidVolumeError) as cm:
            ChangeVolume(-0.1)

        self.assertEqual(cm.exception.volume, -0.1)

        with self.assertRaises(InvalidVolumeError) as cm:
            ChangeVolume(1.1)

        self.assertEqual(cm.exception.volume, 1.1)


class ReversePlayTest(unittest.TestCase):
    def test_reverse_play(self):
        sound = Sound.from_array([-0.1, 0.0, 0.1, 0.2], 4)
        reverse = ReversePlay().apply(sound)

        self.assertEqual(reverse.samplerate, sound.samplerate)
        self.assertEqual(reverse.duration, sound.duration)
        self.assertEqual(tuple(reverse.data[:, 0]), (0.2, 0.1, 0.0, -0.1))

        self.assertEqual(ReversePlay().then(ReversePlay()).apply(sound), sound)


class TrimTest(unittest.TestCase):
    def test_trim_head(self):
        sound = Sound.from_array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4, 0.5))

        sound = Trim(end=2 / 3).apply(sound)

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4))

        sound = Trim(end=1 / 3).apply(sound)

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

        sound = Trim(end=-1 / 3).apply(sound)

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.0, 0.1))
        self.assertEqual(tuple(sound.data[:, 1]), (0.3, 0.4))

        sound = Trim(end=-1 / 3).apply(sound)

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

        sound = Trim(start=1 / 3).apply(sound)

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.4, 0.5))

        sound = Trim(start=1 / 3).apply(sound)

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

        sound = Trim(start=-2 / 3).apply(sound)

        self.assertEqual(sound.duration, 2 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.1, 0.2))
        self.assertEqual(tuple(sound.data[:, 1]), (0.4, 0.5))

        sound = Trim(start=-1 / 3).apply(sound)

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

        sound = Trim(start=1 / 3, end=2 / 3).apply(sound)

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

        sound = Trim(start=-2 / 3, end=-1 / 3).apply(sound)

        self.assertEqual(sound.duration, 1 / 3)
        self.assertEqual(sound.n_channels, 2)
        self.assertEqual(tuple(sound.data[:, 0]), (0.1, ))
        self.assertEqual(tuple(sound.data[:, 1]), (0.4, ))

    def test_trim_between_invalid(self):
        sound = Sound.from_sinwave(440)

        with self.assertRaises(InvalidDurationError) as cm:
            Trim(start=0.6, end=0.5)

        self.assertAlmostEqual(cm.exception.duration, -0.1)

        with self.assertRaises(InvalidDurationError) as cm:
            Trim(start=0.5, end=0.5)

        self.assertAlmostEqual(cm.exception.duration, 0)

        trim = Trim(start=2)
        with self.assertRaises(InvalidDurationError) as cm:
            trim.apply(sound)

        self.assertAlmostEqual(cm.exception.duration, 0)
