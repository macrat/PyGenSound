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
        self.assertTrue(sum((filtered.data - self.ab.data) ** 2) < 0.1)

        filtered = LowPassFilter(110).apply(self.abc)
        self.assertEqual(filtered.samplerate, self.abc.samplerate)
        self.assertEqual(filtered.duration, self.abc.duration)
        self.assertTrue(sum((filtered.data - self.a.data) ** 2) < 0.1)

    def test_high_pass_filter(self):
        filtered = HighPassFilter(190).apply(self.abc)
        self.assertEqual(filtered.samplerate, self.abc.samplerate)
        self.assertEqual(filtered.duration, self.abc.duration)
        self.assertTrue(sum((filtered.data - self.bc.data) ** 2) < 0.1)

        filtered = HighPassFilter(290).apply(self.abc)
        self.assertEqual(filtered.samplerate, self.abc.samplerate)
        self.assertEqual(filtered.duration, self.abc.duration)
        self.assertTrue(sum((filtered.data - self.c.data) ** 2) < 0.1)


class ResamplingTest(unittest.TestCase):
    def test_resampling(self):
        sound = Sound.from_array([-0.4, -0.2, 0.0, 0.2, 0.4], 5)

        self.assertEqual(Resampling(9).apply(sound),
                         Sound.from_array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1,
                                           0.2, 0.3, 0.4], 9))

        self.assertEqual(Resampling(3).apply(sound),
                         Sound.from_array([-0.4, 0.0, 0.4], 3))


class ChangeSpeedTest(unittest.TestCase):
    def test_change_speed(self):
        sound = Sound.from_array([-0.4, -0.2, 0.0, 0.2, 0.4], 5)

        self.assertEqual(ChangeSpeed(5 / 9).apply(sound),
                         Sound.from_array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1,
                                           0.2, 0.3, 0.4], 5))

        self.assertEqual(ChangeSpeed(5 / 3).apply(sound),
                         Sound.from_array([-0.4, 0.0, 0.4], 5))
