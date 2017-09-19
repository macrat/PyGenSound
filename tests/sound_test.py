import tempfile
import unittest

import numpy

from gensound.sound import _repeat_array, overlay, concat, Sound


class SoundUtilsTest(unittest.TestCase):
    def test_repeat_array(self):
        array = numpy.array([1, 2, 3])
        repeated = numpy.array([1, 2, 3] * 3)

        self.assertEqual(tuple(_repeat_array(array, 6)), tuple(repeated[:6]))
        self.assertEqual(tuple(_repeat_array(array, 8)), tuple(repeated[:8]))
        self.assertEqual(tuple(_repeat_array(array, 2)), tuple(repeated[:2]))

    def test_overlay(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5], 2)

        self.assertEqual(overlay(a, b, c).get_samplerate(), 2)
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

        self.assertEqual(overlay(a, b, c).get_samplerate(), 2)
        self.assertEqual(overlay(a, b, c).duration, 2)

        self.assertEqual(overlay(a, b, c),
                         Sound.from_array([0.6, 0.9, 0.6, 0.7], 2))

    def test_concat(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)
        c = Sound.from_array([0.4, 0.5], 2)

        self.assertEqual(concat(a, b, c).get_samplerate(), 2)
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

        self.assertEqual(concat(a, b, c).get_samplerate(), 2)
        self.assertEqual(concat(a, b, c).duration, 4)

        self.assertEqual(concat(a, b, c), Sound.from_array([
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
        ], 2))


class SoundTest(unittest.TestCase):
    def test_constructor(self):
        sound = Sound(numpy.array([-1.0, 0.5, 1.0]), 1)

        self.assertEqual(sound.get_samplerate(), 1)
        self.assertEqual(sound.duration, 3)
        self.assertEqual(tuple(sound.data), (-1.0, 0.5, 1.0))

    def test_constructor_clip(self):
        sound = Sound(numpy.array([-1.1, -1.0, 1.0, 1.1]), 2)

        self.assertEqual(sound.get_samplerate(), 2)
        self.assertEqual(sound.duration, 2)
        self.assertEqual(tuple(sound.data), (-1.0, -1.0, 1.0, 1.0))

    def test_from_array(self):
        self.assertEqual(Sound(numpy.array([-0.5, 0.5]), 44100),
                         Sound.from_array([-0.5, 0.5], 44100))

        self.assertEqual(Sound(numpy.array([0.1, -0.1]), 100),
                         Sound.from_array([0.1, -0.1], 100))

    def test_from_sinwave(self):
        sound = Sound.from_sinwave(440,
                                   duration=1,
                                   volume=0.5,
                                   samplerate=44100)

        self.assertEqual(sound.get_samplerate(), 44100)
        self.assertTrue(abs(sound.duration - 1) < 0.01)
        self.assertTrue((-0.5 <= sound.data).all())
        self.assertTrue((sound.data <= 0.5).all())
        self.assertTrue(abs(sound.data.max() - 0.5) < 1e-04)
        self.assertTrue(abs(sound.data.min() + 0.5) < 1e-04)

        sound = Sound.from_sinwave(880,
                                   duration=2,
                                   volume=0.8,
                                   samplerate=88200)

        self.assertEqual(sound.get_samplerate(), 88200)
        self.assertTrue(abs(sound.duration - 2) < 0.02)
        self.assertTrue((-0.8 <= sound.data).all())
        self.assertTrue((sound.data <= 0.8).all())
        self.assertTrue(abs(sound.data.max() - 0.8) < 1e-04)
        self.assertTrue(abs(sound.data.min() + 0.8) < 1e-04)

    def test_silence(self):
        sound = Sound.silence(100)

        self.assertEqual(sound.get_samplerate(), 100)
        self.assertEqual(sound.duration, 1/100)
        self.assertEqual(tuple(sound.data), (0, ))

        sound = Sound.silence(20)

        self.assertEqual(sound.get_samplerate(), 20)
        self.assertEqual(sound.duration, 1/20)
        self.assertEqual(tuple(sound.data), (0, ))

    def test_whitenoise(self):
        sound = Sound.from_whitenoise(duration=2, volume=0.1, samplerate=100)

        self.assertEqual(sound.get_samplerate(), 100)
        self.assertEqual(sound.duration, 2)
        self.assertTrue((-0.1 <= sound.data).all())
        self.assertTrue((sound.data <= 0.1).all())

    def test_volume(self):
        sound = Sound.from_sinwave(440)

        self.assertTrue(abs(sound.data.max() - 1) < 1e-04)
        self.assertTrue(abs(sound.data.min() + 1) < 1e-04)

        sound = sound.volume(0.8)

        self.assertTrue(abs(sound.data.max() - 0.8) < 1e-04)
        self.assertTrue(abs(sound.data.min() + 0.8) < 1e-04)

        sound = sound.volume(0.3)

        self.assertTrue(abs(sound.data.max() - 0.3) < 1e-04)
        self.assertTrue(abs(sound.data.min() + 0.3) < 1e-04)

        sound = sound.volume(1.0)

        self.assertTrue(abs(sound.data.max() - 1) < 1e-04)
        self.assertTrue(abs(sound.data.min() + 1) < 1e-04)

    def test_repeat(self):
        sound = Sound.from_array([0.0, 0.1, 0.2], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(tuple(sound.data), (0.0, 0.1, 0.2))

        sound = sound.repeat(2)

        self.assertEqual(sound.duration, 2)
        self.assertEqual(tuple(sound.data), (0.0, 0.1, 0.2, 0.0, 0.1, 0.2))

        sound = sound.repeat(2/3)

        self.assertEqual(sound.duration, 2/3)
        self.assertEqual(tuple(sound.data), (0.0, 0.1))

    def test_trim(self):
        sound = Sound.from_array([0.0, 0.1, 0.2], 3)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(tuple(sound.data), (0.0, 0.1, 0.2))

        sound = sound.trim(2/3)

        self.assertEqual(sound.duration, 2/3)
        self.assertEqual(tuple(sound.data), (0.0, 0.1))

        sound = sound.trim(1/3)

        self.assertEqual(sound.duration, 1/3)
        self.assertEqual(tuple(sound.data), (0.0, ))

    def test_split(self):
        sound = Sound.from_array([0.0, 0.1, 0.2, 0.3], 4)

        self.assertEqual(sound.duration, 1)
        self.assertEqual(tuple(sound.data), (0.0, 0.1, 0.2, 0.3))

        a, b = sound.split(1/2)

        self.assertEqual(a, Sound.from_array([0.0, 0.1], 4))
        self.assertEqual(b, Sound.from_array([0.2, 0.3], 4))

        c, d = sound.split(1/4)

        self.assertEqual(c, Sound.from_array([0.0], 4))
        self.assertEqual(d, Sound.from_array([0.1, 0.2, 0.3], 4))

    def test_concat(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3], 2)

        self.assertEqual(a.concat(b),
                         Sound.from_array([0.0, 0.1, 0.2, 0.3], 2))

        self.assertEqual(b.concat(a),
                         Sound.from_array([0.2, 0.3, 0.0, 0.1], 2))

    def test_overlay(self):
        a = Sound.from_array([0.0, 0.1], 2)
        b = Sound.from_array([0.2, 0.3, 0.4], 2)

        self.assertEqual(a.overlay(b),
                         Sound.from_array([0.2, 0.4, 0.4], 2))

        self.assertEqual(b.overlay(a),
                         Sound.from_array([0.2, 0.4, 0.4], 2))


    def test_save_and_load(self):
        original = Sound.from_sinwave(440).concat(Sound.from_sinwave(880))

        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            original.write(f.name)

            loaded = Sound.from_file(f.name)

        self.assertEqual(original.get_samplerate(), loaded.get_samplerate())
        self.assertEqual(original.duration, loaded.duration)
        self.assertTrue(numpy.allclose(original.data, loaded.data, rtol=0.01))
