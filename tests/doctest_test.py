import doctest
import unittest

import gensound
import gensound.effect
import gensound.sound


class DocTest(unittest.TestCase):
    def test_doctest_init(self):
        failure, total = doctest.testmod(gensound)
        self.assertEqual(failure, 0)

    def test_doctest_effect(self):
        failure, total = doctest.testmod(gensound.effect)
        self.assertEqual(failure, 0)

    def test_doctest_sound(self):
        failure, total = doctest.testmod(gensound.sound)
        self.assertEqual(failure, 0)
