import doctest
import unittest

import gensound
import tests


class DocTest(unittest.TestCase):
    def test_doctest_gensound(self):
        failure, total = doctest.testmod(gensound)
        self.assertEqual(failure, 0)

    def test_doctest_tests(self):
        failure, total = doctest.testmod(tests)
        self.assertEqual(failure, 0)
