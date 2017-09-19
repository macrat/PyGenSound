import doctest
import unittest

import gensound


class DocTest(unittest.TestCase):
    def test_doctest(self):
        failure, total = doctest.testmod(gensound)
        self.assertEqual(failure, 0)
