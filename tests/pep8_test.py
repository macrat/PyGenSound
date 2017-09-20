import typing
import unittest

import pep8


class PEP8Test(unittest.TestCase):
    def run_pep8_test(self, modules: typing.Iterable[str]):
        pep8.DEFAULT_IGNORE = 'W503'
        checker = pep8.StyleGuide()

        result = checker.check_files(modules)

        message = 'pep8: {} errors / {} warnings'.format(result.get_count('E'),
                                                         result.get_count('W'))
        self.assertEqual(result.total_errors, 0, message)

    def test_pep8_gensound(self):
        self.run_pep8_test(['gensound'])

    def test_pep8_tests(self):
        self.run_pep8_test(['tests'])
