import unittest

import pep8


class PEP8Test(unittest.TestCase):
    def test_pep8(self):
        pep8.DEFAULT_IGNORE = 'W503'
        checker = pep8.StyleGuide()

        result = checker.check_files(['gensound'])

        message = 'pep8: {} errors / {} warnings'.format(result.get_count('E'),
                                                         result.get_count('W'))
        self.assertEqual(result.total_errors, 0, message)
