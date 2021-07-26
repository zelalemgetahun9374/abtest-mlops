import os
import sys
import unittest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('../scripts')))
from csv_helper import CsvHelper


class TestDfHelper(unittest.TestCase):

    def setUp(self):
        self.helper = CsvHelper()

    def test_save_csv(self):
        df = pd.DataFrame({'col1': range(1,4), 'col2': range(3,6)})
        self.helper.save_csv(df, './test.csv', False)
        df2 = pd.read_csv('test.csv')
        self.assertEqual(df.shape, df2.shape)

    def test_read_csv(self):
        df = self.helper.read_csv('test.csv')
        df2 = pd.read_csv('./test.csv')
        self.assertEqual(df.shape, df2.shape)

if __name__ == '__main__':
    unittest.main()