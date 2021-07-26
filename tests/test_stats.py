import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join('../scripts')))
from stats import pooled_prob, pooled_SE


class TestDfHelper(unittest.TestCase):

    def setUp(self):
        self.N_A = 100
        self.N_B = 100
        self.X_A = 55
        self.X_B = 80

    def test_pooled_prob(self):
        p = pooled_prob(self.N_A, self.N_B, self.X_A, self.X_B)
        self.assertEqual(p, 0.675)

    def test_pooled_SE(self):
        se = pooled_SE(self.N_A, self.N_B, self.X_A, self.X_B)
        self.assertEqual(se, 0.06623820649745885)

if __name__ == '__main__':
    unittest.main()