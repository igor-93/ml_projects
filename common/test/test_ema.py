from unittest import TestCase, main
import numpy as np
import pandas as pd

from ..ema import ewma, ewma_span

TOL = 1e-9


class EWMATest(TestCase):

    def test_ewma_alpha(self):
        alpha = 0.55
        x = np.random.randint(0, 30, 15)
        df = pd.DataFrame(x, columns=['A'])
        expected = df.ewm(alpha=alpha).mean()
        expected = expected.values.squeeze()

        calculated = ewma(x, alpha=alpha)

        error = np.abs(expected - calculated)

        self.assertTrue((error < TOL).all())

        expected2 = df.ewm(alpha=alpha).mean().values.squeeze()
        error = np.abs(expected2 - calculated)

        self.assertTrue((error < TOL).all())

    def test_ewma_span(self):
        span = 30
        x = np.random.randint(0, 30, 150)
        df = pd.DataFrame(x, columns=['A'])
        expected = df.ewm(span=span).mean()
        expected = expected.values.squeeze()

        calculated = ewma(x, span=span)
        error = np.abs(expected - calculated)
        self.assertTrue((error < TOL).all())

        expected2 = df.ewm(span=span).mean().values.squeeze()
        error = np.abs(expected2 - calculated)
        self.assertTrue((error < TOL).all())

    def test_ewma_span_jit(self):
        span = 30
        x = np.random.randint(10, 50, 550)
        df = pd.DataFrame(x, columns=['A'])
        expected = df.ewm(span=span).mean()
        expected = expected.values.squeeze()

        calculated = ewma_span(x, span=span)
        error = np.abs(expected[-10:] - calculated[-10:])
        print(error)
        self.assertTrue((error < TOL).all())

        expected2 = df.ewm(span=span).mean().values.squeeze()
        error = np.abs(expected2[-10:] - calculated[-10:])
        self.assertTrue((error < TOL).all())

    def test_ewma_fail(self):
        x = np.random.randint(0, 30, 150)

        self.assertRaises(ValueError, ewma, x=x)
        self.assertRaises(ValueError, ewma, x=x, alpha=0.5, span=10)
        self.assertRaises(ValueError, ewma, x=x, alpha=1.2)


if __name__ == '__main__':
    main()
