from unittest import TestCase
from FFT import *


class TestFFTAnalyser(TestCase):
    def test_compute_frequencies(self):
        f = FFTAnalyser()
        freqs = f.compute_frequencies()
        nyquist = f.nyquist()
        self.assertGreaterEqual(nyquist, np.max(freqs))

        dt = 0.1
        time = lambda i: i * dt
        twoPiT = lambda i: time(i) * 2 * np.pi
        omegaTForF = lambda i,f : f*twoPiT(i)
        com = lambda i, f: np.cos(omegaTForF(i, f))
        signal = lambda i: com(i, 1) + 0.5 * com(i, 3)
        sigArray = np.fromfunction(signal, (300,))
        f.process_signal(sigArray)
