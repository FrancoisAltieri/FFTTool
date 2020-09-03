from unittest import TestCase
from FFT import *


class TestFFTAnalyser(TestCase):

    @staticmethod
    def _generate_test_signal(duration, dt):
        """
        generate a simple signal equal to the sum of two cosine
        result signal is at f = 1Hz and 2Hz
        :param duration: duration of the generated signal
        :param dt: timestep
        :return: cos(2*pi*1Hz*t)+0.5*cos(2*pi*2Hz*t)
        """
        time = np.arange(0, duration, dt)
        com = lambda f: np.cos(2 * np.pi * f * time)
        sigArray = com(1) + 0.5 * com(2)
        return sigArray

    @staticmethod
    def _generate_test_signal2(duration, dt):
        """
        generate a simple signal equal to the sum of two cosine
        result signal is at f = 1Hz and 2Hz for t = 0:duration
                            f = 1Hz and 3Hz for t = duration:2*duration
        :param duration: duration of the generated signal
        :param dt: timestep
        :return: cos(2*pi*1Hz*t)+0.5*cos(2*pi*2Hz*t) t<= duration
                 cos(2*pi*1Hz*t)+0.5*cos(2*pi*3Hz*t) t> duration
        """
        time = np.arange(0, duration, dt)
        com = lambda f: np.cos(2 * np.pi * f * time)
        sigArray = com(1) + 0.5 * com(2)
        sigArray2 = com(1) + 0.5 * com(3)
        return np.concatenate((sigArray, sigArray2))

    def test_compute_frequencies(self):
        f = FFTAnalyser()
        f.set_timestep(0.1)
        freqs = f.set_moving_average_window(1)
        nyquist = f.nyquist()
        self.assertGreaterEqual(nyquist, np.max(freqs))

    def test_process_signal(self):
        dt = 0.1
        duration = 10
        f = FFTAnalyser()
        f.set_timestep(dt)
        freqs = f.set_moving_average_window(1)
        signal2 = TestFFTAnalyser._generate_test_signal2(duration, dt)
        r = f.process_signal(signal2)
        average = r[0]
        self.assertAlmostEqual(average[0], 0)
        freq1 = r[1]
        self.assertNotAlmostEqual(freq1[0], 0)
        freq2 = r[2]
        # makes sure that 2Hz is found at the beginning of the signal but not at the end
        self.assertNotAlmostEqual(freq2[0], 0)
        self.assertAlmostEqual(freq2[len(freq2)-1], 0)

    def test_process_signal_index(self):
        dt = 0.1
        duration = 10
        f = FFTAnalyser()
        f.set_timestep(dt)
        f.set_moving_average_window(1)
        signal = TestFFTAnalyser._generate_test_signal(duration, dt)
        r = f.process_signal(signal)
        average = r[0]
        average_by_index = f.process_signal_index(signal,0)
        self.assertEqual(average.all(),average_by_index.all())
