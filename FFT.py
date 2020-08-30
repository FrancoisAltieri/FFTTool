import numpy as np


class FFTAnalyser:
    _dt = 0.1
    _window = 1

    def compute_frequencies(self):
        freq = np.fft.rfftfreq(self.window_number_points(), self._dt)
        return freq

    def nyquist(self):
        return 0.5/self._dt

    def window_number_points(self):
        return int(self._window / self._dt)

    def process_signal(self,signal):
        signal_size = len(signal)
        res = np.zeros(signal_size-self.window_number_points(),self.window_number_points()/2)
        for i in range(0,signal_size-self.window_number_points()):
            sample = signal[i:i+self.window_number_points()-1]
            fftsample = np.fft.rfft(sample)
            res[i] = fftsample
