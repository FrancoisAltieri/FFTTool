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
        res = np.zeros((int(signal_size-self.window_number_points()),int(1+self.window_number_points()/2)))
        for i in range(0,signal_size-self.window_number_points()):
            sample = signal[i:i+self.window_number_points()]
            rft = np.fft.rfft(sample)
            fftsample = rft.real**2 + rft.imag**2
            res[i] = fftsample
        return res

    def process_signal_index(self,signal,index):
        signal_size = len(signal)
        res = np.zeros((int(signal_size-self.window_number_points()),1))
        for i in range(0,signal_size-self.window_number_points()):
            sample = signal[i:(i+self.window_number_points())]
            rft = np.fft.rfft(sample)
            fftsample = rft.real**2 + rft.imag**2
            res[i] = fftsample[index]
        return res

    def generate_test_signal(self,duration, dt):
        time = np.arange(0,duration,dt)
        com = lambda f: np.cos(2*np.pi*f*time)
        sigArray = com(1) + 0.5 * com(2)
        return sigArray

    def generate_test_signal2(self,duration, dt):
        time = np.arange(0,duration,dt)
        com = lambda f: np.cos(2*np.pi*f*time)
        sigArray = com(1) + 0.5 * com(2)
        sigArray2 = com(1) + 0.5 * com(3)
        return np.concatenate((sigArray,sigArray2))
