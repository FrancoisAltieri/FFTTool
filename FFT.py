import numpy as np


class FFTAnalyser:
    _dt = 0.1
    _window = 1

    def set_moving_average_window(self,n_seconds):
        """ set the duration in seconds of the window used for moving average
            returns the list of frequencies that can be analysed with that window
        """
        _window = n_seconds
        return self.compute_frequencies()

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
            res[i] = rft.real**2 + rft.imag**2
        return res.transpose()

    def process_signal_index(self,signal,index):
        res = self.process_signal(signal)
        return res[index]

    @staticmethod
    def generate_test_signal(duration, dt):
        time = np.arange(0,duration,dt)
        com = lambda f: np.cos(2*np.pi*f*time)
        sigArray = com(1) + 0.5 * com(2)
        return sigArray

    @staticmethod
    def generate_test_signal2(duration, dt):
        time = np.arange(0,duration,dt)
        com = lambda f: np.cos(2*np.pi*f*time)
        sigArray = com(1) + 0.5 * com(2)
        sigArray2 = com(1) + 0.5 * com(3)
        return np.concatenate((sigArray,sigArray2))
