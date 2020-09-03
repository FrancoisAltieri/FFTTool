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
        res = np.zeros((int(signal_size-self.window_number_points()),int(self.window_number_points()/2)))
        for i in range(0,signal_size-self.window_number_points()):
            sample = signal[i:i+self.window_number_points()-1]
            fftsample = np.fft.rfft(sample)
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
        time = lambda i: i * dt
        twoPiT = lambda i: time(i) * 2 * np.pi
        omegaTForF = lambda i,f : f*twoPiT(i)
        com = lambda i, f: np.cos(omegaTForF(i, f))
        signal = lambda i: com(i, 1) + 0.5 * com(i, 2)
        sigArray = np.fromfunction(signal, (int(duration/dt),))
        return sigArray

    def generate_test_signal2(self,duration, dt):
        time = lambda i: i * dt
        twoPiT = lambda i: time(i) * 2 * np.pi
        omegaTForF = lambda i,f : f*twoPiT(i)
        com = lambda i, f: np.cos(omegaTForF(i, f))
        signal = lambda i: com(i, 1) + 0.5 * com(i, 2)
        signal2 = lambda i: com(i, 1) + 0.5 * com(i, 3)
        sigArray = np.fromfunction(signal, (int(duration/dt),))
        sigArray2 = np.fromfunction(signal2, (int(duration/dt),))
        return np.concatenate((sigArray,sigArray2))
