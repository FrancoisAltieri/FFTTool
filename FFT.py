import numpy as np


class FFTAnalyser:
    """
    helper class to perform Fast Fourier Transform on a signal across a moving window
    provides a more advanced analysis than a simple moving average
    """
    _dt = 0.1
    _window = 1

    def set_timestep(self,timestep_seconds):
        """ set the timestep in second used to sample the signal
        """
        self._dt = timestep_seconds

    def set_moving_average_window(self,n_seconds):
        """ set the duration in seconds of the window used for moving average
            returns the list of frequencies that can be analysed with that window
        """
        _window = n_seconds
        return self.compute_frequencies()

    def compute_frequencies(self):
        """
        :return: list of frequencies that are retrived with fft analysis
        """
        freq = np.fft.rfftfreq(self.window_number_points(), self._dt)
        return freq

    def nyquist(self):
        """
        returns the NyQuist frequency for this analyser
        any signal with a higher frequency cannot be analysed
        """
        return 0.5/self._dt

    def window_number_points(self):
        """
        computes the number of values included in the window for moving average (ie window/timestep)
        """
        return int(self._window / self._dt)

    def process_signal(self,signal):
        """
        analyse the given signal with a moving FFT across its duration
        :param signal: signal to analyse (numpy 1D array)
        :return: np array for all rffts with the moving window for r= process_signal(t) r[0] is the evolution of harmonic 0 over time
        see compute_frequencies to know which frequency corresponds to a given index
        """
        signal_size = len(signal)
        res = np.zeros((int(signal_size-self.window_number_points()),int(1+self.window_number_points()/2)))
        for i in range(0,signal_size-self.window_number_points()):
            sample = signal[i:i+self.window_number_points()]
            rft = np.fft.rfft(sample)
            res[i] = np.sqrt(rft.real**2+rft.imag**2) /self.window_number_points()
        return res.transpose()

    def last_window(self,signal):
        signal_size = len(signal)
        return signal[signal_size-self.window_number_points():signal_size]

    def last_window_fourier(self,signal):
        sample = self.last_window(signal)
        rft = np.fft.rfft(sample)
        return np.sqrt(rft.real**2+rft.imag**2) /self.window_number_points()

    def process_signal_index(self,signal,index):
        """
        same as process_signal but returns only the time evolution of the frequency at the given index
        """
        res = self.process_signal(signal)
        return res[index]

