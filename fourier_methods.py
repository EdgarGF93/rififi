import matplotlib.pyplot as plt
import numpy as np

class Fourier():
    def __init__(self, dt=0.001, t0=0, tf=1) -> None:
        self.dt = dt
        self.t0 = t0
        self.tf = tf
        self.times = np.arange(t0, tf, dt)
        self.N = self.times.size
        self.freqs = np.fft.fftfreq(n=self.N, d=dt)
        self.freqs_shift = np.fft.fftshift(self.freqs)
        self.signal = None

    def generate_signal(self, freqs=np.array([1]), noise_amplitudes=np.array([0.1])):
        # Frequencies
        signal = np.sum([np.cos(2 * np.pi * self.times * f) for f in freqs], axis=0)
        # Noise
        signal += np.sum([np.random.rand(self.N) * f - f/2 for f in noise_amplitudes], axis=0)
        return signal

    def set_signal(self, freqs=np.array([1]), noise_amplitudes=np.array([0.1])):
        self.signal = self.generate_signal(freqs=freqs, noise_amplitudes=noise_amplitudes)

    def generate_signal_2d(self, freqs_1=np.array([1]), freqs_2=np.array([1]), noise_amplitudes_1=np.array([0.1]), noise_amplitudes_2=np.array([0.1])):
        signal_1 = self.generate_signal(freqs=freqs_1, noise_amplitudes=noise_amplitudes_1)
        signal_2 = self.generate_signal(freqs=freqs_2, noise_amplitudes=noise_amplitudes_2)
        signal_2d = np.array([signal_1 for _ in range(self.N)]) + np.array([signal_2 for _ in range(self.N)]).T
        return signal_2d

    def set_signal_2d(self, freqs_1=np.array([1]), freqs_2=np.array([1]), noise_amplitudes_1=np.array([0.1]), noise_amplitudes_2=np.array([0.1])):
        self.signal_2d = self.generate_signal_2d(
            freqs_1=freqs_1,
            freqs_2=freqs_2,
            noise_amplitudes_1=noise_amplitudes_1,
            noise_amplitudes_2=noise_amplitudes_2
        )
    
    def plot_signal(self):
        if not self.signal:
            return
        else:
            plt.plot(self.times, self.signal)
            plt.show()

    def plot_signal_2d(self):
        try:
            plt.imshow(self.signal_2d)
            plt.show()
        except:
            pass

    def plot_fft(self):
        fhat = np.fft.fft(self.signal)
        psd = np.abs(fhat) / self.N
        psd = np.fft.fftshift(psd)
        plt.plot(self.freqs_shift, psd)
        plt.show()

    def plot_fft2d(self, freq_map=False):
        fhat2 = np.fft.fft2(self.signal_2d)
        psd2 = np.abs(fhat2) / self.N
        psd2 = np.fft.fftshift(psd2)
        if not freq_map:
            plt.imshow(psd2)
            plt.show()
        else:
            X, Y = np.meshgrid(self.freqs_shift, self.freqs_shift)
            plt.contourf(X, Y, psd2)
            plt.show()

    def fft_ndim(self):
        pass


from scipy import signal   

class Pulse():
    def __init__(self) -> None:
        self.pulse = signal.gausspulse()
        pass


class Chirp():
    def __init__(self, t0=0, t1=1, dt=0.001) -> None:
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.generate_time_vector()

    def generate_time_vector(self):
        self.times = np.arange(self.t0, self.t1, self.dt)
    
    def generate_signal(self, f0=1, f1=10):
        self.signal = signal.chirp(
            t=self.times,
            f0=f0,
            t1=self.t1,
            f1=f1,
            method='quadratic',
        )

    def plot_1d(self):
        plt.plot(self.times, self.signal)

    def plot_2d(self):
        mat = np.array(
            [
                self.signal for _ in range(self.times.size)
            ]
        )
        plt.imshow(mat)