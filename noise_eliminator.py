import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import butter, filtfilt


filename = "noisy_tlou.wav"
fs, data = wavfile.read(filename)
plt.plot(data)
plt.show()


def hamm(window_size):
    N = window_size;
    output = np.zeros((N, 1));
    if np.mod(N, 2) == 0 :
        m = np.fix(N / 2)
        n = m
    else:
        m = np.fix(N / 2)+1; 
        n = m-1; 
    window = 0.54 - 0.46 * np.cos(2*np.pi*(np.arange(m)) / (N-1))
    tmp1 = window[:int(m)]
    tmp2 = window[np.arange(int(n)-1,-1,-1)]
    return np.hstack((tmp1,tmp2))

def sinc_filter_low(order, fc1, fs):
    Fc1 = fc1 / np.float(fs) 
    M  = order
    B = np.zeros((M+1, 1))
    window = hamm(M+1)
    for i in range(M+1):
        if 2 * i == M:
            B[i] = 2*np.pi*Fc1
        else:
            tmp1 = 2*np.pi*Fc1 *(i-(M/2.))
            tmp2 = (i-(M/2.))
            B[i] = np.sin(tmp1) / tmp2
        B[i] = B[i] * window[i]
    return B / np.sum(B)

def sinc_filter_high(order, fc1, fs):
    Fc1 = fc1 / np.float(fs) 
    M  = order
    B = np.zeros((M+1, 1))
    window = hamm(M+1)
    for i in range(M+1):
        if 2 * i == M:
            B[i] = 2*np.pi*Fc1
        else:
            tmp1 = 2*np.pi*Fc1 *(i-(M/2.))
            tmp2 = (i-(M/2.))
            B[i] = np.sin(tmp1) / tmp2
        B[i] = B[i] * window[i]
    B = B / np.sum(B)
    B = -B
    B[(M/2)] = B[(M/2)] + 1
    return B

def sinc_filter_band(order, fc1, fc2, fs):
    M = order
    A = sinc_filter_low(order, fc1, fs).T[0]
    B = sinc_filter_high(order, fc2, fs).T[0]
    output = A+B
    output = -output
    output[(M/2)] = output[(M/2)] + 1.
    return output

fc1 = 0.1
fc2 = 0.2
x = sinc_filter_band(data,fc1=fc1,fc2=fc2,fs=fs)

# fc = 0.1  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
# b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
# N = int(np.ceil((4 / b)))
# if not N % 2: N += 1  # Make sure that N is odd.
# n = np.arange(N)
 
# # Compute sinc filter.
# h = np.sinc(2 * fc * (n - (N - 1) / 2))
 
# # Compute Blackman window.
# w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
#     0.08 * np.cos(4 * np.pi * n / (N - 1))
 
# # Multiply sinc filter by window.
# h = h * w
 
# # Normalize to get unity gain.
# h = h / np.sum(h)

# s = np.convolve(data, h)
# wavfile.write(filename='sinc_low_pass.wav',rate=fs,data=s)
