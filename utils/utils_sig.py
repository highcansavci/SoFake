import numpy as np
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt


def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter

    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = filtfilt(b, a, sig)
    return y

    # This function implements a Butterworth bandpass filter.
    # It reshapes the input signal sig to a 1D array.
    # Calculates the Nyquist frequency (nyq) which is half of the sampling frequency fs.
    # Computes the low and high cutoff frequencies in terms of the Nyquist frequency.
    # Uses the butter function to get filter coefficients b and a for the specified order and cutoff frequencies.
    # Applies the filter using filtfilt to filter the input signal and returns the filtered signal y.


def hr_fft(sig, fs, num_peaks=3, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD
    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()

    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    top_peak_indices = peak_idx[sort_idx[:num_peaks]]
    top_peak_frequencies = top_peak_indices / sig_f.shape[0] * fs
    top_peak_heart_rates = top_peak_frequencies * 60

    if len(top_peak_heart_rates) == 0:
        hr = 0

    elif harmonics_removal:
        if num_peaks >= 2:
            # Check if the difference between the top two heart rates is small
            if np.abs(top_peak_heart_rates[0] - 2 * top_peak_heart_rates[1]) < 10:
                hr = np.mean(top_peak_heart_rates[1:])
            else:
                hr = np.mean(top_peak_heart_rates[:2])
        else:
            hr = top_peak_heart_rates[0]
    else:
        hr = top_peak_heart_rates[0]

    x_hr = np.arange(len(sig_f)) / len(sig_f) * fs * 60
    return hr, sig_f_original, x_hr

    # This function calculates heart rate and power spectral density (PSD) using FFT.
    # It reshapes the input signal to a 1D array and applies a Hann window to the signal.
    # Computes the FFT of the signal and takes its absolute values to get the spectrum sig_f.
    # Calculates frequency indices low_idx and high_idx based on the specified frequency range.
    # Creates a copy of the original spectrum as sig_f_original.
    # Sets frequencies outside the specified range to zero in sig_f.
    # Identifies peaks in the spectrum using signal.find_peaks.
    # Sorts the peak indices in descending order.
    # Calculates heart rates (hr1 and hr2) corresponding to the top two peaks in the spectrum.
    # Checks if harmonics removal is enabled and chooses the heart rate accordingly.
    # Calculates the x-axis values (x_hr) for the heart rate plot and returns heart rate, original spectrum, and x-axis values.

    # I've added a new parameter num_peaks to the hr_fft function, which specifies how many of the top peaks you want to consider for calculating the mean heart rate. By default, it's set to 3, but you 		 can change it to any other value as needed.

    # The code now finds the top num_peaks peaks in the spectrum, calculates their corresponding heart rates, and then computes the mean heart rate based on these top peaks. If num_peaks is set to 3, it will calculate the mean heart rate of the top 3 peaks.


def normalize(x):
    return (x - x.mean()) / x.std()

    # This function takes an input array x.
    # It normalizes x by subtracting its mean and dividing by its standard deviation.
    # Returns the normalized array.

    # These functions can be used to process heart rate data, filter out unwanted frequencies, and calculate heart rate and PSD using FFT. The normalize function is a simple utility for data normalization.





