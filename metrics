import math
import numpy as np
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as pesq
import torchyin
from torch import Tensor
import torch
import torchaudio
from torch.fft import fft

# include SSNR, SNR, PESQ, CSIG, CBAK, COVL
@torch.no_grad()
def hearing_perception_metrics(ref:Tensor, est:Tensor, Fs):
    # ref: ground truth waveform, Type: 1D-Tensor
    # est: estimated waveform, Type: 1D-Tensor
    alpha = 0.95
    data1, data2 = data_init(ref, est)

    # compute the WSS measure
    wss_dist_vec = wss(data1, data2, Fs)
    wss_dist_vec, _ = torch.sort(wss_dist_vec)
    wss_dist = torch.mean(wss_dist_vec[0: round((wss_dist_vec.size()[0]) * alpha)])

    # compute the LLR measure
    LLR_dist = llr(data1, data2, Fs)
    LLRs, _ = torch.sort(LLR_dist)
    LLR_len = round((LLR_dist.size()[0]) * alpha)
    llr_mean = torch.mean(LLRs[0: LLR_len])

    # compute the SNRseg
    snr_dist, segsnr_dist = snr(data1, data2, Fs)
    SNR = snr_dist
    SSNR = torch.mean(segsnr_dist)

    # compute the pesq
    PESQ = pesq(data2, data1, Fs, 'wb', True)

    # now compute the composite measures
    CSIG = 3.093 - 1.029 * llr_mean + 0.603 * PESQ - 0.009 * wss_dist
    CSIG = max(1, CSIG)
    CSIG = min(5, CSIG)  # limit values to [1, 5]
    CBAK = 1.634 + 0.478 * PESQ - 0.007 * wss_dist + 0.063 * SSNR
    CBAK = max(1, CBAK)
    CBAK = min(5, CBAK)  # limit values to [1, 5]
    COVL = 1.594 + 0.805 * PESQ - 0.512 * llr_mean - 0.007 * wss_dist
    COVL = max(1, COVL)
    COVL = min(5, COVL)  # limit values to [1, 5]

    return SNR.item(), SSNR.item(), PESQ.item(), CSIG.item(), CBAK.item(), COVL.item()


def data_init(ref: Tensor, est: Tensor):
    assert len(ref.size()) < 2, "The shape of all is not correct"

    ref = (ref - torch.mean(ref))/torch.max(torch.abs(ref))
    est = (est - torch.mean(est))/torch.max(torch.abs(est))

    if len(ref) != len(est):
        length = min(len(ref), len(est))
        ref = ref[0: length]
        est = est[0: length]

    return ref, est

def lpcoeff(speech_frame: Tensor, model_order):
    # (1) Compute Autocorrelation Lags
    device = speech_frame.device
    winlength = speech_frame.size()[0]
    R = torch.empty(model_order + 1, device=device)
    E = torch.empty(model_order + 1, device=device)
    for k in range(model_order + 1):
        R[k] = torch.dot(speech_frame[0:winlength - k], speech_frame[k: winlength])

    # (2) Levinson-Durbin
    a = torch.ones(model_order, device=device)
    a_past = torch.empty(model_order, device=device)
    rcoeff = torch.empty(model_order, device=device)
    E[0] = R[0]
    for i in range(model_order):
        a_past[0: i] = a[0: i]
        sum_term = torch.dot(a_past[0: i], R[1: i + 1].flip(0))
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i == 0:
            a[0: i] = a_past[0: i] - torch.multiply(a_past[i - 1:-1].flip(0), rcoeff[i])
        else:
            a[0: i] = a_past[0: i] - torch.multiply(a_past[0: i].flip(0), rcoeff[i])
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = R
    refcoeff = rcoeff
    lpparams = torch.concatenate((Tensor([1], device=device), -a))
    return acorr, refcoeff, lpparams

def toeplitz(c:Tensor, r:Tensor=None):
    if r is None:
        r = torch.conj(c)
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)

def wss(clean_speech: Tensor, processed_speech: Tensor, sample_rate):
    # Check the length of the clean and processed speech, which must be the same.
    device = clean_speech.device
    clean_length = clean_speech.size()[0]
    processed_length = processed_speech.size()[0]
    if clean_length != processed_length:
        raise ValueError('Files must have same length.')

    # Global variables
    winlength = int(np.round(30 * sample_rate / 1000))  # window length in samples
    skiprate = int(np.floor(np.divide(winlength, 4)))  # window skip in samples
    max_freq = int(np.divide(sample_rate, 2))  # maximum bandwidth
    num_crit = 25  # number of critical bands

    USE_FFT_SPECTRUM = 1  # defaults to 10th order LP spectrum
    n_fft = int(np.power(2, np.ceil(np.log2(2 * winlength))))
    n_fftby2 = int(np.multiply(0.5, n_fft))  # FFT size/2
    Kmax = 20.0  # value suggested by Klatt, pg 1280
    Klocmax = 1.0  # value suggested by Klatt, pg 1280

    # Critical Band Filter Definitions (Center Frequency and Bandwidths in Hz)
    cent_freq = Tensor([50.0000, 120.000, 190.000, 260.000, 330.000, 400.000, 470.000,
                          540.000, 617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30,
                          1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71,
                          2701.97, 2978.04, 3276.17, 3597.63], device=device)
    bandwidth = Tensor([70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000,
                          77.3724, 86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423,
                          153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255,
                          276.072, 298.126, 321.465, 346.136], device=device)

    bw_min = bandwidth[0]  # minimum critical bandwidth

    # Set up the critical band filters.
    # Note here that Gaussianly shaped filters are used.
    # Also, the sum of the filter weights are equivalent for each critical band filter.
    # Filter less than -30 dB and set to zero.
    min_factor = math.exp(-30.0 / (2.0 * 2.303))  # -30 dB point of filter
    crit_filter = torch.empty((num_crit, n_fftby2), device=device)
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fftby2
        bw = (bandwidth[i] / max_freq) * n_fftby2
        norm_factor = torch.log(bw_min) - torch.log(bandwidth[i])
        j = torch.arange(n_fftby2, device=device)
        crit_filter[i, :] = torch.exp(-11 * torch.square((j - torch.floor(f0)) / bw) + norm_factor)
        cond = torch.greater(crit_filter[i, :], min_factor)
        crit_filter[i, :] = torch.where(cond, crit_filter[i, :], 0)
    # For each frame of input speech, calculate the Weighted Spectral Slope Measure
    num_frames = int(clean_length / skiprate - (winlength / skiprate))  # number of frames
    start = 0  # starting sample
    window = 0.5 * (1 - torch.cos(2 * math.pi * torch.arange(1, winlength + 1, device=device) / (winlength + 1)))

    distortion = torch.empty(num_frames, device=device)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start: start + winlength] / 32768
        processed_frame = processed_speech[start: start + winlength] / 32768
        clean_frame = torch.multiply(clean_frame, window)
        processed_frame = torch.multiply(processed_frame, window)
        # (2) Compute the Power Spectrum of Clean and Processed
        # if USE_FFT_SPECTRUM:
        clean_spec = torch.square(torch.abs(fft(clean_frame, n_fft)))
        processed_spec = torch.square(torch.abs(fft(processed_frame, n_fft)))

        # (3) Compute Filterbank Output Energies (in dB scale)
        clean_energy = torch.matmul(crit_filter, clean_spec[0:n_fftby2])
        processed_energy = torch.matmul(crit_filter, processed_spec[0:n_fftby2])

        EPS = Tensor([1e-10], device=device)
        clean_energy = 10 * torch.log10(torch.maximum(clean_energy, EPS))
        processed_energy = 10 * torch.log10(torch.maximum(processed_energy, EPS))

        # (4) Compute Spectral Slope (dB[i+1]-dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[0: num_crit - 1]
        processed_slope = processed_energy[1:num_crit] - processed_energy[0: num_crit - 1]

        # (5) Find the nearest peak locations in the spectra to each critical band.
        #     If the slope is negative, we search to the left. If positive, we search to the right.
        clean_loc_peak = torch.empty(num_crit - 1, device=device)
        processed_loc_peak = torch.empty(num_crit - 1, device=device)

        for i in range(num_crit - 1):
            # find the peaks in the clean speech signal
            if clean_slope[i] > 0:  # search to the right
                n = i
                while (n < num_crit - 1) and (clean_slope[n] > 0):
                    n = n + 1
                clean_loc_peak[i] = clean_energy[n - 1]
            else:  # search to the left
                n = i
                while (n >= 0) and (clean_slope[n] <= 0):
                    n = n - 1
                clean_loc_peak[i] = clean_energy[n + 1]

            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:  # search to the right
                n = i
                while (n < num_crit - 1) and (processed_slope[n] > 0):
                    n = n + 1
                processed_loc_peak[i] = processed_energy[n - 1]
            else:  # search to the left
                n = i
                while (n >= 0) and (processed_slope[n] <= 0):
                    n = n - 1
                processed_loc_peak[i] = processed_energy[n + 1]

        # (6) Compute the WSS Measure for this frame. This includes determination of the weighting function.
        dBMax_clean = torch.max(clean_energy)
        dBMax_processed = torch.max(processed_energy)
        '''
        The weights are calculated by averaging individual weighting factors from the clean and processed frame.
        These weights W_clean and W_processed should range from 0 to 1 and place more emphasis on spectral peaks
        and less emphasis on slope differences in spectral valleys.
        This procedure is described on page 1280 of Klatt's 1982 ICASSP paper.
        '''
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[0: num_crit - 1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - clean_energy[0: num_crit - 1])
        W_clean = torch.multiply(Wmax_clean, Wlocmax_clean)

        Wmax_processed = Kmax / (Kmax + dBMax_processed - processed_energy[0: num_crit - 1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - processed_energy[0: num_crit - 1])
        W_processed = torch.multiply(Wmax_processed, Wlocmax_processed)

        W = (W_clean + W_processed) / 2.0
        slope_diff = (clean_slope - processed_slope)[0: num_crit - 1]
        distortion[frame_count] = torch.dot(W, torch.square(slope_diff)) / torch.sum(W)
        # this normalization is not part of Klatt's paper, but helps to normalize the measure.
        # Here we scale the measure by the sum of the weights.
        start = start + skiprate
    return distortion



def llr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech.  Must be the same.
    device = clean_speech.device
    clean_length = clean_speech.size()[0]
    processed_length = processed_speech.size()[0]
    if clean_length != processed_length:
        raise ValueError('Files must have same length.')

    # Global Variables
    winlength = int(np.round(30 * sample_rate / 1000))  # window length in samples
    skiprate = int(np.floor(winlength / 4))  # window skip in samples
    if sample_rate < 10000:
        P = 10  # LPC Analysis Order
    else:
        P = 16  # this could vary depending on sampling frequency.

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int((clean_length - winlength) / skiprate)  # number of frames
    start = 0  # starting sample
    window = 0.5 * (1 - torch.cos(2 * math.pi * torch.arange(1, winlength + 1, device=device) / (winlength + 1)))

    distortion = torch.empty(num_frames, device=device)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start: start + winlength]
        processed_frame = processed_speech[start: start + winlength]
        clean_frame = torch.multiply(clean_frame, window)
        processed_frame = torch.multiply(processed_frame, window)

        # (2) Get the autocorrelation lags and LPC parameters used to compute the LLR measure.
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)

        # (3) Compute the LLR measure
        numerator = torch.dot(torch.matmul(A_processed, toeplitz(R_clean)), A_processed)
        denominator = torch.dot(torch.matmul(A_clean, toeplitz(R_clean)), A_clean)
        distortion[frame_count] = torch.log(numerator / denominator)
        start = start + skiprate
    return distortion



def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    device = clean_speech.device
    clean_length = clean_speech.size()[0]
    processed_length = processed_speech.size()[0]
    if clean_length != processed_length:
        raise ValueError('Files must have same length.')

    overall_snr = 10 * torch.log10(torch.sum(torch.square(clean_speech)) / torch.sum(torch.square(clean_speech - processed_speech)))

    # Global Variables
    winlength = round(30 * sample_rate / 1000)  # window length in samples
    skiprate = math.floor(winlength / 4)  # window skip in samples
    MIN_SNR = Tensor([-10], device=device)  # minimum SNR in dB
    MAX_SNR = Tensor([35], device=device)  # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(clean_length / skiprate - (winlength / skiprate))  # number of frames
    start = 0  # starting sample
    window = 0.5 * (1 - torch.cos(2 * math.pi * torch.arange(1, winlength + 1, device=device) / (winlength + 1)))

    segmental_snr = torch.empty(num_frames, device=device)
    EPS = Tensor([np.finfo(np.float64).eps], device=device)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = torch.multiply(clean_frame, window)
        processed_frame = torch.multiply(processed_frame, window)

        # (2) Compute the Segmental SNR
        signal_energy = torch.sum(torch.square(clean_frame))
        noise_energy = torch.sum(torch.square(clean_frame - processed_frame))
        segmental_snr[frame_count] = 10 * torch.log10(signal_energy / (noise_energy + EPS) + EPS)
        segmental_snr[frame_count] = torch.maximum(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = torch.minimum(segmental_snr[frame_count], MAX_SNR)

        start = start + skiprate

    return overall_snr, segmental_snr

if __name__ == '__main__':
    x = torch.randn(16000)
    y = torch.randn(16000)
    print(hearing_perception_metrics(x,y,16000))

