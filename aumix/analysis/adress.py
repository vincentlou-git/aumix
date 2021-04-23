"""
adress.py

ADRess (Azimuth Discrimination and Resynthesis) algorithm.

@author: DoraMemo
"""

import numpy as np
from scipy import signal


def _null2peak(fas, beta, method="invert_all"):
    """Estimates the magnitude of the null in a frequency-azimuth spectrogram."""

    null_args = np.argmin(fas, axis=-1)  # Index of the smallest freq-azi value ("null") in each frequency bin
    max_fa = np.max(fas, axis=-1)  # Max. freq-azi value in each frequency bin
    min_fa = np.min(fas, axis=-1)  # Min. freq-azi value in each frequency bin

    # Estimated peak magnitude
    azi_peak = np.zeros((fas.shape[0], beta + 1))

    if method == "invert_all":
        for k in range(fas.shape[0]):
            for i in range(beta+1):
                azi_peak[k, i] = max_fa[k] - fas[k, i]

    elif method == "invert_min_only":
        for k in range(fas.shape[0]):
            azi_peak[k, null_args[k]] = max_fa[k] - min_fa[k]

    return azi_peak


def _null2peak_fitz(fas, lfk, rfk, beta):
    """Estimates the magnitude of each null in a 2D frequency-azimuth spectrogram using FitzGerald's formulation without choosing source subspace.

    To filter out the subspace, null_args can be used to indicate the source position of each frequency bin."""

    null_args = np.argmin(fas, axis=-1)  # Dominant stereo position for each frequency
    min_fa = np.min(fas, axis=-1)  # Min. freq-azi value in each frequency bin

    # Estimated peak magnitude
    # Calculate all peaks as from the left channel first
    azi_peak = np.abs(lfk) - min_fa

    # Replace peaks where the min is in the right channel
    right_peak = np.abs(rfk) - min_fa
    right_ids = np.where(null_args <= beta//2)

    azi_peak[right_ids] = right_peak[right_ids]

    # 2D-ify the peak values
    # azi_peak = np.zeros(fas.shape)
    # azi_peak[np.arange(null_args.shape[0]), null_args] = left_peak

    return azi_peak, null_args, right_ids


def adress(left_signal,
           right_signal,
           samp_rate,
           left_ds,
           left_Hs,
           right_ds,
           right_Hs,
           beta=100,
           window="hann",
           nperseg=4096,
           noverlap=3072,
           method="invert_min_only",
           print_options=None,
           more_outputs=None):

    print_options = [] if print_options is None else print_options
    more_outputs = [] if more_outputs is None else more_outputs

    # Pack STFT parameters into a dict
    stft_params = {
        "fs": samp_rate,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap
    }

    # Perform STFT on the left and right signals
    f, t, left_stft = signal.stft(x=left_signal,
                                  **stft_params)

    _, _, right_stft = signal.stft(x=right_signal,
                                   **stft_params)

    # Source cancellation variables
    g = np.array([i / beta for i in range(beta + 1)])

    left_bounds = [(left_ds[i] - int(left_Hs[i]/2), left_ds[i] + int(left_Hs[i]/2)) for i in range(len(left_ds))]
    right_bounds = [(right_ds[i] - int(right_Hs[i]/2), right_ds[i] + int(right_Hs[i]/2)) for i in range(len(right_ds))]

    left_sep_mags = [np.zeros((f.shape[0], t.shape[0])) for _ in range(len(left_ds))]
    right_sep_mags = [np.zeros((f.shape[0], t.shape[0])) for _ in range(len(right_ds))]

    # Resultant STFT variables
    left_sep_stfts = list()
    right_sep_stfts = list()

    # Separated sources
    t_recon = None
    left_recons = list()
    right_recons = list()

    # For each time frame, compute the frequency-azimuth plane
    for tau in range(left_stft.shape[1]):
        if "progress" in print_options:
            print(f"{tau if tau % 10 == 0 else '.'}", end="")

        _, left_azi_peak, _, right_azi_peak = adress_null_peak_at_idx(tau_idx=tau,
                                                                      left_stft=left_stft,
                                                                      right_stft=right_stft,
                                                                      g=g,
                                                                      beta=beta,
                                                                      method=method)

        # Magnitude of the sum of each frequency bin of the azimuth subspace
        # of the chosen source. Equivalent to the short time power spectrum
        # of the separated source - Barry et al. 2004
        for i, (lower, upper) in enumerate(left_bounds):
            left_sep_mags[i][:, tau] = np.sum(left_azi_peak[:, lower:upper], axis=-1)

        for i, (lower, upper) in enumerate(right_bounds):
            right_sep_mags[i][:, tau] = np.sum(right_azi_peak[:, lower:upper], axis=-1)

    # Calculate the phase (unit angle? in complex number form) of the STFT
    # at each time tau
    left_phase = left_stft / np.abs(left_stft)
    right_phase = right_stft / np.abs(right_stft)

    # Combine estimated magnitude and original bin phases
    # Separated range of synthesized STFT values, at time tau
    for i in range(len(left_ds)):
        left_sep_stfts.append( left_sep_mags[i] * left_phase )

    for i in range(len(right_ds)):
        right_sep_stfts.append( right_sep_mags[i] * right_phase )

    # Finish separation on all time frames. Inverse it
    for sep_stft in left_sep_stfts:
        t_recon, x_recon = signal.istft(sep_stft,
                                        **stft_params)
        left_recons.append(x_recon)

    for sep_stft in right_sep_stfts:
        t_recon, x_recon = signal.istft(sep_stft,
                                        **stft_params)
        right_recons.append(x_recon)

    # Output
    if not more_outputs:   # no extra output is specified
        return t_recon, left_recons, right_recons,
    else:
        extra = {}

        if "stfts" in more_outputs:
            extra["stfts"] = {
                "left": left_stft,
                "right": right_stft,
                "left_recons": left_sep_stfts,
                "right_recons": right_sep_stfts,
            }

        if "stft_f" in more_outputs:
            extra["stft_f"] = f

        if "stft_t" in more_outputs:
            extra["stft_t"] = t

        return t_recon, left_recons, right_recons, extra


def adress_fitz(left_signal,
                right_signal,
                samp_rate,
                ds,
                Hs,
                beta=200,
                window="hann",
                nperseg=4096,
                noverlap=3072,
                print_options=None,
                more_outputs=None):

    print_options = [] if print_options is None else print_options
    more_outputs = [] if more_outputs is None else more_outputs

    # Pack STFT parameters into a dict
    stft_params = {
        "fs": samp_rate,
        "window": window,
        "nperseg": nperseg,
        "noverlap": noverlap
    }

    # Perform STFT on the left and right signals
    f, t, left_stft = signal.stft(x=left_signal,
                                  **stft_params)

    _, _, right_stft = signal.stft(x=right_signal,
                                   **stft_params)

    # Calculate the phase (unit angle? in complex number form) of the STFT.
    # Multiply the phase from the dominant channel.
    phase = left_stft / np.abs(left_stft)
    right_phase = right_stft / np.abs(right_stft)

    # Source cancellation variables
    g = np.array([i / beta for i in range(beta // 2)] + [(beta - i) / beta for i in range(beta // 2, beta + 1)])

    bounds = [(ds[i] - int(Hs[i]/2), ds[i] + int(Hs[i]/2)) for i in range(len(ds))]

    sep_mags = [np.zeros((f.shape[0], t.shape[0])) for _ in range(len(ds))]

    # Resultant STFT variables
    sep_stfts = list()

    # Separated sources
    t_recon = None
    recons = list()

    # For each time frame, compute the frequency-azimuth plane
    for tau in range(left_stft.shape[1]):
        if "progress" in print_options:
            print(f"{tau if tau % 10 == 0 else '.'}", end="")

        _, azi_peak, null_args, right_ids = adress_fitz_null_peak_at_idx(tau_idx=tau,
                                                                         left_stft=left_stft,
                                                                         right_stft=right_stft,
                                                                         g=g,
                                                                         beta=beta)

        # Magnitude of the dominant source in each frequency bin
        # found by the azimuth subspace - FitzGerald version 2012
        for i, (lower, upper) in enumerate(bounds):
            src_args = np.where((null_args >= lower) & (null_args <= upper))
            sep_mags[i][src_args, tau] = azi_peak[src_args]

        # Replace the phase at the current time column with right channel STFT values
        # if it's dominant
        phase[right_ids, tau] = right_phase[right_ids, tau]

    # Combine estimated magnitude and original bin phases
    # Separated range of synthesized STFT values, at time tau
    for i in range(len(ds)):
        sep_stfts.append( sep_mags[i] * phase )

    # Finish separation on all time frames. Inverse it
    for sep_stft in sep_stfts:
        t_recon, x_recon = signal.istft(sep_stft,
                                        **stft_params)
        recons.append(x_recon)

    # Output
    if not more_outputs:   # no extra output is specified
        return t_recon, recons
    else:
        extra = {}

        if "stfts" in more_outputs:
            extra["stfts"] = {
                "left": left_stft,
                "right": right_stft,
                "recons": sep_stfts,
            }

        if "stft_f" in more_outputs:
            extra["stft_f"] = f

        if "stft_t" in more_outputs:
            extra["stft_t"] = t

        return t_recon, recons, extra


def adress_null_peak_at_idx(tau_idx,
                            left_stft,
                            right_stft,
                            beta,
                            method,
                            g=None):

    # Compute g if not supplied
    g = np.array([i / beta for i in range(beta + 1)]) if g is None else g

    # Compute the nulls
    left_azi_null = np.abs(np.tile(left_stft[:, tau_idx], (beta + 1, 1)).T - g * np.tile(right_stft[:, tau_idx], (beta + 1, 1)).T)
    right_azi_null = np.abs(np.tile(right_stft[:, tau_idx], (beta + 1, 1)).T - g * np.tile(left_stft[:, tau_idx], (beta + 1, 1)).T)

    # Or, in the more readable format:
    # for i in range(beta + 1):
    #     for freq_idx in range(len(f)):
    #         left_azi_null_curr[freq_idx, i] = np.abs(left_stft[freq_idx, tau] - g[i] * right_stft[freq_idx, tau])
    #         right_azi_null_curr[freq_idx, i] = np.abs(right_stft[freq_idx, tau] - g[i] * left_stft[freq_idx, tau])

    # Estimate the magnitude of the nulls
    left_azi_peak = _null2peak(left_azi_null, beta, method=method)
    right_azi_peak = _null2peak(right_azi_null, beta, method=method)

    return left_azi_null, left_azi_peak, right_azi_null, right_azi_peak


def adress_fitz_null_peak_at_idx(tau_idx,
                                 left_stft,
                                 right_stft,
                                 beta,
                                 g=None):
    """
    Returns the null/peak 2D spectrograms, combining both channels into
    one frequency-azimuth space (does not choose sources in subspace)
    """

    # Cache common intermediate terms
    half_pos = beta//2

    # Compute g if not supplied
    g = np.array([i / beta for i in range(half_pos)] + [(beta - i) / beta for i in range(half_pos, beta + 1)]) if g is None else g

    # Compute the nulls
    azi_null = np.zeros((left_stft.shape[0], beta + 1))
    azi_null[:, :half_pos] = np.abs(np.tile(left_stft[:, tau_idx], (half_pos, 1)).T - g[:half_pos] * np.tile(right_stft[:, tau_idx], (half_pos, 1)).T)
    azi_null[:, half_pos:] = np.abs(np.tile(right_stft[:, tau_idx], (beta + 1 - half_pos, 1)).T - g[half_pos:] * np.tile(left_stft[:, tau_idx], (beta + 1 - half_pos, 1)).T)

    # Or, in the more readable format:
    # Left in stereo space
    # for i in range(int(beta/2)+1):
    #     for freq_idx in range(len(f)):
    #         azi_curr[freq_idx, i] = np.abs(left_stft[freq_idx, tau] - g[i] * right_stft[freq_idx, tau])
    #
    # # Right in stereo space
    # for i in range(int(beta/2)+1, beta+1):
    #     for freq_idx in range(len(f)):
    #         azi_curr[freq_idx, i] = np.abs(right_stft[freq_idx, tau] - g[i] * left_stft[freq_idx, tau])

    # Estimate the magnitude of the nulls
    azi_peak, null_args, right_ids = _null2peak_fitz(azi_null, left_stft[:, tau_idx], right_stft[:, tau_idx], beta)

    return azi_null, azi_peak, null_args, right_ids


def adress_null_peak_at_sec(tau,
                            t,
                            *args,
                            **kwargs):
    """
    Returns the left & right null/peak spectrograms

    Parameters
    ----------
    tau
    t
    kwargs

    Returns
    -------

    """

    tau_idx = np.where(t > tau)[0][0]

    return adress_null_peak_at_idx(tau_idx=tau_idx,
                                   *args,
                                   **kwargs)


def adress_fitz_null_peak_at_sec(tau,
                                 t,
                                 *args,
                                 **kwargs):
    """
    Returns the null/peak 2D spectrograms (does not choose sources in subspace)

    Parameters
    ----------
    tau
    t
    kwargs

    Returns
    -------

    """

    tau_idx = np.where(t > tau)[0][0]

    return adress_fitz_null_peak_at_idx(tau_idx=tau_idx,
                                        *args,
                                        **kwargs)
