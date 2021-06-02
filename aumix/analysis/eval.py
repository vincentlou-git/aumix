"""
eval.py

Evaluation metrics computation.

@author: Chan Wai Lou
"""

from mir_eval.separation import *
from mir_eval.separation import _safe_db
import numpy as np
import pandas as pd


def bss_eval_df(ref, est, compute_permutation=True):
    """
    Evaluate the estimated sources against the true sources using BSS metrics
    (SDR, SIR, SAR, SI-SDR, SD-SDR).

    Parameters
    ----------
    ref : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have same shape as est)
    est : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have same shape as reef)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    df: pd.DataFrame
        BSS metric values table.
    """

    si_sdr_reorder, sd_sdr_reorder, sdr, sir, sar, perm = _bss_eval(ref,
                                                                    est,
                                                                    compute_permutation)
    df = pd.DataFrame(data={
        "SI-SDR": si_sdr_reorder,
        "SD-SDR": sd_sdr_reorder,
        "SDR": sdr,
        "SIR": sir,
        "SAR": sar,
        "Perm": perm  # Best Mean SIR Permutation
    })
    return df


def comp_stats(table, ignore_last_n_rows=3):
    _arr = table.iloc[:, :-ignore_last_n_rows].to_numpy()

    _mean = np.mean(_arr, axis=0)
    _std = np.std(_arr, axis=0)
    _pm = (np.max(_arr, axis=0) - np.min(_arr, axis=0)) / 2

    # TODO: t-test
    # stats.ttest_rel()
    return _arr, _mean, _std, _pm


def _bss_eval(ref, est, compute_permutation=True):
    """
    Evaluate the estimated sources against the true sources using BSS metrics
    (SDR, SIR, SAR, SI-SDR, SD-SDR).
    """
    sdr, sir, sar, perm = bss_eval_sources(ref,
                                           est,
                                           compute_permutation=compute_permutation)
    si_sdr, ip, sd_sdr, dp = bss_scale_sdr(ref,
                                           est,
                                           compute_permutation=compute_permutation)
    si_sdr_reorder = np.zeros(si_sdr.shape)
    si_sdr_reorder[ip] = si_sdr[perm]
    sd_sdr_reorder = np.zeros(sd_sdr.shape)
    sd_sdr_reorder[dp] = sd_sdr[perm]

    return si_sdr_reorder, sd_sdr_reorder, sdr, sir, sar, perm


def bss_scale_sdr(reference_sources, estimated_sources, compute_permutation=True):
    """
    Compute the Scale-Invariant and Scale-Dependent Signal to Distortion Ratios
    as proposed by Jonathan Le Roux et al. This structure of function is identical to
    ``mir_eval.separation.bss_eval_sources''.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have same shape as
        estimated_sources)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have same shape as
        reference_sources)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    si_sdr : np.ndarray, shape=(nsrc,)
        vector of Scale-Invariant Signal to Distortion Ratios (SI-SDR)
    si_perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SI-SDR sense. Note: ``si_perm`` will be ``[0, 1, ...,
        nsrc-1]`` if ``compute_permutation`` is ``False``.
    sd_sdr : np.ndarray, shape=(nsrc,)
        vector of Scale-Dependent Signal to Distortion Ratios (SI-SDR)
    sd_perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SD-SDR sense. Note: ``sd_perm`` will be ``[0, 1, ...,
        nsrc-1]`` if ``compute_permutation`` is ``False``.

    References
    ----------
    .. [#] Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, and John R. Hershey,
        "SDR – Half-Baked or Well Done?", ICASSP 2019 - 2019 IEEE International
        Conference on Acoustics, Speech and Signal Processing (ICASSP),
        pp. 626-630, 2019.
    """
    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = estimated_sources.shape[0]

    # does user desire permutations?
    if compute_permutation:
        # compute criteria for all possible pair matches
        si_sdr = np.empty((nsrc, nsrc))
        sd_sdr = np.empty((nsrc, nsrc))

        for jest in range(nsrc):
            for jtrue in range(nsrc):
                alpha, si_sdr[jest, jtrue] = _si_sdr(reference_sources[jest], estimated_sources[jtrue])
                sd_sdr[jest, jtrue] = snr(reference_sources[jest], estimated_sources[jtrue]) + 10 * np.log10(alpha ** 2)

        # select the best ordering
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_si_sdr = np.empty(len(perms))
        mean_sd_sdr = np.empty(len(perms))
        dum = np.arange(nsrc)
        for (i, perm) in enumerate(perms):
            mean_si_sdr[i] = np.mean(si_sdr[perm, dum])
            mean_sd_sdr[i] = np.mean(sd_sdr[perm, dum])

        si_popt = perms[np.argmax(mean_si_sdr)]
        sd_popt = perms[np.argmax(mean_sd_sdr)]
        si_idx = (si_popt, dum)
        sd_idx = (sd_popt, dum)
        return (si_sdr[si_idx], np.asarray(si_popt), sd_sdr[sd_idx], np.asarray(sd_popt))
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        
        si_sdr = np.empty(nsrc)
        sd_sdr = np.empty(nsrc)
        
        for i in range(nsrc):
            alpha, si_sdr[i] = _si_sdr(reference_sources[i], estimated_sources[i])
            sd_sdr[i] = snr(reference_sources[i], estimated_sources[i]) + 10 * np.log10(alpha**2)

        # return the default permutation for compatibility
        popt = np.arange(nsrc)
        return (si_sdr, popt, sd_sdr, popt)


def _si_sdr(ref, est):
    alpha = est @ ref / np.sum(ref ** 2)
    alpha_ref = alpha * ref
    return alpha, _safe_db( np.sum(alpha_ref ** 2), np.sum((alpha_ref - est) ** 2) )


def snr(ref, est):
    return _safe_db( np.sum(ref ** 2), np.sum((ref - est) ** 2) )
