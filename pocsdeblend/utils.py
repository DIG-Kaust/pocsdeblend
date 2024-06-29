import random
import numpy as np


def set_seed(seed):
    """Set seeds

    Set all random seeds to a fixed value and take out any randomness from cuda kernels

    Parameters
    ----------
    seed : :obj:`int`
        Seed

    """
    random.seed(seed)
    np.random.seed(seed)

    return True


def compute_fold(timings, dt, nt):
    """Fold

    Compute fold of blending data from timings

    Parameters
    ----------
    timings : :obj:`numpy.ndarray`
        Firing times
    dt : :obj:`float`
        Time sampling
    nt : :obj:`float`
        Number of time samples

    Returns
    -------
    fold : :obj:`numpy.ndarray`
        Fold as function of time

    """
    itimings = np.round(timings / dt).astype(np.int32)
    fold = np.zeros(itimings.max()+nt)
    for itiming in itimings:
        fold[itiming:itiming+nt] += 1
    return fold


def snr_time(xref, xcmp):
    """SNR as function of time error

    Compute SNR between two vectors along first axis (second and third axes are avaraged)

    Parameters
    ----------
    xref : :obj:`numpy.ndarray`
        Reference vector of size :math:`nt \times n1 \times n2`
    xcmp : :obj:`numpy.ndarray`
        Comparison vector of size :math:`nt \times n1 \times n2`

    Returns
    -------
    snr : :obj:`numpy.ndarray`
        SNR as function of time

    """
    xrefv = np.mean(np.abs(xref) ** 2, axis=(1, 2))
    snr = 10.0 * np.log10(xrefv / np.mean(np.abs(xref - xcmp) ** 2, axis=(1, 2)))
    return snr


