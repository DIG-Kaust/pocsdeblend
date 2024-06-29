import numpy as np

from pylops.utils.backend import get_array_module
from pyproximal.ProxOperator import _check_tau
from pyproximal.projection import BoxProj, L1BallProj
from pyproximal import ProxOperator


def _hardthreshold(x, thresh):
    r"""Hard thresholding.

    """
    x1 = np.where(np.abs(x) >= thresh, x, 0)     
    return x1


def _softthreshold(x, thresh):
    r"""Soft thresholding.

    """
    if np.iscomplexobj(x):
        # https://stats.stackexchange.com/questions/357339/soft-thresholding-
        # for-the-lasso-with-complex-valued-data
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.exp(1j * np.angle(x))
    else:
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.sign(x)

    return x1


class L01Freq(ProxOperator):
    r""":math:`L_0` or :math:`L_1` norm proximal operator with axis varying threshold.

    Parameters
    ----------
    dims : :obj:`nt=y`, optional
        Dimensions of the input vector (thresholding is varying along last axis)
    sigmamax : :obj:`float`, optional
        Initial value of multiplicative coefficient of norm
    sigmamin : :obj:`np.ndarray`, optional
        Final value of multiplicative coefficient of norm (varying along last axis)
    scaling : :obj:`float`, optional
        Scaling factor applied at each iteration to sigma
    kind : :obj:`int`, optional
        0 (LO Norm) or 1 (L1 Norm)

    """
    def __init__(self, dims, sigmamax, sigmamin, scaling=0.9, kind=1):
        super().__init__(None, False)
        self.dims = dims
        self.sigmamax = sigmamax
        self.sigmamin = sigmamin
        self.scaling = scaling
        self.kind = kind
        self.count = 0

    def __call__(self, x):
        ncp = get_array_module(x)
        sigma = self._current_sigma(ncp)
        if self.kind == 0:
            return float(np.sum(np.abs(x).reshape(self.dims) > sigma))
        else:
            return float(np.sum(sigma * np.abs(x).reshape(self.dims)).real)

    def _current_sigma(self, ncp):
        sigma = (self.scaling ** (self.count + 1)) * self.sigmamax * ncp.ones(self.dims[-1])
        sigma = ncp.maximum(sigma, self.sigmamin)
        return sigma
        
    def _increment_count(func):
        """Increment counter
        """
        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        ncp = get_array_module(x)
        x = x.reshape(self.dims)
        sigma = self._current_sigma(ncp)
        if self.kind == 0:
            x = _hardthreshold(x, tau * sigma)
        else:
            x = _softthreshold(x, tau * sigma)
        return x.ravel()

