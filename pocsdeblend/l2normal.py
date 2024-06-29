import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr
from pylops import MatrixMult, Identity
from pylops.optimization.basic import lsqr
from pylops.utils.backend import get_array_module, get_module_name

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class L2Normal(ProxOperator):
    r"""L2 Norm proximal operator with normal equation.

    The Proximal operator of the :math:`\ell_2` norm is defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`. In this specific 
    implementation, users can provide :math:`\mathbf{Op}^T \mathbf{b}` and the normal
    equations will be solved in the proximal

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    OpTb : :obj:`numpy.ndarray`, optional
        Projected data vector
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    alpha : :obj:`float`, optional
        Multiplicative coefficient of dot product
    niter : :obj:`int` or :obj:`func`, optional
        Number of iterations of iterative scheme used to compute the proximal.
        This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has
        been invoked before and returns the ``niter`` to be used.
    x0 : :obj:`np.ndarray`, optional
        Initial vector
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.
    **kwargs_solver : :obj:`dict`, optional
        Dictionary containing extra arguments for
        :py:func:`scipy.sparse.linalg.lsqr` solver when using
        numpy data (or :py:func:`pylops.optimization.solver.lsqr` and
        when using cupy data)

    """
    def __init__(self, Op, OpTb, sigma=1., alpha=1.,
                 niter=10, x0=None, warm=True,
                 kwargs_solver=None):
        super().__init__(Op, True)
        self.sigma = sigma
        self.alpha = alpha
        self.niter = niter
        self.x0 = x0
        self.warm = warm
        self.count = 0
        self.kwargs_solver = {} if kwargs_solver is None else kwargs_solver

        # create data term
        self.OpTb = self.sigma * OpTb
        
    def __call__(self, x):
        # Alternatively we could compute x^T Op^T Op x + x^T Op^T b
        return (self.sigma / 2.) * (np.linalg.norm(self.Op.H @ self.Op @ x - self.OpTb) ** 2)
        
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
        # define current number of iterations
        if isinstance(self.niter, int):
            niter = self.niter
        else:
            niter = self.niter(self.count)

        # solve proximal optimization
        y = x + tau * self.OpTb
        Op1 = Identity(self.Op.shape[1], dtype=self.Op.dtype) + \
              float(tau * self.sigma) * (self.Op.H * self.Op)
        if get_module_name(get_array_module(x)) == 'numpy':
            x = sp_lsqr(Op1, y, iter_lim=niter, x0=self.x0,
                        **self.kwargs_solver)[0]
        else:
            x = lsqr(Op1, y, niter=niter, x0=self.x0,
                     **self.kwargs_solver)[0].ravel()
        if self.warm:
            self.x0 = x
        
        return x

    def grad(self, x):
        g = self.sigma * (self.Op.H @ self.Op @ x - self.OpTb)
        return g

