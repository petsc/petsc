.. _chapter_kspdm:

High Level Support for Multigrid with KSPSetDM() and SNESSetDM()
----------------------------------------------------------------

This chapter needs to be written. For now, see the manual pages (and
linked examples) for ``KSPSetDM()`` and ``SNESSetDM()``.

Smoothing on each level of the hierarchy is handled by a ``KSP`` held by the ``PCMG``, or in the nonlinear case, a ``SNES`` held by ``SNESFAS``. The ``DM`` for each level is associated with the smoother using ``KSPSetDM()`` and ``SNESSetDM()``.

The linear operators which carry out interpolation and restriction (usually of type ``MATMAIJ``) are held by the ``PCMG``/``SNESFAS``, and generated automatically by the ``DM`` using information about the discretization. Below we briefly discuss the different operations:

**Interpolation** transfers a function from the coarse space to the fine space. We would like this process to be accurate for the functions resolved by the coarse grid, in particular the approximate solution computed there. By default, we create these matrices using local interpolation of the fine grid dual basis functions in the coarse basis. However, an adaptive procedure can optimize the coefficients of the interpolator to reproduce pairs of coarse/fine functions which should approximate the lowest modes of the generalized eigenproblem

.. math::

  A x = \lambda M x

where :math:`A` is the system matrix and :math:`M` is the smoother. Note that for defect-correction MG, the interpolated solution from the coarse space need not be as accurate as the fine solution, for the same reason that updates in iterative refinement can be less accurate. However, in FAS or in the final interpolation step for each level of Full Multigrid, we must have interpolation as accurate as the fine solution since we are moving the entire solution itself.

**Injection** should accurately transfer the fine solution to the coarse grid. Accuracy here means that the action of a coarse dual function on either should produce approximately the same result. In the structured grid case, this means that we just use the same values on coarse points. This can result in aliasing.

**Restriction** is intended to transfer the fine residual to the coarse space. Here we use averaging (often the transpose of the interpolation operation) to damp out the fine space contributions. Thus, it is less accurate than injection, but avoids aliasing of the high modes.

Adaptive Interpolation
``````````````````````

For a multigrid cycle, the interpolator :math:`P` is intended to accurately reproduce "smooth" functions from the coarse space in the fine space, keeping the energy of the interpolant about the same. For the Laplacian on a structured mesh, it is easy to determine what these low-frequency functions are. They are the Fourier modes. However an arbitrary operator :math:`A` will have different coarse modes that we want to resolve accurately on the fine grid, so that our coarse solve produces a good guess for the fine problem. How do we make sure that our interpolator :math:`P` can do this?

We first must decide what we mean by accurate interpolation of some functions. Suppose we know the continuum function :math:`f` that we care about, and we are only interested in a finite element description of discrete functions. Then the coarse function representing :math:`f` is given by

.. math::

  f^C = \sum_i f^C_i \phi^C_i,

and similarly the fine grid form is

.. math::

  f^F = \sum_i f^F_i \phi^F_i.

Now we would like the interpolant of the coarse representer to the fine grid to be as close as possible to the fine representer in a least squares sense, meaning we want to solve the minimization problem

.. math::

  \min_{P} \| f^F - P f^C \|_2

Now we can express :math:`P` as a matrix by looking at the matrix elements :math:`P_{ij} = \phi^F_i P \phi^C_j`. Then we have

.. math::

  \begin{aligned}
    &\phi^F_i f^F - \phi^F_i P f^C \\
  = &f^F_i - \sum_j P_{ij} f^C_j
  \end{aligned}

so that our discrete optimization problem is

.. math::

  \min_{P_{ij}} \| f^F_i - \sum_j P_{ij} f^C_j \|_2

and we will treat each row of the interpolator as a separate optimization problem. We could allow an arbitrary sparsity pattern, or try to determine adaptively, as is done in sparse approximate inverse preconditioning. However, we know the supports of the basis functions in finite elements, and thus the naive sparsity pattern from local interpolation can be used.

We note here that the BAMG framework of Brannick, et. al. :cite:`BrandtBrannickKahlLivshits2011` does not use fine and coarse functions spaces, but rather a fine point/coarse point division which we will not employ here. Our general PETSc routine should work for both since the input would be the checking set (fine basis coefficients or fine space points) and the approximation set (coarse basis coefficients in the support or coarse points in the sparsity pattern).

We can easily solve the above problem using QR factorization. However, there are many smooth functions from the coarse space that we want interpolated accurately, and a single :math:`f` would not constrain the values :math:`P_{ij}`` well. Therefore, we will use several functions :math:`\{f_k\}` in our minimization,

.. math::

  \begin{aligned}
    &\min_{P_{ij}} \sum_k w_k \| f^{F,k}_i - \sum_j P_{ij} f^{C,k}_j \|_2 \\
  = &\min_{P_{ij}} \sum_k \| \sqrt{w_k} f^{F,k}_i - \sqrt{w_k} \sum_j P_{ij} f^{C,k}_j \|_2 \\
  = &\min_{P_{ij}} \| W^{1/2} \mathbf{f}^{F}_i - W^{1/2} \mathbf{f}^{C} p_i \|_2
  \end{aligned}

where

.. math::

  \begin{aligned}
  W         &= \begin{pmatrix} w_0 & & \\ & \ddots & \\ & & w_K \end{pmatrix} \\
  \mathbf{f}^{F}_i &= \begin{pmatrix} f^{F,0}_i \\ \vdots \\ f^{F,K}_i \end{pmatrix} \\
  \mathbf{f}^{C}   &= \begin{pmatrix} f^{C,0}_0 & \cdots & f^{C,0}_n \\ \vdots & \ddots &  \vdots \\ f^{C,K}_0 & \cdots & f^{C,K}_n \end{pmatrix} \\
  p_i       &= \begin{pmatrix} P_{i0} \\ \vdots \\ P_{in} \end{pmatrix}
  \end{aligned}

or alternatively

.. math::

  \begin{aligned}
  [W]_{kk}     &= w_k \\
  [f^{F}_i]_k  &= f^{F,k}_i \\
  [f^{C}]_{kj} &= f^{C,k}_j \\
  [p_i]_j      &= P_{ij}
  \end{aligned}

We thus have a standard least-squares problem

.. math::

  \min_{P_{ij}} \| b - A x \|_2

where

.. math::

  \begin{aligned}
  A &= W^{1/2} f^{C} \\
  b &= W^{1/2} f^{F}_i \\
  x &= p_i
  \end{aligned}

which can be solved using LAPACK.

We will typically perform this optimization on a multigrid level :math:`l` when the change in eigenvalue from level :math:`l+1` is relatively large, meaning

.. math::

  \frac{|\lambda_l - \lambda_{l+1}|}{|\lambda_l|}.

This indicates that the generalized eigenvector associated with that eigenvalue was not adequately represented by :math:`P^l_{l+1}``, and the interpolator should be recomputed.

.. raw:: html

    <hr>

.. bibliography:: /petsc.bib
   :filter: docname in docnames
