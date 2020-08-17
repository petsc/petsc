===========================================================
Guide to the Stokes Equations using Finite Elements in PETSc
===========================================================

This guide accompanies `SNES Example 62 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex62.c.html>`__.

The Stokes equations for a fluid, a steady-state form of the Navier-Stokes equations, start with the balance of momentum, just as in elastostatics,

.. math::

    \nabla \cdot \sigma + f = 0,

where :math:`\sigma` is the stress tensor and :math:`f` is the body force, combined with the conservation of mass

.. math::

    \nabla \cdot (\rho u) = 0,

where :math:`\rho` is the density and :math:`u` is the fluid velocity. If we assume that the density is constant, making the fluid incompressible, and that the rheology is Newtonian, meaning that the viscous stress is linearly proportional to the local strain rate, then we have

.. math::

    \begin{aligned}
      \nabla \cdot \mu \left( \nabla u + \nabla u^T \right) - \nabla p + f &= 0 \\
      \nabla \cdot u &= 0
    \end{aligned}

where :math:`p` is the pressure, :math:`\mu` is the dynamic shear viscosity, with units :math:`N\cdot s/m^2` or :math:`Pa\cdot s`. If we divide by the constant density, we would have the kinematic viscosity :math:`nu` and a force per unit mass. The second equation demands that the velocity field be divergence-free, indicating that the flow is incompressible. The pressure in this case can be thought of as the Lagrange multiplier enforcing the incompressibility constraint. In the compressible case, we would need an equation of state to relate the pressure to the density, and perhaps temperature.

We will discretize our Stokes equation with finite elements, so the first step is to write a variational weak form of the equations. We choose to use a Ritz-Galerkin setup, so let our velocity :math:`u \in V` and pressure :math:`p \in Q`, so that

.. math::

    \begin{aligned}
      \left< \nabla v, \mu \left( \nabla u + \nabla u^T \right) \right> + \left< v, \frac{\partial\sigma}{\partial n} \right>_\Gamma - \left< \nabla\cdot v, p \right> - \left< v, f \right> &= 0 \\
      \left< q, -\nabla \cdot u \right> &= 0
    \end{aligned}

where integration by parts has added a boundary integral over the normal derivative of the stress (traction), and natural boundary conditions correspond to stress-free boundaries. We have multiplied the continuity equation by minus one in order to preserve symmetry.

Equation Definition
-------------------

The test functions :math:`v, q` and their derivatives are determined by the discretization, whereas the integrand on the right is determined by the physics. Given a quadrature rule to evaluate the form integral, we would only need the evaluation of the physics integrand at the quadrature points, given the values of the fields and their derivatives. The entire scheme is detailed in :cite:`KnepleyBrownRuppSmith13`. The kernels paired with test functions we will call :math:`f_0` and those paired with gradients of test functions will be called :math:`f_1`.

For example, the kernel for the continuity equation, paired with the pressure test function, is called ``f0_p`` and can be seen here

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: static void f0_p(
   :end-at: }

We use the components of the Jacobian of :math:`u` to build up its divergence. For the balance of momentum excluding body force, we test against the gradient of the test function, as seen in ``f1_u``,

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :lines: 46-61

Notice how the pressure :math:`p` is referred to using ``u[uOff[1]]`` so that we can have many fields with different numbers of components. ``DMPlex`` uses these point functions to construct the residual. A similar set of point functions is also used to build the Jacobian. The last piece of our physics specification is the construction of exact solutions using the Method of Manufactured Solutions (MMS).

MMS Solutions
-------------

An MMS solution is chosen to elucidate some property of the problem, and to check that it is being solved accurately, since the error can be calculated explicitly. For our Stokes problem, we first choose a solution with quadratic velocity and linear pressure,

.. math::

   u = \begin{pmatrix} x^2 + y^2 \\ 2 x^2 - 2 x y \end{pmatrix} \quad \mathrm{or} \quad \begin{pmatrix} 2 x^2 + y^2 + z^2 \\ 2 x^2 - 2xy \\ 2 x^2 - 2xz \end{pmatrix}

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: static PetscErrorCode quadratic_u
   :end-at: return 0;
   :append: }

.. math::

   p = x + y - 1 \quad \mathrm{or} \quad x + y + z - \frac{3}{2}

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: static PetscErrorCode quadratic_p
   :end-at: return 0;
   :append: }

By plugging these solutions into our equations, assuming that the velocity we choose is divergence-free, we can determine the body force necessary to make them satisfy the Stokes equation. For the quadratic solution above, we find

.. math::

  f = \begin{pmatrix} 1 - 4\mu \\ 1 - 4\mu \end{pmatrix} \quad \mathrm{or} \quad \begin{pmatrix} 1 - 8\mu \\ 1 - 4\mu \\ 1 - 4\mu \end{pmatrix}

which is implemented in our ``f0_u`` pointwise function

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: static void f0_quadratic_u
   :end-at: }

We let PETSc now about these solutions

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: ierr = PetscDSSetExactSolution(ds, 0
   :end-at: ierr = PetscDSSetExactSolution(ds, 1

These solutions will be captured exactly by the :math:`P_2-P_1` finite element space. We can use the ``-dmsnes_check`` option to activate function space checks. It gives the :math:`L_2` error, or *discretizaton* error, of the exact solution, the residual computed using the interpolation of the exact solution into our finite element space, and uses a Taylor test to check that our Jacobian matches the residual. It should converge at order 2, or be exact in the case of linear equations like Stokes. Our :math:`P_2-P_1` runs in the PETSc test section at the bottom of the source file

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :lines: 475-483

verify these claims, as we can see from the output files

.. literalinclude:: /../../../src/snes/tutorials/output/ex62_2d_p2_p1_check.out

.. literalinclude:: /../../../src/snes/tutorials/output/ex62_3d_p2_p1_check.out

We can carry out the same tests for the :math:`Q_2-Q_1` element,

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :lines: 510-516

The quadratic solution, however, cannot tell us whether our discretization is attaining the correct order of convergence, especially for higher order elements. Thus, we will define another solution based on trigonometric functions.

.. math::

  u = \begin{pmatrix} \sin(\pi x) + \sin(\pi y) \\ -\pi \cos(\pi x) y \end{pmatrix} \quad \mathrm{or} \quad
      \begin{pmatrix} 2 \sin(\pi x) + \sin(\pi y) + \sin(\pi z) \\ -\pi \cos(\pi x) y \\ -\pi \cos(\pi x) z \end{pmatrix}

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: static PetscErrorCode trig_u
   :end-at: return 0;
   :append: }

.. math::

  p = \sin(2 \pi x) + \sin(2 \pi y) \quad \mathrm{or} \quad \sin(2 \pi x) + \sin(2 \pi y) + \sin(2 \pi z)

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: static PetscErrorCode trig_p
   :end-at: return 0;
   :append: }

.. math::

  f = \begin{pmatrix} 2 \pi \cos(2 \pi x) + \mu \pi^2 \sin(\pi x) + \mu \pi^2 \sin(\pi y) \\ 2 \pi \cos(2 \pi y) - \mu \pi^3 \cos(\pi x) y \end{pmatrix} \quad \mathrm{or} \quad
  \begin{pmatrix} 2 \pi \cos(2 \pi x) + 2\mu \pi^2 \sin(\pi x) + \mu \pi^2 \sin(\pi y) + \mu \pi^2 \sin(\pi z) \\ 2 \pi \cos(2 \pi y) - \mu \pi^3 cos(\pi x) y \\ 2 \pi \cos(2 \pi z) - \mu \pi^3 \cos(\pi x) z \end{pmatrix}

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: static void f0_quadratic_u
   :end-at: }
   :append: }

We can now use ``-snes_convergence_estimate`` to determine the convergence exponent for the discretization. This options solves the problem on a series of refined meshes, calculates the error on each mesh, and determines the slope on a logarithmic scale. For example, we do this in two dimensions, refining our mesh twice using ``-convest_num_refine 2`` in the following test.

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :lines: 485-492

However, the test needs an accurate linear solver. Sparse LU factorizations do not, in general, do full pivoting. Thus we must deal with the zero pressure block explicitly. We use the ``PCFIELDSPLIT`` preconditioner and the full Schur complement factorization, but we still need a preconditioner for the Schur complement :math:`B^T A^{-1} B`. We can have PETSc construct that matrix automatically, but the cost rises steeply as the problem size increases. Instead, we use the fact that the Schur complement is spectrally equivalent to the pressure mass matrix :math:`M_p`. We can make a preconditioning matrix, which has the diagonal blocks we will use to build the preconditioners, letting PETSc know that we get the off-diagonal blocks from the original system with ``-pc_fieldsplit_off_diag_use_amat`` and to build the Schur complement from the original matrix using ``-pc_use_amat``,

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: ierr = PetscDSSetJacobianPreconditioner(ds, 0
   :end-at: ierr = PetscDSSetJacobianPreconditioner(ds, 1

Putting this all together, and using exact solvers on the subblocks, we have

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :lines: 491-492

and we see it converges, however it is superconverging in the pressure,

.. literalinclude:: /../../../src/snes/tutorials/output/ex62_2d_p2_p1_conv.out

If we refine the mesh using ``-dm_refine 3``, the convergence rates become ``[3.0, 2.1]``.

Dealing with Parameters
-----------------------

Like most physical problems, the Stokes problem has a parameter, the dynamic shear viscosity, which determines what solution regime we are in. To handle these parameters in PETSc, we first define a C struct to hold them,

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: typedef struct {
   :end-at: } Parameter;

and then add a ``PetscBag`` object to our application context. We then setup the parameter object,

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :start-at: static PetscErrorCode SetupParameters
   :end-at: PetscFunctionReturn(0);
   :append: }

which will allow us to set the value from the command line using ``-mu``. The ``PetscBag`` can also be persisted to disk with ``PetscBagLoad/View()``. We can make these values available as constant to our pointwise functions through the ``PetscDS`` object.

.. literalinclude:: /../../../src/snes/tutorials/ex62.c
   :language: c
   :lines: 347-354

Bibliography
------------
.. bibliography:: /../tex/petsc.bib
   :filter: docname in docnames

.. bibliography:: /../tex/petscapp.bib
   :filter: docname in docnames
