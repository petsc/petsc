.. _chapter_snes:

SNES: Nonlinear Solvers
-----------------------

.. note::

  This chapter is being cleaned up by Jed Brown.  Contributions are welcome.

The solution of large-scale nonlinear problems pervades many facets of
computational science and demands robust and flexible solution
strategies. The ``SNES`` library of PETSc provides a powerful suite of
data-structure-neutral numerical routines for such problems. Built on
top of the linear solvers and data structures discussed in preceding
chapters, ``SNES`` enables the user to easily customize the nonlinear
solvers according to the application at hand. Also, the ``SNES``
interface is *identical* for the uniprocess and parallel cases; the only
difference in the parallel version is that each process typically forms
only its local contribution to various matrices and vectors.

The ``SNES`` class includes methods for solving systems of nonlinear
equations of the form

.. math::
   :label: fx0

   \mathbf{F}(\mathbf{x}) = 0,

where :math:`\mathbf{F}: \, \Re^n \to \Re^n`. Newton-like methods provide the
core of the package, including both line search and trust region
techniques. A suite of nonlinear Krylov methods and methods based upon
problem decomposition are also included. The solvers are discussed
further in :any:`sec_nlsolvers`. Following the PETSc design
philosophy, the interfaces to the various solvers are all virtually
identical. In addition, the ``SNES`` software is completely flexible, so
that the user can at runtime change any facet of the solution process.

PETSc’s default method for solving the nonlinear equation is Newton’s
method. The general form of the :math:`n`-dimensional Newton’s method
for solving :math:numref:`fx0` is

.. math::
   :label: newton

   \mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{J}(\mathbf{x}_k)^{-1} \mathbf{F}(\mathbf{x}_k), \;\; k=0,1, \ldots,

where :math:`\mathbf{x}_0` is an initial approximation to the solution and
:math:`\mathbf{J}(\mathbf{x}_k) = \mathbf{F}'(\mathbf{x}_k)`, the Jacobian, is nonsingular at each
iteration. In practice, the Newton iteration :math:numref:`newton` is
implemented by the following two steps:

.. math::

   \begin{aligned}
   1. & \text{(Approximately) solve} & \mathbf{J}(\mathbf{x}_k) \Delta \mathbf{x}_k &= -\mathbf{F}(\mathbf{x}_k). \\
   2. & \text{Update} & \mathbf{x}_{k+1} &\gets \mathbf{x}_k + \Delta \mathbf{x}_k.
   \end{aligned}

Other defect-correction algorithms can be implemented by using different
choices for :math:`J(\mathbf{x}_k)`.

.. _sec_snesusage:

Basic SNES Usage
~~~~~~~~~~~~~~~~

In the simplest usage of the nonlinear solvers, the user must merely
provide a C, C++, or Fortran routine to evaluate the nonlinear function
:math:numref:`fx0`. The corresponding Jacobian matrix
can be approximated with finite differences. For codes that are
typically more efficient and accurate, the user can provide a routine to
compute the Jacobian; details regarding these application-provided
routines are discussed below. To provide an overview of the use of the
nonlinear solvers, browse the concrete example in `ex1.c <#snes-ex1>`_ or skip ahead to the discussion.

.. admonition:: Listing: ``src/snes/tutorials/ex1.c``
   :name: snes-ex1

   .. literalinclude:: ../../../snes/tutorials/ex1.c
      :end-before: /*TEST

To create a ``SNES`` solver, one must first call ``SNESCreate()`` as
follows:

::

   SNESCreate(MPI_Comm comm,SNES *snes);

The user must then set routines for evaluating the residual function :math:numref:`fx0` and its associated Jacobian matrix, as
discussed in the following sections.

To choose a nonlinear solution method, the user can either call

::

   SNESSetType(SNES snes,SNESType method);

or use the option ``-snes_type <method>``, where details regarding the
available methods are presented in :any:`sec_nlsolvers`. The
application code can take complete control of the linear and nonlinear
techniques used in the Newton-like method by calling

::

   SNESSetFromOptions(snes);

This routine provides an interface to the PETSc options database, so
that at runtime the user can select a particular nonlinear solver, set
various parameters and customized routines (e.g., specialized line
search variants), prescribe the convergence tolerance, and set
monitoring routines. With this routine the user can also control all
linear solver options in the ``KSP``, and ``PC`` modules, as discussed
in :any:`chapter_ksp`.

After having set these routines and options, the user solves the problem
by calling

::

   SNESSolve(SNES snes,Vec b,Vec x);

where ``x`` should be initialized to the initial guess before calling and contains the solution on return.
In particular, to employ an initial guess of
zero, the user should explicitly set this vector to zero by calling
``VecZeroEntries(x)``. Finally, after solving the nonlinear system (or several
systems), the user should destroy the ``SNES`` context with

::

   SNESDestroy(SNES *snes);

.. _sec_snesfunction:

Nonlinear Function Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When solving a system of nonlinear equations, the user must provide a
a residual function :math:numref:`fx0`, which is set using

::

   SNESSetFunction(SNES snes,Vec f,PetscErrorCode (*FormFunction)(SNES snes,Vec x,Vec f,void *ctx),void *ctx);

The argument ``f`` is an optional vector for storing the solution; pass ``NULL`` to have the ``SNES`` allocate it for you.
The argument ``ctx`` is an optional user-defined context, which can
store any private, application-specific data required by the function
evaluation routine; ``NULL`` should be used if such information is not
needed. In C and C++, a user-defined context is merely a structure in
which various objects can be stashed; in Fortran a user context can be
an integer array that contains both parameters and pointers to PETSc
objects.
`SNES Tutorial ex5 <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex5.c.html>`__
and
`SNES Tutorial ex5f <https://www.mcs.anl.gov/petsc/petsc-current/src/snes/tutorials/ex5f.F90.html>`__
give examples of user-defined application contexts in C and Fortran,
respectively.

.. _sec_snesjacobian:

Jacobian Evaluation
^^^^^^^^^^^^^^^^^^^

The user must also specify a routine to form some approximation of the
Jacobian matrix, ``A``, at the current iterate, ``x``, as is typically
done with

::

   SNESSetJacobian(SNES snes,Mat Amat,Mat Pmat,PetscErrorCode (*FormJacobian)(SNES snes,Vec x,Mat A,Mat B,void *ctx),void *ctx);

The arguments of the routine ``FormJacobian()`` are the current iterate,
``x``; the (approximate) Jacobian matrix, ``Amat``; the matrix from
which the preconditioner is constructed, ``Pmat`` (which is usually the
same as ``Amat``); and an optional user-defined Jacobian context,
``ctx``, for application-specific data. Note that the ``SNES`` solvers
are all data-structure neutral, so the full range of PETSc matrix
formats (including “matrix-free” methods) can be used.
:any:`chapter_matrices` discusses information regarding
available matrix formats and options, while :any:`sec_nlmatrixfree` focuses on matrix-free methods in
``SNES``. We briefly touch on a few details of matrix usage that are
particularly important for efficient use of the nonlinear solvers.

A common usage paradigm is to assemble the problem Jacobian in the
preconditioner storage ``B``, rather than ``A``. In the case where they
are identical, as in many simulations, this makes no difference.
However, it allows us to check the analytic Jacobian we construct in
``FormJacobian()`` by passing the ``-snes_mf_operator`` flag. This
causes PETSc to approximate the Jacobian using finite differencing of
the function evaluation (discussed in :any:`sec_fdmatrix`),
and the analytic Jacobian becomes merely the preconditioner. Even if the
analytic Jacobian is incorrect, it is likely that the finite difference
approximation will converge, and thus this is an excellent method to
verify the analytic Jacobian. Moreover, if the analytic Jacobian is
incomplete (some terms are missing or approximate),
``-snes_mf_operator`` may be used to obtain the exact solution, where
the Jacobian approximation has been transferred to the preconditioner.

One such approximate Jacobian comes from “Picard linearization” which
writes the nonlinear system as

.. math:: \mathbf{F}(\mathbf{x}) := \mathbf{A}(\mathbf{x}) \mathbf{x} - \mathbf{b} = 0

where :math:`\mathbf{A}(\mathbf{x})` usually contains the lower-derivative parts of the
equation. For example, the nonlinear diffusion problem

.. math:: - \nabla\cdot(\kappa(u) \nabla u) = 0

would be linearized as

.. math:: A(u) v \simeq -\nabla\cdot(\kappa(u) \nabla v).

Usually this linearization is simpler to implement than Newton and the
linear problems are somewhat easier to solve. In addition to using
``-snes_mf_operator`` with this approximation to the Jacobian, the
Picard iterative procedure can be performed by defining :math:`\mathbf{J}(\mathbf{x})`
to be :math:`\mathbf{A}(\mathbf{x})`. Sometimes this iteration exhibits better global
convergence than Newton linearization.

During successive calls to ``FormJacobian()``, the user can either
insert new matrix contexts or reuse old ones, depending on the
application requirements. For many sparse matrix formats, reusing the
old space (and merely changing the matrix elements) is more efficient;
however, if the matrix structure completely changes, creating an
entirely new matrix context may be preferable. Upon subsequent calls to
the ``FormJacobian()`` routine, the user may wish to reinitialize the
matrix entries to zero by calling ``MatZeroEntries()``. See
:any:`sec_othermat` for details on the reuse of the matrix
context.

The directory ``${PETSC_DIR}/src/snes/tutorials`` provides a variety of
examples.

.. _sec_nlsolvers:

The Nonlinear Solvers
~~~~~~~~~~~~~~~~~~~~~

As summarized in Table :any:`tab-snesdefaults`, ``SNES`` includes
several Newton-like nonlinear solvers based on line search techniques
and trust region methods. Also provided are several nonlinear Krylov
methods, as well as nonlinear methods involving decompositions of the
problem.

Each solver may have associated with it a set of options, which can be
set with routines and options database commands provided for this
purpose. A complete list can be found by consulting the manual pages or
by running a program with the ``-help`` option; we discuss just a few in
the sections below.

.. list-table:: PETSc Nonlinear Solvers
   :name: tab-snesdefaults
   :header-rows: 1

   * - Method
     - SNESType
     - Options Name
     - Default Line Search
   * - Line Search Newton
     - ``SNESNEWTONLS``
     - ``newtonls``
     - ``SNESLINESEARCHBT``
   * - Trust region Newton
     - ``SNESNEWTONTR``
     - ``newtontr``
     - —
   * - Nonlinear Richardson
     - ``SNESNRICHARDSON``
     - ``nrichardson``
     - ``SNESLINESEARCHL2``
   * - Nonlinear CG
     - ``SNESNCG``
     - ``ncg``
     - ``SNESLINESEARCHCP``
   * - Nonlinear GMRES
     - ``SNESNGMRES``
     - ``ngmres``
     - ``SNESLINESEARCHL2``
   * - Quasi-Newton
     - ``SNESQN``
     - ``qn``
     - see :any:`tab-qndefaults`
   * - Full Approximation Scheme
     - ``SNESFAS``
     - ``fas``
     - —
   * - Nonlinear ASM
     - ``SNESNASM``
     - ``nasm``
     - –
   * - ASPIN
     - ``SNESASPIN``
     - ``aspin``
     - ``SNESLINESEARCHBT``
   * - Nonlinear Gauss-Seidel
     - ``SNESNGS``
     - ``ngs``
     - –
   * - Anderson Mixing
     - ``SNESANDERSON``
     - ``anderson``
     - –
   * -  Newton with constraints (1)
     - ``SNESVINEWTONRSLS``
     - ``vinewtonrsls``
     - ``SNESLINESEARCHBT``
   * -  Newton with constraints (2)
     - ``SNESVINEWTONSSLS``
     - ``vinewtonssls``
     - ``SNESLINESEARCHBT``
   * - Multi-stage Smoothers
     - ``SNESMS``
     - ``ms``
     - –
   * - Composite
     - ``SNESCOMPOSITE``
     - ``composite``
     - –
   * - Linear solve only
     - ``SNESKSPONLY``
     - ``ksponly``
     - –
   * - Python Shell
     - ``SNESPYTHON``
     - ``python``
     - –
   * - Shell (user-defined)
     - ``SNESSHELL``
     - ``shell``
     - –


Line Search Newton
^^^^^^^^^^^^^^^^^^

The method ``SNESNEWTONLS`` (``-snes_type newtonls``) provides a
line search Newton method for solving systems of nonlinear equations. By
default, this technique employs cubic backtracking
:cite:`dennis:83`. Alternative line search techniques are
listed in Table :any:`tab-linesearches`.

.. table:: PETSc Line Search Methods
   :name: tab-linesearches

   ==================== ======================= ================
   **Line Search**      **SNESLineSearchType**  **Options Name**
   ==================== ======================= ================
   Backtracking         ``SNESLINESEARCHBT``    ``bt``
   (damped) step        ``SNESLINESEARCHBASIC`` ``basic``
   L2-norm Minimization ``SNESLINESEARCHL2``    ``l2``
   Critical point       ``SNESLINESEARCHCP``    ``cp``
   Shell                ``SNESLINESEARCHSHELL`` ``shell``
   ==================== ======================= ================

Every ``SNES`` has a line search context of type ``SNESLineSearch`` that
may be retrieved using

::

   SNESGetLineSearch(SNES snes,SNESLineSearch *ls);.

There are several default options for the line searches. The order of
polynomial approximation may be set with ``-snes_linesearch_order`` or

::

   SNESLineSearchSetOrder(SNESLineSearch ls, PetscInt order);

for instance, 2 for quadratic or 3 for cubic. Sometimes, it may not be
necessary to monitor the progress of the nonlinear iteration. In this
case, ``-snes_linesearch_norms`` or

::

   SNESLineSearchSetComputeNorms(SNESLineSearch ls,PetscBool norms);

may be used to turn off function, step, and solution norm computation at
the end of the linesearch.

The default line search for the line search Newton method,
``SNESLINESEARCHBT`` involves several parameters, which are set to
defaults that are reasonable for many applications. The user can
override the defaults by using the following options:

* ``-snes_linesearch_alpha <alpha>``
* ``-snes_linesearch_maxstep <max>``
* ``-snes_linesearch_minlambda <tol>``

Besides the backtracking linesearch, there are ``SNESLINESEARCHL2``,
which uses a polynomial secant minimization of :math:`||F(x)||_2`, and
``SNESLINESEARCHCP``, which minimizes :math:`F(x) \cdot Y` where
:math:`Y` is the search direction. These are both potentially iterative
line searches, which may be used to find a better-fitted steplength in
the case where a single secant search is not sufficient. The number of
iterations may be set with ``-snes_linesearch_max_it``. In addition, the
convergence criteria of the iterative line searches may be set using
function tolerances ``-snes_linesearch_rtol`` and
``-snes_linesearch_atol``, and steplength tolerance
``snes_linesearch_ltol``.

Custom line search types may either be defined using
``SNESLineSearchShell``, or by creating a custom user line search type
in the model of the preexisting ones and register it using

::

   SNESLineSearchRegister(const char sname[],PetscErrorCode (*function)(SNESLineSearch));.

Trust Region Methods
^^^^^^^^^^^^^^^^^^^^

The trust region method in ``SNES`` for solving systems of nonlinear
equations, ``SNESNEWTONTR`` (``-snes_type newtontr``), is taken from the
MINPACK project :cite:`more84`. Several parameters can be
set to control the variation of the trust region size during the
solution process. In particular, the user can control the initial trust
region radius, computed by

.. math:: \Delta = \Delta_0 \| F_0 \|_2,

by setting :math:`\Delta_0` via the option ``-snes_tr_delta0 <delta0>``.

Nonlinear Krylov Methods
^^^^^^^^^^^^^^^^^^^^^^^^

A number of nonlinear Krylov methods are provided, including Nonlinear
Richardson, conjugate gradient, GMRES, and Anderson Mixing. These
methods are described individually below. They are all instrumental to
PETSc’s nonlinear preconditioning.

**Nonlinear Richardson.** The nonlinear Richardson iteration merely
takes the form of a line search-damped fixed-point iteration of the form

.. math::

   \mathbf{x}_{k+1} = \mathbf{x}_k - \lambda \mathbf{F}(\mathbf{x}_k), \;\; k=0,1, \ldots,

where the default linesearch is ``SNESLINESEARCHL2``. This simple solver
is mostly useful as a nonlinear smoother, or to provide line search
stabilization to an inner method.

**Nonlinear Conjugate Gradients.** Nonlinear CG is equivalent to linear
CG, but with the steplength determined by line search
(``SNESLINESEARCHCP`` by default). Five variants (Fletcher-Reed,
Hestenes-Steifel, Polak-Ribiere-Polyak, Dai-Yuan, and Conjugate Descent)
are implemented in PETSc and may be chosen using

::

   SNESNCGSetType(SNES snes, SNESNCGType btype);

**Anderson Mixing and Nonlinear GMRES Methods.** Nonlinear GMRES and
Anderson Mixing methods combine the last :math:`m` iterates, plus a new
fixed-point iteration iterate, into a residual-minimizing new iterate.

Quasi-Newton Methods
^^^^^^^^^^^^^^^^^^^^

Quasi-Newton methods store iterative rank-one updates to the Jacobian
instead of computing it directly. Three limited-memory quasi-Newton
methods are provided, L-BFGS, which are described in
Table :any:`tab-qndefaults`. These all are encapsulated under
``-snes_type qn`` and may be changed with ``snes_qn_type``. The default
is L-BFGS, which provides symmetric updates to an approximate Jacobian.
This iteration is similar to the line search Newton methods.

.. list-table:: PETSc quasi-Newton solvers
   :name: tab-qndefaults
   :header-rows: 1

   * - QN Method
     - ``SNESQNType``
     - Options Name
     - Default Line Search
   * - L-BFGS
     - ``SNES_QN_LBFGS``
     - ``lbfgs``
     - ``SNESLINESEARCHCP``
   * - “Good” Broyden
     - ``SNES_QN_BROYDEN``
     - ``broyden``
     - ``SNESLINESEARCHBASIC``
   * - “Bad” Broyden
     - ``SNES_QN_BADBROYEN``
     - ``badbroyden``
     - ``SNESLINESEARCHL2``

One may also control the form of the initial Jacobian approximation with

::

   SNESQNSetScaleType(SNES snes, SNESQNScaleType stype);

and the restart type with

::

   SNESQNSetRestartType(SNES snes, SNESQNRestartType rtype);

The Full Approximation Scheme
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Full Approximation Scheme is a nonlinear multigrid correction. At
each level, there is a recursive cycle control ``SNES`` instance, and
either one or two nonlinear solvers as smoothers (up and down). Problems
set up using the ``SNES`` ``DMDA`` interface are automatically
coarsened. FAS differs slightly from ``PCMG``, in that the hierarchy is
constructed recursively. However, much of the interface is a one-to-one
map. We describe the “get” operations here, and it can be assumed that
each has a corresponding “set” operation. For instance, the number of
levels in the hierarchy may be retrieved using

::

   SNESFASGetLevels(SNES snes, PetscInt *levels);

There are four ``SNESFAS`` cycle types, ``SNES_FAS_MULTIPLICATIVE``,
``SNES_FAS_ADDITIVE``, ``SNES_FAS_FULL``, and ``SNES_FAS_KASKADE``. The
type may be set with

::

   SNESFASSetType(SNES snes,SNESFASType fastype);.

and the cycle type, 1 for V, 2 for W, may be set with

::

   SNESFASSetCycles(SNES snes, PetscInt cycles);.

Much like the interface to ``PCMG`` described in :any:`sec_mg`, there are interfaces to recover the
various levels’ cycles and smoothers. The level smoothers may be
accessed with

::

   SNESFASGetSmoother(SNES snes, PetscInt level, SNES *smooth);
   SNESFASGetSmootherUp(SNES snes, PetscInt level, SNES *smooth);
   SNESFASGetSmootherDown(SNES snes, PetscInt level, SNES *smooth);

and the level cycles with

::

   SNESFASGetCycleSNES(SNES snes,PetscInt level,SNES *lsnes);.

Also akin to ``PCMG``, the restriction and prolongation at a level may
be acquired with

::

   SNESFASGetInterpolation(SNES snes, PetscInt level, Mat *mat);
   SNESFASGetRestriction(SNES snes, PetscInt level, Mat *mat);

In addition, FAS requires special restriction for solution-like
variables, called injection. This may be set with

::

   SNESFASGetInjection(SNES snes, PetscInt level, Mat *mat);.

The coarse solve context may be acquired with

::

   SNESFASGetCoarseSolve(SNES snes, SNES *smooth);

Nonlinear Additive Schwarz
^^^^^^^^^^^^^^^^^^^^^^^^^^

Nonlinear Additive Schwarz methods (NASM) take a number of local
nonlinear subproblems, solves them independently in parallel, and
combines those solutions into a new approximate solution.

::

   SNESNASMSetSubdomains(SNES snes,PetscInt n,SNES subsnes[],VecScatter iscatter[],VecScatter oscatter[],VecScatter gscatter[]);

allows for the user to create these local subdomains. Problems set up
using the ``SNES`` ``DMDA`` interface are automatically decomposed. To
begin, the type of subdomain updates to the whole solution are limited
to two types borrowed from ``PCASM``: ``PC_ASM_BASIC``, in which the
overlapping updates added. ``PC_ASM_RESTRICT`` updates in a
nonoverlapping fashion. This may be set with

::

   SNESNASMSetType(SNES snes,PCASMType type);.

``SNESASPIN`` is a helper ``SNES`` type that sets up a nonlinearly
preconditioned Newton’s method using NASM as the preconditioner.

General Options
~~~~~~~~~~~~~~~

This section discusses options and routines that apply to all ``SNES``
solvers and problem classes. In particular, we focus on convergence
tests, monitoring routines, and tools for checking derivative
computations.

.. _sec_snesconvergence:

Convergence Tests
^^^^^^^^^^^^^^^^^

Convergence of the nonlinear solvers can be detected in a variety of
ways; the user can even specify a customized test, as discussed below.
Most of the nonlinear solvers use ``SNESConvergenceTestDefault()``,
however, ``SNESNEWTONTR`` uses a method-specific additional convergence
test as well. The convergence tests involves several parameters, which
are set by default to values that should be reasonable for a wide range
of problems. The user can customize the parameters to the problem at
hand by using some of the following routines and options.

One method of convergence testing is to declare convergence when the
norm of the change in the solution between successive iterations is less
than some tolerance, ``stol``. Convergence can also be determined based
on the norm of the function. Such a test can use either the absolute
size of the norm, ``atol``, or its relative decrease, ``rtol``, from an
initial guess. The following routine sets these parameters, which are
used in many of the default ``SNES`` convergence tests:

::

   SNESSetTolerances(SNES snes,PetscReal atol,PetscReal rtol,PetscReal stol, PetscInt its,PetscInt fcts);

This routine also sets the maximum numbers of allowable nonlinear
iterations, ``its``, and function evaluations, ``fcts``. The
corresponding options database commands for setting these parameters are:

* ``-snes_atol <atol>``
* ``-snes_rtol <rtol>``
* ``-snes_stol <stol>``
* ``-snes_max_it <its>``
* ``-snes_max_funcs <fcts>``

A related routine is ``SNESGetTolerances()``.

Convergence tests for trust regions methods often use an additional
parameter that indicates the minimum allowable trust region radius. The
user can set this parameter with the option ``-snes_trtol <trtol>`` or
with the routine

::

   SNESSetTrustRegionTolerance(SNES snes,PetscReal trtol);

Users can set their own customized convergence tests in ``SNES`` by
using the command

::

   SNESSetConvergenceTest(SNES snes,PetscErrorCode (*test)(SNES snes,PetscInt it,PetscReal xnorm, PetscReal gnorm,PetscReal f,SNESConvergedReason reason, void *cctx),void *cctx,PetscErrorCode (*destroy)(void *cctx));

The final argument of the convergence test routine, ``cctx``, denotes an
optional user-defined context for private data. When solving systems of
nonlinear equations, the arguments ``xnorm``, ``gnorm``, and ``f`` are
the current iterate norm, current step norm, and function norm,
respectively. ``SNESConvergedReason`` should be set positive for
convergence and negative for divergence. See ``include/petscsnes.h`` for
a list of values for ``SNESConvergedReason``.

.. _sec_snesmonitor:

Convergence Monitoring
^^^^^^^^^^^^^^^^^^^^^^

By default the ``SNES`` solvers run silently without displaying
information about the iterations. The user can initiate monitoring with
the command

::

   SNESMonitorSet(SNES snes,PetscErrorCode (*mon)(SNES,PetscInt its,PetscReal norm,void* mctx),void *mctx,PetscErrorCode (*monitordestroy)(void**));

The routine, ``mon``, indicates a user-defined monitoring routine, where
``its`` and ``mctx`` respectively denote the iteration number and an
optional user-defined context for private data for the monitor routine.
The argument ``norm`` is the function norm.

The routine set by ``SNESMonitorSet()`` is called once after every
successful step computation within the nonlinear solver. Hence, the user
can employ this routine for any application-specific computations that
should be done after the solution update. The option ``-snes_monitor``
activates the default ``SNES`` monitor routine,
``SNESMonitorDefault()``, while ``-snes_monitor_lg_residualnorm`` draws
a simple line graph of the residual norm’s convergence.

One can cancel hardwired monitoring routines for ``SNES`` at runtime
with ``-snes_monitor_cancel``.

As the Newton method converges so that the residual norm is small, say
:math:`10^{-10}`, many of the final digits printed with the
``-snes_monitor`` option are meaningless. Worse, they are different on
different machines; due to different round-off rules used by, say, the
IBM RS6000 and the Sun SPARC. This makes testing between different
machines difficult. The option ``-snes_monitor_short`` causes PETSc to
print fewer of the digits of the residual norm as it gets smaller; thus
on most of the machines it will always print the same numbers making
cross-process testing easier.

The routines

::

   SNESGetSolution(SNES snes,Vec *x);
   SNESGetFunction(SNES snes,Vec *r,void *ctx,int(**func)(SNES,Vec,Vec,void*));

return the solution vector and function vector from a ``SNES`` context.
These routines are useful, for instance, if the convergence test
requires some property of the solution or function other than those
passed with routine arguments.

.. _sec_snesderivs:

Checking Accuracy of Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since hand-coding routines for Jacobian matrix evaluation can be error
prone, ``SNES`` provides easy-to-use support for checking these matrices
against finite difference versions. In the simplest form of comparison,
users can employ the option ``-snes_test_jacobian`` to compare the
matrices at several points. Although not exhaustive, this test will
generally catch obvious problems. One can compare the elements of the
two matrices by using the option ``-snes_test_jacobian_view`` , which
causes the two matrices to be printed to the screen.

Another means for verifying the correctness of a code for Jacobian
computation is running the problem with either the finite difference or
matrix-free variant, ``-snes_fd`` or ``-snes_mf``; see :any:`sec_fdmatrix` or :any:`sec_nlmatrixfree`.
If a
problem converges well with these matrix approximations but not with a
user-provided routine, the problem probably lies with the hand-coded
matrix. See the note in :any:`sec_snesjacobian` about
assembling your Jabobian in the "preconditioner" slot ``Pmat``.

The correctness of user provided ``MATSHELL`` Jacobians in general can be
checked with ``MatShellTestMultTranspose()`` and ``MatShellTestMult()``.

The correctness of user provided ``MATSHELL`` Jacobians via ``TSSetRHSJacobian()``
can be checked with ``TSRHSJacobianTestTranspose()`` and ``TSRHSJacobianTest()``
that check the correction of the matrix-transpose vector product and the
matrix-product. From the command line, these can be checked with

* ``-ts_rhs_jacobian_test_mult_transpose``
* ``-mat_shell_test_mult_transpose_view``
* ``-ts_rhs_jacobian_test_mult``
* ``-mat_shell_test_mult_view``

Inexact Newton-like Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since exact solution of the linear Newton systems within :math:numref:`newton`
at each iteration can be costly, modifications
are often introduced that significantly reduce these expenses and yet
retain the rapid convergence of Newton’s method. Inexact or truncated
Newton techniques approximately solve the linear systems using an
iterative scheme. In comparison with using direct methods for solving
the Newton systems, iterative methods have the virtue of requiring
little space for matrix storage and potentially saving significant
computational work. Within the class of inexact Newton methods, of
particular interest are Newton-Krylov methods, where the subsidiary
iterative technique for solving the Newton system is chosen from the
class of Krylov subspace projection methods. Note that at runtime the
user can set any of the linear solver options discussed in :any:`chapter_ksp`,
such as ``-ksp_type <ksp_method>`` and
``-pc_type <pc_method>``, to set the Krylov subspace and preconditioner
methods.

Two levels of iterations occur for the inexact techniques, where during
each global or outer Newton iteration a sequence of subsidiary inner
iterations of a linear solver is performed. Appropriate control of the
accuracy to which the subsidiary iterative method solves the Newton
system at each global iteration is critical, since these inner
iterations determine the asymptotic convergence rate for inexact Newton
techniques. While the Newton systems must be solved well enough to
retain fast local convergence of the Newton’s iterates, use of excessive
inner iterations, particularly when :math:`\| \mathbf{x}_k - \mathbf{x}_* \|` is large,
is neither necessary nor economical. Thus, the number of required inner
iterations typically increases as the Newton process progresses, so that
the truncated iterates approach the true Newton iterates.

A sequence of nonnegative numbers :math:`\{\eta_k\}` can be used to
indicate the variable convergence criterion. In this case, when solving
a system of nonlinear equations, the update step of the Newton process
remains unchanged, and direct solution of the linear system is replaced
by iteration on the system until the residuals

.. math:: \mathbf{r}_k^{(i)} =  \mathbf{F}'(\mathbf{x}_k) \Delta \mathbf{x}_k + \mathbf{F}(\mathbf{x}_k)

satisfy

.. math:: \frac{ \| \mathbf{r}_k^{(i)} \| }{ \| \mathbf{F}(\mathbf{x}_k) \| } \leq \eta_k \leq \eta < 1.

Here :math:`\mathbf{x}_0` is an initial approximation of the solution, and
:math:`\| \cdot \|` denotes an arbitrary norm in :math:`\Re^n` .

By default a constant relative convergence tolerance is used for solving
the subsidiary linear systems within the Newton-like methods of
``SNES``. When solving a system of nonlinear equations, one can instead
employ the techniques of Eisenstat and Walker :cite:`EW96`
to compute :math:`\eta_k` at each step of the nonlinear solver by using
the option ``-snes_ksp_ew_conv`` . In addition, by adding one’s own
``KSP`` convergence test (see :any:`sec_convergencetests`), one can easily create one’s own,
problem-dependent, inner convergence tests.

.. _sec_nlmatrixfree:

Matrix-Free Methods
~~~~~~~~~~~~~~~~~~~

The ``SNES`` class fully supports matrix-free methods. The matrices
specified in the Jacobian evaluation routine need not be conventional
matrices; instead, they can point to the data required to implement a
particular matrix-free method. The matrix-free variant is allowed *only*
when the linear systems are solved by an iterative method in combination
with no preconditioning (``PCNONE`` or ``-pc_type`` ``none``), a
user-provided preconditioner matrix, or a user-provided preconditioner
shell (``PCSHELL``, discussed in :any:`sec_pc`); that
is, obviously matrix-free methods cannot be used with a direct solver,
approximate factorization, or other preconditioner which requires access
to explicit matrix entries.

The user can create a matrix-free context for use within ``SNES`` with
the routine

::

   MatCreateSNESMF(SNES snes,Mat *mat);

This routine creates the data structures needed for the matrix-vector
products that arise within Krylov space iterative
methods :cite:`brownsaad:90` by employing the matrix type
``MATSHELL``, discussed in :any:`sec_matrixfree`.
The default ``SNES``
matrix-free approximations can also be invoked with the command
``-snes_mf``. Or, one can retain the user-provided Jacobian
preconditioner, but replace the user-provided Jacobian matrix with the
default matrix free variant with the option ``-snes_mf_operator``.

See also

::

   MatCreateMFFD(Vec x, Mat *mat);

for users who need a matrix-free matrix but are not using ``SNES``.

The user can set one parameter to control the Jacobian-vector product
approximation with the command

::

   MatMFFDSetFunctionError(Mat mat,PetscReal rerror);

The parameter ``rerror`` should be set to the square root of the
relative error in the function evaluations, :math:`e_{rel}`; the default
is the square root of machine epsilon (about :math:`10^{-8}` in double
precision), which assumes that the functions are evaluated to full
floating-point precision accuracy. This parameter can also be set from
the options database with ``-snes_mf_err <err>``

In addition, ``SNES`` provides a way to register new routines to compute
the differencing parameter (:math:`h`); see the manual page for
``MatMFFDSetType()`` and ``MatMFFDRegister()``. We currently provide two
default routines accessible via ``-snes_mf_type <default or wp>``. For
the default approach there is one “tuning” parameter, set with

::

   MatMFFDDSSetUmin(Mat mat,PetscReal umin);

This parameter, ``umin`` (or :math:`u_{min}`), is a bit involved; its
default is :math:`10^{-6}` . The Jacobian-vector product is approximated
via the formula

.. math:: F'(u) a \approx \frac{F(u + h*a) - F(u)}{h}

where :math:`h` is computed via

.. math::

   h = e_{\text{rel}} \cdot \begin{cases}
   u^{T}a/\lVert a \rVert^2_2                                 & \text{if $|u^T a| > u_{\min} \lVert a \rVert_{1}$} \\
   u_{\min} \operatorname{sign}(u^{T}a) \lVert a \rVert_{1}/\lVert a\rVert^2_2  & \text{otherwise}.
   \end{cases}

This approach is taken from Brown and Saad
:cite:`brownsaad:90`. The parameter can also be set from the
options database with ``-snes_mf_umin <umin>``

The second approach, taken from Walker and Pernice,
:cite:`pw98`, computes :math:`h` via

.. math::

   \begin{aligned}
           h = \frac{\sqrt{1 + ||u||}e_{rel}}{||a||}\end{aligned}

This has no tunable parameters, but note that inside the nonlinear solve
for the entire *linear* iterative process :math:`u` does not change
hence :math:`\sqrt{1 + ||u||}` need be computed only once. This
information may be set with the options

::

   MatMFFDWPSetComputeNormU(Mat mat,PetscBool );

or ``-mat_mffd_compute_normu <true or false>``. This information is used
to eliminate the redundant computation of these parameters, therefore
reducing the number of collective operations and improving the
efficiency of the application code.

It is also possible to monitor the differencing parameters h that are
computed via the routines

::

   MatMFFDSetHHistory(Mat,PetscScalar *,int);
   MatMFFDResetHHistory(Mat,PetscScalar *,int);
   MatMFFDGetH(Mat,PetscScalar *);

We include an explicit example of using matrix-free methods in `ex3.c <#snes-ex3>`_.
Note that by using the option ``-snes_mf`` one can
easily convert any ``SNES`` code to use a matrix-free Newton-Krylov
method without a preconditioner. As shown in this example,
``SNESSetFromOptions()`` must be called *after* ``SNESSetJacobian()`` to
enable runtime switching between the user-specified Jacobian and the
default ``SNES`` matrix-free form.

.. admonition:: Listing: ``src/snes/tutorials/ex3.c``
   :name: snes-ex3

   .. literalinclude:: ../../../snes/tutorials/ex3.c
      :end-before: /*TEST

Table :any:`tab-jacobians` summarizes the various matrix situations
that ``SNES`` supports. In particular, different linear system matrices
and preconditioning matrices are allowed, as well as both matrix-free
and application-provided preconditioners. If `ex3.c <#snes-ex3>`_ is run with
the options ``-snes_mf`` and ``-user_precond`` then it uses a
matrix-free application of the matrix-vector multiple and a user
provided matrix free Jacobian.

.. list-table:: Jacobian Options
   :name: tab-jacobians

   * - Matrix Use
     - Conventional Matrix Formats
     - Matrix-free versions
   * - Jacobian Matrix
     - Create matrix with ``MatCreate()``:math:`^*`.  Assemble matrix with user-defined routine :math:`^\dagger`
     - Create matrix with ``MatCreateShell()``.  Use ``MatShellSetOperation()`` to set various matrix actions, or use ``MatCreateMFFD()`` or ``MatCreateSNESMF()``.
   * - Preconditioning Matrix
     - Create matrix with ``MatCreate()``:math:`^*`.  Assemble matrix with user-defined routine :math:`^\dagger`
     - Use ``SNESGetKSP()`` and ``KSPGetPC()`` to access the ``PC``, then use ``PCSetType(pc, PCSHELL)`` followed by ``PCShellSetApply()``.

| :math:`^*` Use either the generic ``MatCreate()`` or a format-specific variant such as ``MatCreateAIJ()``.
| :math:`^\dagger` Set user-defined matrix formation routine with ``SNESSetJacobian()`` or with a ``DM`` variant such as ``DMDASNESSetJacobianLocal()``

.. _sec_fdmatrix:

Finite Difference Jacobian Approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSc provides some tools to help approximate the Jacobian matrices
efficiently via finite differences. These tools are intended for use in
certain situations where one is unable to compute Jacobian matrices
analytically, and matrix-free methods do not work well without a
preconditioner, due to very poor conditioning. The approximation
requires several steps:

-  First, one colors the columns of the (not yet built) Jacobian matrix,
   so that columns of the same color do not share any common rows.

-  Next, one creates a ``MatFDColoring`` data structure that will be
   used later in actually computing the Jacobian.

-  Finally, one tells the nonlinear solvers of ``SNES`` to use the
   ``SNESComputeJacobianDefaultColor()`` routine to compute the
   Jacobians.

A code fragment that demonstrates this process is given below.

::

   ISColoring    iscoloring;
   MatFDColoring fdcoloring;
   MatColoring   coloring;

   /*
     This initializes the nonzero structure of the Jacobian. This is artificial
     because clearly if we had a routine to compute the Jacobian we wouldn't
     need to use finite differences.
   */
   FormJacobian(snes,x, &J, &J, &user);

   /*
      Color the matrix, i.e. determine groups of columns that share no common
     rows. These columns in the Jacobian can all be computed simultaneously.
   */
   MatColoringCreate(J, &coloring);
   MatColoringSetType(coloring,MATCOLORINGSL);
   MatColoringSetFromOptions(coloring);
   MatColoringApply(coloring, &iscoloring);
   MatColoringDestroy(&coloring);
   /*
      Create the data structure that SNESComputeJacobianDefaultColor() uses
      to compute the actual Jacobians via finite differences.
   */
   MatFDColoringCreate(J,iscoloring, &fdcoloring);
   ISColoringDestroy(&iscoloring);
   MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))FormFunction, &user);
   MatFDColoringSetFromOptions(fdcoloring);

   /*
     Tell SNES to use the routine SNESComputeJacobianDefaultColor()
     to compute Jacobians.
   */
   SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,fdcoloring);

Of course, we are cheating a bit. If we do not have an analytic formula
for computing the Jacobian, then how do we know what its nonzero
structure is so that it may be colored? Determining the structure is
problem dependent, but fortunately, for most structured grid problems
(the class of problems for which PETSc was originally designed) if one
knows the stencil used for the nonlinear function one can usually fairly
easily obtain an estimate of the location of nonzeros in the matrix.
This is harder in the unstructured case, but one typically knows where the nonzero entries are from the mesh topology and distribution of degrees of freedom.
If using ``DMPlex`` (:any:`chapter_unstructured`) for unstructured meshes, the nonzero locations will be identified in ``DMCreateMatrix()`` and the procedure above can be used.
Most external packages for unstructured meshes have similar functionality.

One need not necessarily use a ``MatColoring`` object to determine a
coloring. For example, if a grid can be colored directly (without using
the associated matrix), then that coloring can be provided to
``MatFDColoringCreate()``. Note that the user must always preset the
nonzero structure in the matrix regardless of which coloring routine is
used.

PETSc provides the following coloring algorithms, which can be selected using ``MatColoringSetType()`` or via the command line argument ``-mat_coloring_type``.

.. list-table::
   :header-rows: 1

   * - Algorithm
     - ``MatColoringType``
     - ``-mat_coloring_type``
     - Parallel
   * - smallest-last :cite:`more84`
     - ``MATCOLORINGSL``
     - ``sl``
     - No
   * - largest-first :cite:`more84`
     - ``MATCOLORINGLF``
     - ``lf``
     - No
   * - incidence-degree :cite:`more84`
     - ``MATCOLORINGID``
     - ``id``
     - No
   * - Jones-Plassmann :cite:`jp:pcolor`
     - ``MATCOLORINGJP``
     - ``jp``
     - Yes
   * - Greedy
     - ``MATCOLORINGGREEDY``
     - ``greedy``
     - Yes
   * - Natural (1 color per column)
     - ``MATCOLORINGNATURAL``
     - ``natural``
     - Yes
   * - Power (:math:`A^k` followed by 1-coloring)
     - ``MATCOLORINGPOWER``
     - ``power``
     - Yes

As for the matrix-free computation of Jacobians (:any:`sec_nlmatrixfree`), two parameters affect the accuracy of the
finite difference Jacobian approximation. These are set with the command

::

   MatFDColoringSetParameters(MatFDColoring fdcoloring,PetscReal rerror,PetscReal umin);

The parameter ``rerror`` is the square root of the relative error in the
function evaluations, :math:`e_{rel}`; the default is the square root of
machine epsilon (about :math:`10^{-8}` in double precision), which
assumes that the functions are evaluated approximately to floating-point
precision accuracy. The second parameter, ``umin``, is a bit more
involved; its default is :math:`10e^{-6}` . Column :math:`i` of the
Jacobian matrix (denoted by :math:`F_{:i}`) is approximated by the
formula

.. math:: F'_{:i} \approx \frac{F(u + h*dx_{i}) - F(u)}{h}

where :math:`h` is computed via:

.. math::

   h = e_{\text{rel}} \cdot \begin{cases}
   u_{i}             &    \text{if $|u_{i}| > u_{\min}$} \\
   u_{\min} \cdot \operatorname{sign}(u_{i})  & \text{otherwise}.
   \end{cases}

for ``MATMFFD_DS`` or:

.. math::

   h = e_{\text{rel}} \sqrt(\|u\|)

for ``MATMFFD_WP`` (default). These parameters may be set from the options
database with

::

   -mat_fd_coloring_err <err>
   -mat_fd_coloring_umin <umin>
   -mat_fd_type <htype>

Note that ``MatColoring`` type ``MATCOLORINGSL``, ``MATCOLORINGLF``, and
``MATCOLORINGID`` are sequential algorithms. ``MATCOLORINGJP`` and
``MATCOLORINGGREEDY`` are parallel algorithms, although in practice they
may create more colors than the sequential algorithms. If one computes
the coloring ``iscoloring`` reasonably with a parallel algorithm or by
knowledge of the discretization, the routine ``MatFDColoringCreate()``
is scalable. An example of this for 2D distributed arrays is given below
that uses the utility routine ``DMCreateColoring()``.

::

   DMCreateColoring(da,IS_COLORING_GHOSTED, &iscoloring);
   MatFDColoringCreate(J,iscoloring, &fdcoloring);
   MatFDColoringSetFromOptions(fdcoloring);
   ISColoringDestroy( &iscoloring);

Note that the routine ``MatFDColoringCreate()`` currently is only
supported for the AIJ and BAIJ matrix formats.

.. _sec_vi:

Variational Inequalities
~~~~~~~~~~~~~~~~~~~~~~~~

``SNES`` can also solve variational inequalities with box constraints.
These are nonlinear algebraic systems with additional inequality
constraints on some or all of the variables:
:math:`Lu_i \le u_i \le Hu_i`. Some or all of the lower bounds may be
negative infinity (indicated to PETSc with ``SNES_VI_NINF``) and some or
all of the upper bounds may be infinity (indicated by ``SNES_VI_INF``).
The command

::

   SNESVISetVariableBounds(SNES,Vec Lu,Vec Hu);

is used to indicate that one is solving a variational inequality. The
option ``-snes_vi_monitor`` turns on extra monitoring of the active set
associated with the bounds and ``-snes_vi_type`` allows selecting from
several VI solvers, the default is preferred.

Nonlinear Preconditioning
~~~~~~~~~~~~~~~~~~~~~~~~~

The mathematical framework of nonlinear preconditioning is explained in detail in :cite:`BruneKnepleySmithTu15`.
Nonlinear preconditioning in PETSc involves the use of an inner ``SNES``
instance to define the step for an outer ``SNES`` instance. The inner
instance may be extracted using

::

   SNESGetNPC(SNES snes,SNES *npc);

and passed run-time options using the ``-npc_`` prefix. Nonlinear
preconditioning comes in two flavors: left and right. The side may be
changed using ``-snes_npc_side`` or ``SNESSetNPCSide()``. Left nonlinear
preconditioning redefines the nonlinear function as the action of the
nonlinear preconditioner :math:`\mathbf{M}`;

.. math:: \mathbf{F}_{M}(x) = \mathbf{M}(\mathbf{x},\mathbf{b}) - \mathbf{x}.

Right nonlinear preconditioning redefines the nonlinear function as the
function on the action of the nonlinear preconditioner;

.. math:: \mathbf{F}(\mathbf{M}(\mathbf{x},\mathbf{b})) = \mathbf{b},

which can be interpreted as putting the preconditioner into “striking
distance” of the solution by outer acceleration.

In addition, basic patterns of solver composition are available with the
``SNESType`` ``SNESCOMPOSITE``. This allows for two or more ``SNES``
instances to be combined additively or multiplicatively. By command
line, a set of ``SNES`` types may be given by comma separated list
argument to ``-snes_composite_sneses``. There are additive
(``SNES_COMPOSITE_ADDITIVE``), additive with optimal damping
(``SNES_COMPOSITE_ADDITIVEOPTIMAL``), and multiplicative
(``SNES_COMPOSITE_MULTIPLICATIVE``) variants which may be set with

::

   SNESCompositeSetType(SNES,SNESCompositeType);

New subsolvers may be added to the composite solver with

::

   SNESCompositeAddSNES(SNES,SNESType);

and accessed with

::

   SNESCompositeGetSNES(SNES,PetscInt,SNES *);

References
~~~~~~~~~~

.. bibliography:: ../../tex/petsc.bib
   :filter: docname in docnames

.. bibliography:: ../../tex/petscapp.bib
   :filter: docname in docnames
