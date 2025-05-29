(ch_snes)=

# SNES: Nonlinear Solvers

The solution of large-scale nonlinear problems pervades many facets of
computational science and demands robust and flexible solution
strategies. The `SNES` library of PETSc provides a powerful suite of
data-structure-neutral numerical routines for such problems. Built on
top of the linear solvers and data structures discussed in preceding
chapters, `SNES` enables the user to easily customize the nonlinear
solvers according to the application at hand. Also, the `SNES`
interface is *identical* for the uniprocess and parallel cases; the only
difference in the parallel version is that each process typically forms
only its local contribution to various matrices and vectors.

The `SNES` class includes methods for solving systems of nonlinear
equations of the form

$$
\mathbf{F}(\mathbf{x}) = 0,
$$ (fx0)

where $\mathbf{F}: \, \Re^n \to \Re^n$. Newton-like methods provide the
core of the package, including both line search and trust region
techniques. A suite of nonlinear Krylov methods and methods based upon
problem decomposition are also included. The solvers are discussed
further in {any}`sec_nlsolvers`. Following the PETSc design
philosophy, the interfaces to the various solvers are all virtually
identical. In addition, the `SNES` software is completely flexible, so
that the user can at runtime change any facet of the solution process.

PETSc’s default method for solving the nonlinear equation is Newton’s
method with line search, `SNESNEWTONLS`. The general form of the $n$-dimensional Newton’s method
for solving {math:numref}`fx0` is

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{J}(\mathbf{x}_k)^{-1} \mathbf{F}(\mathbf{x}_k), \;\; k=0,1, \ldots,
$$ (newton)

where $\mathbf{x}_0$ is an initial approximation to the solution and
$\mathbf{J}(\mathbf{x}_k) = \mathbf{F}'(\mathbf{x}_k)$, the Jacobian, is nonsingular at each
iteration. In practice, the Newton iteration {math:numref}`newton` is
implemented by the following two steps:

$$
\begin{aligned}
1. & \text{(Approximately) solve} & \mathbf{J}(\mathbf{x}_k) \Delta \mathbf{x}_k &= -\mathbf{F}(\mathbf{x}_k). \\
2. & \text{Update} & \mathbf{x}_{k+1} &\gets \mathbf{x}_k + \Delta \mathbf{x}_k.
\end{aligned}
$$

Other defect-correction algorithms can be implemented by using different
choices for $J(\mathbf{x}_k)$.

(sec_snesusage)=

## Basic SNES Usage

In the simplest usage of the nonlinear solvers, the user must merely
provide a C, C++, Fortran, or Python routine to evaluate the nonlinear function
{math:numref}`fx0`. The corresponding Jacobian matrix
can be approximated with finite differences. For codes that are
typically more efficient and accurate, the user can provide a routine to
compute the Jacobian; details regarding these application-provided
routines are discussed below. To provide an overview of the use of the
nonlinear solvers, browse the concrete example in {ref}`ex1.c <snes-ex1>` or skip ahead to the discussion.

(snes_ex1)=

:::{admonition} Listing: `src/snes/tutorials/ex1.c`
```{literalinclude} /../src/snes/tutorials/ex1.c
:end-before: /*TEST
```
:::

To create a `SNES` solver, one must first call `SNESCreate()` as
follows:

```
SNESCreate(MPI_Comm comm,SNES *snes);
```

The user must then set routines for evaluating the residual function {math:numref}`fx0`
and, *possibly*, its associated Jacobian matrix, as
discussed in the following sections.

To choose a nonlinear solution method, the user can either call

```
SNESSetType(SNES snes,SNESType method);
```

or use the option `-snes_type <method>`, where details regarding the
available methods are presented in {any}`sec_nlsolvers`. The
application code can take complete control of the linear and nonlinear
techniques used in the Newton-like method by calling

```
SNESSetFromOptions(snes);
```

This routine provides an interface to the PETSc options database, so
that at runtime the user can select a particular nonlinear solver, set
various parameters and customized routines (e.g., specialized line
search variants), prescribe the convergence tolerance, and set
monitoring routines. With this routine the user can also control all
linear solver options in the `KSP`, and `PC` modules, as discussed
in {any}`ch_ksp`.

After having set these routines and options, the user solves the problem
by calling

```
SNESSolve(SNES snes,Vec b,Vec x);
```

where `x` should be initialized to the initial guess before calling and contains the solution on return.
In particular, to employ an initial guess of
zero, the user should explicitly set this vector to zero by calling
`VecZeroEntries(x)`. Finally, after solving the nonlinear system (or several
systems), the user should destroy the `SNES` context with

```
SNESDestroy(SNES *snes);
```

(sec_snesfunction)=

### Nonlinear Function Evaluation

When solving a system of nonlinear equations, the user must provide a
a residual function {math:numref}`fx0`, which is set using

```
SNESSetFunction(SNES snes,Vec f,PetscErrorCode (*FormFunction)(SNES snes,Vec x,Vec f,void *ctx),void *ctx);
```

The argument `f` is an optional vector for storing the solution; pass `NULL` to have the `SNES` allocate it for you.
The argument `ctx` is an optional user-defined context, which can
store any private, application-specific data required by the function
evaluation routine; `NULL` should be used if such information is not
needed. In C and C++, a user-defined context is merely a structure in
which various objects can be stashed; in Fortran a user context can be
an integer array that contains both parameters and pointers to PETSc
objects.
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex5.c.html">SNES Tutorial ex5</a>
and
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/snes/tutorials/ex5f90.F90.html">SNES Tutorial ex5f90</a>
give examples of user-defined application contexts in C and Fortran,
respectively.

(sec_snesjacobian)=

### Jacobian Evaluation

The user may also specify a routine to form some approximation of the
Jacobian matrix, `A`, at the current iterate, `x`, as is typically
done with

```
SNESSetJacobian(SNES snes,Mat Amat,Mat Pmat,PetscErrorCode (*FormJacobian)(SNES snes,Vec x,Mat A,Mat B,void *ctx),void *ctx);
```

The arguments of the routine `FormJacobian()` are the current iterate,
`x`; the (approximate) Jacobian matrix, `Amat`; the matrix from
which the preconditioner is constructed, `Pmat` (which is usually the
same as `Amat`); and an optional user-defined Jacobian context,
`ctx`, for application-specific data. The `FormJacobian()`
callback is only invoked if the solver requires it, always
*after* `FormFunction()` has been called at the current iterate.

Note that the `SNES` solvers
are all data-structure neutral, so the full range of PETSc matrix
formats (including “matrix-free” methods) can be used.
{any}`ch_matrices` discusses information regarding
available matrix formats and options, while {any}`sec_nlmatrixfree` focuses on matrix-free methods in
`SNES`. We briefly touch on a few details of matrix usage that are
particularly important for efficient use of the nonlinear solvers.

A common usage paradigm is to assemble the problem Jacobian in the
preconditioner storage `B`, rather than `A`. In the case where they
are identical, as in many simulations, this makes no difference.
However, it allows us to check the analytic Jacobian we construct in
`FormJacobian()` by passing the `-snes_mf_operator` flag. This
causes PETSc to approximate the Jacobian using finite differencing of
the function evaluation (discussed in {any}`sec_fdmatrix`),
and the analytic Jacobian becomes merely the preconditioner. Even if the
analytic Jacobian is incorrect, it is likely that the finite difference
approximation will converge, and thus this is an excellent method to
verify the analytic Jacobian. Moreover, if the analytic Jacobian is
incomplete (some terms are missing or approximate),
`-snes_mf_operator` may be used to obtain the exact solution, where
the Jacobian approximation has been transferred to the preconditioner.

One such approximate Jacobian comes from “Picard linearization”, use `SNESSetPicard()`, which
writes the nonlinear system as

$$
\mathbf{F}(\mathbf{x}) := \mathbf{A}(\mathbf{x}) \mathbf{x} - \mathbf{b} = 0
$$

where $\mathbf{A}(\mathbf{x})$ usually contains the lower-derivative parts of the
equation. For example, the nonlinear diffusion problem

$$
- \nabla\cdot(\kappa(u) \nabla u) = 0
$$

would be linearized as

$$
A(u) v \simeq -\nabla\cdot(\kappa(u) \nabla v).
$$

Usually this linearization is simpler to implement than Newton and the
linear problems are somewhat easier to solve. In addition to using
`-snes_mf_operator` with this approximation to the Jacobian, the
Picard iterative procedure can be performed by defining $\mathbf{J}(\mathbf{x})$
to be $\mathbf{A}(\mathbf{x})$. Sometimes this iteration exhibits better global
convergence than Newton linearization.

During successive calls to `FormJacobian()`, the user can either
insert new matrix contexts or reuse old ones, depending on the
application requirements. For many sparse matrix formats, reusing the
old space (and merely changing the matrix elements) is more efficient;
however, if the matrix nonzero structure completely changes, creating an
entirely new matrix context may be preferable. Upon subsequent calls to
the `FormJacobian()` routine, the user may wish to reinitialize the
matrix entries to zero by calling `MatZeroEntries()`. See
{any}`sec_othermat` for details on the reuse of the matrix
context.

The directory `$PETSC_DIR/src/snes/tutorials` provides a variety of
examples.

Sometimes a nonlinear solver may produce a step that is not within the domain
of a given function, for example one with a negative pressure. When this occurs
one can call `SNESSetFunctionDomainError()` or `SNESSetJacobianDomainError()`
to indicate to `SNES` the step is not valid. One must also use `SNESGetConvergedReason()`
and check the reason to confirm if the solver succeeded. See {any}`sec_vi` for how to
provide `SNES` with bounds on the variables to solve (differential) variational inequalities
and how to control properties of the line step computed.

(sec_nlsolvers)=

## The Nonlinear Solvers

As summarized in Table {any}`tab-snesdefaults`, `SNES` includes
several Newton-like nonlinear solvers based on line search techniques
and trust region methods. Also provided are several nonlinear Krylov
methods, as well as nonlinear methods involving decompositions of the
problem.

Each solver may have associated with it a set of options, which can be
set with routines and options database commands provided for this
purpose. A complete list can be found by consulting the manual pages or
by running a program with the `-help` option; we discuss just a few in
the sections below.

```{eval-rst}
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
   * - Newton with Arc Length Continuation
     - ``SNESNEWTONAL``
     - ``newtonal``
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

```

### Line Search Newton

The method `SNESNEWTONLS` (`-snes_type newtonls`) provides a
line search Newton method for solving systems of nonlinear equations. By
default, this technique employs cubic backtracking
{cite}`dennis:83`. Alternative line search techniques are
listed in Table {any}`tab-linesearches`.

```{eval-rst}
.. table:: PETSc Line Search Methods
   :name: tab-linesearches

   ==================== =========================== ================
   **Line Search**      **SNESLineSearchType**      **Options Name**
   ==================== =========================== ================
   Backtracking         ``SNESLINESEARCHBT``        ``bt``
   (damped) step        ``SNESLINESEARCHBASIC``     ``basic``
   identical to above   ``SNESLINESEARCHNONE``      ``none``
   L2-norm Minimization ``SNESLINESEARCHL2``        ``l2``
   Critical point       ``SNESLINESEARCHCP``        ``cp``
   Bisection            ``SNESLINESEARCHBISECTION`` ``bisection``
   Shell                ``SNESLINESEARCHSHELL``     ``shell``
   ==================== =========================== ================
```

Every `SNES` has a line search context of type `SNESLineSearch` that
may be retrieved using

```
SNESGetLineSearch(SNES snes,SNESLineSearch *ls);.
```

There are several default options for the line searches. The order of
polynomial approximation may be set with `-snes_linesearch_order` or

```
SNESLineSearchSetOrder(SNESLineSearch ls, PetscInt order);
```

for instance, 2 for quadratic or 3 for cubic. Sometimes, it may not be
necessary to monitor the progress of the nonlinear iteration. In this
case, `-snes_linesearch_norms` or

```
SNESLineSearchSetComputeNorms(SNESLineSearch ls,PetscBool norms);
```

may be used to turn off function, step, and solution norm computation at
the end of the linesearch.

The default line search for the line search Newton method,
`SNESLINESEARCHBT` involves several parameters, which are set to
defaults that are reasonable for many applications. The user can
override the defaults by using the following options:

- `-snes_linesearch_alpha <alpha>`
- `-snes_linesearch_maxstep <max>`
- `-snes_linesearch_minlambda <tol>`

Besides the backtracking linesearch, there are `SNESLINESEARCHL2`,
which uses a polynomial secant minimization of $||F(x)||_2$, and
`SNESLINESEARCHCP`, which minimizes $F(x) \cdot Y$ where
$Y$ is the search direction. These are both potentially iterative
line searches, which may be used to find a better-fitted steplength in
the case where a single secant search is not sufficient. The number of
iterations may be set with `-snes_linesearch_max_it`. In addition, the
convergence criteria of the iterative line searches may be set using
function tolerances `-snes_linesearch_rtol` and
`-snes_linesearch_atol`, and steplength tolerance
`snes_linesearch_ltol`.

For highly non-linear problems, the bisection line search `SNESLINESEARCHBISECTION`
may prove useful due to its robustness. Similar to the critical point line search
`SNESLINESEARCHCP`, it seeks to find the root of $F(x) \cdot Y$.
While the latter does so through a secant method, the bisection line search
does so by iteratively bisecting the step length interval.
It works as follows (with $f(\lambda)=F(x-\lambda Y) \cdot Y / ||Y||$ for brevity):

1. initialize: $j=1$, $\lambda_0 = \lambda_{\text{left}} = 0.0$, $\lambda_j = \lambda_{\text{right}} = \alpha$, compute $f(\lambda_0)$ and $f(\lambda_j)$

2. check whether there is a change of sign in the interval: $f(\lambda_{\text{left}}) f(\lambda_j) \leq 0$; if not accept the full step length $\lambda_1$

3. if there is a change of sign, enter iterative bisection procedure

   1. check convergence/ exit criteria:

      - absolute tolerance $f(\lambda_j) < \mathtt{atol}$
      - relative tolerance $f(\lambda_j) < \mathtt{rtol} \cdot f(\lambda_0)$
      - change of step length $\lambda_j - \lambda_{j-1} < \mathtt{ltol}$
      - number of iterations $j < \mathtt{max\_it}$

   2. if $j > 1$, determine direction of bisection

   $$
   \begin{aligned}\lambda_{\text{left}} &= \begin{cases}\lambda_{\text{left}} &f(\lambda_{\text{left}}) f(\lambda_j) \leq 0\\\lambda_{j} &\text{else}\\ \end{cases}\\ \lambda_{\text{right}} &= \begin{cases} \lambda_j &f(\lambda_{\text{left}}) f(\lambda_j) \leq 0\\\lambda_{\text{right}} &\text{else}\\ \end{cases}\\\end{aligned}
   $$

   3. bisect the interval: $\lambda_{j+1} = (\lambda_{\text{left}} + \lambda_{\text{right}})/2$, compute $f(\lambda_{j+1})$
   4. update variables for the next iteration: $\lambda_j \gets \lambda_{j+1}$, $f(\lambda_j) \gets f(\lambda_{j+1})$, $j \gets j+1$

Custom line search types may either be defined using
`SNESLineSearchShell`, or by creating a custom user line search type
in the model of the preexisting ones and register it using

```
SNESLineSearchRegister(const char sname[],PetscErrorCode (*function)(SNESLineSearch));.
```

### Trust Region Methods

The trust region method in `SNES` for solving systems of nonlinear
equations, `SNESNEWTONTR` (`-snes_type newtontr`), is similar to the one developed in the
MINPACK project {cite}`more84`. Several parameters can be
set to control the variation of the trust region size during the
solution process. In particular, the user can control the initial trust
region radius, computed by

$$
\Delta = \Delta_0 \| F_0 \|_2,
$$

by setting $\Delta_0$ via the option `-snes_tr_delta0 <delta0>`.

### Newton with Arc Length Continuation

The Newton method with arc length continuation reformulates the linearized system
$K\delta \mathbf x = -\mathbf F(\mathbf x)$ by introducing the load parameter
$\lambda$ and splitting the residual into two components, commonly
corresponding to internal and external forces:

$$
\mathbf F(x, \lambda) = \mathbf F^{\mathrm{int}}(\mathbf x) - \mathbf F^{\mathrm{ext}}(\mathbf x, \lambda)
$$

Often, $\mathbf F^{\mathrm{ext}}(\mathbf x, \lambda)$ is linear in $\lambda$,
which can be thought of as applying the external force in proportional load
increments. By default, this is how the right-hand side vector is handled in the
implemented method. Generally, however, $\mathbf F^{\mathrm{ext}}(\mathbf x, \lambda)$
may depend non-linearly on $\lambda$ or $\mathbf x$, or both.
To accommodate this possibility, we provide the `SNESNewtonALGetLoadParameter()`
function, which allows for the current value of $\lambda$ to be queried in the
functions provided to `SNESSetFunction()` and `SNESSetJacobian()`.

Additionally, we split the solution update into two components:

$$
\delta \mathbf x = \delta s\delta\mathbf x^F + \delta\lambda\delta\mathbf x^Q,
$$

where $\delta s = 1$ unless partial corrections are used (discussed more
below). Each of $\delta \mathbf x^F$ and $\delta \mathbf x^Q$ are found via
solving a linear system with the Jacobian $K$:

- $\delta \mathbf x^F$ is the full Newton step for a given value of $\lambda$: $K \delta \mathbf x^F = -\mathbf F(\mathbf x, \lambda)$
- $\delta \mathbf x^Q$ is the variation in $\mathbf x$ with respect to $\lambda$, computed by $K \delta\mathbf x^Q = \mathbf Q(\mathbf x, \lambda)$, where $\mathbf Q(\mathbf x, \lambda) = -\partial \mathbf F (\mathbf x, \lambda) / \partial \lambda$ is the tangent load vector.

Often, the tangent load vector $\mathbf Q$ is constant within a load increment,
which corresponds to the case of proportional loading discussed above. By default,
$\mathbf Q$ is the full right-hand-side vector, if one was provided.
The user can also provide a function which computes $\mathbf Q$ to
`SNESNewtonALSetFunction()`. This function should have the same signature as for
`SNESSetFunction`, and the user should use `SNESNewtonALGetLoadParameter()` to get
$\lambda$ if it is needed.

**The Constraint Surface.** Considering the $n+1$ dimensional space of
$\mathbf x$ and $\lambda$, we define the linearized equilibrium line to be
the set of points for which the linearized equilibrium equations are satisfied.
Given the previous iterative solution
$\mathbf t^{(j-1)} = [\mathbf x^{(j-1)}, \lambda^{(j-1)}]$,
this line is defined by the point $\mathbf t^{(j-1)} + [\delta\mathbf x^F, 0]$ and
the vector $\mathbf t^Q [\delta\mathbf x^Q, 1]$.
The arc length method seeks the intersection of this linearized equilibrium line
with a quadratic constraint surface, defined by

% math::L^2 = \|\Delta x\|^2 + \psi^2 (\Delta\lambda)^2,

where $L$ is a user-provided step size corresponding to the radius of the
constraint surface, $\Delta\mathbf x$ and $\Delta\lambda$ are the
accumulated updates over the current load step, and $\psi^2$ is a
user-provided consistency parameter determining the shape of the constraint surface.
Generally, $\psi^2 > 0$ leads to a hyper-sphere constraint surface, while
$\psi^2 = 0$ leads to a hyper-cylinder constraint surface.

Since the solution will always fall on the constraint surface, the method will often
require multiple incremental steps to fully solve the non-linear problem.
This is necessary to accurately trace the equilibrium path.
Importantly, this is fundamentally different from time stepping.
While a similar process could be implemented as a `TS`, this method is
particularly designed to be used as a SNES, either standalone or within a `TS`.

To this end, by default, the load parameter is used such that the full external
forces are applied at $\lambda = 1$, although we allow for the user to specify
a different value via `-snes_newtonal_lambda_max`.
To ensure that the solution corresponds exactly to the external force prescribed by
the user, i.e. that the load parameter is exactly $\lambda_{max}$ at the end
of the SNES solve, we clamp the value before computing the solution update.
As such, the final increment will likely be a hybrid of arc length continuation and
normal Newton iterations.

**Choosing the Continuation Step.** For the first iteration from an equilibrium
point, there is a single correct way to choose $\delta\lambda$, which follows
from the constraint equations. Specifically the constraint equations yield the
quadratic equation $a\delta\lambda^2 + b\delta\lambda + c = 0$, where

$$
\begin{aligned}
a &= \|\delta\mathbf x^Q\|^2 + \psi^2,\\
b &= 2\delta\mathbf x^Q\cdot (\Delta\mathbf x + \delta s\delta\mathbf x^F) + 2\psi^2 \Delta\lambda,\\
c &= \|\Delta\mathbf x + \delta s\delta\mathbf x^F\|^2 + \psi^2 \Delta\lambda^2 - L^2.
\end{aligned}
$$

Since in the first iteration, $\Delta\mathbf x = \delta\mathbf x^F = \mathbf 0$ and
$\Delta\lambda = 0$, $b = 0$ and the equation simplifies to a pair of
real roots:

$$
\delta\lambda = \pm\frac{L}{\sqrt{\|\delta\mathbf x^Q\|^2 + \psi^2}},
$$

where the sign is positive for the first increment and is determined by the previous
increment otherwise as

$$
\text{sign}(\delta\lambda) = \text{sign}\big(\delta\mathbf x^Q \cdot (\Delta\mathbf x)_{i-1} + \psi^2(\Delta\lambda)_{i-1}\big),
$$

where $(\Delta\mathbf x)_{i-1}$ and $(\Delta\lambda)_{i-1}$ are the
accumulated updates over the previous load step.

In subsequent iterations, there are different approaches to selecting
$\delta\lambda$, all of which have trade-offs.
The main difference is whether the iterative solution falls on the constraint
surface at every iteration, or only when fully converged.
PETSc implements two approaches, set via
`SNESNewtonALSetCorrectionType()` or
`-snes_newtonal_correction_type <normal|exact>` on the command line.

**Corrections in the Normal Hyperplane.** The `SNES_NEWTONAL_CORRECTION_NORMAL`
option is simpler and computationally less expensive, but may fail to converge, as
the constraint equation is not satisfied at every iteration.
The update $\delta \lambda$ is chosen such that the update is within the
normal hyper-surface to the quadratic constraint surface.
Mathematically, that is

$$
\delta \lambda = -\frac{\Delta \mathbf x \cdot \delta \mathbf x^F}{\Delta\mathbf x \cdot \delta\mathbf x^Q + \psi^2 \Delta\lambda}.
$$

This implementation is based on {cite}`LeonPaulinoPereiraMenezesLages_2011`.

**Exact Corrections.** The `SNES_NEWTONAL_CORRECTION_EXACT` option is far more
complex, but ensures that the constraint is exactly satisfied at every Newton
iteration. As such, it is generally more robust.
By evaluating the intersection of constraint surface and equilibrium line at each
iteration, $\delta\lambda$ is chosen as one of the roots of the above
quadratic equation $a\delta\lambda^2 + b\delta\lambda + c = 0$.
This method encounters issues, however, if the linearized equilibrium line and
constraint surface do not intersect due to particularly large linearized error.
In this case, the roots are complex.
To continue progressing toward a solution, this method uses a partial correction by
choosing $\delta s$ such that the quadratic equation has a single real root.
Geometrically, this is selecting the point on the constraint surface closest to the
linearized equilibrium line. See the code or {cite}`Ritto-CorreaCamotim2008` for a
mathematical description of these partial corrections.

### Nonlinear Krylov Methods

A number of nonlinear Krylov methods are provided, including Nonlinear
Richardson (`SNESNRICHARDSON`), nonlinear conjugate gradient (`SNESNCG`), nonlinear GMRES (`SNESNGMRES`), and Anderson Mixing (`SNESANDERSON`). These
methods are described individually below. They are all instrumental to
PETSc’s nonlinear preconditioning.

**Nonlinear Richardson.** The nonlinear Richardson iteration, `SNESNRICHARDSON`, merely
takes the form of a line search-damped fixed-point iteration of the form

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \lambda \mathbf{F}(\mathbf{x}_k), \;\; k=0,1, \ldots,
$$

where the default linesearch is `SNESLINESEARCHL2`. This simple solver
is mostly useful as a nonlinear smoother, or to provide line search
stabilization to an inner method.

**Nonlinear Conjugate Gradients.** Nonlinear CG, `SNESNCG`, is equivalent to linear
CG, but with the steplength determined by line search
(`SNESLINESEARCHCP` by default). Five variants (Fletcher-Reed,
Hestenes-Steifel, Polak-Ribiere-Polyak, Dai-Yuan, and Conjugate Descent)
are implemented in PETSc and may be chosen using

```
SNESNCGSetType(SNES snes, SNESNCGType btype);
```

**Anderson Mixing and Nonlinear GMRES Methods.** Nonlinear GMRES (`SNESNGMRES`), and
Anderson Mixing (`SNESANDERSON`) methods combine the last $m$ iterates, plus a new
fixed-point iteration iterate, into an approximate residual-minimizing new iterate.

All of the above methods have support for using a nonlinear preconditioner to compute the preliminary update step, rather than the default
which is the nonlinear function's residual, \$ mathbf\{F}(mathbf\{x}\_k)\$. The different update is obtained by solving a nonlinear preconditioner nonlinear problem, which has its own
`SNES` object that may be obtained with `SNESGetNPC()`.
Quasi-Newton Methods
^^^^^^^^^^^^^^^^^^^^

Quasi-Newton methods store iterative rank-one updates to the Jacobian
instead of computing the Jacobian directly. Three limited-memory quasi-Newton
methods are provided, L-BFGS, which are described in
Table {any}`tab-qndefaults`. These all are encapsulated under
`-snes_type qn` and may be changed with `snes_qn_type`. The default
is L-BFGS, which provides symmetric updates to an approximate Jacobian.
This iteration is similar to the line search Newton methods.

The quasi-Newton methods support the use of a nonlinear preconditioner that can be obtained with `SNESGetNPC()` and then configured; or that can be configured with
`SNES`, `KSP`, and `PC` options using the options database prefix `-npc_`.

```{eval-rst}
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
     - ``SNESLINESEARCHBASIC`` (or equivalently ``SNESLINESEARCHNONE``
   * - “Bad” Broyden
     - ``SNES_QN_BADBROYDEN``
     - ``badbroyden``
     - ``SNESLINESEARCHL2``
```

One may also control the form of the initial Jacobian approximation with

```
SNESQNSetScaleType(SNES snes, SNESQNScaleType stype);
```

and the restart type with

```
SNESQNSetRestartType(SNES snes, SNESQNRestartType rtype);
```

### The Full Approximation Scheme

The Nonlinear Full Approximation Scheme (FAS) `SNESFAS`, is a nonlinear multigrid method. At
each level, there is a recursive cycle control `SNES` instance, and
either one or two nonlinear solvers that act as smoothers (up and down). Problems
set up using the `SNES` `DMDA` interface are automatically
coarsened. FAS, `SNESFAS`, differs slightly from linear multigrid `PCMG`, in that the hierarchy is
constructed recursively. However, much of the interface is a one-to-one
map. We describe the “get” operations here, and it can be assumed that
each has a corresponding “set” operation. For instance, the number of
levels in the hierarchy may be retrieved using

```
SNESFASGetLevels(SNES snes, PetscInt *levels);
```

There are four `SNESFAS` cycle types, `SNES_FAS_MULTIPLICATIVE`,
`SNES_FAS_ADDITIVE`, `SNES_FAS_FULL`, and `SNES_FAS_KASKADE`. The
type may be set with

```
SNESFASSetType(SNES snes,SNESFASType fastype);.
```

and the cycle type, 1 for V, 2 for W, may be set with

```
SNESFASSetCycles(SNES snes, PetscInt cycles);.
```

Much like the interface to `PCMG` described in {any}`sec_mg`, there are interfaces to recover the
various levels’ cycles and smoothers. The level smoothers may be
accessed with

```
SNESFASGetSmoother(SNES snes, PetscInt level, SNES *smooth);
SNESFASGetSmootherUp(SNES snes, PetscInt level, SNES *smooth);
SNESFASGetSmootherDown(SNES snes, PetscInt level, SNES *smooth);
```

and the level cycles with

```
SNESFASGetCycleSNES(SNES snes,PetscInt level,SNES *lsnes);.
```

Also akin to `PCMG`, the restriction and prolongation at a level may
be acquired with

```
SNESFASGetInterpolation(SNES snes, PetscInt level, Mat *mat);
SNESFASGetRestriction(SNES snes, PetscInt level, Mat *mat);
```

In addition, FAS requires special restriction for solution-like
variables, called injection. This may be set with

```
SNESFASGetInjection(SNES snes, PetscInt level, Mat *mat);.
```

The coarse solve context may be acquired with

```
SNESFASGetCoarseSolve(SNES snes, SNES *smooth);
```

### Nonlinear Additive Schwarz

Nonlinear Additive Schwarz methods (NASM) take a number of local
nonlinear subproblems, solves them independently in parallel, and
combines those solutions into a new approximate solution.

```
SNESNASMSetSubdomains(SNES snes,PetscInt n,SNES subsnes[],VecScatter iscatter[],VecScatter oscatter[],VecScatter gscatter[]);
```

allows for the user to create these local subdomains. Problems set up
using the `SNES` `DMDA` interface are automatically decomposed. To
begin, the type of subdomain updates to the whole solution are limited
to two types borrowed from `PCASM`: `PC_ASM_BASIC`, in which the
overlapping updates added. `PC_ASM_RESTRICT` updates in a
nonoverlapping fashion. This may be set with

```
SNESNASMSetType(SNES snes,PCASMType type);.
```

`SNESASPIN` is a helper `SNES` type that sets up a nonlinearly
preconditioned Newton’s method using NASM as the preconditioner.

## General Options

This section discusses options and routines that apply to all `SNES`
solvers and problem classes. In particular, we focus on convergence
tests, monitoring routines, and tools for checking derivative
computations.

(sec_snesconvergence)=

### Convergence Tests

Convergence of the nonlinear solvers can be detected in a variety of
ways; the user can even specify a customized test, as discussed below.
Most of the nonlinear solvers use `SNESConvergenceTestDefault()`,
however, `SNESNEWTONTR` uses a method-specific additional convergence
test as well. The convergence tests involves several parameters, which
are set by default to values that should be reasonable for a wide range
of problems. The user can customize the parameters to the problem at
hand by using some of the following routines and options.

One method of convergence testing is to declare convergence when the
norm of the change in the solution between successive iterations is less
than some tolerance, `stol`. Convergence can also be determined based
on the norm of the function. Such a test can use either the absolute
size of the norm, `atol`, or its relative decrease, `rtol`, from an
initial guess. The following routine sets these parameters, which are
used in many of the default `SNES` convergence tests:

```
SNESSetTolerances(SNES snes,PetscReal atol,PetscReal rtol,PetscReal stol, PetscInt its,PetscInt fcts);
```

This routine also sets the maximum numbers of allowable nonlinear
iterations, `its`, and function evaluations, `fcts`. The
corresponding options database commands for setting these parameters are:

- `-snes_atol <atol>`
- `-snes_rtol <rtol>`
- `-snes_stol <stol>`
- `-snes_max_it <its>`
- `-snes_max_funcs <fcts>` (use `unlimited` for no maximum)

A related routine is `SNESGetTolerances()`. `PETSC_CURRENT` may be used
for any parameter to indicate the current value should be retained; use `PETSC_DETERMINE` to restore to the default value from when the object was created.

Users can set their own customized convergence tests in `SNES` by
using the command

```
SNESSetConvergenceTest(SNES snes,PetscErrorCode (*test)(SNES snes,PetscInt it,PetscReal xnorm, PetscReal gnorm,PetscReal f,SNESConvergedReason reason, void *cctx),void *cctx,PetscErrorCode (*destroy)(void *cctx));
```

The final argument of the convergence test routine, `cctx`, denotes an
optional user-defined context for private data. When solving systems of
nonlinear equations, the arguments `xnorm`, `gnorm`, and `f` are
the current iterate norm, current step norm, and function norm,
respectively. `SNESConvergedReason` should be set positive for
convergence and negative for divergence. See `include/petscsnes.h` for
a list of values for `SNESConvergedReason`.

(sec_snesmonitor)=

### Convergence Monitoring

By default the `SNES` solvers run silently without displaying
information about the iterations. The user can initiate monitoring with
the command

```
SNESMonitorSet(SNES snes, PetscErrorCode (*mon)(SNES snes, PetscInt its, PetscReal norm, void* mctx), void *mctx, (PetscCtxDestroyFn *)*monitordestroy);
```

The routine, `mon`, indicates a user-defined monitoring routine, where
`its` and `mctx` respectively denote the iteration number and an
optional user-defined context for private data for the monitor routine.
The argument `norm` is the function norm.

The routine set by `SNESMonitorSet()` is called once after every
successful step computation within the nonlinear solver. Hence, the user
can employ this routine for any application-specific computations that
should be done after the solution update. The option `-snes_monitor`
activates the default `SNES` monitor routine,
`SNESMonitorDefault()`, while `-snes_monitor_lg_residualnorm` draws
a simple line graph of the residual norm’s convergence.

One can cancel hardwired monitoring routines for `SNES` at runtime
with `-snes_monitor_cancel`.

As the Newton method converges so that the residual norm is small, say
$10^{-10}$, many of the final digits printed with the
`-snes_monitor` option are meaningless. Worse, they are different on
different machines; due to different round-off rules used by, say, the
IBM RS6000 and the Sun SPARC. This makes testing between different
machines difficult. The option `-snes_monitor_short` causes PETSc to
print fewer of the digits of the residual norm as it gets smaller; thus
on most of the machines it will always print the same numbers making
cross-process testing easier.

The routines

```
SNESGetSolution(SNES snes,Vec *x);
SNESGetFunction(SNES snes,Vec *r,void *ctx,int(**func)(SNES,Vec,Vec,void*));
```

return the solution vector and function vector from a `SNES` context.
These routines are useful, for instance, if the convergence test
requires some property of the solution or function other than those
passed with routine arguments.

(sec_snesderivs)=

### Checking Accuracy of Derivatives

Since hand-coding routines for Jacobian matrix evaluation can be error
prone, `SNES` provides easy-to-use support for checking these matrices
against finite difference versions. In the simplest form of comparison,
users can employ the option `-snes_test_jacobian` to compare the
matrices at several points. Although not exhaustive, this test will
generally catch obvious problems. One can compare the elements of the
two matrices by using the option `-snes_test_jacobian_view` , which
causes the two matrices to be printed to the screen.

Another means for verifying the correctness of a code for Jacobian
computation is running the problem with either the finite difference or
matrix-free variant, `-snes_fd` or `-snes_mf`; see {any}`sec_fdmatrix` or {any}`sec_nlmatrixfree`.
If a
problem converges well with these matrix approximations but not with a
user-provided routine, the problem probably lies with the hand-coded
matrix. See the note in {any}`sec_snesjacobian` about
assembling your Jabobian in the "preconditioner" slot `Pmat`.

The correctness of user provided `MATSHELL` Jacobians in general can be
checked with `MatShellTestMultTranspose()` and `MatShellTestMult()`.

The correctness of user provided `MATSHELL` Jacobians via `TSSetRHSJacobian()`
can be checked with `TSRHSJacobianTestTranspose()` and `TSRHSJacobianTest()`
that check the correction of the matrix-transpose vector product and the
matrix-product. From the command line, these can be checked with

- `-ts_rhs_jacobian_test_mult_transpose`
- `-mat_shell_test_mult_transpose_view`
- `-ts_rhs_jacobian_test_mult`
- `-mat_shell_test_mult_view`

## Inexact Newton-like Methods

Since exact solution of the linear Newton systems within {math:numref}`newton`
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
user can set any of the linear solver options discussed in {any}`ch_ksp`,
such as `-ksp_type <ksp_method>` and
`-pc_type <pc_method>`, to set the Krylov subspace and preconditioner
methods.

Two levels of iterations occur for the inexact techniques, where during
each global or outer Newton iteration a sequence of subsidiary inner
iterations of a linear solver is performed. Appropriate control of the
accuracy to which the subsidiary iterative method solves the Newton
system at each global iteration is critical, since these inner
iterations determine the asymptotic convergence rate for inexact Newton
techniques. While the Newton systems must be solved well enough to
retain fast local convergence of the Newton’s iterates, use of excessive
inner iterations, particularly when $\| \mathbf{x}_k - \mathbf{x}_* \|$ is large,
is neither necessary nor economical. Thus, the number of required inner
iterations typically increases as the Newton process progresses, so that
the truncated iterates approach the true Newton iterates.

A sequence of nonnegative numbers $\{\eta_k\}$ can be used to
indicate the variable convergence criterion. In this case, when solving
a system of nonlinear equations, the update step of the Newton process
remains unchanged, and direct solution of the linear system is replaced
by iteration on the system until the residuals

$$
\mathbf{r}_k^{(i)} =  \mathbf{F}'(\mathbf{x}_k) \Delta \mathbf{x}_k + \mathbf{F}(\mathbf{x}_k)
$$

satisfy

$$
\frac{ \| \mathbf{r}_k^{(i)} \| }{ \| \mathbf{F}(\mathbf{x}_k) \| } \leq \eta_k \leq \eta < 1.
$$

Here $\mathbf{x}_0$ is an initial approximation of the solution, and
$\| \cdot \|$ denotes an arbitrary norm in $\Re^n$ .

By default a constant relative convergence tolerance is used for solving
the subsidiary linear systems within the Newton-like methods of
`SNES`. When solving a system of nonlinear equations, one can instead
employ the techniques of Eisenstat and Walker {cite}`ew96`
to compute $\eta_k$ at each step of the nonlinear solver by using
the option `-snes_ksp_ew` . In addition, by adding one’s own
`KSP` convergence test (see {any}`sec_convergencetests`), one can easily create one’s own,
problem-dependent, inner convergence tests.

(sec_nlmatrixfree)=

## Matrix-Free Methods

The `SNES` class fully supports matrix-free methods. The matrices
specified in the Jacobian evaluation routine need not be conventional
matrices; instead, they can point to the data required to implement a
particular matrix-free method. The matrix-free variant is allowed *only*
when the linear systems are solved by an iterative method in combination
with no preconditioning (`PCNONE` or `-pc_type` `none`), a
user-provided matrix from which to construct the preconditioner, or a user-provided preconditioner
shell (`PCSHELL`, discussed in {any}`sec_pc`); that
is, obviously matrix-free methods cannot be used with a direct solver,
approximate factorization, or other preconditioner which requires access
to explicit matrix entries.

The user can create a matrix-free context for use within `SNES` with
the routine

```
MatCreateSNESMF(SNES snes,Mat *mat);
```

This routine creates the data structures needed for the matrix-vector
products that arise within Krylov space iterative
methods {cite}`brownsaad:90`.
The default `SNES`
matrix-free approximations can also be invoked with the command
`-snes_mf`. Or, one can retain the user-provided Jacobian
preconditioner, but replace the user-provided Jacobian matrix with the
default matrix-free variant with the option `-snes_mf_operator`.

`MatCreateSNESMF()` uses

```
MatCreateMFFD(Vec x, Mat *mat);
```

which can also be used directly for users who need a matrix-free matrix but are not using `SNES`.

The user can set one parameter to control the Jacobian-vector product
approximation with the command

```
MatMFFDSetFunctionError(Mat mat,PetscReal rerror);
```

The parameter `rerror` should be set to the square root of the
relative error in the function evaluations, $e_{rel}$; the default
is the square root of machine epsilon (about $10^{-8}$ in double
precision), which assumes that the functions are evaluated to full
floating-point precision accuracy. This parameter can also be set from
the options database with `-mat_mffd_err <err>`

In addition, PETSc provides ways to register new routines to compute
the differencing parameter ($h$); see the manual page for
`MatMFFDSetType()` and `MatMFFDRegister()`. We currently provide two
default routines accessible via `-mat_mffd_type <ds or wp>`. For
the default approach there is one “tuning” parameter, set with

```
MatMFFDDSSetUmin(Mat mat,PetscReal umin);
```

This parameter, `umin` (or $u_{min}$), is a bit involved; its
default is $10^{-6}$ . Its command line form is `-mat_mffd_umin <umin>`.

The Jacobian-vector product is approximated
via the formula

$$
F'(u) a \approx \frac{F(u + h*a) - F(u)}{h}
$$

where $h$ is computed via

$$
h = e_{\text{rel}} \cdot \begin{cases}
u^{T}a/\lVert a \rVert^2_2                                 & \text{if $|u^T a| > u_{\min} \lVert a \rVert_{1}$} \\
u_{\min} \operatorname{sign}(u^{T}a) \lVert a \rVert_{1}/\lVert a\rVert^2_2  & \text{otherwise}.
\end{cases}
$$

This approach is taken from Brown and Saad
{cite}`brownsaad:90`. The second approach, taken from Walker and Pernice,
{cite}`pw98`, computes $h$ via

$$
\begin{aligned}
        h = \frac{\sqrt{1 + ||u||}e_{rel}}{||a||}\end{aligned}
$$

This has no tunable parameters, but note that inside the nonlinear solve
for the entire *linear* iterative process $u$ does not change
hence $\sqrt{1 + ||u||}$ need be computed only once. This
information may be set with the options

```
MatMFFDWPSetComputeNormU(Mat mat,PetscBool );
```

or `-mat_mffd_compute_normu <true or false>`. This information is used
to eliminate the redundant computation of these parameters, therefore
reducing the number of collective operations and improving the
efficiency of the application code. This takes place automatically for the PETSc GMRES solver with left preconditioning.

It is also possible to monitor the differencing parameters h that are
computed via the routines

```
MatMFFDSetHHistory(Mat,PetscScalar *,int);
MatMFFDResetHHistory(Mat,PetscScalar *,int);
MatMFFDGetH(Mat,PetscScalar *);
```

We include an explicit example of using matrix-free methods in {any}`ex3.c <snes_ex3>`.
Note that by using the option `-snes_mf` one can
easily convert any `SNES` code to use a matrix-free Newton-Krylov
method without a preconditioner. As shown in this example,
`SNESSetFromOptions()` must be called *after* `SNESSetJacobian()` to
enable runtime switching between the user-specified Jacobian and the
default `SNES` matrix-free form.

(snes_ex3)=

:::{admonition} Listing: `src/snes/tutorials/ex3.c`
```{literalinclude} /../src/snes/tutorials/ex3.c
:end-before: /*TEST
```
:::

Table {any}`tab-jacobians` summarizes the various matrix situations
that `SNES` supports. In particular, different linear system matrices
and preconditioning matrices are allowed, as well as both matrix-free
and application-provided preconditioners. If {any}`ex3.c <snes_ex3>` is run with
the options `-snes_mf` and `-user_precond` then it uses a
matrix-free application of the matrix-vector multiple and a user
provided matrix-free Jacobian.

```{eval-rst}
.. list-table:: Jacobian Options
   :name: tab-jacobians

   * - Matrix Use
     - Conventional Matrix Formats
     - Matrix-free versions
   * - Jacobian Matrix
     - Create matrix with ``MatCreate()``:math:`^*`.  Assemble matrix with user-defined routine :math:`^\dagger`
     - Create matrix with ``MatCreateShell()``.  Use ``MatShellSetOperation()`` to set various matrix actions, or use ``MatCreateMFFD()`` or ``MatCreateSNESMF()``.
   * - Matrix used to construct the preconditioner
     - Create matrix with ``MatCreate()``:math:`^*`.  Assemble matrix with user-defined routine :math:`^\dagger`
     - Use ``SNESGetKSP()`` and ``KSPGetPC()`` to access the ``PC``, then use ``PCSetType(pc, PCSHELL)`` followed by ``PCShellSetApply()``.
```

$^*$ Use either the generic `MatCreate()` or a format-specific variant such as `MatCreateAIJ()`.

$^\dagger$ Set user-defined matrix formation routine with `SNESSetJacobian()` or with a `DM` variant such as `DMDASNESSetJacobianLocal()`

SNES also provides some less well-integrated code to apply matrix-free finite differencing using an automatically computed measurement of the
noise of the functions. This can be selected with `-snes_mf_version 2`; it does not use `MatCreateMFFD()` but has similar options that start with
`-snes_mf_` instead of `-mat_mffd_`. Note that this alternative prefix **only** works for version 2 differencing.

(sec_fdmatrix)=

## Finite Difference Jacobian Approximations

PETSc provides some tools to help approximate the Jacobian matrices
efficiently via finite differences. These tools are intended for use in
certain situations where one is unable to compute Jacobian matrices
analytically, and matrix-free methods do not work well without a
preconditioner, due to very poor conditioning. The approximation
requires several steps:

- First, one colors the columns of the (not yet built) Jacobian matrix,
  so that columns of the same color do not share any common rows.
- Next, one creates a `MatFDColoring` data structure that will be
  used later in actually computing the Jacobian.
- Finally, one tells the nonlinear solvers of `SNES` to use the
  `SNESComputeJacobianDefaultColor()` routine to compute the
  Jacobians.

A code fragment that demonstrates this process is given below.

```
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
MatFDColoringCreate(J, iscoloring, &fdcoloring);
ISColoringDestroy(&iscoloring);
MatFDColoringSetFunction(fdcoloring, (MatFDColoringFn *)FormFunction, &user);
MatFDColoringSetFromOptions(fdcoloring);

/*
  Tell SNES to use the routine SNESComputeJacobianDefaultColor()
  to compute Jacobians.
*/
SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,fdcoloring);
```

Of course, we are cheating a bit. If we do not have an analytic formula
for computing the Jacobian, then how do we know what its nonzero
structure is so that it may be colored? Determining the structure is
problem dependent, but fortunately, for most structured grid problems
(the class of problems for which PETSc was originally designed) if one
knows the stencil used for the nonlinear function one can usually fairly
easily obtain an estimate of the location of nonzeros in the matrix.
This is harder in the unstructured case, but one typically knows where the nonzero entries are from the mesh topology and distribution of degrees of freedom.
If using `DMPlex` ({any}`ch_unstructured`) for unstructured meshes, the nonzero locations will be identified in `DMCreateMatrix()` and the procedure above can be used.
Most external packages for unstructured meshes have similar functionality.

One need not necessarily use a `MatColoring` object to determine a
coloring. For example, if a grid can be colored directly (without using
the associated matrix), then that coloring can be provided to
`MatFDColoringCreate()`. Note that the user must always preset the
nonzero structure in the matrix regardless of which coloring routine is
used.

PETSc provides the following coloring algorithms, which can be selected using `MatColoringSetType()` or via the command line argument `-mat_coloring_type`.

```{eval-rst}
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
```

As for the matrix-free computation of Jacobians ({any}`sec_nlmatrixfree`), two parameters affect the accuracy of the
finite difference Jacobian approximation. These are set with the command

```
MatFDColoringSetParameters(MatFDColoring fdcoloring,PetscReal rerror,PetscReal umin);
```

The parameter `rerror` is the square root of the relative error in the
function evaluations, $e_{rel}$; the default is the square root of
machine epsilon (about $10^{-8}$ in double precision), which
assumes that the functions are evaluated approximately to floating-point
precision accuracy. The second parameter, `umin`, is a bit more
involved; its default is $10^{-6}$. Column $i$ of the
Jacobian matrix (denoted by $F_{:i}$) is approximated by the
formula

$$
F'_{:i} \approx \frac{F(u + h*dx_{i}) - F(u)}{h}
$$

where $h$ is computed via:

$$
h = e_{\text{rel}} \cdot \begin{cases}
u_{i}             &    \text{if $|u_{i}| > u_{\min}$} \\
u_{\min} \cdot \operatorname{sign}(u_{i})  & \text{otherwise}.
\end{cases}
$$

for `MATMFFD_DS` or:

$$
h = e_{\text{rel}} \sqrt{\|u\|}
$$

for `MATMFFD_WP` (default). These parameters may be set from the options
database with

```
-mat_fd_coloring_err <err>
-mat_fd_coloring_umin <umin>
-mat_fd_type <htype>
```

Note that `MatColoring` type `MATCOLORINGSL`, `MATCOLORINGLF`, and
`MATCOLORINGID` are sequential algorithms. `MATCOLORINGJP` and
`MATCOLORINGGREEDY` are parallel algorithms, although in practice they
may create more colors than the sequential algorithms. If one computes
the coloring `iscoloring` reasonably with a parallel algorithm or by
knowledge of the discretization, the routine `MatFDColoringCreate()`
is scalable. An example of this for 2D distributed arrays is given below
that uses the utility routine `DMCreateColoring()`.

```
DMCreateColoring(da,IS_COLORING_GHOSTED, &iscoloring);
MatFDColoringCreate(J,iscoloring, &fdcoloring);
MatFDColoringSetFromOptions(fdcoloring);
ISColoringDestroy( &iscoloring);
```

Note that the routine `MatFDColoringCreate()` currently is only
supported for the AIJ and BAIJ matrix formats.

(sec_vi)=

## Variational Inequalities

`SNES` can also solve (differential) variational inequalities with box (bound) constraints.
These are nonlinear algebraic systems with additional inequality
constraints on some or all of the variables:
$L_i \le u_i \le H_i$. For example, the pressure variable cannot be negative.
Some, or all, of the lower bounds may be
negative infinity (indicated to PETSc with `SNES_VI_NINF`) and some, or
all, of the upper bounds may be infinity (indicated by `SNES_VI_INF`).
The commands

```
SNESVISetVariableBounds(SNES,Vec L,Vec H);
SNESVISetComputeVariableBounds(SNES snes, PetscErrorCode (*compute)(SNES,Vec,Vec))
```

are used to indicate that one is solving a variational inequality. Problems with box constraints can be solved with
the reduced space, `SNESVINEWTONRSLS`, and semi-smooth `SNESVINEWTONSSLS` solvers.

The
option `-snes_vi_monitor` turns on extra monitoring of the active set
associated with the bounds and `-snes_vi_type` allows selecting from
several VI solvers, the default is preferred.

`SNESLineSearchSetPreCheck()` and `SNESLineSearchSetPostCheck()` can also be used to control properties
of the steps selected by `SNES`.

(sec_snespc)=

## Nonlinear Preconditioning

The mathematical framework of nonlinear preconditioning is explained in detail in {cite}`bruneknepleysmithtu15`.
Nonlinear preconditioning in PETSc involves the use of an inner `SNES`
instance to define the step for an outer `SNES` instance. The inner
instance may be extracted using

```
SNESGetNPC(SNES snes,SNES *npc);
```

and passed run-time options using the `-npc_` prefix. Nonlinear
preconditioning comes in two flavors: left and right. The side may be
changed using `-snes_npc_side` or `SNESSetNPCSide()`. Left nonlinear
preconditioning redefines the nonlinear function as the action of the
nonlinear preconditioner $\mathbf{M}$;

$$
\mathbf{F}_{M}(x) = \mathbf{M}(\mathbf{x},\mathbf{b}) - \mathbf{x}.
$$

Right nonlinear preconditioning redefines the nonlinear function as the
function on the action of the nonlinear preconditioner;

$$
\mathbf{F}(\mathbf{M}(\mathbf{x},\mathbf{b})) = \mathbf{b},
$$

which can be interpreted as putting the preconditioner into “striking
distance” of the solution by outer acceleration.

In addition, basic patterns of solver composition are available with the
`SNESType` `SNESCOMPOSITE`. This allows for two or more `SNES`
instances to be combined additively or multiplicatively. By command
line, a set of `SNES` types may be given by comma separated list
argument to `-snes_composite_sneses`. There are additive
(`SNES_COMPOSITE_ADDITIVE`), additive with optimal damping
(`SNES_COMPOSITE_ADDITIVEOPTIMAL`), and multiplicative
(`SNES_COMPOSITE_MULTIPLICATIVE`) variants which may be set with

```
SNESCompositeSetType(SNES,SNESCompositeType);
```

New subsolvers may be added to the composite solver with

```
SNESCompositeAddSNES(SNES,SNESType);
```

and accessed with

```
SNESCompositeGetSNES(SNES,PetscInt,SNES *);
```

```{eval-rst}
.. bibliography:: /petsc.bib
   :filter: docname in docnames
```
