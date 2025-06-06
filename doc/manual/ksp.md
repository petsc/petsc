(ch_ksp)=

# KSP: Linear System Solvers

The `KSP` object is the heart of PETSc, because it provides uniform
and efficient access to all of the package’s linear system solvers,
including parallel and sequential, direct and iterative. `KSP` is
intended for solving systems of the form

$$
A x = b,
$$ (eq_axeqb)

where $A$ denotes the matrix representation of a linear operator,
$b$ is the right-hand-side vector, and $x$ is the solution
vector. `KSP` uses the same calling sequence for both direct and
iterative solution of a linear system. In addition, particular solution
techniques and their associated options can be selected at runtime.

`KSP` can also be used to solve least squares problems, using, for example, `KSPLSQR`. See
`PETSCREGRESSORLINEAR` for tools focusing on linear regression.

The combination of a Krylov subspace method and a preconditioner is at
the center of most modern numerical codes for the iterative solution of
linear systems. Many textbooks (e.g. {cite}`fgn` {cite}`vandervorst2003`, or {cite}`saad2003`) provide an
overview of the theory of such methods.
The `KSP` package, discussed in
{any}`sec_ksp`, provides many popular Krylov subspace
iterative methods; the `PC` module, described in
{any}`sec_pc`, includes a variety of preconditioners.

(sec_usingksp)=

## Using KSP

To solve a linear system with `KSP`, one must first create a solver
context with the command

```
KSPCreate(MPI_Comm comm,KSP *ksp);
```

Here `comm` is the MPI communicator and `ksp` is the newly formed
solver context. Before actually solving a linear system with `KSP`,
the user must call the following routine to set the matrices associated
with the linear system:

```
KSPSetOperators(KSP ksp,Mat Amat,Mat Pmat);
```

The argument `Amat`, representing the matrix that defines the linear
system, is a symbolic placeholder for any kind of matrix or operator. In
particular, `KSP` *does* support matrix-free methods. The routine
`MatCreateShell()` in {any}`sec_matrixfree`
provides further information regarding matrix-free methods. Typically,
the matrix from which the preconditioner is to be constructed, `Pmat`,
is the same as the matrix that defines the linear system, `Amat`;
however, occasionally these matrices differ (for instance, when a
matrix used to compute the preconditioner is obtained from a lower order method than that
employed to form the linear system matrix).

Much of the power of `KSP` can be accessed through the single routine

```
KSPSetFromOptions(KSP ksp);
```

This routine accepts the option `-help` as well as any of
the `KSP` and `PC` options discussed below. To solve a linear
system, one sets the right hand size and solution vectors using the
command

```
KSPSolve(KSP ksp,Vec b,Vec x);
```

where `b` and `x` respectively denote the right-hand side and
solution vectors. On return, the iteration number at which the iterative
process stopped can be obtained using

```
KSPGetIterationNumber(KSP ksp, PetscInt *its);
```

Note that this does not state that the method converged at this
iteration: it can also have reached the maximum number of iterations, or
have diverged.

{any}`sec_convergencetests` gives more details
regarding convergence testing. Note that multiple linear solves can be
performed by the same `KSP` context. Once the `KSP` context is no
longer needed, it should be destroyed with the command

```
KSPDestroy(KSP *ksp);
```

The above procedure is sufficient for general use of the `KSP`
package. One additional step is required for users who wish to customize
certain preconditioners (e.g., see {any}`sec_bjacobi`) or
to log certain performance data using the PETSc profiling facilities (as
discussed in {any}`ch_profiling`). In this case, the user can
optionally explicitly call

```
KSPSetUp(KSP ksp);
```

before calling `KSPSolve()` to perform any setup required for the
linear solvers. The explicit call of this routine enables the separate
profiling of any computations performed during the set up phase, such
as incomplete factorization for the ILU preconditioner.

The default solver within `KSP` is restarted GMRES, `KSPGMRES`, preconditioned for
the uniprocess case with ILU(0), and for the multiprocess case with the
block Jacobi method (with one block per process, each of which is solved
with ILU(0)). A variety of other solvers and options are also available.
To allow application programmers to set any of the preconditioner or
Krylov subspace options directly within the code, we provide routines
that extract the `PC` and `KSP` contexts,

```
KSPGetPC(KSP ksp,PC *pc);
```

The application programmer can then directly call any of the `PC` or
`KSP` routines to modify the corresponding default options.

To solve a linear system with a direct solver (supported by
PETSc for sequential matrices, and by several external solvers through
PETSc interfaces, see {any}`sec_externalsol`) one may use
the options `-ksp_type` `preonly` (or the equivalent `-ksp_type` `none`)
`-pc_type` `lu` or `-pc_type` `cholesky` (see below).

By default, if a direct solver is used, the factorization is *not* done
in-place. This approach prevents the user from the unexpected surprise
of having a corrupted matrix after a linear solve. The routine
`PCFactorSetUseInPlace()`, discussed below, causes factorization to be
done in-place.

## Solving Successive Linear Systems

When solving multiple linear systems of the same size with the same
method, several options are available. To solve successive linear
systems having the *same* matrix from which to construct the preconditioner (i.e., the same data
structure with exactly the same matrix elements) but different
right-hand-side vectors, the user should simply call `KSPSolve()`
multiple times. The preconditioner setup operations (e.g., factorization
for ILU) will be done during the first call to `KSPSolve()` only; such
operations will *not* be repeated for successive solves.

To solve successive linear systems that have *different* matrix values, because you
have changed the matrix values in the `Mat` objects you passed to `KSPSetOperators()`,
still simply call `KPSSolve()`. In this case the preconditioner will be recomputed
automatically. Use the option `-ksp_reuse_preconditioner true`, or call
`KSPSetReusePreconditioner()`, to reuse the previously computed preconditioner.
For many problems, if the matrix changes values only slightly, reusing the
old preconditioner can be more efficient.

If you wish to reuse the `KSP` with a different sized matrix and vectors, you must
call `KSPReset()` before calling `KSPSetOperators()` with the new matrix.

(sec_ksp)=

## Krylov Methods

The Krylov subspace methods accept a number of options, many of which
are discussed below. First, to set the Krylov subspace method that is to
be used, one calls the command

```
KSPSetType(KSP ksp,KSPType method);
```

The type can be one of `KSPRICHARDSON`, `KSPCHEBYSHEV`, `KSPCG`,
`KSPGMRES`, `KSPTCQMR`, `KSPBCGS`, `KSPCGS`, `KSPTFQMR`,
`KSPCR`, `KSPLSQR`, `KSPBICG`, `KSPPREONLY` (or the equivalent `KSPNONE`), or others; see
{any}`tab-kspdefaults` or the `KSPType` man page for more.
The `KSP` method can also be set with the options database command
`-ksp_type`, followed by one of the options `richardson`,
`chebyshev`, `cg`, `gmres`, `tcqmr`, `bcgs`, `cgs`,
`tfqmr`, `cr`, `lsqr`, `bicg`, `preonly` (or the equivalent `none`), or others (see
{any}`tab-kspdefaults` or the `KSPType` man page). There are
method-specific options. For instance, for the Richardson, Chebyshev, and
GMRES methods:

```
KSPRichardsonSetScale(KSP ksp,PetscReal scale);
KSPChebyshevSetEigenvalues(KSP ksp,PetscReal emax,PetscReal emin);
KSPGMRESSetRestart(KSP ksp,PetscInt max_steps);
```

The default parameter values are
`scale=1.0, emax=0.01, emin=100.0`, and `max_steps=30`. The
GMRES restart and Richardson damping factor can also be set with the
options `-ksp_gmres_restart <n>` and
`-ksp_richardson_scale <factor>`.

The default technique for orthogonalization of the Krylov vectors in
GMRES is the unmodified (classical) Gram-Schmidt method, which can be
set with

```
KSPGMRESSetOrthogonalization(KSP ksp,KSPGMRESClassicalGramSchmidtOrthogonalization);
```

or the options database command `-ksp_gmres_classicalgramschmidt`. By
default this will *not* use iterative refinement to improve the
stability of the orthogonalization. This can be changed with the option

```
KSPGMRESSetCGSRefinementType(KSP ksp,KSPGMRESCGSRefinementType type)
```

or via the options database with

```
-ksp_gmres_cgs_refinement_type <refine_never,refine_ifneeded,refine_always>
```

The values for `KSPGMRESCGSRefinementType()` are
`KSP_GMRES_CGS_REFINE_NEVER`, `KSP_GMRES_CGS_REFINE_IFNEEDED`
and `KSP_GMRES_CGS_REFINE_ALWAYS`.

One can also use modified Gram-Schmidt, by using the orthogonalization
routine `KSPGMRESModifiedGramSchmidtOrthogonalization()` or by using
the command line option `-ksp_gmres_modifiedgramschmidt`.

For the conjugate gradient method with complex numbers, there are two
slightly different algorithms depending on whether the matrix is
Hermitian symmetric or truly symmetric (the default is to assume that it
is Hermitian symmetric). To indicate that it is symmetric, one uses the
command

```
KSPCGSetType(ksp,KSP_CG_SYMMETRIC);
```

Note that this option is not valid for all matrices.

Some `KSP` types do not support preconditioning. For instance,
the CGLS algorithm does not involve a preconditioner; any preconditioner
set to work with the `KSP` object is ignored if `KSPCGLS` was
selected.

By default, `KSP` assumes an initial guess of zero by zeroing the
initial value for the solution vector that is given; this zeroing is
done at the call to `KSPSolve()`. To use a nonzero initial guess, the
user *must* call

```
KSPSetInitialGuessNonzero(KSP ksp,PetscBool flg);
```

(sec_ksppc)=

### Preconditioning within KSP

Since the rate of convergence of Krylov projection methods for a
particular linear system is strongly dependent on its spectrum,
preconditioning is typically used to alter the spectrum and hence
accelerate the convergence rate of iterative techniques. Preconditioning
can be applied to the system {eq}`eq_axeqb` by

$$
(M_L^{-1} A M_R^{-1}) \, (M_R x) = M_L^{-1} b,
$$ (eq_prec)

where $M_L$ and $M_R$ indicate preconditioning matrices (or,
matrices from which the preconditioner is to be constructed). If
$M_L = I$ in {eq}`eq_prec`, right preconditioning
results, and the residual of {eq}`eq_axeqb`,

$$
r \equiv b - Ax = b - A M_R^{-1} \, M_R x,
$$

is preserved. In contrast, the residual is altered for left
($M_R = I$) and symmetric preconditioning, as given by

$$
r_L \equiv M_L^{-1} b - M_L^{-1} A x = M_L^{-1} r.
$$

By default, most KSP implementations use left preconditioning. Some more
naturally use other options, though. For instance, `KSPQCG` defaults
to use symmetric preconditioning and `KSPFGMRES` uses right
preconditioning by default. Right preconditioning can be activated for
some methods by using the options database command
`-ksp_pc_side right` or calling the routine

```
KSPSetPCSide(ksp,PC_RIGHT);
```

Attempting to use right preconditioning for a method that does not
currently support it results in an error message of the form

```none
KSPSetUp_Richardson:No right preconditioning for KSPRICHARDSON
```

```{eval-rst}
.. list-table:: KSP Objects
  :name: tab-kspdefaults
  :header-rows: 1

  * - Method
    - KSPType
    - Options Database
  * - Richardson
    - ``KSPRICHARDSON``
    - ``richardson``
  * - Chebyshev
    - ``KSPCHEBYSHEV``
    - ``chebyshev``
  * - Conjugate Gradient :cite:`hs:52`
    - ``KSPCG``
    - ``cg``
  * - Pipelined Conjugate Gradients :cite:`ghyselsvanroose2014`
    - ``KSPPIPECG``
    - ``pipecg``
  * - Pipelined Conjugate Gradients (Gropp)
    - ``KSPGROPPCG``
    - ``groppcg``
  * - Pipelined Conjugate Gradients with Residual Replacement
    - ``KSPPIPECGRR``
    - ``pipecgrr``
  * - Conjugate Gradients for the Normal Equations
    - ``KSPCGNE``
    - ``cgne``
  * - Flexible Conjugate Gradients :cite:`flexiblecg`
    - ``KSPFCG``
    - ``fcg``
  * -  Pipelined, Flexible Conjugate Gradients :cite:`sananschneppmay2016`
    - ``KSPPIPEFCG``
    - ``pipefcg``
  * - Conjugate Gradients for Least Squares
    - ``KSPCGLS``
    - ``cgls``
  * - Conjugate Gradients with Constraint (1)
    - ``KSPNASH``
    - ``nash``
  * - Conjugate Gradients with Constraint (2)
    - ``KSPSTCG``
    - ``stcg``
  * - Conjugate Gradients with Constraint (3)
    - ``KSPGLTR``
    - ``gltr``
  * - Conjugate Gradients with Constraint (4)
    - ``KSPQCG``
    - ``qcg``
  * - BiConjugate Gradient
    - ``KSPBICG``
    - ``bicg``
  * - BiCGSTAB :cite:`v:92`
    - ``KSPBCGS``
    - ``bcgs``
  * - Improved BiCGSTAB
    - ``KSPIBCGS``
    - ``ibcgs``
  * - QMRCGSTAB :cite:`chan1994qmrcgs`
    - ``KSPQMRCGS``
    - ``qmrcgs``
  * - Flexible BiCGSTAB
    - ``KSPFBCGS``
    - ``fbcgs``
  * - Flexible BiCGSTAB (variant)
    - ``KSPFBCGSR``
    - ``fbcgsr``
  * - Enhanced BiCGSTAB(L)
    - ``KSPBCGSL``
    - ``bcgsl``
  * - Minimal Residual Method :cite:`paige.saunders:solution`
    - ``KSPMINRES``
    - ``minres``
  * - Generalized Minimal Residual :cite:`saad.schultz:gmres`
    - ``KSPGMRES``
    - ``gmres``
  * - Flexible Generalized Minimal Residual :cite:`saad1993`
    - ``KSPFGMRES``
    - ``fgmres``
  * - Deflated Generalized Minimal Residual
    - ``KSPDGMRES``
    - ``dgmres``
  * - Pipelined Generalized Minimal Residual :cite:`ghyselsashbymeerbergenvanroose2013`
    - ``KSPPGMRES``
    - ``pgmres``
  * - Pipelined, Flexible Generalized Minimal Residual :cite:`sananschneppmay2016`
    - ``KSPPIPEFGMRES``
    - ``pipefgmres``
  * - Generalized Minimal Residual with Accelerated Restart
    - ``KSPLGMRES``
    - ``lgmres``
  * - Conjugate Residual :cite:`eisenstat1983variational`
    - ``KSPCR``
    - ``cr``
  * - Generalized Conjugate Residual
    - ``KSPGCR``
    - ``gcr``
  * - Pipelined Conjugate Residual
    - ``KSPPIPECR``
    - ``pipecr``
  * - Pipelined, Flexible Conjugate Residual :cite:`sananschneppmay2016`
    - ``KSPPIPEGCR``
    - ``pipegcr``
  * - FETI-DP
    - ``KSPFETIDP``
    - ``fetidp``
  * - Conjugate Gradient Squared :cite:`so:89`
    - ``KSPCGS``
    - ``cgs``
  * - Transpose-Free Quasi-Minimal Residual (1) :cite:`f:93`
    - ``KSPTFQMR``
    - ``tfqmr``
  * - Transpose-Free Quasi-Minimal Residual (2)
    - ``KSPTCQMR``
    - ``tcqmr``
  * - Least Squares Method
    - ``KSPLSQR``
    - ``lsqr``
  * - Symmetric LQ Method :cite:`paige.saunders:solution`
    - ``KSPSYMMLQ``
    - ``symmlq``
  * - TSIRM
    - ``KSPTSIRM``
    - ``tsirm``
  * - Python Shell
    - ``KSPPYTHON``
    - ``python``
  * - Shell for no ``KSP`` method
    - ``KSPNONE``
    - ``none``

```

Note: the bi-conjugate gradient method requires application of both the
matrix and its transpose plus the preconditioner and its transpose.
Currently not all matrices and preconditioners provide this support and
thus the `KSPBICG` cannot always be used.

Note: PETSc implements the FETI-DP (Finite Element Tearing and
Interconnecting Dual-Primal) method as an implementation of `KSP` since it recasts the
original problem into a constrained minimization one with Lagrange
multipliers. The only matrix type supported is `MATIS`. Support for
saddle point problems is provided. See the man page for `KSPFETIDP` for
further details.

(sec_convergencetests)=

### Convergence Tests

The default convergence test, `KSPConvergedDefault()`, uses the \$ l_2 \$ norm of the preconditioned \$ B(b - A x) \$ or unconditioned residual \$ b - Ax\$, depending on the `KSPType` and the value of `KSPNormType` set with `KSPSetNormType`. For `KSPCG` and `KSPGMRES` the default is the norm of the preconditioned residual.
The preconditioned residual is used by default for
convergence testing of all left-preconditioned `KSP` methods. For the
conjugate gradient, Richardson, and Chebyshev methods the true residual
can be used by the options database command
`-ksp_norm_type unpreconditioned` or by calling the routine

```
KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
```

`KSPCG` also supports using the natural norm induced by the symmetric positive-definite
matrix that defines the linear system with the options database command `-ksp_norm_type natural` or by calling the routine

```
KSPSetNormType(ksp, KSP_NORM_NATURAL);
```

Convergence (or divergence) is decided
by three quantities: the decrease of the residual norm relative to the
norm of the right-hand side, `rtol`, the absolute size of the residual
norm, `atol`, and the relative increase in the residual, `dtol`.
Convergence is detected at iteration $k$ if

$$
\| r_k \|_2 < {\rm max} ( \text{rtol} * \| b \|_2, \text{atol}),
$$

where $r_k = b - A x_k$. Divergence is detected if

$$
\| r_k \|_2 > \text{dtol} * \| b \|_2.
$$

These parameters, as well as the maximum number of allowable iterations,
can be set with the routine

```
KSPSetTolerances(KSP ksp,PetscReal rtol,PetscReal atol,PetscReal dtol,PetscInt maxits);
```

The user can retain the current value of any of these parameters by
specifying `PETSC_CURRENT` as the corresponding tolerance; the
defaults are `rtol=1e-5`, `atol=1e-50`, `dtol=1e5`, and
`maxits=1e4`. Using `PETSC_DETERMINE` will set the parameters back to their
initial values when the object's type was set. These parameters can also be set from the options
database with the commands `-ksp_rtol` `<rtol>`, `-ksp_atol`
`<atol>`, `-ksp_divtol` `<dtol>`, and `-ksp_max_it` `<its>`.

In addition to providing an interface to a simple convergence test,
`KSP` allows the application programmer the flexibility to provide
customized convergence-testing routines. The user can specify a
customized routine with the command

```
KSPSetConvergenceTest(KSP ksp,PetscErrorCode (*test)(KSP ksp,PetscInt it,PetscReal rnorm, KSPConvergedReason *reason,void *ctx),void *ctx,PetscErrorCode (*destroy)(void *ctx));
```

The final routine argument, `ctx`, is an optional context for private
data for the user-defined convergence routine, `test`. Other `test`
routine arguments are the iteration number, `it`, and the residual’s
norm, `rnorm`. The routine for detecting convergence,
`test`, should set `reason` to positive for convergence, 0 for no
convergence, and negative for failure to converge. A full list of
possible values is given in the `KSPConvergedReason` manual page.
You can use `KSPGetConvergedReason()` after
`KSPSolve()` to see why convergence/divergence was detected.

(sec_kspmonitor)=

### Convergence Monitoring

By default, the Krylov solvers, `KSPSolve()`, run silently without displaying
information about the iterations. The user can indicate that the norms
of the residuals should be displayed at each iteration by using `-ksp_monitor` with
the options database. To display the residual norms in a graphical
window (running under X Windows), one should use
`-ksp_monitor draw::draw_lg`. Application programmers can also
provide their own routines to perform the monitoring by using the
command

```
KSPMonitorSet(KSP ksp, PetscErrorCode (*mon)(KSP ksp, PetscInt it, PetscReal rnorm, void *ctx), void *ctx, (PetscCtxDestroyFn *)mondestroy);
```

The final routine argument, `ctx`, is an optional context for private
data for the user-defined monitoring routine, `mon`. Other `mon`
routine arguments are the iteration number (`it`) and the residual’s
norm (`rnorm`), as discussed above in {any}`sec_convergencetests`.
A helpful routine within user-defined
monitors is `PetscObjectGetComm((PetscObject)ksp,MPI_Comm *comm)`,
which returns in `comm` the MPI communicator for the `KSP` context.
See {any}`sec_writing` for more discussion of the use of
MPI communicators within PETSc.

Many monitoring routines are supplied with PETSc, including

```
KSPMonitorResidual(KSP,PetscInt,PetscReal, void *);
KSPMonitorSingularValue(KSP,PetscInt,PetscReal,void *);
KSPMonitorTrueResidual(KSP,PetscInt,PetscReal, void *);
```

The default monitor simply prints an estimate of a norm of
the residual at each iteration. The routine
`KSPMonitorSingularValue()` is appropriate only for use with the
conjugate gradient method or GMRES, since it prints estimates of the
extreme singular values of the preconditioned operator at each
iteration computed via the Lanczos or Arnoldi algorithms.

Since `KSPMonitorTrueResidual()` prints the true
residual at each iteration by actually computing the residual using the
formula $r = b - Ax$, the routine is slow and should be used only
for testing or convergence studies, not for timing. These `KSPSolve()` monitors may
be accessed with the command line options `-ksp_monitor`,
`-ksp_monitor_singular_value`, and `-ksp_monitor_true_residual`.

To employ the default graphical monitor, one should use the command
`-ksp_monitor draw::draw_lg`.

One can cancel hardwired monitoring routines for KSP at runtime with
`-ksp_monitor_cancel`.

### Understanding the Operator’s Spectrum

Since the convergence of Krylov subspace methods depends strongly on the
spectrum (eigenvalues) of the preconditioned operator, PETSc has
specific routines for eigenvalue approximation via the Arnoldi or
Lanczos iteration. First, before the linear solve one must call

```
KSPSetComputeEigenvalues(ksp,PETSC_TRUE);
```

Then after the `KSP` solve one calls

```
KSPComputeEigenvalues(KSP ksp,PetscInt n,PetscReal *realpart,PetscReal *complexpart,PetscInt *neig);
```

Here, `n` is the size of the two arrays and the eigenvalues are
inserted into those two arrays. `neig` is the number of eigenvalues
computed; this number depends on the size of the Krylov space generated
during the linear system solution, for GMRES it is never larger than the
`restart` parameter. There is an additional routine

```
KSPComputeEigenvaluesExplicitly(KSP ksp, PetscInt n,PetscReal *realpart,PetscReal *complexpart);
```

that is useful only for very small problems. It explicitly computes the
full representation of the preconditioned operator and calls LAPACK to
compute its eigenvalues. It should be only used for matrices of size up
to a couple hundred. The `PetscDrawSP*()` routines are very useful for
drawing scatter plots of the eigenvalues.

The eigenvalues may also be computed and displayed graphically with the
options data base commands `-ksp_view_eigenvalues draw` and
`-ksp_view_eigenvalues_explicit draw`. Or they can be dumped to the
screen in ASCII text via `-ksp_view_eigenvalues` and
`-ksp_view_eigenvalues_explicit`.

(sec_flexibleksp)=

### Flexible Krylov Methods

Standard Krylov methods require that the preconditioner be a linear operator, thus, for example, a standard `KSP` method
cannot use a `KSP` in its preconditioner, as is common in the Block-Jacobi method `PCBJACOBI`, for example.
Flexible Krylov methods are a subset of methods that allow (with modest additional requirements
on memory) the preconditioner to be nonlinear. For example, they can be used with the `PCKSP` preconditioner.
The flexible `KSP` methods have the label "Flexible" in {any}`tab-kspdefaults`.

One can use `KSPMonitorDynamicTolerance()` to control the tolerances used by inner `KSP` solvers in `PCKSP`, `PCBJACOBI`, and `PCDEFLATION`.

In addition to supporting `PCKSP`, the flexible methods support `KSPFlexibleSetModifyPC()` to
allow the user to provide a callback function that changes the preconditioner at each Krylov iteration. Its calling sequence is as follows.

```
PetscErrorCode f(KSP ksp,PetscInt total_its,PetscInt its_since_restart,PetscReal res_norm,void *ctx);
```

(sec_pipelineksp)=

### Pipelined Krylov Methods

Standard Krylov methods have one or more global reductions resulting from the computations of inner products or norms in each iteration.
These reductions need to block until all MPI processes have received the results. For a large number of MPI processes (this number is machine dependent
but can be above 10,000 processes) this synchronization is very time consuming and can significantly slow the computation. Pipelined Krylov
methods overlap the reduction operations with local computations (generally the application of the matrix-vector products and precondtiioners)
thus effectively "hiding" the time of the reductions. In addition, they may reduce the number of global synchronizations by rearranging the
computations in a way that some of them can be collapsed, e.g., two or more calls to `MPI_Allreduce()` may be combined into one call.
The pipeline `KSP` methods have the label "Pipeline" in {any}`tab-kspdefaults`.

Special configuration of MPI may be necessary for reductions to make asynchronous progress, which is important for
performance of pipelined methods. See {any}`doc_faq_pipelined` for details.

### Other KSP Options

To obtain the solution vector and right-hand side from a `KSP`
context, one uses

```
KSPGetSolution(KSP ksp,Vec *x);
KSPGetRhs(KSP ksp,Vec *rhs);
```

During the iterative process the solution may not yet have been
calculated or it may be stored in a different location. To access the
approximate solution during the iterative process, one uses the command

```
KSPBuildSolution(KSP ksp,Vec w,Vec *v);
```

where the solution is returned in `v`. The user can optionally provide
a vector in `w` as the location to store the vector; however, if `w`
is `NULL`, space allocated by PETSc in the `KSP` context is used.
One should not destroy this vector. For certain `KSP` methods (e.g.,
GMRES), the construction of the solution is expensive, while for many
others it doesn’t even require a vector copy.

Access to the residual is done in a similar way with the command

```
KSPBuildResidual(KSP ksp,Vec t,Vec w,Vec *v);
```

Again, for GMRES and certain other methods this is an expensive
operation.

(sec_pc)=

## Preconditioners

As discussed in {any}`sec_ksppc`, Krylov subspace methods
are typically used in conjunction with a preconditioner. To employ a
particular preconditioning method, the user can either select it from
the options database using input of the form `-pc_type <methodname>`
or set the method with the command

```
PCSetType(PC pc,PCType method);
```

In {any}`tab-pcdefaults` we summarize the basic
preconditioning methods supported in PETSc. See the `PCType` manual
page for a complete list.

The `PCSHELL` preconditioner allows users to provide their own
specific, application-provided custom preconditioner.

The direct
preconditioner, `PCLU` , is, in fact, a direct solver for the linear
system that uses LU factorization. `PCLU` is included as a
preconditioner so that PETSc has a consistent interface among direct and
iterative linear solvers.

PETSc provides several domain decomposition methods/preconditioners including
`PCASM`, `PCGASM`, `PCBDDC`, and `PCHPDDM`. In addition PETSc provides
multiple multigrid solvers/preconditioners including `PCMG`, `PCGAMG`, `PCHYPRE`,
and `PCML`. See further discussion below.

```{eval-rst}
.. list-table:: PETSc Preconditioners (partial list)
   :name: tab-pcdefaults
   :header-rows: 1

   * - Method
     - PCType
     - Options Database
   * - Jacobi
     - ``PCJACOBI``
     - ``jacobi``
   * - Block Jacobi
     - ``PCBJACOBI``
     - ``bjacobi``
   * - SOR (and SSOR)
     - ``PCSOR``
     - ``sor``
   * - SOR with Eisenstat trick
     - ``PCEISENSTAT``
     - ``eisenstat``
   * - Incomplete Cholesky
     - ``PCICC``
     - ``icc``
   * - Incomplete LU
     - ``PCILU``
     - ``ilu``
   * - Additive Schwarz
     - ``PCASM``
     - ``asm``
   * - Generalized Additive Schwarz
     - ``PCGASM``
     - ``gasm``
   * - Algebraic Multigrid
     - ``PCGAMG``
     - ``gamg``
   * - Balancing Domain Decomposition by Constraints
     - ``PCBDDC``
     - ``bddc``
   * - Linear solver
     - ``PCKSP``
     - ``ksp``
   * - Combination of preconditioners
     - ``PCCOMPOSITE``
     - ``composite``
   * - LU
     - ``PCLU``
     - ``lu``
   * - Cholesky
     - ``PCCHOLESKY``
     - ``cholesky``
   * - No preconditioning
     - ``PCNONE``
     - ``none``
   * - Shell for user-defined ``PC``
     - ``PCSHELL``
     - ``shell``
```

Each preconditioner may have associated with it a set of options, which
can be set with routines and options database commands provided for this
purpose. Such routine names and commands are all of the form
`PC<TYPE><Option>` and `-pc_<type>_<option> [value]`. A complete
list can be found by consulting the `PCType` manual page; we discuss
just a few in the sections below.

(sec_ilu_icc)=

### ILU and ICC Preconditioners

Some of the options for ILU preconditioner are

```
PCFactorSetLevels(PC pc,PetscInt levels);
PCFactorSetReuseOrdering(PC pc,PetscBool flag);
PCFactorSetDropTolerance(PC pc,PetscReal dt,PetscReal dtcol,PetscInt dtcount);
PCFactorSetReuseFill(PC pc,PetscBool flag);
PCFactorSetUseInPlace(PC pc,PetscBool flg);
PCFactorSetAllowDiagonalFill(PC pc,PetscBool flg);
```

When repeatedly solving linear systems with the same `KSP` context,
one can reuse some information computed during the first linear solve.
In particular, `PCFactorSetReuseOrdering()` causes the ordering (for
example, set with `-pc_factor_mat_ordering_type` `order`) computed
in the first factorization to be reused for later factorizations.
`PCFactorSetUseInPlace()` is often used with `PCASM` or
`PCBJACOBI` when zero fill is used, since it reuses the matrix space
to store the incomplete factorization it saves memory and copying time.
Note that in-place factorization is not appropriate with any ordering
besides natural and cannot be used with the drop tolerance
factorization. These options may be set in the database with

- `-pc_factor_levels <levels>`
- `-pc_factor_reuse_ordering`
- `-pc_factor_reuse_fill`
- `-pc_factor_in_place`
- `-pc_factor_nonzeros_along_diagonal`
- `-pc_factor_diagonal_fill`

See {any}`sec_symbolfactor` for information on
preallocation of memory for anticipated fill during factorization. By
alleviating the considerable overhead for dynamic memory allocation,
such tuning can significantly enhance performance.

PETSc supports incomplete factorization preconditioners
for several matrix types for sequential matrices (for example
`MATSEQAIJ`, `MATSEQBAIJ`, and `MATSEQSBAIJ`).

### SOR and SSOR Preconditioners

PETSc provides only a sequential SOR preconditioner; it can only be
used with sequential matrices or as the subblock preconditioner when
using block Jacobi or ASM preconditioning (see below).

The options for SOR preconditioning with `PCSOR` are

```
PCSORSetOmega(PC pc,PetscReal omega);
PCSORSetIterations(PC pc,PetscInt its,PetscInt lits);
PCSORSetSymmetric(PC pc,MatSORType type);
```

The first of these commands sets the relaxation factor for successive
over (under) relaxation. The second command sets the number of inner
iterations `its` and local iterations `lits` (the number of
smoothing sweeps on a process before doing a ghost point update from the
other processes) to use between steps of the Krylov space method. The
total number of SOR sweeps is given by `its*lits`. The third command
sets the kind of SOR sweep, where the argument `type` can be one of
`SOR_FORWARD_SWEEP`, `SOR_BACKWARD_SWEEP` or
`SOR_SYMMETRIC_SWEEP`, the default being `SOR_FORWARD_SWEEP`.
Setting the type to be `SOR_SYMMETRIC_SWEEP` produces the SSOR method.
In addition, each process can locally and independently perform the
specified variant of SOR with the types `SOR_LOCAL_FORWARD_SWEEP`,
`SOR_LOCAL_BACKWARD_SWEEP`, and `SOR_LOCAL_SYMMETRIC_SWEEP`. These
variants can also be set with the options `-pc_sor_omega <omega>`,
`-pc_sor_its <its>`, `-pc_sor_lits <lits>`, `-pc_sor_backward`,
`-pc_sor_symmetric`, `-pc_sor_local_forward`,
`-pc_sor_local_backward`, and `-pc_sor_local_symmetric`.

The Eisenstat trick {cite}`eisenstat81` for SSOR
preconditioning can be employed with the method `PCEISENSTAT`
(`-pc_type` `eisenstat`). By using both left and right
preconditioning of the linear system, this variant of SSOR requires
about half of the floating-point operations for conventional SSOR. The
option `-pc_eisenstat_no_diagonal_scaling` (or the routine
`PCEisenstatSetNoDiagonalScaling()`) turns off diagonal scaling in
conjunction with Eisenstat SSOR method, while the option
`-pc_eisenstat_omega <omega>` (or the routine
`PCEisenstatSetOmega(PC pc,PetscReal omega)`) sets the SSOR relaxation
coefficient, `omega`, as discussed above.

(sec_factorization)=

### LU Factorization

The LU preconditioner provides several options. The first, given by the
command

```
PCFactorSetUseInPlace(PC pc,PetscBool flg);
```

causes the factorization to be performed in-place and hence destroys the
original matrix. The options database variant of this command is
`-pc_factor_in_place`. Another direct preconditioner option is
selecting the ordering of equations with the command
`-pc_factor_mat_ordering_type <ordering>`. The possible orderings are

- `MATORDERINGNATURAL` - Natural
- `MATORDERINGND` - Nested Dissection
- `MATORDERING1WD` - One-way Dissection
- `MATORDERINGRCM` - Reverse Cuthill-McKee
- `MATORDERINGQMD` - Quotient Minimum Degree

These orderings can also be set through the options database by
specifying one of the following: `-pc_factor_mat_ordering_type`
`natural`, or `nd`, or `1wd`, or `rcm`, or `qmd`. In addition,
see `MatGetOrdering()`, discussed in {any}`sec_matfactor`.

The sparse LU factorization provided in PETSc does not perform pivoting
for numerical stability (since they are designed to preserve nonzero
structure), and thus occasionally an LU factorization will fail with a
zero pivot when, in fact, the matrix is non-singular. The option
`-pc_factor_nonzeros_along_diagonal <tol>` will often help eliminate
the zero pivot, by preprocessing the column ordering to remove small
values from the diagonal. Here, `tol` is an optional tolerance to
decide if a value is nonzero; by default it is `1.e-10`.

In addition, {any}`sec_symbolfactor` provides information
on preallocation of memory for anticipated fill during factorization.
Such tuning can significantly enhance performance, since it eliminates
the considerable overhead for dynamic memory allocation.

(sec_bjacobi)=

### Block Jacobi and Overlapping Additive Schwarz Preconditioners

The block Jacobi and overlapping additive Schwarz (domain decomposition) methods in PETSc are
supported in parallel; however, only the uniprocess version of the block
Gauss-Seidel method is available. By default, the PETSc
implementations of these methods employ ILU(0) factorization on each
individual block (that is, the default solver on each subblock is
`PCType=PCILU`, `KSPType=KSPPREONLY` (or equivalently `KSPType=KSPNONE`); the user can set alternative
linear solvers via the options `-sub_ksp_type` and `-sub_pc_type`.
In fact, all of the `KSP` and `PC` options can be applied to the
subproblems by inserting the prefix `-sub_` at the beginning of the
option name. These options database commands set the particular options
for *all* of the blocks within the global problem. In addition, the
routines

```
PCBJacobiGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **subksp);
PCASMGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP **subksp);
```

extract the `KSP` context for each local block. The argument
`n_local` is the number of blocks on the calling process, and
`first_local` indicates the global number of the first block on the
process. The blocks are numbered successively by processes from zero
through $b_g-1$, where $b_g$ is the number of global blocks.
The array of `KSP` contexts for the local blocks is given by
`subksp`. This mechanism enables the user to set different solvers for
the various blocks. To set the appropriate data structures, the user
*must* explicitly call `KSPSetUp()` before calling
`PCBJacobiGetSubKSP()` or `PCASMGetSubKSP(`). For further details,
see
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex7.c.html">KSP Tutorial ex7</a>
or
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex8.c.html">KSP Tutorial ex8</a>.

The block Jacobi, block Gauss-Seidel, and additive Schwarz
preconditioners allow the user to set the number of blocks into which
the problem is divided. The options database commands to set this value
are `-pc_bjacobi_blocks` `n` and `-pc_bgs_blocks` `n`, and,
within a program, the corresponding routines are

```
PCBJacobiSetTotalBlocks(PC pc,PetscInt blocks,PetscInt *size);
PCASMSetTotalSubdomains(PC pc,PetscInt n,IS *is,IS *islocal);
PCASMSetType(PC pc,PCASMType type);
```

The optional argument `size` is an array indicating the size of each
block. Currently, for certain parallel matrix formats, only a single
block per process is supported. However, the `MATMPIAIJ` and
`MATMPIBAIJ` formats support the use of general blocks as long as no
blocks are shared among processes. The `is` argument contains the
index sets that define the subdomains.

The object `PCASMType` is one of `PC_ASM_BASIC`,
`PC_ASM_INTERPOLATE`, `PC_ASM_RESTRICT`, or `PC_ASM_NONE` and may
also be set with the options database `-pc_asm_type` `[basic`,
`interpolate`, `restrict`, `none]`. The type `PC_ASM_BASIC` (or
`-pc_asm_type` `basic`) corresponds to the standard additive Schwarz
method that uses the full restriction and interpolation operators. The
type `PC_ASM_RESTRICT` (or `-pc_asm_type` `restrict`) uses a full
restriction operator, but during the interpolation process ignores the
off-process values. Similarly, `PC_ASM_INTERPOLATE` (or
`-pc_asm_type` `interpolate`) uses a limited restriction process in
conjunction with a full interpolation, while `PC_ASM_NONE` (or
`-pc_asm_type` `none`) ignores off-process values for both
restriction and interpolation. The ASM types with limited restriction or
interpolation were suggested by Xiao-Chuan Cai and Marcus Sarkis
{cite}`cs99`. `PC_ASM_RESTRICT` is the PETSc default, as
it saves substantial communication and for many problems has the added
benefit of requiring fewer iterations for convergence than the standard
additive Schwarz method.

The user can also set the number of blocks and sizes on a per-process
basis with the commands

```
PCBJacobiSetLocalBlocks(PC pc,PetscInt blocks,PetscInt *size);
PCASMSetLocalSubdomains(PC pc,PetscInt N,IS *is,IS *islocal);
```

For the ASM preconditioner one can use the following command to set the
overlap to compute in constructing the subdomains.

```
PCASMSetOverlap(PC pc,PetscInt overlap);
```

The overlap defaults to 1, so if one desires that no additional overlap
be computed beyond what may have been set with a call to
`PCASMSetTotalSubdomains()` or `PCASMSetLocalSubdomains()`, then
`overlap` must be set to be 0. In particular, if one does *not*
explicitly set the subdomains in an application code, then all overlap
would be computed internally by PETSc, and using an overlap of 0 would
result in an ASM variant that is equivalent to the block Jacobi
preconditioner. Note that one can define initial index sets `is` with
*any* overlap via `PCASMSetTotalSubdomains()` or
`PCASMSetLocalSubdomains()`; the routine `PCASMSetOverlap()` merely
allows PETSc to extend that overlap further if desired.

`PCGASM` is a generalization of `PCASM` that allows
the user to specify subdomains that span multiple MPI processes. This can be
useful for problems where small subdomains result in poor convergence.
To be effective, the multi-processor subproblems must be solved using a
sufficiently strong subsolver, such as `PCLU`, for which `SuperLU_DIST` or a
similar parallel direct solver could be used; other choices may include
a multigrid solver on the subdomains.

The interface for `PCGASM` is similar to that of `PCASM`. In
particular, `PCGASMType` is one of `PC_GASM_BASIC`,
`PC_GASM_INTERPOLATE`, `PC_GASM_RESTRICT`, `PC_GASM_NONE`. These
options have the same meaning as with `PCASM` and may also be set with
the options database `-pc_gasm_type` `[basic`, `interpolate`,
`restrict`, `none]`.

Unlike `PCASM`, however, `PCGASM` allows the user to define
subdomains that span multiple MPI processes. The simplest way to do this is
using a call to `PCGASMSetTotalSubdomains(PC pc,PetscInt N)` with
the total number of subdomains `N` that is smaller than the MPI
communicator `size`. In this case `PCGASM` will coalesce `size/N`
consecutive single-rank subdomains into a single multi-rank subdomain.
The single-rank subdomains contain the degrees of freedom corresponding
to the locally-owned rows of the `PCGASM` matrix used to compute the preconditioner –
these are the subdomains `PCASM` and `PCGASM` use by default.

Each of the multirank subdomain subproblems is defined on the
subcommunicator that contains the coalesced `PCGASM` processes. In general
this might not result in a very good subproblem if the single-rank
problems corresponding to the coalesced processes are not very strongly
connected. In the future this will be addressed with a hierarchical
partitioner that generates well-connected coarse subdomains first before
subpartitioning them into the single-rank subdomains.

In the meantime the user can provide his or her own multi-rank
subdomains by calling `PCGASMSetSubdomains(PC,IS[],IS[])` where each
of the `IS` objects on the list defines the inner (without the
overlap) or the outer (including the overlap) subdomain on the
subcommunicator of the `IS` object. A helper subroutine
`PCGASMCreateSubdomains2D()` is similar to PCASM’s but is capable of
constructing multi-rank subdomains that can be then used with
`PCGASMSetSubdomains()`. An alternative way of creating multi-rank
subdomains is by using the underlying `DM` object, if it is capable of
generating such decompositions via `DMCreateDomainDecomposition()`.
Ordinarily the decomposition specified by the user via
`PCGASMSetSubdomains()` takes precedence, unless
`PCGASMSetUseDMSubdomains()` instructs `PCGASM` to prefer
`DM`-created decompositions.

Currently there is no support for increasing the overlap of multi-rank
subdomains via `PCGASMSetOverlap()` – this functionality works only
for subdomains that fit within a single MPI process, exactly as in
`PCASM`.

Examples of the described `PCGASM` usage can be found in
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex62.c.html">KSP Tutorial ex62</a>.
In particular, `runex62_superlu_dist` illustrates the use of
`SuperLU_DIST` as the subdomain solver on coalesced multi-rank
subdomains. The `runex62_2D_*` examples illustrate the use of
`PCGASMCreateSubdomains2D()`.

(sec_amg)=

### Algebraic Multigrid (AMG) Preconditioners

PETSc has a native algebraic multigrid preconditioner `PCGAMG` –
*gamg* – and interfaces to three external AMG packages: *hypre*, *ML*
and *AMGx* (CUDA platforms only) that can be downloaded in the
configuration phase (e.g., `--download-hypre` ) and used by
specifying that command line parameter (e.g., `-pc_type hypre`).
*Hypre* is relatively monolithic in that a PETSc matrix is converted into a hypre
matrix, and then *hypre* is called to solve the entire problem. *ML* is more
modular because PETSc only has *ML* generate the coarse grid spaces
(columns of the prolongation operator), which is the core of an AMG method,
and then constructs a `PCMG` with Galerkin coarse grid operator
construction. `PCGAMG` is designed from the beginning to be modular, to
allow for new components to be added easily and also populates a
multigrid preconditioner `PCMG` so generic multigrid parameters are
used (see {any}`sec_mg`). PETSc provides a fully supported (smoothed) aggregation AMG, but supports the addition of new methods
(`-pc_type gamg -pc_gamg_type agg` or `PCSetType(pc,PCGAMG)` and
`PCGAMGSetType(pc, PCGAMGAGG)`. Examples of extension are reference implementations of
a classical AMG method (`-pc_gamg_type classical`), a (2D) hybrid geometric
AMG method (`-pc_gamg_type geo`) that are not supported. A 2.5D AMG method DofColumns
{cite}`isaacstadlerghattas2015` supports 2D coarsenings extruded in the third dimension. `PCGAMG` does require the use
of `MATAIJ` matrices. For instance, `MATBAIJ` matrices are not supported. One
can use `MATAIJ` instead of `MATBAIJ` without changing any code other than the
constructor (or the `-mat_type` from the command line). For instance,
`MatSetValuesBlocked` works with `MATAIJ` matrices.

**Important parameters for PCGAMGAGG**

- Control the generation of the coarse grid

  > - `-pc_gamg_aggressive_coarsening` \<n:int:1> Use aggressive coarsening on the finest n levels to construct the coarser mesh.
  >   See `PCGAMGAGGSetNSmooths()`. The larger value produces a faster preconditioner to create and solve, but the convergence may be slower.
  > - `-pc_gamg_low_memory_threshold_filter` \<bool:false> Filter small matrix entries before coarsening the mesh.
  >   See `PCGAMGSetLowMemoryFilter()`.
  > - `-pc_gamg_threshold` \<tol:real:0.0> The threshold of small values to drop when `-pc_gamg_low_memory_threshold_filter` is used. A
  >   negative value means keeping even the locations with 0.0. See `PCGAMGSetThreshold()`
  > - `-pc_gamg_threshold_scale` \<v>:real:1.0> Set a scale factor applied to each coarser level when `-pc_gamg_low_memory_threshold_filter` is used.
  >   See `PCGAMGSetThresholdScale()`.
  > - `-pc_gamg_mat_coarsen_type` \<mis|hem|misk:misk> Algorithm used to coarsen the matrix graph. See `MatCoarsenSetType()`.
  > - `-pc_gamg_mat_coarsen_max_it` \<it:int:4> Maximum HEM iterations to use. See `MatCoarsenSetMaximumIterations()`.
  > - `-pc_gamg_aggressive_mis_k` \<k:int:2> k distance in MIS coarsening (>2 is 'aggressive') to use in coarsening.
  >   See `PCGAMGMISkSetAggressive()`. The larger value produces a preconditioner that is faster to create and solve with but the convergence may be slower.
  >   This option and the previous option work to determine how aggressively the grids are coarsened.
  > - `-pc_gamg_mis_k_minimum_degree_ordering` \<bool:true> Use a minimum degree ordering in the greedy MIS algorithm used to coarsen.
  >   See `PCGAMGMISkSetMinDegreeOrdering()`

- Control the generation of the prolongation for `PCGAMGAGG`

  > - `-pc_gamg_agg_nsmooths` \<n:int:1> Number of smoothing steps to be used in constructing the prolongation. For symmetric problems,
  >   generally, one or more is best. For some strongly nonsymmetric problems, 0 may be best. See `PCGAMGSetNSmooths()`.

- Control the amount of parallelism on the levels

  > - `-pc_gamg_process_eq_limit` \<n:int:50> Sets the minimum number of equations allowed per process when coarsening (otherwise, fewer MPI processes
  >   are used for the coarser mesh). A larger value will cause the coarser problems to be run on fewer MPI processes, resulting
  >   in less communication and possibly a faster time to solution. See `PCGAMGSetProcEqLim()`.
  >
  > - `-pc_gamg_rank_reduction_factors` \<rn,rn-1,...,r1:int> Set a schedule for MPI rank reduction on coarse grids. `See PCGAMGSetRankReductionFactors()`
  >   This overrides the lessening of processes that would arise from `-pc_gamg_process_eq_limit`.
  >
  > - `-pc_gamg_repartition` \<bool:false> Run a partitioner on each coarser mesh generated rather than using the default partition arising from the
  >   finer mesh. See `PCGAMGSetRepartition()`. This increases the preconditioner setup time but will result in less time per
  >   iteration of the solver.
  >
  > - `-pc_gamg_parallel_coarse_grid_solver` \<bool:false> Allow the coarse grid solve to run in parallel, depending on the value of `-pc_gamg_coarse_eq_limit`.
  >   See `PCGAMGSetParallelCoarseGridSolve()`. If the coarse grid problem is large then this can
  >   improve the time to solution.
  >
  >   - `-pc_gamg_coarse_eq_limit` \<n:int:50> Sets the minimum number of equations allowed per process on the coarsest level when coarsening
  >     (otherwise fewer MPI processes will be used). A larger value will cause the coarse problems to be run on fewer MPI processes.
  >     This only applies if `-pc_gamg_parallel_coarse_grid_solver` is set to true. See `PCGAMGSetCoarseEqLim()`.

- Control the smoothers

  > - `-pc_mg_levels` \<n:int> Set the maximum number of levels to use.
  > - `-mg_levels_ksp_type` \<KSPType:chebyshev> If `KSPCHEBYSHEV` or `KSPRICHARDSON` is not used, then the Krylov
  >   method for the entire multigrid solve has to be a flexible method such as `KSPFGMRES`. Generally, the
  >   stronger the Krylov method the faster the convergence, but with more cost per iteration. See `KSPSetType()`.
  > - `-mg_levels_ksp_max_it` \<its:int:2> Sets the number of iterations to run the smoother on each level. Generally, the more iterations
  >   , the faster the convergence, but with more cost per multigrid iteration. See `PCMGSetNumberSmooth()`.
  > - `-mg_levels_ksp_xxx` Sets options for the `KSP` in the smoother on the levels.
  > - `-mg_levels_pc_type` \<PCType:jacobi> Sets the smoother to use on each level. See `PCSetType()`. Generally, the
  >   stronger the preconditioner the faster the convergence, but with more cost per iteration.
  > - `-mg_levels_pc_xxx` Sets options for the `PC` in the smoother on the levels.
  > - `-mg_coarse_ksp_type` \<KSPType:none> Sets the solver `KSPType` to use on the coarsest level.
  > - `-mg_coarse_pc_type` \<PCType:lu> Sets the solver `PCType` to use on the coarsest level.
  > - `-pc_gamg_asm_use_agg` \<bool:false> Use `PCASM` as the smoother on each level with the aggregates defined by the coarsening process are
  >   the subdomains. This option automatically switches the smoother on the levels to be `PCASM`.
  > - `-mg_levels_pc_asm_overlap` \<n:int:0> Use non-zero overlap with `-pc_gamg_asm_use_agg`. See `PCASMSetOverlap()`.

- Control the multigrid algorithm

  > - `-pc_mg_type` \<additive|multiplicative|full|kaskade:multiplicative> The type of multigrid to use. Usually, multiplicative is the fastest.
  > - `-pc_mg_cycle_type` \<v|w:v> Use V- or W-cycle with `-pc_mg_type` `multiplicative`

`PCGAMG` provides unsmoothed aggregation (`-pc_gamg_agg_nsmooths 0`) and
smoothed aggregation (`-pc_gamg_agg_nsmooths 1` or
`PCGAMGSetNSmooths(pc,1)`). Smoothed aggregation (SA), {cite}`vanek1996algebraic`, {cite}`vanek2001convergence`, is recommended
for symmetric positive definite systems. Unsmoothed aggregation can be
useful for asymmetric problems and problems where the highest eigenestimates are problematic. If poor convergence rates are observed using
the smoothed version, one can test unsmoothed aggregation.

**Eigenvalue estimates:** The parameters for the KSP eigen estimator,
used for SA, can be set with `-pc_gamg_esteig_ksp_max_it` and
`-pc_gamg_esteig_ksp_type`. For example, CG generally converges to the
highest eigenvalue faster than GMRES (the default for KSP) if your problem
is symmetric positive definite. One can specify CG with
`-pc_gamg_esteig_ksp_type cg`. The default for
`-pc_gamg_esteig_ksp_max_it` is 10, which we have found is pretty safe
with a (default) safety factor of 1.1. One can specify the range of real
eigenvalues in the same way as with Chebyshev KSP solvers
(smoothers), with `-pc_gamg_eigenvalues <emin,emax>`. GAMG sets the MG
smoother type to chebyshev by default. By default, GAMG uses its eigen
estimate, if it has one, for Chebyshev smoothers if the smoother uses
Jacobi preconditioning. This can be overridden with
`-pc_gamg_use_sa_esteig  <true,false>`.

AMG methods require knowledge of the number of degrees of freedom per
vertex; the default is one (a scalar problem). Vector problems like
elasticity should set the block size of the matrix appropriately with
`-mat_block_size bs` or `MatSetBlockSize(mat,bs)`. Equations must be
ordered in “vertex-major” ordering (e.g.,
$x_1,y_1,z_1,x_2,y_2,...$).

**Near null space:** Smoothed aggregation requires an explicit
representation of the (near) null space of the operator for optimal
performance. One can provide an orthonormal set of null space vectors
with `MatSetNearNullSpace()`. The vector of all ones is the default
for each variable given by the block size (e.g., the translational rigid
body modes). For elasticity, where rotational rigid body modes are
required to complete the near null-space you can use
`MatNullSpaceCreateRigidBody()` to create the null space vectors and
then `MatSetNearNullSpace()`.

**Coarse grid data model:** The GAMG framework provides for reducing the
number of active processes on coarse grids to reduce communication costs
when there is not enough parallelism to keep relative communication
costs down. Most AMG solvers reduce to just one active process on the
coarsest grid (the PETSc MG framework also supports redundantly solving
the coarse grid on all processes to reduce communication
costs potentially). However, this forcing to one process can be overridden if one
wishes to use a parallel coarse grid solver. GAMG generalizes this by
reducing the active number of processes on other coarse grids.
GAMG will select the number of active processors by fitting the desired
number of equations per process (set with
`-pc_gamg_process_eq_limit <50>,`) at each level given that size of
each level. If $P_i < P$ processors are desired on a level
$i$, then the first $P_i$ processes are populated with the grid
and the remaining are empty on that grid. One can, and probably should,
repartition the coarse grids with `-pc_gamg_repartition <true>`,
otherwise an integer process reduction factor ($q$) is selected
and the equations on the first $q$ processes are moved to process
0, and so on. As mentioned, multigrid generally coarsens the problem
until it is small enough to be solved with an exact solver (e.g., LU or
SVD) in a relatively short time. GAMG will stop coarsening when the
number of the equation on a grid falls below the threshold given by
`-pc_gamg_coarse_eq_limit <50>,`.

**Coarse grid parameters:** There are several options to provide
parameters to the coarsening algorithm and parallel data layout. Run a
code using `PCGAMG` with `-help` to get a full listing of GAMG
parameters with short descriptions. The rate of coarsening is
critical in AMG performance – too slow coarsening will result in an
overly expensive solver per iteration and too fast coarsening will
result in decrease in the convergence rate. `-pc_gamg_threshold <-1>`
and `-pc_gamg_aggressive_coarsening <N>` are the primary parameters that
control coarsening rates, which is very important for AMG performance. A
greedy maximal independent set (MIS) algorithm is used in coarsening.
Squaring the graph implements MIS-2; the root vertex in an
aggregate is more than two edges away from another root vertex instead
of more than one in MIS. The threshold parameter sets a normalized
threshold for which edges are removed from the MIS graph, thereby
coarsening slower. Zero will keep all non-zero edges, a negative number
will keep zero edges, and a positive number will drop small edges. Typical
finite threshold values are in the range of $0.01 - 0.05$. There
are additional parameters for changing the weights on coarse grids.

The parallel MIS algorithms require symmetric weights/matrices. Thus `PCGAMG`
will automatically make the graph symmetric if it is not symmetric. Since this
has additional cost, users should indicate the symmetry of the matrices they
provide by calling

```
MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE (or PETSC_FALSE))
```

or

```
MatSetOption(mat,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE (or PETSC_FALSE)).
```

If they know that the matrix will always have symmetry despite future changes
to the matrix (with, for example, `MatSetValues()`) then they should also call

```
MatSetOption(mat,MAT_SYMMETRY_ETERNAL,PETSC_TRUE (or PETSC_FALSE))
```

or

```
MatSetOption(mat,MAT_STRUCTURAL_SYMMETRY_ETERNAL,PETSC_TRUE (or PETSC_FALSE)).
```

Using this information allows the algorithm to skip unnecessary computations.

**Troubleshooting algebraic multigrid methods:** If `PCGAMG`, *ML*, *AMGx* or
*hypre* does not perform well; the first thing to try is one of the other
methods. Often, the default parameters or just the strengths of different
algorithms can fix performance problems or provide useful information to
guide further debugging. There are several sources of poor performance
of AMG solvers and often special purpose methods must be developed to
achieve the full potential of multigrid. To name just a few sources of
performance degradation that may not be fixed with parameters in PETSc
currently: non-elliptic operators, curl/curl operators, highly stretched
grids or highly anisotropic problems, large jumps in material
coefficients with complex geometry (AMG is particularly well suited to
jumps in coefficients, but it is not a perfect solution), highly
incompressible elasticity, not to mention ill-posed problems and many
others. For Grad-Div and Curl-Curl operators, you may want to try the
Auxiliary-space Maxwell Solver (AMS,
`-pc_type hypre -pc_hypre_type ams`) or the Auxiliary-space Divergence
Solver (ADS, `-pc_type hypre -pc_hypre_type ads`) solvers. These
solvers need some additional information on the underlying mesh;
specifically, AMS needs the discrete gradient operator, which can be
specified via `PCHYPRESetDiscreteGradient()`. In addition to the
discrete gradient, ADS also needs the specification of the discrete curl
operator, which can be set using `PCHYPRESetDiscreteCurl()`.

**I am converging slowly, what do I do?** AMG methods are sensitive to
coarsening rates and methods; for GAMG use `-pc_gamg_threshold <x>`
or `PCGAMGSetThreshold()` to regulate coarsening rates; higher values decrease
coarsening rate. Squaring the graph is the second mechanism for
increasing the coarsening rate. Use `-pc_gamg_aggressive_coarsening <N>`, or
`PCGAMGSetAggressiveLevels(pc,N)`, to aggressive ly coarsen (MIS-2) the graph on the finest N
levels. A high threshold (e.g., $x=0.08$) will result in an
expensive but potentially powerful preconditioner, and a low threshold
(e.g., $x=0.0$) will result in faster coarsening, fewer levels,
cheaper solves, and generally worse convergence rates.

One can run with `-info :pc` and grep for `PCGAMG` to get statistics on
each level, which can be used to see if you are coarsening at an
appropriate rate. With smoothed aggregation, you generally want to coarse
at about a rate of 3:1 in each dimension. Coarsening too slowly will
result in large numbers of non-zeros per row on coarse grids (this is
reported). The number of non-zeros can go up very high, say about 300
(times the degrees of freedom per vertex) on a 3D hex mesh. One can also
look at the grid complexity, which is also reported (the ratio of the
total number of matrix entries for all levels to the number of matrix
entries on the fine level). Grid complexity should be well under 2.0 and
preferably around $1.3$ or lower. If convergence is poor and the
Galerkin coarse grid construction is much smaller than the time for each
solve, one can safely decrease the coarsening rate.
`-pc_gamg_threshold` $-1.0$ is the simplest and most robust
option and is recommended if poor convergence rates are observed, at
least until the source of the problem is discovered. In conclusion, decreasing the coarsening rate (increasing the
threshold) should be tried if convergence is slow.

**A note on Chebyshev smoothers.** Chebyshev solvers are attractive as
multigrid smoothers because they can target a specific interval of the
spectrum, which is the purpose of a smoother. The spectral bounds for
Chebyshev solvers are simple to compute because they rely on the highest
eigenvalue of your (diagonally preconditioned) operator, which is
conceptually simple to compute. However, if this highest eigenvalue
estimate is not accurate (too low), the solvers can fail with an
indefinite preconditioner message. One can run with `-info` and grep
for `PCGAMG` to get these estimates or use `-ksp_view`. These highest
eigenvalues are generally between 1.5-3.0. For symmetric positive
definite systems, CG is a better eigenvalue estimator
`-mg_levels_esteig_ksp_type cg`. Bad Eigen estimates often cause indefinite matrix messages. Explicitly damped Jacobi or Krylov
smoothers can provide an alternative to Chebyshev, and *hypre* has
alternative smoothers.

**Now, am I solving alright? Can I expect better?** If you find that you
are getting nearly one digit in reduction of the residual per iteration
and are using a modest number of point smoothing steps (e.g., 1-4
iterations of SOR), then you may be fairly close to textbook multigrid
efficiency. However, you also need to check the setup costs. This can be
determined by running with `-log_view` and check that the time for the
Galerkin coarse grid construction (`MatPtAP()`) is not (much) more than
the time spent in each solve (`KSPSolve()`). If the `MatPtAP()` time is
too large, then one can increase the coarsening rate by decreasing the
threshold and using aggressive coarsening
(`-pc_gamg_aggressive_coarsening <N>`, squares the graph on the finest N
levels). Likewise, if your `MatPtAP()` time is short and your convergence
If the rate is not ideal, you could decrease the coarsening rate.

PETSc’s AMG solver is a framework for developers to
easily add AMG capabilities, like new AMG methods or an AMG component
like a matrix triple product. Contact us directly if you are interested
in contributing.

Using algebraic multigrid as a "standalone" solver is possible but not recommended, as it does not accelerate it with a Krylov method.
Use a `KSPType` of `KSPRICHARDSON`
(or equivalently `-ksp_type richardson`) to achieve this. Using `KSPPREONLY` will not work since it only applies a single multigrid cycle.

#### Adaptive Interpolation

**Interpolation** transfers a function from the coarse space to the fine space. We would like this process to be accurate for the functions resolved by the coarse grid, in particular the approximate solution computed there. By default, we create these matrices using local interpolation of the fine grid dual basis functions in the coarse basis. However, an adaptive procedure can optimize the coefficients of the interpolator to reproduce pairs of coarse/fine functions which should approximate the lowest modes of the generalized eigenproblem

$$
A x = \lambda M x
$$

where $A$ is the system matrix and $M$ is the smoother. Note that for defect-correction MG, the interpolated solution from the coarse space need not be as accurate as the fine solution, for the same reason that updates in iterative refinement can be less accurate. However, in FAS or in the final interpolation step for each level of Full Multigrid, we must have interpolation as accurate as the fine solution since we are moving the entire solution itself.

**Injection** should accurately transfer the fine solution to the coarse grid. Accuracy here means that the action of a coarse dual function on either should produce approximately the same result. In the structured grid case, this means that we just use the same values on coarse points. This can result in aliasing.

**Restriction** is intended to transfer the fine residual to the coarse space. Here we use averaging (often the transpose of the interpolation operation) to damp out the fine space contributions. Thus, it is less accurate than injection, but avoids aliasing of the high modes.

For a multigrid cycle, the interpolator $P$ is intended to accurately reproduce "smooth" functions from the coarse space in the fine space, keeping the energy of the interpolant about the same. For the Laplacian on a structured mesh, it is easy to determine what these low-frequency functions are. They are the Fourier modes. However an arbitrary operator $A$ will have different coarse modes that we want to resolve accurately on the fine grid, so that our coarse solve produces a good guess for the fine problem. How do we make sure that our interpolator $P$ can do this?

We first must decide what we mean by accurate interpolation of some functions. Suppose we know the continuum function $f$ that we care about, and we are only interested in a finite element description of discrete functions. Then the coarse function representing $f$ is given by

$$
f^C = \sum_i f^C_i \phi^C_i,
$$

and similarly the fine grid form is

$$
f^F = \sum_i f^F_i \phi^F_i.
$$

Now we would like the interpolant of the coarse representer to the fine grid to be as close as possible to the fine representer in a least squares sense, meaning we want to solve the minimization problem

$$
\min_{P} \| f^F - P f^C \|_2
$$

Now we can express $P$ as a matrix by looking at the matrix elements $P_{ij} = \phi^F_i P \phi^C_j$. Then we have

$$
\begin{aligned}
  &\phi^F_i f^F - \phi^F_i P f^C \\
= &f^F_i - \sum_j P_{ij} f^C_j
\end{aligned}
$$

so that our discrete optimization problem is

$$
\min_{P_{ij}} \| f^F_i - \sum_j P_{ij} f^C_j \|_2
$$

and we will treat each row of the interpolator as a separate optimization problem. We could allow an arbitrary sparsity pattern, or try to determine adaptively, as is done in sparse approximate inverse preconditioning. However, we know the supports of the basis functions in finite elements, and thus the naive sparsity pattern from local interpolation can be used.

We note here that the BAMG framework of Brannick et al. {cite}`brandtbrannickkahllivshits2011` does not use fine and coarse functions spaces, but rather a fine point/coarse point division which we will not employ here. Our general PETSc routine should work for both since the input would be the checking set (fine basis coefficients or fine space points) and the approximation set (coarse basis coefficients in the support or coarse points in the sparsity pattern).

We can easily solve the above problem using QR factorization. However, there are many smooth functions from the coarse space that we want interpolated accurately, and a single $f$ would not constrain the values $P_{ij}`$ well. Therefore, we will use several functions $\{f_k\}$ in our minimization,

$$
\begin{aligned}
  &\min_{P_{ij}} \sum_k w_k \| f^{F,k}_i - \sum_j P_{ij} f^{C,k}_j \|_2 \\
= &\min_{P_{ij}} \sum_k \| \sqrt{w_k} f^{F,k}_i - \sqrt{w_k} \sum_j P_{ij} f^{C,k}_j \|_2 \\
= &\min_{P_{ij}} \| W^{1/2} \mathbf{f}^{F}_i - W^{1/2} \mathbf{f}^{C} p_i \|_2
\end{aligned}
$$

where

$$
\begin{aligned}
W         &= \begin{pmatrix} w_0 & & \\ & \ddots & \\ & & w_K \end{pmatrix} \\
\mathbf{f}^{F}_i &= \begin{pmatrix} f^{F,0}_i \\ \vdots \\ f^{F,K}_i \end{pmatrix} \\
\mathbf{f}^{C}   &= \begin{pmatrix} f^{C,0}_0 & \cdots & f^{C,0}_n \\ \vdots & \ddots &  \vdots \\ f^{C,K}_0 & \cdots & f^{C,K}_n \end{pmatrix} \\
p_i       &= \begin{pmatrix} P_{i0} \\ \vdots \\ P_{in} \end{pmatrix}
\end{aligned}
$$

or alternatively

$$
\begin{aligned}
[W]_{kk}     &= w_k \\
[f^{F}_i]_k  &= f^{F,k}_i \\
[f^{C}]_{kj} &= f^{C,k}_j \\
[p_i]_j      &= P_{ij}
\end{aligned}
$$

We thus have a standard least-squares problem

$$
\min_{P_{ij}} \| b - A x \|_2
$$

where

$$
\begin{aligned}
A &= W^{1/2} f^{C} \\
b &= W^{1/2} f^{F}_i \\
x &= p_i
\end{aligned}
$$

which can be solved using LAPACK.

We will typically perform this optimization on a multigrid level $l$ when the change in eigenvalue from level $l+1$ is relatively large, meaning

$$
\frac{|\lambda_l - \lambda_{l+1}|}{|\lambda_l|}.
$$

This indicates that the generalized eigenvector associated with that eigenvalue was not adequately represented by $P^l_{l+1}`$, and the interpolator should be recomputed.

```{raw} html
<hr>
```

### Balancing Domain Decomposition by Constraints

PETSc provides the Balancing Domain Decomposition by Constraints (`PCBDDC`)
method for preconditioning parallel finite element problems stored in
unassembled format (see `MATIS`). `PCBDDC` is a 2-level non-overlapping
domain decomposition method which can be easily adapted to different
problems and discretizations by means of few user customizations. The
application of the preconditioner to a vector consists in the static
condensation of the residual at the interior of the subdomains by means
of local Dirichlet solves, followed by an additive combination of Neumann
local corrections and the solution of a global coupled coarse problem.
Command line options for the underlying `KSP` objects are prefixed by
`-pc_bddc_dirichlet`, `-pc_bddc_neumann`, and `-pc_bddc_coarse`
respectively.

The implementation supports any kind of linear system, and
assumes a one-to-one mapping between subdomains and MPI processes.
Complex numbers are supported as well. For non-symmetric problems, use
the runtime option `-pc_bddc_symmetric 0`.

Unlike conventional non-overlapping methods that iterates just on the
degrees of freedom at the interface between subdomain, `PCBDDC`
iterates on the whole set of degrees of freedom, allowing the use of
approximate subdomain solvers. When using approximate solvers, the
command line switches `-pc_bddc_dirichlet_approximate` and/or
`-pc_bddc_neumann_approximate` should be used to inform `PCBDDC`. If
any of the local problems is singular, the nullspace of the local
operator should be attached to the local matrix via
`MatSetNullSpace()`.

At the basis of the method there’s the analysis of the connected
components of the interface for the detection of vertices, edges and
faces equivalence classes. Additional information on the degrees of
freedom can be supplied to `PCBDDC` by using the following functions:

- `PCBDDCSetDofsSplitting()`
- `PCBDDCSetLocalAdjacencyGraph()`
- `PCBDDCSetPrimalVerticesLocalIS()`
- `PCBDDCSetNeumannBoundaries()`
- `PCBDDCSetDirichletBoundaries()`
- `PCBDDCSetNeumannBoundariesLocal()`
- `PCBDDCSetDirichletBoundariesLocal()`

Crucial for the convergence of the iterative process is the
specification of the primal constraints to be imposed at the interface
between subdomains. `PCBDDC` uses by default vertex continuities and
edge arithmetic averages, which are enough for the three-dimensional
Poisson problem with constant coefficients. The user can switch on and
off the usage of vertices, edges or face constraints by using the
command line switches `-pc_bddc_use_vertices`, `-pc_bddc_use_edges`,
`-pc_bddc_use_faces`. A customization of the constraints is available
by attaching a `MatNullSpace` object to the  matrix used to compute the preconditioner via
`MatSetNearNullSpace()`. The vectors of the `MatNullSpace` object
should represent the constraints in the form of quadrature rules;
quadrature rules for different classes of the interface can be listed in
the same vector. The number of vectors of the `MatNullSpace` object
corresponds to the maximum number of constraints that can be imposed for
each class. Once all the quadrature rules for a given interface class
have been extracted, an SVD operation is performed to retain the
non-singular modes. As an example, the rigid body modes represent an
effective choice for elasticity, even in the almost incompressible case.
For particular problems, e.g. edge-based discretization with Nedelec
elements, a user defined change of basis of the degrees of freedom can
be beneficial for `PCBDDC`; use `PCBDDCSetChangeOfBasisMat()` to
customize the change of basis.

The `PCBDDC` method is usually robust with respect to jumps in the material
parameters aligned with the interface; for PDEs with more than one
material parameter you may also consider to use the so-called deluxe
scaling, available via the command line switch
`-pc_bddc_use_deluxe_scaling`. Other scalings are available, see
`PCISSetSubdomainScalingFactor()`,
`PCISSetSubdomainDiagonalScaling()` or
`PCISSetUseStiffnessScaling()`. However, the convergence properties of
the `PCBDDC` method degrades in presence of large jumps in the material
coefficients not aligned with the interface; for such cases, PETSc has
the capability of adaptively computing the primal constraints. Adaptive
selection of constraints could be requested by specifying a threshold
value at command line by using `-pc_bddc_adaptive_threshold x`. Valid
values for the threshold `x` ranges from 1 to infinity, with smaller
values corresponding to more robust preconditioners. For SPD problems in
2D, or in 3D with only face degrees of freedom (like in the case of
Raviart-Thomas or Brezzi-Douglas-Marini elements), such a threshold is a
very accurate estimator of the condition number of the resulting
preconditioned operator. Since the adaptive selection of constraints for
`PCBDDC` methods is still an active topic of research, its implementation is
currently limited to SPD problems; moreover, because the technique
requires the explicit knowledge of the local Schur complements, it needs
the external package MUMPS.

When solving problems decomposed in thousands of subdomains or more, the
solution of the `PCBDDC` coarse problem could become a bottleneck; in order
to overcome this issue, the user could either consider to solve the
parallel coarse problem on a subset of the communicator associated with
`PCBDDC` by using the command line switch
`-pc_bddc_coarse_redistribute`, or instead use a multilevel approach.
The latter can be requested by specifying the number of requested level
at command line (`-pc_bddc_levels`) or by using `PCBDDCSetLevels()`.
An additional parameter (see `PCBDDCSetCoarseningRatio()`) controls
the number of subdomains that will be generated at the next level; the
larger the coarsening ratio, the lower the number of coarser subdomains.

For further details, see the example
<a href="PETSC_DOC_OUT_ROOT_PLACEHOLDER/src/ksp/ksp/tutorials/ex59.c">KSP Tutorial ex59</a>
and the online documentation for `PCBDDC`.

### Shell Preconditioners

The shell preconditioner simply uses an application-provided routine to
implement the preconditioner. That is, it allows users to write or wrap their
own custom preconditioners as a `PC` and use it with `KSP`, etc.

To provide a custom preconditioner application, use

```
PCShellSetApply(PC pc,PetscErrorCode (*apply)(PC,Vec,Vec));
```

Often a preconditioner needs access to an application-provided data
structured. For this, one should use

```
PCShellSetContext(PC pc,void *ctx);
```

to set this data structure and

```
PCShellGetContext(PC pc,void *ctx);
```

to retrieve it in `apply`. The three routine arguments of `apply()`
are the `PC`, the input vector, and the output vector, respectively.

For a preconditioner that requires some sort of “setup” before being
used, that requires a new setup every time the operator is changed, one
can provide a routine that is called every time the operator is changed
(usually via `KSPSetOperators()`).

```
PCShellSetSetUp(PC pc,PetscErrorCode (*setup)(PC));
```

The argument to the `setup` routine is the same `PC` object which
can be used to obtain the operators with `PCGetOperators()` and the
application-provided data structure that was set with
`PCShellSetContext()`.

(sec_combining_pcs)=

### Combining Preconditioners

The `PC` type `PCCOMPOSITE` allows one to form new preconditioners
by combining already-defined preconditioners and solvers. Combining
preconditioners usually requires some experimentation to find a
combination of preconditioners that works better than any single method.
It is a tricky business and is not recommended until your application
code is complete and running and you are trying to improve performance.
In many cases using a single preconditioner is better than a
combination; an exception is the multigrid/multilevel preconditioners
(solvers) that are always combinations of some sort, see {any}`sec_mg`.

Let $B_1$ and $B_2$ represent the application of two
preconditioners of type `type1` and `type2`. The preconditioner
$B = B_1 + B_2$ can be obtained with

```
PCSetType(pc,PCCOMPOSITE);
PCCompositeAddPCType(pc,type1);
PCCompositeAddPCType(pc,type2);
```

Any number of preconditioners may added in this way.

This way of combining preconditioners is called additive, since the
actions of the preconditioners are added together. This is the default
behavior. An alternative can be set with the option

```
PCCompositeSetType(pc,PC_COMPOSITE_MULTIPLICATIVE);
```

In this form the new residual is updated after the application of each
preconditioner and the next preconditioner applied to the next residual.
For example, with two composed preconditioners: $B_1$ and
$B_2$; $y = B x$ is obtained from

$$
\begin{aligned}
y    = B_1 x \\
w_1  = x - A y \\
y    = y + B_2 w_1\end{aligned}
$$

Loosely, this corresponds to a Gauss-Seidel iteration, while additive
corresponds to a Jacobi iteration.

Under most circumstances, the multiplicative form requires one-half the
number of iterations as the additive form; however, the multiplicative
form does require the application of $A$ inside the
preconditioner.

In the multiplicative version, the calculation of the residual inside
the preconditioner can be done in two ways: using the original linear
system matrix or using the matrix used to build the preconditioners
$B_1$, $B_2$, etc. By default it uses the “preconditioner
matrix”, to use the `Amat` matrix use the option

```
PCSetUseAmat(PC pc);
```

The individual preconditioners can be accessed (in order to set options)
via

```
PCCompositeGetPC(PC pc,PetscInt count,PC *subpc);
```

For example, to set the first sub preconditioners to use ILU(1)

```
PC subpc;
PCCompositeGetPC(pc,0,&subpc);
PCFactorSetFill(subpc,1);
```

One can also change the operator that is used to construct a particular
`PC` in the composite `PC` calling `PCSetOperators()` on the obtained `PC`.
`PCFIELDSPLIT`, {any}`sec_block_matrices`, provides an alternative approach to defining composite preconditioners
with a variety of pre-defined compositions.

These various options can also be set via the options database. For
example, `-pc_type` `composite` `-pc_composite_pcs` `jacobi,ilu`
causes the composite preconditioner to be used with two preconditioners:
Jacobi and ILU. The option `-pc_composite_type` `multiplicative`
initiates the multiplicative version of the algorithm, while
`-pc_composite_type` `additive` the additive version. Using the
`Amat` matrix is obtained with the option `-pc_use_amat`. One sets
options for the sub-preconditioners with the extra prefix `-sub_N_`
where `N` is the number of the sub-preconditioner. For example,
`-sub_0_pc_ifactor_fill` `0`.

PETSc also allows a preconditioner to be a complete `KSPSolve()` linear solver. This
is achieved with the `PCKSP` type.

```
PCSetType(PC pc,PCKSP);
PCKSPGetKSP(pc,&ksp);
 /* set any KSP/PC options */
```

From the command line one can use 5 iterations of biCG-stab with ILU(0)
preconditioning as the preconditioner with
`-pc_type ksp -ksp_pc_type ilu -ksp_ksp_max_it 5 -ksp_ksp_type bcgs`.

By default the inner `KSP` solver uses the outer preconditioner
matrix, `Pmat`, as the matrix to be solved in the linear system; to
use the matrix that defines the linear system, `Amat` use the option

```
PCSetUseAmat(PC pc);
```

or at the command line with `-pc_use_amat`.

Naturally, one can use a `PCKSP` preconditioner inside a composite
preconditioner. For example,
`-pc_type composite -pc_composite_pcs ilu,ksp -sub_1_pc_type jacobi -sub_1_ksp_max_it 10`
uses two preconditioners: ILU(0) and 10 iterations of GMRES with Jacobi
preconditioning. However, it is not clear whether one would ever wish to
do such a thing.

(sec_mg)=

### Multigrid Preconditioners

A large suite of routines is available for using geometric multigrid as
a preconditioner [^id3]. In the `PC` framework, the user is required to
provide the coarse grid solver, smoothers, restriction and interpolation
operators, and code to calculate residuals. The `PC` package allows
these components to be encapsulated within a PETSc-compliant
preconditioner. We fully support both matrix-free and matrix-based
multigrid solvers.

A multigrid preconditioner is created with the four commands

```
KSPCreate(MPI_Comm comm,KSP *ksp);
KSPGetPC(KSP ksp,PC *pc);
PCSetType(PC pc,PCMG);
PCMGSetLevels(pc,PetscInt levels,MPI_Comm *comms);
```

A large number of parameters affect the multigrid behavior. The command

```
PCMGSetType(PC pc,PCMGType mode);
```

indicates which form of multigrid to apply {cite}`1sbg`.

For standard V or W-cycle multigrids, one sets the `mode` to be
`PC_MG_MULTIPLICATIVE`; for the additive form (which in certain cases
reduces to the BPX method, or additive multilevel Schwarz, or multilevel
diagonal scaling), one uses `PC_MG_ADDITIVE` as the `mode`. For a
variant of full multigrid, one can use `PC_MG_FULL`, and for the
Kaskade algorithm `PC_MG_KASKADE`. For the multiplicative and full
multigrid options, one can use a W-cycle by calling

```
PCMGSetCycleType(PC pc,PCMGCycleType ctype);
```

with a value of `PC_MG_CYCLE_W` for `ctype`. The commands above can
also be set from the options database. The option names are
`-pc_mg_type [multiplicative, additive, full, kaskade]`, and
`-pc_mg_cycle_type` `<ctype>`.

The user can control the amount of smoothing by configuring the solvers
on the levels. By default, the up and down smoothers are identical. If
separate configuration of up and down smooths is required, it can be
requested with the option `-pc_mg_distinct_smoothup` or the routine

```
PCMGSetDistinctSmoothUp(PC pc);
```

The multigrid routines, which determine the solvers and
interpolation/restriction operators that are used, are mandatory. To set
the coarse grid solver, one must call

```
PCMGGetCoarseSolve(PC pc,KSP *ksp);
```

and set the appropriate options in `ksp`. Similarly, the smoothers are
controlled by first calling

```
PCMGGetSmoother(PC pc,PetscInt level,KSP *ksp);
```

and then setting the various options in the `ksp.` For example,

```
PCMGGetSmoother(pc,1,&ksp);
KSPSetOperators(ksp,A1,A1);
```

sets the matrix that defines the smoother on level 1 of the multigrid.
While

```
PCMGGetSmoother(pc,1,&ksp);
KSPGetPC(ksp,&pc);
PCSetType(pc,PCSOR);
```

sets SOR as the smoother to use on level 1.

To use a different pre- or postsmoother, one should call the following
routines instead.

```
PCMGGetSmootherUp(PC pc,PetscInt level,KSP *upksp);
PCMGGetSmootherDown(PC pc,PetscInt level,KSP *downksp);
```

Use

```
PCMGSetInterpolation(PC pc,PetscInt level,Mat P);
```

and

```
PCMGSetRestriction(PC pc,PetscInt level,Mat R);
```

to define the intergrid transfer operations. If only one of these is
set, its transpose will be used for the other.

It is possible for these interpolation operations to be matrix-free (see
{any}`sec_matrixfree`); One should then make
sure that these operations are defined for the (matrix-free) matrices
passed in. Note that this system is arranged so that if the
interpolation is the transpose of the restriction, you can pass the same
`mat` argument to both `PCMGSetRestriction()` and
`PCMGSetInterpolation()`.

On each level except the coarsest, one must also set the routine to
compute the residual. The following command suffices:

```
PCMGSetResidual(PC pc,PetscInt level,PetscErrorCode (*residual)(Mat,Vec,Vec,Vec),Mat mat);
```

The `residual()` function normally does not need to be set if one’s
operator is stored in `Mat` format. In certain circumstances, where it
is much cheaper to calculate the residual directly, rather than through
the usual formula $b - Ax$, the user may wish to provide an
alternative.

Finally, the user may provide three work vectors for each level (except
on the finest, where only the residual work vector is required). The
work vectors are set with the commands

```
PCMGSetRhs(PC pc,PetscInt level,Vec b);
PCMGSetX(PC pc,PetscInt level,Vec x);
PCMGSetR(PC pc,PetscInt level,Vec r);
```

The `PC` references these vectors, so you should call `VecDestroy()`
when you are finished with them. If any of these vectors are not
provided, the preconditioner will allocate them.

One can control the `KSP` and `PC` options used on the various
levels (as well as the coarse grid) using the prefix `mg_levels_`
(`mg_coarse_` for the coarse grid). For example,
`-mg_levels_ksp_type cg` will cause the CG method to be used as the
Krylov method for each level. Or
`-mg_levels_pc_type ilu -mg_levels_pc_factor_levels 2` will cause the
ILU preconditioner to be used on each level with two levels of fill in
the incomplete factorization.

(sec_block_matrices)=

## Solving Block Matrices with PCFIELDSPLIT

Block matrices represent an important class of problems in numerical
linear algebra and offer the possibility of far more efficient iterative
solvers than just treating the entire matrix as a black box. In this
section, we use the common linear algebra definition of block matrices, where matrices are divided into a small, problem-size independent (two,
three, or so) number of very large blocks. These blocks arise naturally
from the underlying physics or discretization of the problem, such as the velocity and pressure. Under a certain numbering of
unknowns, the matrix can be written as

$$
\left( \begin{array}{cccc}
A_{00}   & A_{01} & A_{02} & A_{03} \\
A_{10}   & A_{11} & A_{12} & A_{13} \\
A_{20}   & A_{21} & A_{22} & A_{23} \\
A_{30}   & A_{31} & A_{32} & A_{33} \\
\end{array} \right),
$$

where each $A_{ij}$ is an entire block. The matrices on a parallel computer are not explicitly stored this way. Instead, each process will
own some rows of $A_{0*}$, $A_{1*}$ etc. On a
process, the blocks may be stored in one block followed by another

$$
\left( \begin{array}{ccccccc}
A_{{00}_{00}}   & A_{{00}_{01}} & A_{{00}_{02}} & ... & A_{{01}_{00}} & A_{{01}_{01}} & ...  \\
A_{{00}_{10}}   & A_{{00}_{11}} & A_{{00}_{12}} & ... & A_{{01}_{10}} & A_{{01}_{11}} & ... \\
A_{{00}_{20}}   & A_{{00}_{21}} & A_{{00}_{22}} & ... & A_{{01}_{20}} & A_{{01}_{21}}  & ...\\
... \\
A_{{10}_{00}}   & A_{{10}_{01}} & A_{{10}_{02}} & ... & A_{{11}_{00}} & A_{{11}_{01}}  & ... \\
A_{{10}_{10}}   & A_{{10}_{11}} & A_{{10}_{12}} & ... & A_{{11}_{10}} & A_{{11}_{11}}  & ... \\
... \\
\end{array} \right)
$$

or interlaced, for example, with four blocks

$$
\left( \begin{array}{ccccc}
A_{{00}_{00}}   & A_{{01}_{00}} &  A_{{00}_{01}} & A_{{01}_{01}} &  ... \\
A_{{10}_{00}}   & A_{{11}_{00}} &  A_{{10}_{01}} & A_{{11}_{01}} &  ... \\
A_{{00}_{10}}   & A_{{01}_{10}} & A_{{00}_{11}} & A_{{01}_{11}} & ...\\
A_{{10}_{10}}   & A_{{11}_{10}} & A_{{10}_{11}} & A_{{11}_{11}} & ...\\
...
\end{array} \right).
$$

Note that for interlaced storage, the number of rows/columns of each
block must be the same size. Matrices obtained with `DMCreateMatrix()`
where the `DM` is a `DMDA` are always stored interlaced. Block
matrices can also be stored using the `MATNEST` format, which holds
separate assembled blocks. Each of these nested matrices is itself
distributed in parallel. It is more efficient to use `MATNEST` with
the methods described in this section because there are fewer copies and
better formats (e.g., `MATBAIJ` or `MATSBAIJ`) can be used for the
components, but it is not possible to use many other methods with
`MATNEST`. See {any}`sec_matnest` for more on assembling
block matrices without depending on a specific matrix format.

The PETSc `PCFIELDSPLIT` preconditioner implements the
“block” solvers in PETSc, {cite}`elman2008tcp`. There are three ways to provide the
information that defines the blocks. If the matrices are stored as
interlaced then `PCFieldSplitSetFields()` can be called repeatedly to
indicate which fields belong to each block. More generally
`PCFieldSplitSetIS()` can be used to indicate exactly which
rows/columns of the matrix belong to a particular block (field). You can provide
names for each block with these routines; if you do not, they are numbered from 0. With these two approaches, the blocks may
overlap (though they generally will not overlap). If only one block is defined,
then the complement of the matrices is used to define the other block.
Finally, the option `-pc_fieldsplit_detect_saddle_point` causes two
diagonal blocks to be found, one associated with all rows/columns that
have zeros on the diagonals and the rest.

**Important parameters for PCFIELDSPLIT**

- Control the fields used

  - `-pc_fieldsplit_detect_saddle_point` \<bool:false> Generate two fields, the first consists of all rows with a nonzero on the diagonal, and the second will be all rows
    with zero on the diagonal. See `PCFieldSplitSetDetectSaddlePoint()`.

  - `-pc_fieldsplit_dm_splits` \<bool:true> Use the `DM` attached to the preconditioner to determine the fields. See `PCFieldSplitSetDMSplits()` and
    `DMCreateFieldDecomposition()`.

  - `-pc_fieldsplit_%d_fields` \<f1,f2,...:int> Use f1, f2, .. to define field `d`. The `fn` are in the range of 0, ..., bs-1 where bs is the block size
    of the matrix or set with `PCFieldSplitSetBlockSize()`. See `PCFieldSplitSetFields()`.

    - `-pc_fieldsplit_default` \<bool:true> Automatically add any fields needed that have not been supplied explicitly by `-pc_fieldsplit_%d_fields`.

  - `DMFieldsplitSetIS()` Provide the `IS` that defines a particular field.

- Control the type of the block preconditioner

  - `-pc_fieldsplit_type` \<additive|multiplicative|symmetric_multiplicative|schur|gkb:multiplicative> The order in which the field solves are applied.
    For symmetric problems where `KSPCG` is used `symmetric_multiplicative` must be used instead of `multiplicative`. `additive` is the least expensive
    to apply but provides the worst convergence. `schur` requires either a good preconditioner for the Schur complement or a naturally well-conditioned
    Schur complement, but when it works well can be extremely effective. See `PCFieldSplitSetType()`. `gkb` is for symmetric saddle-point problems (the lower-right
    the block is zero).

  - `-pc_fieldsplit_diag_use_amat` \<bool:false> Use the first matrix that is passed to `KSPSetJacobian()` to construct the block-diagonal sub-matrices used in the algorithms,
    by default, the second matrix is used.

  - Options for Schur preconditioner: `-pc_fieldsplit_type`
    `schur`

    - `-pc_fieldsplit_schur_fact_type` \<diag|lower|upper|full:diag> See `PCFieldSplitSetSchurFactType()`. `full` reduces the iterations but each iteration requires additional
      field solves.

    - `-pc_fieldsplit_schur_precondition` \<self|selfp|user|a11|full:user> How the Schur complement is preconditioned. See `PCFieldSplitSetSchurPre()`.

      - `-fieldsplit_1_mat_schur_complement_ainv_type` \<diag|lump:diag> Use the lumped diagonal of $A_{00}$ when `-pc_fieldsplit_schur_precondition`
        `selfp` is used.

    - `-pc_fieldsplit_schur_scale` \<real:-1.0> Controls the sign flip of S for `-pc_fieldsplit_schur_fact_type` `diag`.
      See `PCFieldSplitSetSchurScale()`

    - `fieldsplit_1_xxx` controls the solver for the Schur complement system.
      If a `DM` provided the fields, use the second field name set in the `DM` instead of 1.

      - `-fieldsplit_1_pc_type` `lsc` `-fieldsplit_1_lsc_pc_xxx` use
        the least squares commutators {cite}`elmanhowleshadidshuttleworthtuminaro2006` {cite}`silvester2001efficient`
        preconditioner for the Schur complement with any preconditioner for the least-squares matrix, see `PCLSC`.
        If a `DM` provided the fields, use the second field name set in the `DM` instead of 1.

    - `-fieldsplit_upper_xxx` Set options for the solver in the upper solver when `-pc_fieldsplit_schur_fact_type`
      `upper` or `full` is used. Defaults to
      using the solver as provided with `-fieldsplit_0_xxx`.

    - `-fieldsplit_1_inner_xxx` Set the options for the solver inside the application of the Schur complement;
      defaults to using the solver as provided with `-fieldsplit_0_xxx`. If a `DM` provides the fields use the name of the second field name set in the `DM` instead of 1.

  - Options for GKB preconditioner: `-pc_fieldsplit_type` gkb

    - `-pc_fieldsplit_gkb_tol` \<real:1e-5> See `PCFieldSplitSetGKBTol()`.
    - `-pc_fieldsplit_gkb_delay` \<int:5> See `PCFieldSplitSetGKBDelay()`.
    - `-pc_fieldsplit_gkb_nu` \<real:1.0> See `PCFieldSplitSetGKBNu()`.
    - `-pc_fieldsplit_gkb_maxit` \<int:100> See `PCFieldSplitSetGKBMaxit()`.
    - `-pc_fieldsplit_gkb_monitor` \<bool:false> Monitor the convergence of the inner solver.

- Options for additive and multiplication field solvers:

   - `-fieldsplit_%d_xxx` Set options for the solver for field number `d`. For example, `-fieldsplit_0_pc_type`
    `jacobi`. When the fields are obtained from a `DM` use the
    field name instead of `d`.

For simplicity, we restrict our matrices to two-by-two blocks in the rest of the section. So the matrix is

$$
\left( \begin{array}{cc}
A_{00}   & A_{01} \\
A_{10}   & A_{11} \\
\end{array} \right).
$$

On occasion, the user may provide another matrix that is used to
construct parts of the preconditioner

$$
\left( \begin{array}{cc}
Ap_{00}   & Ap_{01} \\
Ap_{10}   & Ap_{11} \\
\end{array} \right).
$$

For notational simplicity define $\text{ksp}(A,Ap)$ to mean
approximately solving a linear system using `KSP` with the operator
$A$ and preconditioner built from matrix $Ap$.

For matrices defined with any number of blocks, there are three “block”
algorithms available: block Jacobi,

$$
\left( \begin{array}{cc}
  \text{ksp}(A_{00},Ap_{00})   & 0 \\
  0   & \text{ksp}(A_{11},Ap_{11}) \\
\end{array} \right)
$$

block Gauss-Seidel,

$$
\left( \begin{array}{cc}
I   & 0 \\
0 & A^{-1}_{11} \\
\end{array} \right)
\left( \begin{array}{cc}
I   & 0 \\
-A_{10} & I \\
\end{array} \right)
\left( \begin{array}{cc}
A^{-1}_{00}   & 0 \\
0 & I \\
\end{array} \right)
$$

which is implemented [^id4] as

$$
\left( \begin{array}{cc}
I   & 0 \\
  0 & \text{ksp}(A_{11},Ap_{11}) \\
\end{array} \right)
$$

$$
\left[
\left( \begin{array}{cc}
0   & 0 \\
0 & I \\
\end{array} \right) +
\left( \begin{array}{cc}
I   & 0 \\
-A_{10} & -A_{11} \\
\end{array} \right)
\left( \begin{array}{cc}
I   & 0 \\
0 & 0 \\
\end{array} \right)
\right]
$$

$$
\left( \begin{array}{cc}
  \text{ksp}(A_{00},Ap_{00})   & 0 \\
0 & I \\
\end{array} \right)
$$

and symmetric block Gauss-Seidel

$$
\left( \begin{array}{cc}
A_{00}^{-1}   & 0 \\
0 & I \\
\end{array} \right)
\left( \begin{array}{cc}
I   & -A_{01} \\
0 & I \\
\end{array} \right)
\left( \begin{array}{cc}
A_{00}   & 0 \\
0 & A_{11}^{-1} \\
\end{array} \right)
\left( \begin{array}{cc}
I   & 0 \\
-A_{10} & I \\
\end{array} \right)
\left( \begin{array}{cc}
A_{00}^{-1}   & 0 \\
0 & I \\
\end{array} \right).
$$

These can be accessed with
`-pc_fieldsplit_type <additive,multiplicative,symmetric_multiplicative>`
or the function `PCFieldSplitSetType()`. The option prefixes for the
internal KSPs are given by `-fieldsplit_name_`.

By default blocks $A_{00}, A_{01}$ and so on are extracted out of
`Pmat`, the matrix that the `KSP` uses to build the preconditioner,
and not out of `Amat` (i.e., $A$ itself). As discussed above, in
{any}`sec_combining_pcs`, however, it is
possible to use `Amat` instead of `Pmat` by calling
`PCSetUseAmat(pc)` or using `-pc_use_amat` on the command line.
Alternatively, you can have `PCFIELDSPLIT` extract the diagonal blocks
$A_{00}, A_{11}$ etc. out of `Amat` by calling
`PCFieldSplitSetDiagUseAmat(pc,PETSC_TRUE)` or supplying command-line
argument `-pc_fieldsplit_diag_use_amat`. Similarly,
`PCFieldSplitSetOffDiagUseAmat(pc,{PETSC_TRUE`) or
`-pc_fieldsplit_off_diag_use_amat` will cause the off-diagonal blocks
$A_{01},A_{10}$ etc. to be extracted out of `Amat`.

For two-by-two blocks only, there is another family of solvers based on
Schur complements. The inverse of the Schur complement factorization is

$$
\left[
\left( \begin{array}{cc}
I   & 0 \\
A_{10}A_{00}^{-1} & I \\
\end{array} \right)
\left( \begin{array}{cc}
A_{00}  & 0 \\
0 & S \\
\end{array} \right)
\left( \begin{array}{cc}
I   & A_{00}^{-1} A_{01} \\
0 & I \\
\end{array} \right)
\right]^{-1} =
$$

$$
\left( \begin{array}{cc}
I   & A_{00}^{-1} A_{01} \\
0 & I \\
\end{array} \right)^{-1}
\left( \begin{array}{cc}
A_{00}^{-1}  & 0 \\
0 & S^{-1} \\
\end{array} \right)
\left( \begin{array}{cc}
I   & 0 \\
A_{10}A_{00}^{-1} & I \\
\end{array} \right)^{-1} =
$$

$$
\left( \begin{array}{cc}
I   & -A_{00}^{-1} A_{01} \\
0 & I \\
\end{array} \right)
\left( \begin{array}{cc}
A_{00}^{-1}  & 0 \\
0 & S^{-1} \\
\end{array} \right)
\left( \begin{array}{cc}
I   & 0 \\
-A_{10}A_{00}^{-1} & I \\
\end{array} \right) =
$$

$$
\left( \begin{array}{cc}
A_{00}^{-1}   & 0 \\
0 & I \\
\end{array} \right)
\left( \begin{array}{cc}
I   & -A_{01} \\
0 & I \\
\end{array} \right)
\left( \begin{array}{cc}
A_{00} & 0 \\
0 & S^{-1} \\
\end{array} \right)
\left( \begin{array}{cc}
I   & 0 \\
-A_{10} & I \\
\end{array} \right)
\left( \begin{array}{cc}
A_{00}^{-1}   & 0 \\
0 & I \\
\end{array} \right).
$$

The preconditioner is accessed with `-pc_fieldsplit_type` `schur` and is
implemented as

$$
\left( \begin{array}{cc}
  \text{ksp}(A_{00},Ap_{00})   & 0 \\
0 & I \\
\end{array} \right)
\left( \begin{array}{cc}
I   & -A_{01} \\
0 & I \\
\end{array} \right)
$$

$$
\left( \begin{array}{cc}
I  & 0 \\
  0 & \text{ksp}(\hat{S},\hat{S}p) \\
\end{array} \right)
\left( \begin{array}{cc}
I   & 0 \\
  -A_{10} \text{ksp}(A_{00},Ap_{00}) & I \\
\end{array} \right).
$$

Where
$\hat{S} = A_{11} - A_{10} \text{ksp}(A_{00},Ap_{00}) A_{01}$ is
the approximate Schur complement.

There are several variants of the Schur complement preconditioner
obtained by dropping some of the terms; these can be obtained with
`-pc_fieldsplit_schur_fact_type <diag,lower,upper,full>` or the
function `PCFieldSplitSetSchurFactType()`. Note that the `diag` form
uses the preconditioner

$$
\left( \begin{array}{cc}
  \text{ksp}(A_{00},Ap_{00})   & 0 \\
  0 & -\text{ksp}(\hat{S},\hat{S}p) \\
\end{array} \right).
$$

This is done to ensure the preconditioner is positive definite for a
a common class of problems, saddle points with a positive definite
$A_{00}$: for these, the Schur complement is negative definite.

The effectiveness of the Schur complement preconditioner depends on the
availability of a good preconditioner $\hat Sp$ for the Schur
complement matrix. In general, you are responsible for supplying
$\hat Sp$ via
`PCFieldSplitSetSchurPre(pc,PC_FIELDSPLIT_SCHUR_PRE_USER,Sp)`.
Without a good problem-specific $\hat Sp$, you can use
some built-in options.

Using `-pc_fieldsplit_schur_precondition user` on the command line
activates the matrix supplied programmatically, as explained above.

With `-pc_fieldsplit_schur_precondition a11` (default)
$\hat Sp = A_{11}$ is used to build a preconditioner for
$\hat S$.

Otherwise, `-pc_fieldsplit_schur_precondition self` will set
$\hat Sp = \hat S$ and use the Schur complement matrix itself to
build the preconditioner.

The problem with the last approach is that $\hat S$ is used in
the unassembled, matrix-free form, and many preconditioners (e.g., ILU)
cannot be built out of such matrices. Instead, you can *assemble* an
approximation to $\hat S$ by inverting $A_{00}$, but only
approximately, to ensure the sparsity of $\hat Sp$ as much
as possible. Specifically, using
`-pc_fieldsplit_schur_precondition selfp` will assemble
$\hat Sp = A_{11} - A_{10} \text{inv}(A_{00}) A_{01}$.

By default $\text{inv}(A_{00})$ is the inverse of the diagonal of
$A_{00}$, but using
`-fieldsplit_1_mat_schur_complement_ainv_type lump` will lump
$A_{00}$ first. Using
`-fieldsplit_1_mat_schur_complement_ainv_type blockdiag` will use the
inverse of the block diagonal of $A_{00}$. Option
`-mat_schur_complement_ainv_type` applies to any matrix of
`MatSchurComplement` type and here it is used with the prefix
`-fieldsplit_1` of the linear system in the second split.

Finally, you can use the `PCLSC` preconditioner for the Schur
complement with `-pc_fieldsplit_type schur -fieldsplit_1_pc_type lsc`.
This uses for the preconditioner to $\hat{S}$ the operator

$$
\text{ksp}(A_{10} A_{01},A_{10} A_{01}) A_{10} A_{00} A_{01} \text{ksp}(A_{10} A_{01},A_{10} A_{01})
$$

Which, of course, introduces two additional inner solves for each
application of the Schur complement. The options prefix for this inner
`KSP` is `-fieldsplit_1_lsc_`. Instead of constructing the matrix
$A_{10} A_{01}$, users can provide their own matrix. This is
done by attaching the matrix/matrices to the $Sp$ matrix they
provide with

```
PetscObjectCompose((PetscObject)Sp,"LSC_L",(PetscObject)L);
PetscObjectCompose((PetscObject)Sp,"LSC_Lp",(PetscObject)Lp);
```

(sec_singular)=

## Solving Singular Systems

Sometimes one is required to solver singular linear systems. In this
case, the system matrix has a nontrivial null space. For example, the
discretization of the Laplacian operator with Neumann boundary
conditions has a null space of the constant functions. PETSc has tools
to help solve these systems. This approach is only guaranteed to work for left preconditioning (see `KSPSetPCSide()`); for example it
may not work in some situations with `KSPFGMRES`.

First, one must know what the null space is and store it using an
orthonormal basis in an array of PETSc Vecs. The constant functions can
be handled separately, since they are such a common case. Create a
`MatNullSpace` object with the command

```
MatNullSpaceCreate(MPI_Comm,PetscBool hasconstants,PetscInt dim,Vec *basis,MatNullSpace *nsp);
```

Here, `dim` is the number of vectors in `basis` and `hasconstants`
indicates if the null space contains the constant functions. If the null
space contains the constant functions you do not need to include it in
the `basis` vectors you provide, nor in the count `dim`.

One then tells the `KSP` object you are using what the null space is
with the call

```
MatSetNullSpace(Mat Amat,MatNullSpace nsp);
```

The `Amat` should be the *first* matrix argument used with
`KSPSetOperators()`, `SNESSetJacobian()`, or `TSSetIJacobian()`.
The PETSc solvers will now
handle the null space during the solution process.

If the right-hand side of linear system is not in the range of `Amat`, that is it is not
orthogonal to the null space of `Amat` transpose, then the residual
norm of the Krylov iteration will not converge to zero; it will converge to a non-zero value while the
solution is converging to the least squares solution of the linear system. One can, if one desires,
apply `MatNullSpaceRemove()` with the null space of `Amat` transpose to the right-hand side before calling
`KSPSolve()`. Then the residual norm will converge to zero.

If one chooses a direct solver (or an incomplete factorization) it may
still detect a zero pivot. You can run with the additional options or
`-pc_factor_shift_type NONZERO`
`-pc_factor_shift_amount  <dampingfactor>` to prevent the zero pivot.
A good choice for the `dampingfactor` is 1.e-10.

If the matrix is non-symmetric and you wish to solve the transposed linear system
you must provide the null space of the transposed matrix with `MatSetTransposeNullSpace()`.

(sec_externalsol)=

## Using External Linear Solvers

PETSc interfaces to several external linear solvers (also see {any}`acknowledgements`).
To use these solvers, one may:

1. Run `configure` with the additional options
   `--download-packagename` e.g. `--download-superlu_dist`
   `--download-parmetis` (SuperLU_DIST needs ParMetis) or
   `--download-mumps` `--download-scalapack` (MUMPS requires
   ScaLAPACK).
2. Build the PETSc libraries.
3. Use the runtime option: `-ksp_type preonly` (or equivalently `-ksp_type none`) `-pc_type <pctype>`
   `-pc_factor_mat_solver_type <packagename>`. For eg:
   `-ksp_type preonly` `-pc_type lu`
   `-pc_factor_mat_solver_type superlu_dist`.

```{eval-rst}
.. list-table:: Options for External Solvers
   :name: tab-externaloptions
   :header-rows: 1

   * - MatType
     - PCType
     - MatSolverType
     - Package
   * - ``seqaij``
     - ``lu``
     - ``MATSOLVERESSL``
     - ``essl``
   * - ``seqaij``
     - ``lu``
     - ``MATSOLVERLUSOL``
     -  ``lusol``
   * - ``seqaij``
     - ``lu``
     - ``MATSOLVERMATLAB``
     - ``matlab``
   * - ``aij``
     - ``lu``
     - ``MATSOLVERMUMPS``
     - ``mumps``
   * - ``aij``
     - ``cholesky``
     - -
     - -
   * - ``sbaij``
     - ``cholesky``
     - -
     - -
   * - ``seqaij``
     - ``lu``
     - ``MATSOLVERSUPERLU``
     - ``superlu``
   * - ``aij``
     - ``lu``
     - ``MATSOLVERSUPERLU_DIST``
     - ``superlu_dist``
   * - ``seqaij``
     - ``lu``
     - ``MATSOLVERUMFPACK``
     - ``umfpack``
   * - ``seqaij``
     - ``cholesky``
     - ``MATSOLVERCHOLMOD``
     - ``cholmod``
   * - ``seqaij``
     - ``lu``
     - ``MATSOLVERKLU``
     -  ``klu``
   * - ``dense``
     - ``lu``
     - ``MATSOLVERELEMENTAL``
     -  ``elemental``
   * - ``dense``
     - ``cholesky``
     - -
     - -
   * - ``seqaij``
     - ``lu``
     - ``MATSOLVERMKL_PARDISO``
     - ``mkl_pardiso``
   * - ``aij``
     - ``lu``
     - ``MATSOLVERMKL_CPARDISO``
     - ``mkl_cpardiso``
   * - ``aij``
     - ``lu``
     - ``MATSOLVERPASTIX``
     -  ``pastix``
   * - ``aij``
     - ``cholesky``
     - ``MATSOLVERBAS``
     -  ``bas``
   * - ``aijcusparse``
     - ``lu``
     - ``MATSOLVERCUSPARSE``
     - ``cusparse``
   * - ``aijcusparse``
     - ``cholesky``
     -  -
     -  -
   * - ``aij``
     - ``lu``, ``cholesky``
     - ``MATSOLVERPETSC``
     - ``petsc``
   * - ``baij``
     - -
     - -
     - -
   * - ``aijcrl``
     - -
     - -
     - -
   * - ``aijperm``
     - -
     - -
     - -
   * - ``seqdense``
     - -
     - -
     - -
   * - ``aij``
     - -
     - -
     - -
   * - ``baij``
     - -
     - -
     - -
   * - ``aijcrl``
     - -
     - -
     - -
   * - ``aijperm``
     - -
     - -
     - -
   * - ``seqdense``
     - -
     - -
     - -
```

The default and available input options for each external software can
be found by specifying `-help` at runtime.

As an alternative to using runtime flags to employ these external
packages, procedural calls are provided for some packages. For example,
the following procedural calls are equivalent to runtime options
`-ksp_type preonly` (or equivalently `-ksp_type none`) `-pc_type lu`
`-pc_factor_mat_solver_type mumps` `-mat_mumps_icntl_7 3`:

```
KSPSetType(ksp, KSPPREONLY); // (or equivalently KSPSetType(ksp, KSPNONE))
KSPGetPC(ksp, &pc);
PCSetType(pc, PCLU);
PCFactorSetMatSolverType(pc, MATSOLVERMUMPS);
PCFactorSetUpMatSolverType(pc);
PCFactorGetMatrix(pc, &F);
icntl=7; ival = 3;
MatMumpsSetIcntl(F, icntl, ival);
```

One can also create matrices with the appropriate capabilities by
calling `MatCreate()` followed by `MatSetType()` specifying the
desired matrix type from {any}`tab-externaloptions`. These
matrix types inherit capabilities from their PETSc matrix parents:
`MATSEQAIJ`, `MATMPIAIJ`, etc. As a result, the preallocation routines
`MatSeqAIJSetPreallocation()`, `MatMPIAIJSetPreallocation()`, etc.
and any other type specific routines of the base class are supported.
One can also call `MatConvert()` inplace to convert the matrix to and
from its base class without performing an expensive data copy.
`MatConvert()` cannot be called on matrices that have already been
factored.

In {any}`tab-externaloptions`, the base class `aij` refers
to the fact that inheritance is based on `MATSEQAIJ` when constructed
with a single process communicator, and from `MATMPIAIJ` otherwise.
The same holds for `baij` and `sbaij`. For codes that are intended
to be run as both a single process or with multiple processes, depending
on the `mpiexec` command, it is recommended that both sets of
preallocation routines are called for these communicator morphing types.
The call for the incorrect type will simply be ignored without any harm
or message.

(sec_pcmpi)=

## Using PETSc's MPI parallel linear solvers from a non-MPI program

Using PETSc's MPI linear solver server it is possible to use multiple MPI processes to solve a
a linear system when the application code, including the matrix generation, is run on a single
MPI process (with or without OpenMP). The application code must be built with MPI and must call
`PetscInitialize()` at the very beginning of the program and end with `PetscFinalize()`. The
application code may utilize OpenMP.
The code may create multiple matrices and `KSP` objects and call `KSPSolve()`, similarly the
code may utilize the `SNES` nonlinear solvers, the `TS` ODE integrators, and the `Tao` optimization algorithms
which use `KSP`.

The program must then be launched using the standard approaches for launching MPI programs with the additional
PETSc option `-mpi_linear_solver_server`. The linear solves are controlled via the options database in the usual manner (using any options prefix
you may have provided via `KSPSetOptionsPrefix()`, for example `-ksp_type cg -ksp_monitor -pc_type bjacobi -ksp_view`. The solver options cannot be set via
the functional interface, for example `KSPSetType()` etc.

The option `-mpi_linear_solver_server_view` will print
a summary of all the systems solved by the MPI linear solver server when the program completes. By default the linear solver server
will only parallelize the linear solve to the extent that it believes is appropriate to obtain speedup for the parallel solve, for example, if the
matrix has 1,000 rows and columns the solution will not be parallelized by default. One can use the option `-mpi_linear_solver_server_minimum_count_per_rank 5000`
to cause the linear solver server to allow as few as 5,000 unknowns per MPI process in the parallel solve.

See `PCMPI`, `PCMPIServerBegin()`, and `PCMPIServerEnd()` for more details on the solvers.

For help when anything goes wrong with the MPI linear solver server see `PCMPIServerBegin()`.

Amdahl's law makes clear that parallelizing only a portion of a numerical code can only provide a limited improvement
in the computation time; thus it is crucial to understand what phases of a computation must be parallelized (via MPI, OpenMP, or some other model)
to ensure a useful increase in performance. One of the crucial phases is likely the generation of the matrix entries; the
use of `MatSetPreallocationCOO()` and `MatSetValuesCOO()` in an OpenMP code allows parallelizing the generation of the matrix.

See {any}`sec_pcmpi_study` for a study of the use of `PCMPI` on a specific PETSc application.

```{rubric} Footnotes
```

[^id3]: See {any}`sec_amg` for information on using algebraic multigrid.

[^id4]: This may seem an odd way to implement since it involves the "extra"
    multiply by $-A_{11}$. The reason is this is implemented this
    way is that this approach works for any number of blocks that may
    overlap.

```{rubric} References
```

```{eval-rst}
.. bibliography:: /petsc.bib
   :filter: docname in docnames
```
