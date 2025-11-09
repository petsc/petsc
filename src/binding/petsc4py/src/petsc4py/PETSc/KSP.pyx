# --------------------------------------------------------------------

class KSPType(object):
    """KSP Type.

    The available types are:

    `RICHARDSON`
        The preconditioned Richardson iterative method
        `petsc.KSPRICHARDSON`.
    `CHEBYSHEV`
        The preconditioned Chebyshev iterative method.
        `petsc.KSPCHEBYSHEV`.
    `CG`
        The Preconditioned Conjugate Gradient (PCG) iterative method.
        `petsc.KSPCG`
    `GROPPCG`
        A pipelined conjugate gradient method (Gropp).
        `petsc.KSPGROPPCG`
    `PIPECG`
        A pipelined conjugate gradient method.
        `petsc.KSPPIPECG`
    `PIPECGRR`
        Pipelined Conjugate Gradients with Residual Replacement.
        `petsc.KSPPIPECGRR`
    `PIPELCG`
        Deep pipelined (length l) Conjugate Gradient method.
        `petsc.KSPPIPELCG`
    `PIPEPRCG`
        Pipelined predict-and-recompute conjugate gradient method.
        `petsc.KSPPIPEPRCG`
    `PIPECG2`
        Pipelined conjugate gradient method with a single non-blocking
        reduction per two iterations. `petsc.KSPPIPECG2`
    `CGNE`
        Applies the preconditioned conjugate gradient method to the
        normal equations without explicitly forming AᵀA. `petsc.KSPCGNE`
    `NASH`
        Conjugate gradient method subject to a constraint
        on the solution norm. `petsc.KSPNASH`
    `STCG`
        Conjugate gradient method subject to a constraint on the
        solution norm. `petsc.KSPSTCG`
    `GLTR`
        Conjugate gradient method subject to a constraint on the
        solution norm. `petsc.KSPGLTR`
    `FCG`
        Flexible Conjugate Gradient method (FCG). Unlike most KSP
        methods this allows the preconditioner to be nonlinear.
        `petsc.KSPFCG`
    `PIPEFCG`
        Pipelined, Flexible Conjugate Gradient method.
        `petsc.KSPPIPEFCG`
    `GMRES`
        Generalized Minimal Residual method with restart.
        `petsc.KSPGMRES`
    `PIPEFGMRES`
        Pipelined (1-stage) Flexible Generalized Minimal Residual
        method. `petsc.KSPPIPEFGMRES`
    `FGMRES`
        Implements the Flexible Generalized Minimal Residual method.
        `petsc.KSPFGMRES`
    `LGMRES`
        Augments the standard Generalized Minimal Residual method
        approximation space with approximations to the error from
        previous restart cycles. `petsc.KSPLGMRES`
    `DGMRES`
        Deflated Generalized Minimal Residual method. In this
        implementation, the adaptive strategy allows to switch to the
        deflated GMRES when the stagnation occurs. `petsc.KSPDGMRES`
    `PGMRES`
        Pipelined Generalized Minimal Residual method.
        `petsc.KSPPGMRES`
    `TCQMR`
        A variant of Quasi Minimal Residual (QMR).
        `petsc.KSPTCQMR`
    `BCGS`
        Stabilized version of Biconjugate Gradient (BiCGStab) method.
        `petsc.KSPBCGS`
    `IBCGS`
        Improved Stabilized version of BiConjugate Gradient (IBiCGStab)
        method in an alternative form to have only a single global
        reduction operation instead of the usual 3 (or 4).
        `petsc.KSPIBCGS`
    `QMRCGS`
        Quasi- Minimal Residual variant of the Bi-CGStab algorithm
        (QMRCGStab) method. `petsc.KSPQMRCGS`
    `FBCGS`
        Flexible Stabilized version of BiConjugate Gradient (BiCGStab)
        method. `petsc.KSPFBCGS`
    `FBCGSR`
        A mathematically equivalent variant of flexible stabilized
        BiConjugate Gradient (BiCGStab). `petsc.KSPFBCGSR`
    `BCGSL`
        Variant of the L-step stabilized BiConjugate Gradient
        (BiCGStab(L)) algorithm. Uses "L-step" Minimal Residual (MR)
        polynomials. The variation concerns cases when some parameters
        are negative due to round-off. `petsc.KSPBCGSL`
    `PIPEBCGS`
        Pipelined stabilized BiConjugate Gradient (BiCGStab) method.
        `petsc.KSPPIPEBCGS`
    `CGS`
        Conjugate Gradient Squared method.
        `petsc.KSPCGS`
    `TFQMR`
        A Transpose Tree Quasi- Minimal Residual (QMR).
        `petsc.KSPCR`
    `CR`
        (Preconditioned) Conjugate Residuals (CR) method.
        `petsc.KSPCR`
    `PIPECR`
        Pipelined Conjugate Residual (CR) method.
        `petsc.KSPPIPECR`
    `LSQR`
        Least squares solver.
        `petsc.KSPLSQR`
    `PREONLY`
        Applies ONLY the preconditioner exactly once. This may be used
        in inner iterations, where it is desired to allow multiple
        iterations as well as the "0-iteration" case. It is commonly
        used with the direct solver preconditioners like PCLU and
        PCCHOLESKY. There is an alias of KSPNONE.
        `petsc.KSPPREONLY`
    `NONE`
        No solver
        ``KSPNONE``
    `QCG`
        Conjugate Gradient (CG) method subject to a constraint on the
        solution norm. `petsc.KSPQCG`
    `BICG`
        Implements the Biconjugate gradient method (BiCG).
        Similar to running the conjugate gradient on the normal equations.
        `petsc.KSPBICG`
    `MINRES`
        Minimum Residual (MINRES) method.
        `petsc.KSPMINRES`
    `SYMMLQ`
        Symmetric LQ method (SymmLQ). Uses LQ decomposition (lower
        trapezoidal).
        `petsc.KSPSYMMLQ`
    `LCD`
        Left Conjugate Direction (LCD) method.
        `petsc.KSPLCD`
    `PYTHON`
        Python shell solver. Call Python function to implement solver.
        ``KSPPYTHON``
    `GCR`
        Preconditioned flexible Generalized Conjugate Residual (GCR)
        method.
        `petsc.KSPGCR`
    `PIPEGCR`
        Pipelined Generalized Conjugate Residual method.
        `petsc.KSPPIPEGCR`
    `TSIRM`
        Two-Stage Iteration with least-squares Residual Minimization
        method. `petsc.KSPTSIRM`
    `CGLS`
        Conjugate Gradient method for Least-Squares problems. Supports
        non-square (rectangular) matrices. `petsc.KSPCGLS`
    `FETIDP`
        Dual-Primal (DP) Finite Element Tearing and Interconnect (FETI)
        method. `petsc.KSPFETIDP`
    `HPDDM`
        Interface with the HPDDM library. This KSP may be used to
        further select methods that are currently not implemented
        natively in PETSc, e.g., GCRODR, a recycled Krylov
        method which is similar to KSPLGMRES. `petsc.KSPHPDDM`

    See Also
    --------
    petsc_options, petsc.KSPType

    """
    RICHARDSON = S_(KSPRICHARDSON)
    CHEBYSHEV  = S_(KSPCHEBYSHEV)
    CG         = S_(KSPCG)
    GROPPCG    = S_(KSPGROPPCG)
    PIPECG     = S_(KSPPIPECG)
    PIPECGRR   = S_(KSPPIPECGRR)
    PIPELCG    = S_(KSPPIPELCG)
    PIPEPRCG   = S_(KSPPIPEPRCG)
    PIPECG2    = S_(KSPPIPECG2)
    CGNE       = S_(KSPCGNE)
    NASH       = S_(KSPNASH)
    STCG       = S_(KSPSTCG)
    GLTR       = S_(KSPGLTR)
    FCG        = S_(KSPFCG)
    PIPEFCG    = S_(KSPPIPEFCG)
    GMRES      = S_(KSPGMRES)
    PIPEFGMRES = S_(KSPPIPEFGMRES)
    FGMRES     = S_(KSPFGMRES)
    LGMRES     = S_(KSPLGMRES)
    DGMRES     = S_(KSPDGMRES)
    PGMRES     = S_(KSPPGMRES)
    TCQMR      = S_(KSPTCQMR)
    BCGS       = S_(KSPBCGS)
    IBCGS      = S_(KSPIBCGS)
    QMRCGS     = S_(KSPQMRCGS)
    FBCGS      = S_(KSPFBCGS)
    FBCGSR     = S_(KSPFBCGSR)
    BCGSL      = S_(KSPBCGSL)
    PIPEBCGS   = S_(KSPPIPEBCGS)
    CGS        = S_(KSPCGS)
    TFQMR      = S_(KSPTFQMR)
    CR         = S_(KSPCR)
    PIPECR     = S_(KSPPIPECR)
    LSQR       = S_(KSPLSQR)
    PREONLY    = S_(KSPPREONLY)
    NONE       = S_(KSPNONE)
    QCG        = S_(KSPQCG)
    BICG       = S_(KSPBICG)
    MINRES     = S_(KSPMINRES)
    SYMMLQ     = S_(KSPSYMMLQ)
    LCD        = S_(KSPLCD)
    PYTHON     = S_(KSPPYTHON)
    GCR        = S_(KSPGCR)
    PIPEGCR    = S_(KSPPIPEGCR)
    TSIRM      = S_(KSPTSIRM)
    CGLS       = S_(KSPCGLS)
    FETIDP     = S_(KSPFETIDP)
    HPDDM      = S_(KSPHPDDM)


class KSPNormType(object):
    """KSP norm type.

    The available norm types are:

    `NONE`
        Skips computing the norm, this should generally only be used if
        you are using the Krylov method as a smoother with a fixed
        small number of iterations. Implicitly sets
        `petsc.KSPConvergedSkip` as KSP convergence test. Note that
        certain algorithms such as `Type.GMRES` ALWAYS require the norm
        calculation, for these methods the norms are still computed,
        they are just not used in the convergence test.
    `PRECONDITIONED`
        The default for left preconditioned solves, uses the l₂ norm of
        the preconditioned residual P⁻¹(b - Ax).
    `UNPRECONDITIONED`
        Uses the l₂ norm of the true b - Ax residual.
    `NATURAL`
        Supported by `Type.CG`, `Type.CR`, `Type.CGNE`, `Type.CGS`.

    """
    # native
    NORM_DEFAULT          = KSP_NORM_DEFAULT
    NORM_NONE             = KSP_NORM_NONE
    NORM_PRECONDITIONED   = KSP_NORM_PRECONDITIONED
    NORM_UNPRECONDITIONED = KSP_NORM_UNPRECONDITIONED
    NORM_NATURAL          = KSP_NORM_NATURAL
    # aliases
    DEFAULT          = NORM_DEFAULT
    NONE = NO        = NORM_NONE
    PRECONDITIONED   = NORM_PRECONDITIONED
    UNPRECONDITIONED = NORM_UNPRECONDITIONED
    NATURAL          = NORM_NATURAL


class KSPConvergedReason(object):
    """KSP Converged Reason.

    `CONVERGED_ITERATING`
        Still iterating
    `ITERATING`
        Still iterating

    `CONVERGED_RTOL_NORMAL_EQUATIONS`
        Undocumented.
    `CONVERGED_ATOL_NORMAL_EQUATIONS`
        Undocumented.
    `CONVERGED_RTOL`
        ∥r∥ <= rtolnorm(b) or rtolnorm(b - Ax₀)
    `CONVERGED_ATOL`
        ∥r∥ <= atol
    `CONVERGED_ITS`
        Used by the `Type.PREONLY` solver after the single iteration of the
        preconditioner is applied. Also used when the
        `petsc.KSPConvergedSkip` convergence test routine is set in KSP.
    `CONVERGED_NEG_CURVE`
        Undocumented.
    `CONVERGED_STEP_LENGTH`
        Undocumented.
    `CONVERGED_HAPPY_BREAKDOWN`
        Undocumented.

    `DIVERGED_NULL`
        Undocumented.
    `DIVERGED_MAX_IT`
        Ran out of iterations before any convergence criteria was
        reached.
    `DIVERGED_DTOL`
        norm(r) >= dtol*norm(b)
    `DIVERGED_BREAKDOWN`
        A breakdown in the Krylov method was detected so the method
        could not continue to enlarge the Krylov space. Could be due to
        a singular matrix or preconditioner. In KSPHPDDM, this is also
        returned when some search directions within a block are
        collinear.
    `DIVERGED_BREAKDOWN_BICG`
        A breakdown in the KSPBICG method was detected so the method
        could not continue to enlarge the Krylov space.
    `DIVERGED_NONSYMMETRIC`
        It appears the operator or preconditioner is not symmetric and
        this Krylov method (`Type.CG`, `Type.MINRES`, `Type.CR`)
        requires symmetry.
    `DIVERGED_INDEFINITE_PC`
        It appears the preconditioner is indefinite (has both positive
        and negative eigenvalues) and this Krylov method (`Type.CG`)
        requires it to be positive definite.
    `DIVERGED_NANORINF`
        Undocumented.
    `DIVERGED_INDEFINITE_MAT`
        Undocumented.
    `DIVERGED_PCSETUP_FAILED`
        It was not possible to build or use the requested
        preconditioner. This is usually due to a zero pivot in a
        factorization. It can also result from a failure in a
        subpreconditioner inside a nested preconditioner such as
        `PC.Type.FIELDSPLIT`.

    See Also
    --------
    `petsc.KSPConvergedReason`

    """
    # iterating
    CONVERGED_ITERATING       = KSP_CONVERGED_ITERATING
    ITERATING                 = KSP_CONVERGED_ITERATING
    # converged
    CONVERGED_RTOL_NORMAL_EQUATIONS = KSP_CONVERGED_RTOL_NORMAL_EQUATIONS
    CONVERGED_ATOL_NORMAL_EQUATIONS = KSP_CONVERGED_ATOL_NORMAL_EQUATIONS
    CONVERGED_RTOL            = KSP_CONVERGED_RTOL
    CONVERGED_ATOL            = KSP_CONVERGED_ATOL
    CONVERGED_ITS             = KSP_CONVERGED_ITS
    CONVERGED_NEG_CURVE       = KSP_CONVERGED_NEG_CURVE
    CONVERGED_STEP_LENGTH     = KSP_CONVERGED_STEP_LENGTH
    CONVERGED_HAPPY_BREAKDOWN = KSP_CONVERGED_HAPPY_BREAKDOWN
    # diverged
    DIVERGED_NULL             = KSP_DIVERGED_NULL
    DIVERGED_MAX_IT           = KSP_DIVERGED_MAX_IT
    DIVERGED_DTOL             = KSP_DIVERGED_DTOL
    DIVERGED_BREAKDOWN        = KSP_DIVERGED_BREAKDOWN
    DIVERGED_BREAKDOWN_BICG   = KSP_DIVERGED_BREAKDOWN_BICG
    DIVERGED_NONSYMMETRIC     = KSP_DIVERGED_NONSYMMETRIC
    DIVERGED_INDEFINITE_PC    = KSP_DIVERGED_INDEFINITE_PC
    DIVERGED_NANORINF         = KSP_DIVERGED_NANORINF
    DIVERGED_INDEFINITE_MAT   = KSP_DIVERGED_INDEFINITE_MAT
    DIVERGED_PCSETUP_FAILED   = KSP_DIVERGED_PC_FAILED


class KSPHPDDMType(object):
    """The *HPDDM* Krylov solver type."""
    GMRES                     = KSP_HPDDM_TYPE_GMRES
    BGMRES                    = KSP_HPDDM_TYPE_BGMRES
    CG                        = KSP_HPDDM_TYPE_CG
    BCG                       = KSP_HPDDM_TYPE_BCG
    GCRODR                    = KSP_HPDDM_TYPE_GCRODR
    BGCRODR                   = KSP_HPDDM_TYPE_BGCRODR
    BFBCG                     = KSP_HPDDM_TYPE_BFBCG
    PREONLY                   = KSP_HPDDM_TYPE_PREONLY

# --------------------------------------------------------------------


cdef class KSP(Object):
    """Abstract PETSc object that manages all Krylov methods.

    This is the object that manages the linear solves in PETSc (even
    those such as direct solvers that do no use Krylov accelerators).

    Notes
    -----
    When a direct solver is used, but no Krylov solver is used, the KSP
    object is still used but with a `Type.PREONLY`, meaning that
    only application of the preconditioner is used as the linear
    solver.

    See Also
    --------
    create, setType, SNES, TS, PC, Type.CG, Type.GMRES,
    petsc.KSP

    """

    Type            = KSPType
    NormType        = KSPNormType
    ConvergedReason = KSPConvergedReason
    HPDDMType       = KSPHPDDMType

    # --- xxx ---

    def __cinit__(self):
        self.obj = <PetscObject*> &self.ksp
        self.ksp = NULL

    def __call__(self, Vec b, Vec x = None) -> Vec:
        """Solve linear system.

        Collective.

        Parameters
        ----------
        b
            Right hand side vector.
        x
            Solution vector.

        Notes
        -----
        Shortcut for `solve`, which returns the solution vector.

        See Also
        --------
        solve, petsc_options, petsc.KSPSolve

        """
        if x is None: # XXX do this better
            x = self.getOperators()[0].createVecLeft()
        self.solve(b, x)
        return x

    # --- xxx ---

    def view(self, Viewer viewer=None) -> None:
        """Print the KSP data structure.

        Collective.

        Parameters
        ----------
        viewer
            Viewer used to display the KSP.

        See Also
        --------
        petsc.KSPView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(KSPView(self.ksp, vwr))

    def destroy(self) -> Self:
        """Destroy KSP context.

        Collective.

        See Also
        --------
        petsc.KSPDestroy

        """
        CHKERR(KSPDestroy(&self.ksp))
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create the KSP context.

        Collective.

        See Also
        --------
        petsc.KSPCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscKSP newksp = NULL
        CHKERR(KSPCreate(ccomm, &newksp))
        CHKERR(PetscCLEAR(self.obj)); self.ksp = newksp
        return self

    def setType(self, ksp_type: Type | str) -> None:
        """Build the `KSP` data structure for a particular `Type`.

        Logically collective.

        Parameters
        ----------
        ksp_type
            KSP Type object

        Notes
        -----
        See `Type` for available methods (for instance, `Type.CG` or
        `Type.GMRES`).

        Normally, it is best to use the `setFromOptions` command
        and then set the KSP type from the options database rather than
        by using this routine. Using the options database provides the
        user with maximum flexibility in evaluating the many different
        Krylov methods. This method is provided for those situations
        where it is necessary to set the iterative solver independently
        of the command line or options database. This might be the
        case, for example, when the choice of iterative solver changes
        during the execution of the program, and the user's application
        is taking responsibility for choosing the appropriate method.
        In other words, this routine is not for beginners.

        See Also
        --------
        petsc.KSPSetType

        """
        cdef PetscKSPType cval = NULL
        ksp_type = str2bytes(ksp_type, &cval)
        CHKERR(KSPSetType(self.ksp, cval))

    def getType(self) -> str:
        """Return the KSP type as a string from the `KSP` object.

        Not collective.

        See Also
        --------
        petsc.KSPGetType

        """
        cdef PetscKSPType cval = NULL
        CHKERR(KSPGetType(self.ksp, &cval))
        return bytes2str(cval)

    def setOptionsPrefix(self, prefix: str | None) -> None:
        """Set the prefix used for all `KSP` options in the database.

        Logically collective.

        Parameters
        ----------
        prefix
            The options prefix.

        Notes
        -----
        A hyphen (-) must NOT be given at the beginning of the prefix
        name. The first character of all runtime options is
        AUTOMATICALLY the hyphen. For example, to distinguish between
        the runtime options for two different `KSP` contexts, one could
        call
        ```
        KSPSetOptionsPrefix(ksp1, "sys1_")
        KSPSetOptionsPrefix(ksp2, "sys2_")
        ```

        This would enable use of different options for each system,
        such as
        ```
        -sys1_ksp_type gmres -sys1_ksp_rtol 1.e-3
        -sys2_ksp_type bcgs  -sys2_ksp_rtol 1.e-4
        ```

        See Also
        --------
        petsc_options, petsc.KSPSetOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR(KSPSetOptionsPrefix(self.ksp, cval))

    def getOptionsPrefix(self) -> str:
        """Return the prefix used for all `KSP` options in the database.

        Not collective.

        See Also
        --------
        petsc.KSPGetOptionsPrefix

        """
        cdef const char *cval = NULL
        CHKERR(KSPGetOptionsPrefix(self.ksp, &cval))
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix: str | None) -> None:
        """Append to prefix used for all `KSP` options in the database.

        Logically collective.

        Parameters
        ----------
        prefix
            The options prefix to append.

        Notes
        -----
        A hyphen (-) must NOT be given at the beginning of the prefix
        name. The first character of all runtime options is
        AUTOMATICALLY the hyphen.

        See Also
        --------
        petsc.KSPAppendOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR(KSPAppendOptionsPrefix(self.ksp, cval))

    def setFromOptions(self) -> None:
        """Set `KSP` options from the options database.

        Collective.

        This routine must be called before `setUp` if the user is
        to be allowed to set the Krylov type.

        See Also
        --------
        petsc_options, petsc.KSPSetFromOptions

        """
        CHKERR(KSPSetFromOptions(self.ksp))

    # --- application context ---

    def setAppCtx(self, appctx: Any) -> None:
        """Set the optional user-defined context for the linear solver.

        Not collective.

        Parameters
        ----------
        appctx
            The user defined context

        Notes
        -----
        The user context is a way for users to attach any information
        to the `KSP` that they may need later when interacting with
        the solver.

        See Also
        --------
        getAppCtx

        """
        self.set_attr('__appctx__', appctx)

    def getAppCtx(self) -> Any:
        """Return the user-defined context for the linear solver.

        Not collective.

        See Also
        --------
        setAppCtx

        """
        return self.get_attr('__appctx__')

    # --- discretization space ---

    def getDM(self) -> DM:
        """Return the `DM` that may be used by some preconditioners.

        Not collective.

        See Also
        --------
        PETSc.KSP, DM, petsc.KSPGetDM

        """
        cdef PetscDM newdm = NULL
        CHKERR(KSPGetDM(self.ksp, &newdm))
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        CHKERR(PetscINCREF(dm.obj))
        return dm

    def setDM(self, DM dm) -> None:
        """Set the `DM` that may be used by some preconditioners.

        Logically collective.

        Parameters
        ----------
        dm
            The `DM` object, cannot be `None`.

        Notes
        -----
        If this is used then the `KSP` will attempt to use the `DM` to
        create the matrix and use the routine set with
        `DM.setKSPComputeOperators`. Use ``setDMActive(False)``
        to instead use the matrix you have provided with
        `setOperators`.

        A `DM` can only be used for solving one problem at a time
        because information about the problem is stored on the `DM`,
        even when not using interfaces like
        `DM.setKSPComputeOperators`. Use `DM.clone` to get a distinct
        `DM` when solving different problems using the same function
        space.

        See Also
        --------
        PETSc.KSP, DM, DM.setKSPComputeOperators, setOperators, DM.clone
        petsc.KSPSetDM

        """
        CHKERR(KSPSetDM(self.ksp, dm.dm))

    def setDMActive(self, flag: bool) -> None:
        """`DM` should be used to generate system matrix & RHS vector.

        Logically collective.

        Parameters
        ----------
        flag
            Whether to use the `DM`.

        Notes
        -----
        By default `setDM` sets the `DM` as active, call
        ``setDMActive(False)`` after ``setDM(dm)`` to not
        have the `KSP` object use the `DM` to generate the matrices.

        See Also
        --------
        PETSc.KSP, DM, setDM, petsc.KSPSetDMActive

        """
        cdef PetscBool cflag = asBool(flag)
        CHKERR(KSPSetDMActive(self.ksp, cflag))

    # --- operators and preconditioner ---

    def setComputeRHS(
        self,
        rhs: KSPRHSFunction,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set routine to compute the right-hand side of the linear system.

        Logically collective.

        Parameters
        ----------
        rhs
            Function which computes the right-hand side.
        args
            Positional arguments for callback function ``rhs``.
        kargs
            Keyword arguments for callback function ``rhs``.

        Notes
        -----
        The routine you provide will be called each time you call `solve`
        to prepare the new right-hand side for that solve.

        See Also
        --------
        PETSc.KSP, solve, petsc.KSPSetComputeRHS

        """
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (rhs, args, kargs)
        self.set_attr('__rhs__', context)
        CHKERR(KSPSetComputeRHS(self.ksp, KSP_ComputeRHS, <void*>context))

    def setComputeOperators(
        self,
        operators: KSPOperatorsFunction,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set routine to compute the linear operators.

        Logically collective.

        Parameters
        ----------
        operators
            Function which computes the operators.
        args
            Positional arguments for callback function ``operators``.
        kargs
            Keyword arguments for callback function ``operators``.

        Notes
        -----
        The user provided function `operators` will be called
        automatically at the very next call to `solve`. It will NOT
        be called at future `solve` calls unless either
        `setComputeOperators` or `setOperators` is called
        before that `solve` is called. This allows the same system
        to be solved several times with different right-hand side
        functions, but is a confusing API since one might expect it to
        be called for each `solve`.

        To reuse the same preconditioner for the next `solve` and
        not compute a new one based on the most recently computed
        matrix call `petsc.KSPSetReusePreconditioner`.

        See Also
        --------
        PETSc.KSP, solve, setOperators, petsc.KSPSetComputeOperators
        petsc.KSPSetReusePreconditioner

        """
        if args  is None: args  = ()
        if kargs is None: kargs = {}
        context = (operators, args, kargs)
        self.set_attr('__operators__', context)
        CHKERR(KSPSetComputeOperators(self.ksp, KSP_ComputeOps, <void*>context))

    def setOperators(self, Mat A=None, Mat P=None) -> None:
        """Set matrix associated with the linear system.

        Collective.

        Set the matrix associated with the linear system and a
        (possibly) different one from which the preconditioner will be
        built.

        Parameters
        ----------
        A
            Matrix that defines the linear system.
        P
            Matrix to be used in constructing the preconditioner,
            usually the same as ``A``.

        Notes
        -----
        This is equivalent to ``pc = ksp.getPC(); pc.setOperators(A, P)``
        but is the preferred approach.

        If you know the operator ``A`` has a null space you can use
        `Mat.setNullSpace` and `Mat.setTransposeNullSpace` to supply the
        null space to ``A`` and the `KSP` solvers will automatically use
        that null space as needed during the solution process.

        All future calls to `setOperators` must use the same size
        matrices!

        Passing `None` for ``A`` or ``P`` removes the matrix that is
        currently used.

        See Also
        --------
        PETSc.KSP, solve, setComputeOperators, petsc.KSPSetOperators

        """
        cdef PetscMat amat=NULL
        if A is not None: amat = A.mat
        cdef PetscMat pmat=amat
        if P is not None: pmat = P.mat
        CHKERR(KSPSetOperators(self.ksp, amat, pmat))

    def getOperators(self) -> tuple[Mat, Mat]:
        """Return the matrix associated with the linear system.

        Collective.

        Return the matrix associated with the linear system and a
        (possibly) different one used to construct the preconditioner.

        Returns
        -------
        A : Mat
            Matrix that defines the linear system.
        P : Mat
            Matrix to be used in constructing the preconditioner,
            usually the same as ``A``.

        See Also
        --------
        PETSc.KSP, solve, setOperators, petsc.KSPGetOperators

        """
        cdef Mat A = Mat(), P = Mat()
        CHKERR(KSPGetOperators(self.ksp, &A.mat, &P.mat))
        CHKERR(PetscINCREF(A.obj))
        CHKERR(PetscINCREF(P.obj))
        return (A, P)

    def setPC(self, PC pc) -> None:
        """Set the preconditioner.

        Collective.

        Set the preconditioner to be used to calculate the application
        of the preconditioner on a vector.

        Parameters
        ----------
        pc
            The preconditioner object

        See Also
        --------
        PETSc.KSP, getPC, petsc.KSPSetPC

        """
        CHKERR(KSPSetPC(self.ksp, pc.pc))

    def getPC(self) -> PC:
        """Return the preconditioner.

        Not collective.

        See Also
        --------
        PETSc.KSP, setPC, petsc.KSPGetPC

        """
        cdef PC pc = PC()
        CHKERR(KSPGetPC(self.ksp, &pc.pc))
        CHKERR(PetscINCREF(pc.obj))
        return pc

    # --- tolerances and convergence ---

    def setTolerances(
        self,
        rtol: float | None = None,
        atol: float | None = None,
        divtol: float | None = None,
        max_it: int | None = None) -> None:
        """Set various tolerances used by the KSP convergence testers.

        Logically collective.

        Set the relative, absolute, divergence, and maximum iteration
        tolerances used by the default KSP convergence testers.

        Parameters
        ----------
        rtol
            The relative convergence tolerance, relative decrease in
            the (possibly preconditioned) residual norm.
            Or `DETERMINE` to use the value when
            the object's type was set.
        atol
            The absolute convergence tolerance absolute size of the
            (possibly preconditioned) residual norm.
            Or `DETERMINE` to use the value when
            the object's type was set.
        dtol
            The divergence tolerance, amount (possibly preconditioned)
            residual norm can increase before
            `petsc.KSPConvergedDefault` concludes that the method is
            diverging.
            Or `DETERMINE` to use the value when
            the object's type was set.
        max_it
            Maximum number of iterations to use.
            Or `DETERMINE` to use the value when
            the object's type was set.

        Notes
        -----
        Use `None` to retain the default value of any of the
        tolerances.

        See Also
        --------
        petsc_options, getTolerances, setConvergenceTest
        petsc.KSPSetTolerances, petsc.KSPConvergedDefault

        """
        cdef PetscReal crtol, catol, cdivtol
        crtol = catol = cdivtol = PETSC_CURRENT
        if rtol   is not None: crtol   = asReal(rtol)
        if atol   is not None: catol   = asReal(atol)
        if divtol is not None: cdivtol = asReal(divtol)
        cdef PetscInt cmaxits = PETSC_CURRENT
        if max_it is not None: cmaxits = asInt(max_it)
        CHKERR(KSPSetTolerances(self.ksp, crtol, catol, cdivtol, cmaxits))

    def getTolerances(self) -> tuple[float, float, float, int]:
        """Return various tolerances used by the KSP convergence tests.

        Not collective.

        Return the relative, absolute, divergence, and maximum iteration
        tolerances used by the default KSP convergence tests.

        Returns
        -------
        rtol : float
            The relative convergence tolerance
        atol : float
            The absolute convergence tolerance
        dtol : float
            The divergence tolerance
        maxits : int
            Maximum number of iterations

        See Also
        --------
        setTolerances, petsc.KSPGetTolerances

        """
        cdef PetscReal crtol=0, catol=0, cdivtol=0
        cdef PetscInt cmaxits=0
        CHKERR(KSPGetTolerances(self.ksp, &crtol, &catol, &cdivtol, &cmaxits))
        return (toReal(crtol), toReal(catol), toReal(cdivtol), toInt(cmaxits))

    def setConvergenceTest(
        self,
        converged: KSPConvergenceTestFunction,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the function to be used to determine convergence.

        Logically collective.

        Parameters
        ----------
        converged
            Callback which computes the convergence.
        args
            Positional arguments for callback function.
        kargs
            Keyword arguments for callback function.

        Notes
        -----
        Must be called after the KSP type has been set so put this
        after a call to `setType`, or `setFromOptions`.

        The default is a combination of relative and absolute
        tolerances. The residual value that is tested may be an
        approximation; routines that need exact values should compute
        them.

        See Also
        --------
        addConvergenceTest, ConvergedReason, setTolerances,
        getConvergenceTest, buildResidual,
        petsc.KSPSetConvergenceTest, petsc.KSPConvergedDefault

        """
        cdef PetscKSPNormType normtype = KSP_NORM_NONE
        cdef void* cctx = NULL
        cdef PetscBool islsqr = PETSC_FALSE
        if converged is not None:
            CHKERR(KSPSetConvergenceTest(
                    self.ksp, KSP_Converged, NULL, NULL))
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__converged__', (converged, args, kargs))
        else:
            # this is wrong in general, since different KSP may use
            # different convergence tests (like KSPLSQR for example)
            # Now we handle LSQR explicitly, but a proper mechanism,
            # say KSPGetDefaultConverged would be more appropriate
            CHKERR(KSPGetNormType(self.ksp, &normtype))
            if normtype != KSP_NORM_NONE:
                CHKERR(PetscObjectTypeCompare(<PetscObject>self.ksp,
                                              KSPLSQR,  &islsqr))
                CHKERR(KSPConvergedDefaultCreate(&cctx))
                if not islsqr:
                    CHKERR(KSPSetConvergenceTest(self.ksp, KSPConvergedDefault,
                                                 cctx, KSPConvergedDefaultDestroy))
                else:
                    CHKERR(KSPSetConvergenceTest(self.ksp, KSPLSQRConvergedDefault,
                                                 cctx, KSPConvergedDefaultDestroy))
            else:
                CHKERR(KSPSetConvergenceTest(self.ksp, KSPConvergedSkip,
                                             NULL, NULL))
            self.set_attr('__converged__', None)

    def addConvergenceTest(
        self,
        converged: KSPConvergenceTestFunction,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None,
        prepend: bool = False) -> None:
        """Add the function to be used to determine convergence.

        Logically collective.

        Parameters
        ----------
        converged
            Callback which computes the convergence.
        args
            Positional arguments for callback function.
        kargs
            Keyword arguments for callback function.
        prepend
            Whether to prepend this call before the default
            convergence test or call it after.

        Notes
        -----
        Cannot be mixed with a call to `setConvergenceTest`.
        It can only be called once. If called multiple times, it will
        generate an error.

        See Also
        --------
        setTolerances, getConvergenceTest, setConvergenceTest,
        petsc.KSPSetConvergenceTest, petsc.KSPConvergedDefault

        """
        cdef object oconverged = self.get_attr("__converged__")
        cdef PetscBool pre = asBool(prepend)
        if converged is None: return
        if oconverged is not None: raise NotImplementedError("converged callback already set or added")
        CHKERR(KSPAddConvergenceTest(self.ksp, KSP_Converged, pre))
        if args is None: args = ()
        if kargs is None: kargs = {}
        self.set_attr('__converged__', (converged, args, kargs))

    def getConvergenceTest(self) -> KSPConvergenceTestFunction:
        """Return the function to be used to determine convergence.

        Logically collective.

        See Also
        --------
        setTolerances, setConvergenceTest, petsc.KSPGetConvergenceTest
        petsc.KSPConvergedDefault

        """
        return self.get_attr('__converged__')

    def callConvergenceTest(self, its: int, rnorm: float) -> None:
        """Call the convergence test callback.

        Collective.

        Parameters
        ----------
        its
            Number of iterations.
        rnorm
            The residual norm.

        Notes
        -----
        This functionality is implemented in petsc4py.

        """
        cdef PetscInt  ival = asInt(its)
        cdef PetscReal rval = asReal(rnorm)
        cdef PetscKSPConvergedReason reason = KSP_CONVERGED_ITERATING
        CHKERR(KSPConvergenceTestCall(self.ksp, ival, rval, &reason))
        return reason

    def setConvergenceHistory(
        self,
        length: int | None = None,
        reset: bool = False) -> None:
        """Set the array used to hold the residual history.

        Not collective.

        If set, this array will contain the residual norms computed at
        each iteration of the solver.

        Parameters
        ----------
        length
            Length of array to store history in.
        reset
            `True` indicates the history counter is reset to zero for
            each new linear solve.

        Notes
        -----
        If ``length`` is not provided or `None` then a default array
        of length 10000 is allocated.

        If the array is not long enough then once the iterations is
        longer than the array length `solve` stops recording the
        history.

        See Also
        --------
        getConvergenceHistory, petsc.KSPSetResidualHistory

        """
        cdef PetscReal *data = NULL
        cdef PetscInt   size = 10000
        cdef PetscBool flag = PETSC_FALSE
        if   length is True:     pass
        elif length is not None: size = asInt(length)
        if size < 0: size = 10000
        if reset: flag = PETSC_TRUE
        cdef object hist = oarray_r(empty_r(size), NULL, &data)
        self.set_attr('__history__', hist)
        CHKERR(KSPSetResidualHistory(self.ksp, data, size, flag))

    def getConvergenceHistory(self) -> ArrayReal:
        """Return array containing the residual history.

        Not collective.

        See Also
        --------
        setConvergenceHistory, petsc.KSPGetResidualHistory

        """
        cdef const PetscReal *data = NULL
        cdef PetscInt   size = 0
        CHKERR(KSPGetResidualHistory(self.ksp, &data, &size))
        return array_r(size, data)

    def logConvergenceHistory(self, rnorm: float) -> None:
        """Add residual to convergence history.

        Logically collective.

        Parameters
        ----------
        rnorm
            Residual norm to be added to convergence history.

        """
        cdef PetscReal rval = asReal(rnorm)
        CHKERR(KSPLogResidualHistory(self.ksp, rval))

    # --- monitoring ---

    def setMonitor(self,
                   monitor: KSPMonitorFunction,
                   args: tuple[Any, ...] | None = None,
                   kargs: dict[str, Any] | None = None) -> None:
        """Set additional function to monitor the residual.

        Logically collective.

        Set an ADDITIONAL function to be called at every iteration to
        monitor the residual/error etc.

        Parameters
        ----------
        monitor
            Callback which monitors the convergence.
        args
            Positional arguments for callback function.
        kargs
            Keyword arguments for callback function.

        Notes
        -----
        The default is to do nothing. To print the residual, or
        preconditioned residual if
        ``setNormType(NORM_PRECONDITIONED)`` was called, use
        `monitor` as the monitoring routine, with a
        `PETSc.Viewer.ASCII` as the context.

        Several different monitoring routines may be set by calling
        `setMonitor` multiple times; all will be called in the order
        in which they were set.

        See Also
        --------
        petsc_options, getMonitor, monitor, monitorCancel, petsc.KSPMonitorSet

        """
        if monitor is None: return
        cdef object monitorlist = self.get_attr('__monitor__')
        if monitorlist is None:
            monitorlist = []
            self.set_attr('__monitor__', monitorlist)
            CHKERR(KSPMonitorSet(self.ksp, KSP_Monitor, NULL, NULL))
        if args is None: args = ()
        if kargs is None: kargs = {}
        monitorlist.append((monitor, args, kargs))

    def getMonitor(self) -> KSPMonitorFunction:
        """Return function used to monitor the residual.

        Not collective.

        See Also
        --------
        petsc_options, setMonitor, monitor, monitorCancel
        petsc.KSPGetMonitorContext

        """
        return self.get_attr('__monitor__')

    def monitorCancel(self) -> None:
        """Clear all monitors for a `KSP` object.

        Logically collective.

        See Also
        --------
        petsc_options, getMonitor, setMonitor, monitor, petsc.KSPMonitorCancel

        """
        CHKERR(KSPMonitorCancel(self.ksp))
        self.set_attr('__monitor__', None)

    cancelMonitor = monitorCancel

    def monitor(self, its: int, rnorm: float) -> None:
        """Run the user provided monitor routines, if they exist.

        Collective.

        Notes
        -----
        This routine is called by the `KSP` implementations. It does not
        typically need to be called by the user.

        See Also
        --------
        setMonitor, petsc.KSPMonitor

        """
        cdef PetscInt  ival = asInt(its)
        cdef PetscReal rval = asReal(rnorm)
        CHKERR(KSPMonitor(self.ksp, ival, rval))

    # --- customization ---

    def setPCSide(self, side: PC.Side) -> None:
        """Set the preconditioning side.

        Logically collective.

        Parameters
        ----------
        side
            The preconditioning side (see `PC.Side`).

        Notes
        -----
        Left preconditioning is used by default for most Krylov methods
        except `Type.FGMRES` which only supports right preconditioning.

        For methods changing the side of the preconditioner changes the
        norm type that is used, see `setNormType`.

        Symmetric preconditioning is currently available only for the
        `Type.QCG` method. Note, however, that symmetric preconditioning
        can be emulated by using either right or left preconditioning
        and a pre or post processing step.

        Setting the PC side often affects the default norm type. See
        `setNormType` for details.

        See Also
        --------
        PC.Side, petsc_options, getPCSide, setNormType, getNormType
        petsc.KSPSetPCSide

        """
        CHKERR(KSPSetPCSide(self.ksp, side))

    def getPCSide(self) -> PC.Side:
        """Return the preconditioning side.

        Not collective.

        See Also
        --------
        petsc_options, setPCSide, setNormType, getNormType, petsc.KSPGetPCSide

        """
        cdef PetscPCSide side = PC_LEFT
        CHKERR(KSPGetPCSide(self.ksp, &side))
        return side

    def setNormType(self, normtype: NormType) -> None:
        """Set the norm that is used for convergence testing.

        Logically collective.

        Parameters
        ----------
        normtype
            The norm type to use (see `NormType`).

        Notes
        -----
        Not all combinations of preconditioner side (see
        `setPCSide`) and norm type are supported by all Krylov
        methods. If only one is set, PETSc tries to automatically
        change the other to find a compatible pair. If no such
        combination is supported, PETSc will generate an error.

        See Also
        --------
        NormType, petsc_options, setUp, solve, destroy, setPCSide, getPCSide
        NormType, petsc.KSPSetNormType, petsc.KSPConvergedSkip
        petsc.KSPSetCheckNormIteration

        """
        CHKERR(KSPSetNormType(self.ksp, normtype))

    def getNormType(self) -> NormType:
        """Return the norm that is used for convergence testing.

        Not collective.

        See Also
        --------
        NormType, setNormType, petsc.KSPGetNormType, petsc.KSPConvergedSkip

        """
        cdef PetscKSPNormType normtype = KSP_NORM_NONE
        CHKERR(KSPGetNormType(self.ksp, &normtype))
        return normtype

    def setComputeEigenvalues(self, flag: bool) -> None:
        """Set a flag to compute eigenvalues.

        Logically collective.

        Set a flag so that the extreme eigenvalues values will be
        calculated via a Lanczos or Arnoldi process as the linear
        system is solved.

        Parameters
        ----------
        flag
            Boolean whether to compute eigenvalues (or not).

        Notes
        -----
        Currently this option is not valid for all iterative methods.

        See Also
        --------
        getComputeEigenvalues, petsc.KSPSetComputeEigenvalues

        """
        cdef PetscBool compute = asBool(flag)
        CHKERR(KSPSetComputeEigenvalues(self.ksp, compute))

    def getComputeEigenvalues(self) -> bool:
        """Return flag indicating whether eigenvalues will be calculated.

        Not collective.

        Return the flag indicating that the extreme eigenvalues values
        will be calculated via a Lanczos or Arnoldi process as the
        linear system is solved.

        See Also
        --------
        setComputeEigenvalues, petsc.KSPSetComputeEigenvalues

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(KSPGetComputeEigenvalues(self.ksp, &flag))
        return toBool(flag)

    def setComputeSingularValues(self, flag: bool) -> None:
        """Set flag to calculate singular values.

        Logically collective.

        Set a flag so that the extreme singular values will be
        calculated via a Lanczos or Arnoldi process as the linear
        system is solved.

        Parameters
        ----------
        flag
            Boolean whether to compute singular values (or not).

        Notes
        -----
        Currently this option is not valid for all iterative methods.

        See Also
        --------
        getComputeSingularValues, petsc.KSPSetComputeSingularValues

        """
        cdef PetscBool compute = asBool(flag)
        CHKERR(KSPSetComputeSingularValues(self.ksp, compute))

    def getComputeSingularValues(self) -> bool:
        """Return flag indicating whether singular values will be calculated.

        Not collective.

        Return the flag indicating whether the extreme singular values
        will be calculated via a Lanczos or Arnoldi process as the
        linear system is solved.

        See Also
        --------
        setComputeSingularValues, petsc.KSPGetComputeSingularValues

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(KSPGetComputeSingularValues(self.ksp, &flag))
        return toBool(flag)

    # --- initial guess ---

    def setInitialGuessNonzero(self, flag: bool) -> None:
        """Tell the iterative solver that the initial guess is nonzero.

        Logically collective.

        Otherwise KSP assumes the initial guess is to be zero (and thus
        zeros it out before solving).

        Parameters
        ----------
        flag
            `True` indicates the guess is non-zero, `False`
            indicates the guess is zero.

        See Also
        --------
        petsc.KSPSetInitialGuessNonzero

        """
        cdef PetscBool guess_nonzero = asBool(flag)
        CHKERR(KSPSetInitialGuessNonzero(self.ksp, guess_nonzero))

    def getInitialGuessNonzero(self) -> bool:
        """Determine whether the KSP solver uses a zero initial guess.

        Not collective.

        See Also
        --------
        petsc.KSPGetInitialGuessNonzero

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(KSPGetInitialGuessNonzero(self.ksp, &flag))
        return toBool(flag)

    def setInitialGuessKnoll(self, flag: bool) -> None:
        """Tell solver to use `PC.apply` to compute the initial guess.

        Logically collective.

        This is the Knoll trick.

        Parameters
        ----------
        flag
            `True` uses Knoll trick.

        See Also
        --------
        petsc.KSPSetInitialGuessKnoll

        """
        cdef PetscBool guess_knoll = asBool(flag)
        CHKERR(KSPSetInitialGuessKnoll(self.ksp, guess_knoll))

    def getInitialGuessKnoll(self) -> bool:
        """Determine whether the KSP solver is using the Knoll trick.

        Not collective.

        This uses the Knoll trick; using `PC.apply` to compute the
        initial guess.

        See Also
        --------
        petsc.KSPGetInitialGuessKnoll

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(KSPGetInitialGuessKnoll(self.ksp, &flag))
        return toBool(flag)

    def setUseFischerGuess(self, model: int, size: int) -> None:
        """Use the Paul Fischer algorithm to compute initial guesses.

        Logically collective.

        Use the Paul Fischer algorithm or its variants to compute
        initial guesses for a set of solves with related right hand
        sides.

        Parameters
        ----------
        model
            Use model ``1``, model ``2``, model ``3``, any other number
            to turn it off.
        size
            Size of subspace used to generate initial guess.

        See Also
        --------
        petsc.KSPSetUseFischerGuess

        """
        cdef PetscInt ival1 = asInt(model)
        cdef PetscInt ival2 = asInt(size)
        CHKERR(KSPSetUseFischerGuess(self.ksp, ival1, ival2))

    # --- solving ---

    def setUp(self) -> None:
        """Set up internal data structures for an iterative solver.

        Collective.

        See Also
        --------
        petsc.KSPSetUp

        """
        CHKERR(KSPSetUp(self.ksp))

    def reset(self) -> None:
        """Resets a KSP context.

        Collective.

        Resets a KSP context to the ``kspsetupcalled = 0`` state and
        removes any allocated Vecs and Mats.

        See Also
        --------
        petsc.KSPReset

        """
        CHKERR(KSPReset(self.ksp))

    def setUpOnBlocks(self) -> None:
        """Set up the preconditioner for each block in a block method.

        Collective.

        Methods include: block Jacobi, block Gauss-Seidel, and
        overlapping Schwarz methods.

        See Also
        --------
        petsc.KSPSetUpOnBlocks

        """
        CHKERR(KSPSetUpOnBlocks(self.ksp))

    def setPreSolve(
        self,
        presolve: KSPPreSolveFunction | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the function that is called at the beginning of each `KSP.solve`.

        Logically collective.

        Parameters
        ----------
        presolve
            The callback function.
        args
            Positional arguments for the callback function.
        kargs
            Keyword arguments for the callback function.

        See Also
        --------
        solve, petsc.KSPSetPreSolve, petsc.KSPSetPostSolve

        """
        if presolve is not None:
            if args is None: args = ()
            if kargs is None: kargs = {}
            context = (presolve, args, kargs)
            self.set_attr('__presolve__', context)
            CHKERR(KSPSetPreSolve(self.ksp, KSP_PreSolve, <void*>context))
        else:
            self.set_attr('__presolve__', None)
            CHKERR(KSPSetPreSolve(self.ksp, NULL, NULL))

    def setPostSolve(
        self,
        postsolve: KSPPostSolveFunction | None,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None) -> None:
        """Set the function that is called at the end of each `KSP.solve`.

        Logically collective.

        Parameters
        ----------
        postsolve
            The callback function.
        args
            Positional arguments for the callback function.
        kargs
            Keyword arguments for the callback function.

        See Also
        --------
        solve, petsc.KSPSetPreSolve, petsc.KSPSetPostSolve

        """
        if postsolve is not None:
            if args is None: args = ()
            if kargs is None: kargs = {}
            context = (postsolve, args, kargs)
            self.set_attr('__postsolve__', context)
            CHKERR(KSPSetPostSolve(self.ksp, KSP_PostSolve, <void*>context))
        else:
            self.set_attr('__postsolve__', None)
            CHKERR(KSPSetPostSolve(self.ksp, NULL, NULL))

    def solve(self, Vec b, Vec x) -> None:
        """Solve the linear system.

        Collective.

        Parameters
        ----------
        b
            Right hand side vector.
        x
            Solution vector.

        Notes
        -----
        If one uses `setDM` then ``x`` or ``b`` need not be passed. Use
        `getSolution` to access the solution in this case.

        The operator is specified with `setOperators`.

        `solve` will normally return without generating an error
        regardless of whether the linear system was solved or if
        constructing the preconditioner failed. Call
        `getConvergedReason` to determine if the solver converged or
        failed and why. The option ``-ksp_error_if_not_converged`` or
        function `setErrorIfNotConverged` will cause `solve` to error
        as soon as an error occurs in the linear solver. In inner
        solves, ``DIVERGED_MAX_IT`` is not treated as an error
        because when using nested solvers it may be fine that inner
        solvers in the preconditioner do not converge during the
        solution process.

        The number of iterations can be obtained from `its`.

        If you provide a matrix that has a `Mat.setNullSpace` and
        `Mat.setTransposeNullSpace` this will use that information to
        solve singular systems in the least squares sense with a norm
        minimizing solution.

        Ax = b where b = bₚ + bₜ where bₜ is not in the range of A
        (and hence by the fundamental theorem of linear algebra is in
        the nullspace(Aᵀ), see `Mat.setNullSpace`.

        KSP first removes bₜ producing the linear system Ax = bₚ (which
        has multiple solutions) and solves this to find the ∥x∥
        minimizing solution (and hence it finds the solution x
        orthogonal to the nullspace(A). The algorithm is simply in each
        iteration of the Krylov method we remove the nullspace(A) from
        the search direction thus the solution which is a linear
        combination of the search directions has no component in the
        nullspace(A).

        We recommend always using `Type.GMRES` for such singular
        systems. If nullspace(A) = nullspace(Aᵀ) (note symmetric
        matrices always satisfy this property) then both left and right
        preconditioning will work If nullspace(A) != nullspace(Aᵀ) then
        left preconditioning will work but right preconditioning may
        not work (or it may).

        If using a direct method (e.g., via the KSP solver
        `Type.PREONLY` and a preconditioner such as `PC.Type.LU` or
        `PC.Type.ILU`, then its=1. See `setTolerances` for more details.

        **Understanding Convergence**

        The routines `setMonitor` and `computeEigenvalues` provide
        information on additional options to monitor convergence and
        print eigenvalue information.

        See Also
        --------
        create, setUp, destroy, setTolerances, is_converged, solveTranspose, its
        Mat.setNullSpace, Mat.setTransposeNullSpace, Type,
        setErrorIfNotConverged petsc_options, petsc.KSPSolve

        """
        cdef PetscVec b_vec = NULL
        cdef PetscVec x_vec = NULL
        if b is not None: b_vec = b.vec
        if x is not None: x_vec = x.vec
        CHKERR(KSPSolve(self.ksp, b_vec, x_vec))

    def solveTranspose(self, Vec b, Vec x) -> None:
        """Solve the transpose of a linear system.

        Collective.

        Parameters
        ----------
        b
            Right hand side vector.
        x
            Solution vector.

        Notes
        -----
        For complex numbers this solve the non-Hermitian transpose
        system.

        See Also
        --------
        solve, petsc.KSPSolveTranspose

        """
        CHKERR(KSPSolveTranspose(self.ksp, b.vec, x.vec))

    def matSolve(self, Mat B, Mat X) -> None:
        """Solve a linear system with multiple right-hand sides.

        Collective.

        These are stored as a `Mat.Type.DENSE`. Unlike `solve`,
        ``B`` and ``X`` must be different matrices.

        Parameters
        ----------
        B
            Block of right-hand sides.
        X
            Block of solutions.

        See Also
        --------
        solve, petsc.KSPMatSolve

        """
        CHKERR(KSPMatSolve(self.ksp, B.mat, X.mat))

    def matSolveTranspose(self, Mat B, Mat X) -> None:
        """Solve the transpose of a linear system with multiple RHS.

        Collective.

        Parameters
        ----------
        B
            Block of right-hand sides.
        X
            Block of solutions.

        See Also
        --------
        solveTranspose, petsc.KSPMatSolve

        """
        CHKERR(KSPMatSolveTranspose(self.ksp, B.mat, X.mat))

    def setIterationNumber(self, its: int) -> None:
        """Use `its` property."""
        cdef PetscInt ival = asInt(its)
        CHKERR(KSPSetIterationNumber(self.ksp, ival))

    def getIterationNumber(self) -> int:
        """Use `its` property."""
        cdef PetscInt ival = 0
        CHKERR(KSPGetIterationNumber(self.ksp, &ival))
        return toInt(ival)

    def setResidualNorm(self, rnorm: float) -> None:
        """Use `norm` property."""
        cdef PetscReal rval = asReal(rnorm)
        CHKERR(KSPSetResidualNorm(self.ksp, rval))

    def getResidualNorm(self) -> float:
        """Use `norm` property."""
        cdef PetscReal rval = 0
        CHKERR(KSPGetResidualNorm(self.ksp, &rval))
        return toReal(rval)

    def setConvergedReason(self, reason: KSP.ConvergedReason) -> None:
        """Use `reason` property."""
        cdef PetscKSPConvergedReason val = reason
        CHKERR(KSPSetConvergedReason(self.ksp, val))

    def getConvergedReason(self) -> KSP.ConvergedReason:
        """Use `reason` property."""
        cdef PetscKSPConvergedReason reason = KSP_CONVERGED_ITERATING
        CHKERR(KSPGetConvergedReason(self.ksp, &reason))
        return reason

    def getCGObjectiveValue(self) -> float:
        """Return the CG objective function value.

        Not collective.

        See Also
        --------
        petsc.KSPCGGetObjFcn

        """
        cdef PetscReal cval = 0
        CHKERR(KSPCGGetObjFcn(self.ksp, &cval))
        return cval

    def setHPDDMType(self, hpddm_type: HPDDMType) -> None:
        """Set the Krylov solver type.

        Collective.

        Parameters
        ----------
        hpddm_type
            The type of Krylov solver to use.

        See Also
        --------
        petsc.KSPHPDDMSetType

        """
        cdef PetscKSPHPDDMType ctype = hpddm_type
        CHKERR(KSPHPDDMSetType(self.ksp, ctype))

    def getHPDDMType(self) -> HPDDMType:
        """Return the Krylov solver type.

        Not collective.

        See Also
        --------
        petsc.KSPHPDDMGetType

        """
        cdef PetscKSPHPDDMType cval = KSP_HPDDM_TYPE_GMRES
        CHKERR(KSPHPDDMGetType(self.ksp, &cval))
        return cval

    def setErrorIfNotConverged(self, flag: bool) -> None:
        """Cause `solve` to generate an error if not converged.

        Logically collective.

        Parameters
        ----------
        flag
            `True` enables this behavior.

        See Also
        --------
        petsc.KSPSetErrorIfNotConverged

        """
        cdef PetscBool ernc = asBool(flag)
        CHKERR(KSPSetErrorIfNotConverged(self.ksp, ernc))

    def getErrorIfNotConverged(self) -> bool:
        """Return the flag indicating the solver will error if divergent.

        Not collective.

        See Also
        --------
        petsc.KSPGetErrorIfNotConverged

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(KSPGetErrorIfNotConverged(self.ksp, &flag))
        return toBool(flag)

    def getRhs(self) -> Vec:
        """Return the right-hand side vector for the linear system.

        Not collective.

        See Also
        --------
        petsc.KSPGetRhs

        """
        cdef Vec vec = Vec()
        CHKERR(KSPGetRhs(self.ksp, &vec.vec))
        CHKERR(PetscINCREF(vec.obj))
        return vec

    def getSolution(self) -> Vec:
        """Return the solution for the linear system to be solved.

        Not collective.

        Note that this may not be the solution that is stored during
        the iterative process.

        See Also
        --------
        petsc.KSPGetSolution

        """
        cdef Vec vec = Vec()
        CHKERR(KSPGetSolution(self.ksp, &vec.vec))
        CHKERR(PetscINCREF(vec.obj))
        return vec

    def getWorkVecs(
        self,
        right: int | None = None,
        left: int | None = None) -> tuple[list[Vec], list[Vec]] | list[Vec] | None:
        """Create working vectors.

        Collective.

        Parameters
        ----------
        right
            Number of right hand vectors to allocate.
        left
            Number of left hand vectors to allocate.

        Returns
        -------
        R : list of Vec
            List of correctly allocated right hand vectors.
        L : list of Vec
            List of correctly allocated left hand vectors.

        """
        cdef bint R = right is not None
        cdef bint L = left  is not None
        cdef PetscInt i=0, nr=0, nl=0
        cdef PetscVec *vr=NULL, *vl=NULL
        if R: nr = asInt(right)
        if L: nl = asInt(left)
        cdef object vecsr = [] if R else None
        cdef object vecsl = [] if L else None
        CHKERR(KSPCreateVecs(self.ksp, nr, &vr, nl, &vr))
        try:
            for i from 0 <= i < nr:
                vecsr.append(ref_Vec(vr[i]))
            for i from 0 <= i < nl:
                vecsl.append(ref_Vec(vl[i]))
        finally:
            if nr > 0 and vr != NULL:
                VecDestroyVecs(nr, &vr) # XXX errors?
            if nl > 0 and vl !=NULL:
                VecDestroyVecs(nl, &vl) # XXX errors?
        #
        if R and L: return (vecsr, vecsl)
        elif R:     return vecsr
        elif L:     return vecsl
        else:       return None

    def buildSolution(self, Vec x=None) -> Vec:
        """Return the solution vector.

        Collective.

        Parameters
        ----------
        x
            Optional vector to store the solution.

        See Also
        --------
        buildResidual, petsc.KSPBuildSolution

        """
        if x is None: x = Vec()
        if x.vec == NULL:
            CHKERR(KSPGetSolution(self.ksp, &x.vec))
            CHKERR(VecDuplicate(x.vec, &x.vec))
        CHKERR(KSPBuildSolution(self.ksp, x.vec, NULL))
        return x

    def buildResidual(self, Vec r=None) -> Vec:
        """Return the residual of the linear system.

        Collective.

        Parameters
        ----------
        r
            Optional vector to use for the result.

        See Also
        --------
        buildSolution, petsc.KSPBuildResidual

        """
        if r is None: r = Vec()
        if r.vec == NULL:
            CHKERR(KSPGetRhs(self.ksp, &r.vec))
            CHKERR(VecDuplicate(r.vec, &r.vec))
        CHKERR(KSPBuildResidual(self.ksp , NULL, r.vec, &r.vec))
        return r

    def computeEigenvalues(self) -> ArrayComplex:
        """Compute the extreme eigenvalues for the preconditioned operator.

        Not collective.

        See Also
        --------
        petsc.KSPComputeEigenvalues

        """
        cdef PetscInt its = 0
        cdef PetscInt neig = 0
        cdef PetscReal *rdata = NULL
        cdef PetscReal *idata = NULL
        CHKERR(KSPGetIterationNumber(self.ksp, &its))
        cdef ndarray r = oarray_r(empty_r(its), NULL, &rdata)
        cdef ndarray i = oarray_r(empty_r(its), NULL, &idata)
        CHKERR(KSPComputeEigenvalues(self.ksp, its, rdata, idata, &neig))
        eigen = empty_c(neig)
        eigen.real = r[:neig]
        eigen.imag = i[:neig]
        return eigen

    def computeExtremeSingularValues(self) -> tuple[float, float]:
        """Compute the extreme singular values for the preconditioned operator.

        Collective.

        Returns
        -------
        smax : float
            The maximum singular value.
        smin : float
            The minimum singular value.

        See Also
        --------
        petsc.KSPComputeExtremeSingularValues

        """
        cdef PetscReal smax = 0
        cdef PetscReal smin = 0
        CHKERR(KSPComputeExtremeSingularValues(self.ksp, &smax, &smin))
        return toReal(smax), toReal(smin)

    # --- GMRES ---

    def setGMRESRestart(self, restart: int) -> None:
        """Set number of iterations at which KSP restarts.

        Logically collective.

        Suitable KSPs are: KSPGMRES, KSPFGMRES and KSPLGMRES.

        Parameters
        ----------
        restart
            Integer restart value.

        See Also
        --------
        petsc.KSPGMRESSetRestart

        """
        cdef PetscInt ival = asInt(restart)
        CHKERR(KSPGMRESSetRestart(self.ksp, ival))

    # --- Python ---

    def createPython(
        self,
        context: Any = None,
        comm: Comm | None = None) -> Self:
        """Create a linear solver of Python type.

        Collective.

        Parameters
        ----------
        context
            An instance of the Python class implementing the required
            methods.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc_python_ksp, setType, setPythonContext, Type.PYTHON

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscKSP newksp = NULL
        CHKERR(KSPCreate(ccomm, &newksp))
        CHKERR(PetscCLEAR(self.obj)); self.ksp = newksp
        CHKERR(KSPSetType(self.ksp, KSPPYTHON))
        CHKERR(KSPPythonSetContext(self.ksp, <void*>context))
        return self

    def setPythonContext(self, context: Any | None = None) -> None:
        """Set the instance of the class implementing Python methods.

        Not collective.

        See Also
        --------
        petsc_python_ksp, getPythonContext

        """
        CHKERR(KSPPythonSetContext(self.ksp, <void*>context))

    def getPythonContext(self) -> Any:
        """Return the instance of the class implementing Python methods.

        Not collective.

        See Also
        --------
        petsc_python_ksp, setPythonContext

        """
        cdef void *context = NULL
        CHKERR(KSPPythonGetContext(self.ksp, &context))
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type: str) -> None:
        """Set the fully qualified Python name of the class to be used.

        Collective.

        See Also
        --------
        petsc_python_ksp, setPythonContext, getPythonType
        petsc.KSPPythonSetType

        """
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR(KSPPythonSetType(self.ksp, cval))

    def getPythonType(self) -> str:
        """Return the fully qualified Python name of the class used by the solver.

        Not collective.

        See Also
        --------
        petsc_python_ksp, setPythonContext, setPythonType
        petsc.KSPPythonGetType

        """
        cdef const char *cval = NULL
        CHKERR(KSPPythonGetType(self.ksp, &cval))
        return bytes2str(cval)

    # --- application context ---

    property appctx:
        """The solver application context."""
        def __get__(self) -> Any:
            return self.getAppCtx()

        def __set__(self, value):
            self.setAppCtx(value)

    # --- discretization space ---

    property dm:
        """The solver `DM`."""
        def __get__(self) -> DM:
            return self.getDM()

        def __set__(self, value):
            self.setDM(value)

    # --- vectors ---

    property vec_sol:
        """The solution vector."""
        def __get__(self) -> Vec:
            return self.getSolution()

    property vec_rhs:
        """The right-hand side vector."""
        def __get__(self) -> Vec:
            return self.getRhs()

    # --- operators ---

    property mat_op:
        """The system matrix operator."""
        def __get__(self) -> Mat:
            return self.getOperators()[0]

    property mat_pc:
        """The preconditioner operator."""
        def __get__(self) -> Mat:
            return self.getOperators()[1]

    # --- initial guess ---

    property guess_nonzero:
        """Whether guess is non-zero."""
        def __get__(self) -> bool:
            return self.getInitialGuessNonzero()

        def __set__(self, value):
            self.setInitialGuessNonzero(value)

    property guess_knoll:
        """Whether solver uses Knoll trick."""
        def __get__(self) -> bool:
            return self.getInitialGuessKnoll()

        def __set__(self, value):
            self.setInitialGuessKnoll(value)

    # --- preconditioner ---

    property pc:
        """The `PC` of the solver."""
        def __get__(self) -> PC:
            return self.getPC()

    property pc_side:
        """The side on which preconditioning is performed."""
        def __get__(self) -> PC.Side:
            return self.getPCSide()

        def __set__(self, value):
            self.setPCSide(value)

    property norm_type:
        """The norm used by the solver."""
        def __get__(self) -> NormType:
            return self.getNormType()

        def __set__(self, value):
            self.setNormType(value)

    # --- tolerances ---

    property rtol:
        """The relative tolerance of the solver."""
        def __get__(self) -> float:
            return self.getTolerances()[0]

        def __set__(self, value):
            self.setTolerances(rtol=value)

    property atol:
        """The absolute tolerance of the solver."""
        def __get__(self) -> float:
            return self.getTolerances()[1]

        def __set__(self, value):
            self.setTolerances(atol=value)

    property divtol:
        """The divergence tolerance of the solver."""
        def __get__(self) -> float:
            return self.getTolerances()[2]

        def __set__(self, value):
            self.setTolerances(divtol=value)

    property max_it:
        """The maximum number of iteration the solver may take."""
        def __get__(self) -> int:
            return self.getTolerances()[3]

        def __set__(self, value):
            self.setTolerances(max_it=value)

    # --- iteration ---

    property its:
        """The current number of iterations the solver has taken."""
        def __get__(self) -> int:
            return self.getIterationNumber()

        def __set__(self, value):
            self.setIterationNumber(value)

    property norm:
        """The norm of the residual at the current iteration."""
        def __get__(self) -> float:
            return self.getResidualNorm()

        def __set__(self, value):
            self.setResidualNorm(value)

    property history:
        """The convergence history of the solver."""
        def __get__(self) -> ndarray:
            return self.getConvergenceHistory()

    # --- convergence ---

    property reason:
        """The converged reason."""
        def __get__(self) -> KSP.ConvergedReason:
            return self.getConvergedReason()

        def __set__(self, value):
            self.setConvergedReason(value)

    property is_iterating:
        """Boolean indicating if the solver has not converged yet."""
        def __get__(self) -> bool:
            return self.reason == 0

    property is_converged:
        """Boolean indicating if the solver has converged."""
        def __get__(self) -> bool:
            return self.reason > 0

    property is_diverged:
        """Boolean indicating if the solver has failed."""
        def __get__(self) -> bool:
            return self.reason < 0

# --------------------------------------------------------------------

del KSPType
del KSPNormType
del KSPConvergedReason
del KSPHPDDMType

# --------------------------------------------------------------------
