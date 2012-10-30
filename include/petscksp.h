/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __PETSCKSP_H
#define __PETSCKSP_H
#include <petscpc.h>

PETSC_EXTERN PetscErrorCode KSPInitializePackage(const char[]);

/*S
     KSP - Abstract PETSc object that manages all Krylov methods

   Level: beginner

  Concepts: Krylov methods

.seealso:  KSPCreate(), KSPSetType(), KSPType, SNES, TS, PC, KSP
S*/
typedef struct _p_KSP*     KSP;

/*J
    KSPType - String with the name of a PETSc Krylov method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mykspcreate()

   Level: beginner

.seealso: KSPSetType(), KSP
J*/
typedef const char* KSPType;
#define KSPRICHARDSON "richardson"
#define KSPCHEBYSHEV  "chebyshev"
#define KSPCG         "cg"
#define   KSPCGNE       "cgne"
#define   KSPNASH       "nash"
#define   KSPSTCG       "stcg"
#define   KSPGLTR       "gltr"
#define KSPGMRES      "gmres"
#define   KSPFGMRES     "fgmres"
#define   KSPLGMRES     "lgmres"
#define   KSPDGMRES     "dgmres"
#define   KSPPGMRES     "pgmres"
#define KSPTCQMR      "tcqmr"
#define KSPBCGS       "bcgs"
#define   KSPIBCGS      "ibcgs"
#define   KSPFBCGS      "fbcgs"
#define   KSPIFBCGS     "ifbcgs"
#define   KSPTFIBCGS    "tfibcgs"
#define   KSPBCGSL      "bcgsl"
#define KSPCGS        "cgs"
#define KSPTFQMR      "tfqmr"
#define KSPCR         "cr"
#define KSPLSQR       "lsqr"
#define KSPPREONLY    "preonly"
#define KSPQCG        "qcg"
#define KSPBICG       "bicg"
#define KSPMINRES     "minres"
#define KSPSYMMLQ     "symmlq"
#define KSPLCD        "lcd"
#define KSPPYTHON     "python"
#define KSPGCR        "gcr"
#define KSPSPECEST    "specest"

/* Logging support */
PETSC_EXTERN PetscClassId KSP_CLASSID;

PETSC_EXTERN PetscErrorCode KSPCreate(MPI_Comm,KSP *);
PETSC_EXTERN PetscErrorCode KSPSetType(KSP,KSPType);
PETSC_EXTERN PetscErrorCode KSPSetUp(KSP);
PETSC_EXTERN PetscErrorCode KSPSetUpOnBlocks(KSP);
PETSC_EXTERN PetscErrorCode KSPSolve(KSP,Vec,Vec);
PETSC_EXTERN PetscErrorCode KSPSolveTranspose(KSP,Vec,Vec);
PETSC_EXTERN PetscErrorCode KSPReset(KSP);
PETSC_EXTERN PetscErrorCode KSPDestroy(KSP*);

PETSC_EXTERN PetscFList KSPList;
PETSC_EXTERN PetscBool KSPRegisterAllCalled;
PETSC_EXTERN PetscErrorCode KSPRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode KSPRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode KSPRegister(const char[],const char[],const char[],PetscErrorCode (*)(KSP));
PETSC_EXTERN PetscErrorCode KSPMatRegisterAll(const char[]);

/*MC
   KSPRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   PetscErrorCode KSPRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(KSP))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   KSPRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   KSPRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     KSPSetType(ksp,"my_solver")
   or at runtime via the option
$     -ksp_type my_solver

   Level: advanced

   Notes: Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
          and others of the form ${any_environmental_variable} occuring in pathname will be
          replaced with appropriate values.
         If your function is not being put into a shared library then use KSPRegister() instead

.keywords: KSP, register

.seealso: KSPRegisterAll(), KSPRegisterDestroy()

M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define KSPRegisterDynamic(a,b,c,d) KSPRegister(a,b,c,0)
#else
#define KSPRegisterDynamic(a,b,c,d) KSPRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode KSPGetType(KSP,KSPType *);
PETSC_EXTERN PetscErrorCode KSPSetPCSide(KSP,PCSide);
PETSC_EXTERN PetscErrorCode KSPGetPCSide(KSP,PCSide*);
PETSC_EXTERN PetscErrorCode KSPGetTolerances(KSP,PetscReal*,PetscReal*,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode KSPSetTolerances(KSP,PetscReal,PetscReal,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode KSPSetInitialGuessNonzero(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPGetInitialGuessNonzero(KSP,PetscBool  *);
PETSC_EXTERN PetscErrorCode KSPSetInitialGuessKnoll(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPGetInitialGuessKnoll(KSP,PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetErrorIfNotConverged(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPGetErrorIfNotConverged(KSP,PetscBool  *);
PETSC_EXTERN PetscErrorCode KSPGetComputeEigenvalues(KSP,PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetComputeEigenvalues(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPGetComputeSingularValues(KSP,PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetComputeSingularValues(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPGetRhs(KSP,Vec *);
PETSC_EXTERN PetscErrorCode KSPGetSolution(KSP,Vec *);
PETSC_EXTERN PetscErrorCode KSPGetResidualNorm(KSP,PetscReal*);
PETSC_EXTERN PetscErrorCode KSPGetIterationNumber(KSP,PetscInt*);
PETSC_EXTERN PetscErrorCode KSPSetNullSpace(KSP,MatNullSpace);
PETSC_EXTERN PetscErrorCode KSPGetNullSpace(KSP,MatNullSpace*);
PETSC_EXTERN PetscErrorCode KSPGetVecs(KSP,PetscInt,Vec**,PetscInt,Vec**);

PETSC_EXTERN PetscErrorCode KSPSetPC(KSP,PC);
PETSC_EXTERN PetscErrorCode KSPGetPC(KSP,PC*);

PETSC_EXTERN PetscErrorCode KSPMonitor(KSP,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode KSPMonitorSet(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode KSPMonitorCancel(KSP);
PETSC_EXTERN PetscErrorCode KSPGetMonitorContext(KSP,void **);
PETSC_EXTERN PetscErrorCode KSPGetResidualHistory(KSP,PetscReal*[],PetscInt *);
PETSC_EXTERN PetscErrorCode KSPSetResidualHistory(KSP,PetscReal[],PetscInt,PetscBool );

/* not sure where to put this */
PETSC_EXTERN PetscErrorCode PCKSPGetKSP(PC,KSP*);
PETSC_EXTERN PetscErrorCode PCBJacobiGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
PETSC_EXTERN PetscErrorCode PCASMGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
PETSC_EXTERN PetscErrorCode PCGASMGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
PETSC_EXTERN PetscErrorCode PCFieldSplitGetSubKSP(PC,PetscInt*,KSP*[]);

PETSC_EXTERN PetscErrorCode PCGalerkinGetKSP(PC,KSP *);

PETSC_EXTERN PetscErrorCode KSPBuildSolution(KSP,Vec,Vec *);
PETSC_EXTERN PetscErrorCode KSPBuildResidual(KSP,Vec,Vec,Vec *);

PETSC_EXTERN PetscErrorCode KSPRichardsonSetScale(KSP,PetscReal);
PETSC_EXTERN PetscErrorCode KSPRichardsonSetSelfScale(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPChebyshevSetEigenvalues(KSP,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode KSPChebyshevSetEstimateEigenvalues(KSP,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode KSPChebyshevSetNewMatrix(KSP);
PETSC_EXTERN PetscErrorCode KSPComputeExtremeSingularValues(KSP,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode KSPComputeEigenvalues(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt *);
PETSC_EXTERN PetscErrorCode KSPComputeEigenvaluesExplicitly(KSP,PetscInt,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode KSPGMRESSetRestart(KSP, PetscInt);
PETSC_EXTERN PetscErrorCode KSPGMRESGetRestart(KSP, PetscInt*);
PETSC_EXTERN PetscErrorCode KSPGMRESSetHapTol(KSP,PetscReal);

PETSC_EXTERN PetscErrorCode KSPGMRESSetPreAllocateVectors(KSP);
PETSC_EXTERN PetscErrorCode KSPGMRESSetOrthogonalization(KSP,PetscErrorCode (*)(KSP,PetscInt));
PETSC_EXTERN PetscErrorCode KSPGMRESGetOrthogonalization(KSP,PetscErrorCode (**)(KSP,PetscInt));
PETSC_EXTERN PetscErrorCode KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,PetscInt);
PETSC_EXTERN PetscErrorCode KSPGMRESClassicalGramSchmidtOrthogonalization(KSP,PetscInt);

PETSC_EXTERN PetscErrorCode KSPLGMRESSetAugDim(KSP,PetscInt);
PETSC_EXTERN PetscErrorCode KSPLGMRESSetConstant(KSP);

PETSC_EXTERN PetscErrorCode KSPGCRSetRestart(KSP,PetscInt);
PETSC_EXTERN PetscErrorCode KSPGCRGetRestart(KSP,PetscInt*);
PETSC_EXTERN PetscErrorCode KSPGCRSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

/*E
    KSPGMRESCGSRefinementType - How the classical (unmodified) Gram-Schmidt is performed.

   Level: advanced

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSPGMRESModifiedGramSchmidtOrthogonalization()

E*/
typedef enum {KSP_GMRES_CGS_REFINE_NEVER, KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS} KSPGMRESCGSRefinementType;
PETSC_EXTERN const char *const KSPGMRESCGSRefinementTypes[];
/*MC
    KSP_GMRES_CGS_REFINE_NEVER - Just do the classical (unmodified) Gram-Schmidt process

   Level: advanced

   Note: Possible unstable, but the fastest to compute

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS,
          KSPGMRESModifiedGramSchmidtOrthogonalization()
M*/

/*MC
    KSP_GMRES_CGS_REFINE_IFNEEDED - Do the classical (unmodified) Gram-Schmidt process and one step of
          iterative refinement if an estimate of the orthogonality of the resulting vectors indicates
          poor orthogonality.

   Level: advanced

   Note: This is slower than KSP_GMRES_CGS_REFINE_NEVER because it requires an extra norm computation to
     estimate the orthogonality but is more stable.

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSP_GMRES_CGS_REFINE_NEVER, KSP_GMRES_CGS_REFINE_ALWAYS,
          KSPGMRESModifiedGramSchmidtOrthogonalization()
M*/

/*MC
    KSP_GMRES_CGS_REFINE_NEVER - Do two steps of the classical (unmodified) Gram-Schmidt process.

   Level: advanced

   Note: This is roughly twice the cost of KSP_GMRES_CGS_REFINE_NEVER because it performs the process twice
     but it saves the extra norm calculation needed by KSP_GMRES_CGS_REFINE_IFNEEDED.

        You should only use this if you absolutely know that the iterative refinement is needed.

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS,
          KSPGMRESModifiedGramSchmidtOrthogonalization()
M*/

PETSC_EXTERN PetscErrorCode KSPGMRESSetCGSRefinementType(KSP,KSPGMRESCGSRefinementType);
PETSC_EXTERN PetscErrorCode KSPGMRESGetCGSRefinementType(KSP,KSPGMRESCGSRefinementType*);

PETSC_EXTERN PetscErrorCode KSPFGMRESModifyPCNoChange(KSP,PetscInt,PetscInt,PetscReal,void*);
PETSC_EXTERN PetscErrorCode KSPFGMRESModifyPCKSP(KSP,PetscInt,PetscInt,PetscReal,void*);
PETSC_EXTERN PetscErrorCode KSPFGMRESSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

PETSC_EXTERN PetscErrorCode KSPQCGSetTrustRegionRadius(KSP,PetscReal);
PETSC_EXTERN PetscErrorCode KSPQCGGetQuadratic(KSP,PetscReal*);
PETSC_EXTERN PetscErrorCode KSPQCGGetTrialStepNorm(KSP,PetscReal*);

PETSC_EXTERN PetscErrorCode KSPBCGSLSetXRes(KSP,PetscReal);
PETSC_EXTERN PetscErrorCode KSPBCGSLSetPol(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPBCGSLSetEll(KSP,PetscInt);

PETSC_EXTERN PetscErrorCode KSPSetFromOptions(KSP);
PETSC_EXTERN PetscErrorCode KSPAddOptionsChecker(PetscErrorCode (*)(KSP));

PETSC_EXTERN PetscErrorCode KSPMonitorSingularValue(KSP,PetscInt,PetscReal,void *);
PETSC_EXTERN PetscErrorCode KSPMonitorDefault(KSP,PetscInt,PetscReal,void *);
PETSC_EXTERN PetscErrorCode KSPLSQRMonitorDefault(KSP,PetscInt,PetscReal,void *);
PETSC_EXTERN PetscErrorCode KSPMonitorRange(KSP,PetscInt,PetscReal,void *);
PETSC_EXTERN PetscErrorCode KSPMonitorDynamicTolerance(KSP ksp,PetscInt its,PetscReal fnorm,void *dummy);
PETSC_EXTERN PetscErrorCode KSPMonitorDynamicToleranceDestroy(void **dummy);
PETSC_EXTERN PetscErrorCode KSPMonitorTrueResidualNorm(KSP,PetscInt,PetscReal,void *);
PETSC_EXTERN PetscErrorCode KSPMonitorTrueResidualMaxNorm(KSP,PetscInt,PetscReal,void *);
PETSC_EXTERN PetscErrorCode KSPMonitorDefaultShort(KSP,PetscInt,PetscReal,void *);
PETSC_EXTERN PetscErrorCode KSPMonitorSolution(KSP,PetscInt,PetscReal,void *);
PETSC_EXTERN PetscErrorCode KSPMonitorAMS(KSP,PetscInt,PetscReal,void*);
PETSC_EXTERN PetscErrorCode KSPMonitorAMSCreate(KSP,const char*,void**);
PETSC_EXTERN PetscErrorCode KSPMonitorAMSDestroy(void**);
PETSC_EXTERN PetscErrorCode KSPGMRESMonitorKrylov(KSP,PetscInt,PetscReal,void *);

PETSC_EXTERN PetscErrorCode KSPUnwindPreconditioner(KSP,Vec,Vec);
PETSC_EXTERN PetscErrorCode KSPDefaultBuildSolution(KSP,Vec,Vec*);
PETSC_EXTERN PetscErrorCode KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);
PETSC_EXTERN PetscErrorCode KSPInitialResidual(KSP,Vec,Vec,Vec,Vec,Vec);

PETSC_EXTERN PetscErrorCode KSPSetOperators(KSP,Mat,Mat,MatStructure);
PETSC_EXTERN PetscErrorCode KSPGetOperators(KSP,Mat*,Mat*,MatStructure*);
PETSC_EXTERN PetscErrorCode KSPGetOperatorsSet(KSP,PetscBool *,PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetOptionsPrefix(KSP,const char[]);
PETSC_EXTERN PetscErrorCode KSPAppendOptionsPrefix(KSP,const char[]);
PETSC_EXTERN PetscErrorCode KSPGetOptionsPrefix(KSP,const char*[]);
PETSC_EXTERN PetscErrorCode KSPSetTabLevel(KSP,PetscInt);
PETSC_EXTERN PetscErrorCode KSPGetTabLevel(KSP,PetscInt*);

PETSC_EXTERN PetscErrorCode KSPSetDiagonalScale(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPGetDiagonalScale(KSP,PetscBool *);
PETSC_EXTERN PetscErrorCode KSPSetDiagonalScaleFix(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPGetDiagonalScaleFix(KSP,PetscBool *);

PETSC_EXTERN PetscErrorCode KSPView(KSP,PetscViewer);

PETSC_EXTERN PetscErrorCode KSPLSQRSetStandardErrorVec(KSP,Vec);
PETSC_EXTERN PetscErrorCode KSPLSQRGetStandardErrorVec(KSP,Vec*);

PETSC_EXTERN PetscErrorCode PCRedundantGetKSP(PC,KSP*);
PETSC_EXTERN PetscErrorCode PCRedistributeGetKSP(PC,KSP*);

/*E
    KSPNormType - Norm that is passed in the Krylov convergence
       test routines.

   Level: advanced

   Each solver only supports a subset of these and some may support different ones
   depending on left or right preconditioning, see KSPSetPCSide()

   Notes: this must match finclude/petscksp.h

.seealso: KSPSolve(), KSPGetConvergedReason(), KSPSetNormType(),
          KSPSetConvergenceTest(), KSPSetPCSide()
E*/
typedef enum {KSP_NORM_DEFAULT = -1,KSP_NORM_NONE = 0,KSP_NORM_PRECONDITIONED = 1,KSP_NORM_UNPRECONDITIONED = 2,KSP_NORM_NATURAL = 3} KSPNormType;
#define KSP_NORM_MAX (KSP_NORM_NATURAL + 1)
PETSC_EXTERN const char *const*const KSPNormTypes;

/*MC
    KSP_NORM_NONE - Do not compute a norm during the Krylov process. This will
          possibly save some computation but means the convergence test cannot
          be based on a norm of a residual etc.

   Level: advanced

    Note: Some Krylov methods need to compute a residual norm and then this option is ignored

.seealso: KSPNormType, KSPSetNormType(), KSP_NORM_PRECONDITIONED, KSP_NORM_UNPRECONDITIONED, KSP_NORM_NATURAL
M*/

/*MC
    KSP_NORM_PRECONDITIONED - Compute the norm of the preconditioned residual and pass that to the
       convergence test routine.

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSP_NORM_NONE, KSP_NORM_UNPRECONDITIONED, KSP_NORM_NATURAL, KSPSetConvergenceTest()
M*/

/*MC
    KSP_NORM_UNPRECONDITIONED - Compute the norm of the true residual (b - A*x) and pass that to the
       convergence test routine.

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSP_NORM_NONE, KSP_NORM_PRECONDITIONED, KSP_NORM_NATURAL, KSPSetConvergenceTest()
M*/

/*MC
    KSP_NORM_NATURAL - Compute the 'natural norm' of residual sqrt((b - A*x)*B*(b - A*x)) and pass that to the
       convergence test routine. This is only supported by  KSPCG, KSPCR, KSPCGNE, KSPCGS

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSP_NORM_NONE, KSP_NORM_PRECONDITIONED, KSP_NORM_UNPRECONDITIONED, KSPSetConvergenceTest()
M*/

PETSC_EXTERN PetscErrorCode KSPSetNormType(KSP,KSPNormType);
PETSC_EXTERN PetscErrorCode KSPGetNormType(KSP,KSPNormType*);
PETSC_EXTERN PetscErrorCode KSPSetSupportedNorm(KSP ksp,KSPNormType,PCSide,PetscInt);
PETSC_EXTERN PetscErrorCode KSPSetCheckNormIteration(KSP,PetscInt);
PETSC_EXTERN PetscErrorCode KSPSetLagNorm(KSP,PetscBool );

/*E
    KSPConvergedReason - reason a Krylov method was said to
         have converged or diverged

   Level: beginner

   Notes: See KSPGetConvergedReason() for explanation of each value

   Developer notes: this must match finclude/petscksp.h

      The string versions of these are KSPConvergedReasons; if you change
      any of the values here also change them that array of names.

.seealso: KSPSolve(), KSPGetConvergedReason(), KSPSetTolerances()
E*/
typedef enum {/* converged */
              KSP_CONVERGED_RTOL_NORMAL        =  1,
              KSP_CONVERGED_ATOL_NORMAL        =  9,
              KSP_CONVERGED_RTOL               =  2,
              KSP_CONVERGED_ATOL               =  3,
              KSP_CONVERGED_ITS                =  4,
              KSP_CONVERGED_CG_NEG_CURVE       =  5,
              KSP_CONVERGED_CG_CONSTRAINED     =  6,
              KSP_CONVERGED_STEP_LENGTH        =  7,
              KSP_CONVERGED_HAPPY_BREAKDOWN    =  8,
              /* diverged */
              KSP_DIVERGED_NULL                = -2,
              KSP_DIVERGED_ITS                 = -3,
              KSP_DIVERGED_DTOL                = -4,
              KSP_DIVERGED_BREAKDOWN           = -5,
              KSP_DIVERGED_BREAKDOWN_BICG      = -6,
              KSP_DIVERGED_NONSYMMETRIC        = -7,
              KSP_DIVERGED_INDEFINITE_PC       = -8,
              KSP_DIVERGED_NAN                 = -9,
              KSP_DIVERGED_INDEFINITE_MAT      = -10,

              KSP_CONVERGED_ITERATING          =  0} KSPConvergedReason;
PETSC_EXTERN const char *const*KSPConvergedReasons;

/*MC
     KSP_CONVERGED_RTOL - norm(r) <= rtol*norm(b)

   Level: beginner

   See KSPNormType and KSPSetNormType() for possible norms that may be used. By default
       for left preconditioning it is the 2-norm of the preconditioned residual, and the
       2-norm of the residual for right preconditioning

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_CONVERGED_ATOL - norm(r) <= atol

   Level: beginner

   See KSPNormType and KSPSetNormType() for possible norms that may be used. By default
       for left preconditioning it is the 2-norm of the preconditioned residual, and the
       2-norm of the residual for right preconditioning

   Level: beginner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_DIVERGED_DTOL - norm(r) >= dtol*norm(b)

   Level: beginner

   See KSPNormType and KSPSetNormType() for possible norms that may be used. By default
       for left preconditioning it is the 2-norm of the preconditioned residual, and the
       2-norm of the residual for right preconditioning

   Level: beginner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_DIVERGED_ITS - Ran out of iterations before any convergence criteria was
      reached

   Level: beginner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_CONVERGED_ITS - Used by the KSPPREONLY solver after the single iteration of
           the preconditioner is applied. Also used when the KSPSkipConverged() convergence
           test routine is set in KSP.


   Level: beginner


.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_DIVERGED_BREAKDOWN - A breakdown in the Krylov method was detected so the
          method could not continue to enlarge the Krylov space. Could be due to a singlular matrix or
          preconditioner.

   Level: beginner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_DIVERGED_BREAKDOWN_BICG - A breakdown in the KSPBICG method was detected so the
          method could not continue to enlarge the Krylov space.


   Level: beginner


.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_DIVERGED_NONSYMMETRIC - It appears the operator or preconditioner is not
        symmetric and this Krylov method (KSPCG, KSPMINRES, KSPCR) requires symmetry

   Level: beginner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_DIVERGED_INDEFINITE_PC - It appears the preconditioner is indefinite (has both
        positive and negative eigenvalues) and this Krylov method (KSPCG) requires it to
        be positive definite

   Level: beginner

     Notes: This can happen with the PCICC preconditioner, use -pc_factor_shift_positive_definite to force
  the PCICC preconditioner to generate a positive definite preconditioner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_CONVERGED_ITERATING - This flag is returned if you call KSPGetConvergedReason()
        while the KSPSolve() is still running.

   Level: beginner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

PETSC_EXTERN PetscErrorCode KSPSetConvergenceTest(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),void *,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode KSPGetConvergenceContext(KSP,void **);
PETSC_EXTERN PetscErrorCode KSPDefaultConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
PETSC_EXTERN PetscErrorCode KSPConvergedLSQR(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
PETSC_EXTERN PetscErrorCode KSPDefaultConvergedDestroy(void *);
PETSC_EXTERN PetscErrorCode KSPDefaultConvergedCreate(void **);
PETSC_EXTERN PetscErrorCode KSPDefaultConvergedSetUIRNorm(KSP);
PETSC_EXTERN PetscErrorCode KSPDefaultConvergedSetUMIRNorm(KSP);
PETSC_EXTERN PetscErrorCode KSPSkipConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
PETSC_EXTERN PetscErrorCode KSPGetConvergedReason(KSP,KSPConvergedReason *);

PETSC_EXTERN PetscErrorCode KSPComputeExplicitOperator(KSP,Mat *);

/*E
    KSPCGType - Determines what type of CG to use

   Level: beginner

.seealso: KSPCGSetType()
E*/
typedef enum {KSP_CG_SYMMETRIC=0,KSP_CG_HERMITIAN=1} KSPCGType;
PETSC_EXTERN const char *const KSPCGTypes[];

PETSC_EXTERN PetscErrorCode KSPCGSetType(KSP,KSPCGType);
PETSC_EXTERN PetscErrorCode KSPCGUseSingleReduction(KSP,PetscBool );

PETSC_EXTERN PetscErrorCode KSPNASHSetRadius(KSP,PetscReal);
PETSC_EXTERN PetscErrorCode KSPNASHGetNormD(KSP,PetscReal *);
PETSC_EXTERN PetscErrorCode KSPNASHGetObjFcn(KSP,PetscReal *);

PETSC_EXTERN PetscErrorCode KSPSTCGSetRadius(KSP,PetscReal);
PETSC_EXTERN PetscErrorCode KSPSTCGGetNormD(KSP,PetscReal *);
PETSC_EXTERN PetscErrorCode KSPSTCGGetObjFcn(KSP,PetscReal *);

PETSC_EXTERN PetscErrorCode KSPGLTRSetRadius(KSP,PetscReal);
PETSC_EXTERN PetscErrorCode KSPGLTRGetNormD(KSP,PetscReal *);
PETSC_EXTERN PetscErrorCode KSPGLTRGetObjFcn(KSP,PetscReal *);
PETSC_EXTERN PetscErrorCode KSPGLTRGetMinEig(KSP,PetscReal *);
PETSC_EXTERN PetscErrorCode KSPGLTRGetLambda(KSP,PetscReal *);

PETSC_EXTERN PetscErrorCode KSPPythonSetType(KSP,const char[]);

PETSC_EXTERN PetscErrorCode PCPreSolve(PC,KSP);
PETSC_EXTERN PetscErrorCode PCPostSolve(PC,KSP);

PETSC_EXTERN PetscErrorCode KSPMonitorLGResidualNormCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
PETSC_EXTERN PetscErrorCode KSPMonitorLGResidualNorm(KSP,PetscInt,PetscReal,void*);
PETSC_EXTERN PetscErrorCode KSPMonitorLGResidualNormDestroy(PetscDrawLG*);
PETSC_EXTERN PetscErrorCode KSPMonitorLGTrueResidualNormCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
PETSC_EXTERN PetscErrorCode KSPMonitorLGTrueResidualNorm(KSP,PetscInt,PetscReal,void*);
PETSC_EXTERN PetscErrorCode KSPMonitorLGTrueResidualNormDestroy(PetscDrawLG*);
PETSC_EXTERN PetscErrorCode KSPMonitorLGRange(KSP,PetscInt,PetscReal,void*);

PETSC_EXTERN PetscErrorCode PCShellSetPreSolve(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec));
PETSC_EXTERN PetscErrorCode PCShellSetPostSolve(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec));

/* see src/ksp/ksp/interface/iguess.c */
typedef struct _p_KSPFischerGuess {PetscInt method,curl,maxl,refcnt;PetscBool  monitor;Mat mat; KSP ksp;}* KSPFischerGuess;

PETSC_EXTERN PetscErrorCode KSPFischerGuessCreate(KSP,PetscInt,PetscInt,KSPFischerGuess*);
PETSC_EXTERN PetscErrorCode KSPFischerGuessDestroy(KSPFischerGuess*);
PETSC_EXTERN PetscErrorCode KSPFischerGuessReset(KSPFischerGuess);
PETSC_EXTERN PetscErrorCode KSPFischerGuessUpdate(KSPFischerGuess,Vec);
PETSC_EXTERN PetscErrorCode KSPFischerGuessFormGuess(KSPFischerGuess,Vec,Vec);
PETSC_EXTERN PetscErrorCode KSPFischerGuessSetFromOptions(KSPFischerGuess);

PETSC_EXTERN PetscErrorCode KSPSetUseFischerGuess(KSP,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode KSPSetFischerGuess(KSP,KSPFischerGuess);
PETSC_EXTERN PetscErrorCode KSPGetFischerGuess(KSP,KSPFischerGuess*);

PETSC_EXTERN PetscErrorCode MatCreateSchurComplement(Mat,Mat,Mat,Mat,Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatSchurComplementGetKSP(Mat,KSP*);
PETSC_EXTERN PetscErrorCode MatSchurComplementSetKSP(Mat,KSP);
PETSC_EXTERN PetscErrorCode MatSchurComplementSet(Mat,Mat,Mat,Mat,Mat,Mat);
PETSC_EXTERN PetscErrorCode MatSchurComplementUpdate(Mat,Mat,Mat,Mat,Mat,Mat,MatStructure);
PETSC_EXTERN PetscErrorCode MatSchurComplementGetSubmatrices(Mat,Mat*,Mat*,Mat*,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode MatGetSchurComplement(Mat,IS,IS,IS,IS,MatReuse,Mat *,MatReuse,Mat *);

PETSC_EXTERN PetscErrorCode MatGetSchurComplement_Basic(Mat mat,IS isrow0,IS iscol0,IS isrow1,IS iscol1,MatReuse mreuse,Mat *newmat,MatReuse preuse,Mat *newpmat);

PETSC_EXTERN PetscErrorCode KSPSetDM(KSP,DM);
PETSC_EXTERN PetscErrorCode KSPSetDMActive(KSP,PetscBool );
PETSC_EXTERN PetscErrorCode KSPGetDM(KSP,DM*);
PETSC_EXTERN PetscErrorCode KSPSetApplicationContext(KSP,void*);
PETSC_EXTERN PetscErrorCode KSPGetApplicationContext(KSP,void*);
PETSC_EXTERN PetscErrorCode KSPSetComputeOperators(KSP,PetscErrorCode(*)(KSP,Mat,Mat,MatStructure*,void*),void*);
PETSC_EXTERN PetscErrorCode KSPSetComputeRHS(KSP,PetscErrorCode(*)(KSP,Vec,void*),void*);
PETSC_EXTERN PetscErrorCode DMKSPSetComputeOperators(DM,PetscErrorCode(*)(KSP,Mat,Mat,MatStructure*,void*),void*);
PETSC_EXTERN PetscErrorCode DMKSPGetComputeOperators(DM,PetscErrorCode(**)(KSP,Mat,Mat,MatStructure*,void*),void*);
PETSC_EXTERN PetscErrorCode DMKSPSetComputeRHS(DM,PetscErrorCode(*)(KSP,Vec,void*),void*);
PETSC_EXTERN PetscErrorCode DMKSPGetComputeRHS(DM,PetscErrorCode(**)(KSP,Vec,void*),void*);

#endif
