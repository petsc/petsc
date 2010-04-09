/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __PETSCKSP_H
#define __PETSCKSP_H
#include "petscpc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPInitializePackage(const char[]);

/*S
     KSP - Abstract PETSc object that manages all Krylov methods

   Level: beginner

  Concepts: Krylov methods

.seealso:  KSPCreate(), KSPSetType(), KSPType, SNES, TS, PC, KSP
S*/
typedef struct _p_KSP*     KSP;

/*E
    KSPType - String with the name of a PETSc Krylov method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mykspcreate()

   Level: beginner

.seealso: KSPSetType(), KSP
E*/
#define KSPType char*
#define KSPRICHARDSON "richardson"
#define KSPCHEBYCHEV  "chebychev"
#define KSPCG         "cg"
#define   KSPCGNE       "cgne"
#define   KSPNASH       "nash"
#define   KSPSTCG       "stcg"
#define   KSPGLTR       "gltr"
#define KSPGMRES      "gmres"
#define   KSPFGMRES     "fgmres" 
#define   KSPLGMRES     "lgmres"
#define KSPTCQMR      "tcqmr"
#define KSPBCGS       "bcgs"
#define KSPIBCGS        "ibcgs"
#define KSPBCGSL        "bcgsl"
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
#define KSPBROYDEN    "broyden"
#define KSPGCR        "gcr"

/* Logging support */
extern PetscCookie PETSCKSP_DLLEXPORT KSP_COOKIE;

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate(MPI_Comm,KSP *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetType(KSP,const KSPType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetUp(KSP);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetUpOnBlocks(KSP);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSolve(KSP,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSolveTranspose(KSP,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPDestroy(KSP);

extern PetscFList KSPList;
extern PetscTruth KSPRegisterAllCalled;
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPRegisterDestroy(void);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPRegister(const char[],const char[],const char[],PetscErrorCode (*)(KSP));

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

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetType(KSP,const KSPType *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetPreconditionerSide(KSP,PCSide);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetPreconditionerSide(KSP,PCSide*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetTolerances(KSP,PetscReal*,PetscReal*,PetscReal*,PetscInt*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetTolerances(KSP,PetscReal,PetscReal,PetscReal,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetInitialGuessNonzero(KSP,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetInitialGuessNonzero(KSP,PetscTruth *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetInitialGuessKnoll(KSP,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetInitialGuessKnoll(KSP,PetscTruth*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetComputeEigenvalues(KSP,PetscTruth*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetComputeEigenvalues(KSP,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetComputeSingularValues(KSP,PetscTruth*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetComputeSingularValues(KSP,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetRhs(KSP,Vec *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetSolution(KSP,Vec *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetResidualNorm(KSP,PetscReal*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetIterationNumber(KSP,PetscInt*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetNullSpace(KSP,MatNullSpace);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetNullSpace(KSP,MatNullSpace*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetVecs(KSP,PetscInt,Vec**,PetscInt,Vec**);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetPC(KSP,PC);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetPC(KSP,PC*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorSet(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void*));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorCancel(KSP);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetMonitorContext(KSP,void **);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetResidualHistory(KSP,PetscReal*[],PetscInt *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetResidualHistory(KSP,PetscReal[],PetscInt,PetscTruth);

/* not sure where to put this */
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCKSPGetKSP(PC,KSP*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASMGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitGetSubKSP(PC,PetscInt*,KSP*[]);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCGalerkinGetKSP(PC,KSP *);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPBuildSolution(KSP,Vec,Vec *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPBuildResidual(KSP,Vec,Vec,Vec *);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPRichardsonSetScale(KSP,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPChebychevSetEigenvalues(KSP,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPComputeExtremeSingularValues(KSP,PetscReal*,PetscReal*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPComputeEigenvalues(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPComputeEigenvaluesExplicitly(KSP,PetscInt,PetscReal*,PetscReal*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetRestart(KSP, PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetHapTol(KSP,PetscReal);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetPreAllocateVectors(KSP);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetOrthogonalization(KSP,PetscErrorCode (*)(KSP,PetscInt));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESClassicalGramSchmidtOrthogonalization(KSP,PetscInt);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPLGMRESSetAugDim(KSP,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPLGMRESSetConstant(KSP);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGCRSetRestart(KSP,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGCRSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

/*E
    KSPGMRESCGSRefinementType - How the classical (unmodified) Gram-Schmidt is performed.

   Level: advanced

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSPGMRESModifiedGramSchmidtOrthogonalization()

E*/
typedef enum {KSP_GMRES_CGS_REFINE_NEVER, KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS} KSPGMRESCGSRefinementType;
extern const char *KSPGMRESCGSRefinementTypes[];
/*MC
    KSP_GMRES_CGS_REFINE_NEVER - Just do the classical (unmodified) Gram-Schmidt process

   Level: advanced

   Note: Possible unstable, but the fastest to compute

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS,
          KSPGMRESModifiedGramSchmidtOrthogonalization()
M*/

/*MC
    KSP_GMRES_CGS_REFINE_IFNEEDED - Do the classical (unmodified) Gram-Schmidt process and one step of 
          iterative refinement if an estimate of the orthogonality of the resulting vectors indicates
          poor orthogonality.

   Level: advanced

   Note: This is slower than KSP_GMRES_CGS_REFINE_NEVER because it requires an extra norm computation to 
     estimate the orthogonality but is more stable.

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSP_GMRES_CGS_REFINE_NEVER, KSP_GMRES_CGS_REFINE_ALWAYS,
          KSPGMRESModifiedGramSchmidtOrthogonalization()
M*/

/*MC
    KSP_GMRES_CGS_REFINE_NEVER - Do two steps of the classical (unmodified) Gram-Schmidt process.

   Level: advanced

   Note: This is roughly twice the cost of KSP_GMRES_CGS_REFINE_NEVER because it performs the process twice
     but it saves the extra norm calculation needed by KSP_GMRES_CGS_REFINE_IFNEEDED.

        You should only use this if you absolutely know that the iterative refinement is needed.

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS,
          KSPGMRESModifiedGramSchmidtOrthogonalization()
M*/

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetCGSRefinementType(KSP,KSPGMRESCGSRefinementType);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFGMRESModifyPCNoChange(KSP,PetscInt,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFGMRESModifyPCKSP(KSP,PetscInt,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFGMRESSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGSetTrustRegionRadius(KSP,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGGetQuadratic(KSP,PetscReal*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGGetTrialStepNorm(KSP,PetscReal*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetXRes(KSP,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetPol(KSP,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetEll(KSP,int);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetFromOptions(KSP);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPAddOptionsChecker(PetscErrorCode (*)(KSP));

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorSingularValue(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorDefault(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorDefaultLSQR(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorRange(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorTrueResidualNorm(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorDefaultShort(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorSolution(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESMonitorKrylov(KSP,PetscInt,PetscReal,void *);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPUnwindPreconditioner(KSP,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultBuildSolution(KSP,Vec,Vec*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPInitialResidual(KSP,Vec,Vec,Vec,Vec,Vec);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetOperators(KSP,Mat,Mat,MatStructure);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOperators(KSP,Mat*,Mat*,MatStructure*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOperatorsSet(KSP,PetscTruth*,PetscTruth*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetOptionsPrefix(KSP,const char[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPAppendOptionsPrefix(KSP,const char[]);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOptionsPrefix(KSP,const char*[]);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetDiagonalScale(KSP,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetDiagonalScale(KSP,PetscTruth*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetDiagonalScaleFix(KSP,PetscTruth);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetDiagonalScaleFix(KSP,PetscTruth*);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPView(KSP,PetscViewer);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPLSQRSetStandardErrorVec(KSP,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPLSQRGetStandardErrorVec(KSP,Vec*);

/*E
    KSPNormType - Norm that is passed in the Krylov convergence
       test routines.

   Level: advanced

   Each solver only supports a subset of these and some may support different ones
   depending on left or right preconditioning, see KSPSetPreconditionerSide()

   Notes: this must match finclude/petscksp.h 

.seealso: KSPSolve(), KSPGetConvergedReason(), KSPSetNormType(),
          KSPSetConvergenceTest(), KSPSetPreconditionerSide()
E*/
typedef enum {KSP_NORM_NO = 0,KSP_NORM_PRECONDITIONED = 1,KSP_NORM_UNPRECONDITIONED = 2,KSP_NORM_NATURAL = 3} KSPNormType;
extern const char *KSPNormTypes[];
/*MC
    KSP_NORM_NO - Do not compute a norm during the Krylov process. This will 
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

.seealso: KSPNormType, KSPSetNormType(), KSP_NORM_NO, KSP_NORM_UNPRECONDITIONED, KSP_NORM_NATURAL, KSPSetConvergenceTest()
M*/

/*MC
    KSP_NORM_UNPRECONDITIONED - Compute the norm of the true residual (b - A*x) and pass that to the 
       convergence test routine.

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSP_NORM_NO, KSP_NORM_PRECONDITIONED, KSP_NORM_NATURAL, KSPSetConvergenceTest()
M*/

/*MC
    KSP_NORM_NATURAL - Compute the 'natural norm' of residual sqrt((b - A*x)*B*(b - A*x)) and pass that to the 
       convergence test routine. This is only supported by  KSPCG, KSPCR, KSPCGNE, KSPCGS

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSP_NORM_NO, KSP_NORM_PRECONDITIONED, KSP_NORM_UNPRECONDITIONED, KSPSetConvergenceTest()
M*/

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetNormType(KSP,KSPNormType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetNormType(KSP,KSPNormType*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetCheckNormIteration(KSP,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetLagNorm(KSP,PetscTruth);

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
extern const char **KSPConvergedReasons;

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
          method could not continue to enlarge the Krylov space.

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

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetConvergenceTest(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),void *,PetscErrorCode (*)(void*));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetConvergenceContext(KSP,void **);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPConvergedLSQR(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConvergedDestroy(void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConvergedCreate(void **);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConvergedSetUIRNorm(KSP);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConvergedSetUMIRNorm(KSP);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSkipConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetConvergedReason(KSP,KSPConvergedReason *);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPComputeExplicitOperator(KSP,Mat *);

/*E
    KSPCGType - Determines what type of CG to use

   Level: beginner

.seealso: KSPCGSetType()
E*/
typedef enum {KSP_CG_SYMMETRIC=0,KSP_CG_HERMITIAN=1} KSPCGType;
extern const char *KSPCGTypes[];

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPCGSetType(KSP,KSPCGType);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPCGUseSingleReduction(KSP,PetscTruth);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPNASHSetRadius(KSP,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPNASHGetNormD(KSP,PetscReal *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPNASHGetObjFcn(KSP,PetscReal *);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGSetRadius(KSP,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGGetNormD(KSP,PetscReal *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGGetObjFcn(KSP,PetscReal *);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRSetRadius(KSP,PetscReal);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetNormD(KSP,PetscReal *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetObjFcn(KSP,PetscReal *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetMinEig(KSP,PetscReal *);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetLambda(KSP,PetscReal *);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetType(KSP,const char[]);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPreSolve(PC,KSP);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCPostSolve(PC,KSP);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLG(KSP,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGDestroy(PetscDrawLG);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGTrueResidualNormCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGTrueResidualNorm(KSP,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGTrueResidualNormDestroy(PetscDrawLG);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGRangeCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGRange(KSP,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGRangeDestroy(PetscDrawLG);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetPreSolve(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec));
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetPostSolve(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec));

/* see src/ksp/ksp/interface/iguess.c */
typedef struct _p_KSPFischerGuess {PetscInt method,curl,maxl,refcnt;PetscTruth monitor;Mat mat; KSP ksp;}* KSPFischerGuess;

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessCreate(KSP,PetscInt,PetscInt,KSPFischerGuess*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessDestroy(KSPFischerGuess);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessReset(KSPFischerGuess);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessUpdate(KSPFischerGuess,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessFormGuess(KSPFischerGuess,Vec,Vec);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessSetFromOptions(KSPFischerGuess);

EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetUseFischerGuess(KSP,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPSetFischerGuess(KSP,KSPFischerGuess);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT KSPGetFischerGuess(KSP,KSPFischerGuess*);

EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSchurComplement(Mat,Mat,Mat,Mat,Mat,Mat*);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT MatSchurComplementGetKSP(Mat,KSP*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSchurComplementUpdate(Mat,Mat,Mat,Mat,Mat,Mat,MatStructure);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatSchurComplementGetSubmatrices(Mat,Mat*,Mat*,Mat*,Mat*,Mat*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatGetSchurComplement(Mat,IS,IS,IS,IS,MatReuse,Mat *,MatReuse,Mat *);

PETSC_EXTERN_CXX_END
#endif
