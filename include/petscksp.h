/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __PETSCKSP_H
#define __PETSCKSP_H
#include "petscpc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode KSPInitializePackage(const char[]);

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
#define KSPRICHARDSON "richardson"
#define KSPCHEBYCHEV  "chebychev"
#define KSPCG         "cg"
#define KSPCGNE       "cgne"
#define KSPGMRES      "gmres"
#define KSPTCQMR      "tcqmr"
#define KSPBCGS       "bcgs"
#define KSPBCGSL      "bcgsl"
#define KSPCGS        "cgs"
#define KSPTFQMR      "tfqmr"
#define KSPCR         "cr"
#define KSPLSQR       "lsqr"
#define KSPPREONLY    "preonly"
#define KSPQCG        "qcg"
#define KSPBICG       "bicg"
#define KSPFGMRES     "fgmres" 
#define KSPMINRES     "minres"
#define KSPSYMMLQ     "symmlq"
#define KSPLGMRES     "lgmres"
#define KSPType char*

/* Logging support */
extern PetscCookie KSP_COOKIE;
extern PetscEvent    KSP_GMRESOrthogonalization, KSP_SetUp, KSP_Solve;

EXTERN PetscErrorCode KSPCreate(MPI_Comm,KSP *);
EXTERN PetscErrorCode KSPSetType(KSP,const KSPType);
EXTERN PetscErrorCode KSPSetUp(KSP);
EXTERN PetscErrorCode KSPSetUpOnBlocks(KSP);
EXTERN PetscErrorCode KSPSolve(KSP,Vec,Vec);
EXTERN PetscErrorCode KSPSolveTranspose(KSP,Vec,Vec);
EXTERN PetscErrorCode KSPDestroy(KSP);

extern PetscFList KSPList;
EXTERN PetscErrorCode KSPRegisterAll(const char[]);
EXTERN PetscErrorCode KSPRegisterDestroy(void);

EXTERN PetscErrorCode KSPRegister(const char[],const char[],const char[],PetscErrorCode (*)(KSP));

/*MC
   KSPRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   PetscErrorCode KSPRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(KSP))

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

EXTERN PetscErrorCode KSPGetType(KSP,KSPType *);
EXTERN PetscErrorCode KSPSetPreconditionerSide(KSP,PCSide);
EXTERN PetscErrorCode KSPGetPreconditionerSide(KSP,PCSide*);
EXTERN PetscErrorCode KSPGetTolerances(KSP,PetscReal*,PetscReal*,PetscReal*,PetscInt*);
EXTERN PetscErrorCode KSPSetTolerances(KSP,PetscReal,PetscReal,PetscReal,PetscInt);
EXTERN PetscErrorCode KSPSetInitialGuessNonzero(KSP,PetscTruth);
EXTERN PetscErrorCode KSPGetInitialGuessNonzero(KSP,PetscTruth *);
EXTERN PetscErrorCode KSPSetInitialGuessKnoll(KSP,PetscTruth);
EXTERN PetscErrorCode KSPGetInitialGuessKnoll(KSP,PetscTruth*);
EXTERN PetscErrorCode KSPSetComputeEigenvalues(KSP,PetscTruth);
EXTERN PetscErrorCode KSPSetComputeSingularValues(KSP,PetscTruth);
EXTERN PetscErrorCode KSPGetRhs(KSP,Vec *);
EXTERN PetscErrorCode KSPGetSolution(KSP,Vec *);
EXTERN PetscErrorCode KSPGetResidualNorm(KSP,PetscReal*);
EXTERN PetscErrorCode KSPGetIterationNumber(KSP,PetscInt*);
EXTERN PetscErrorCode KSPSetNullSpace(KSP,MatNullSpace);
EXTERN PetscErrorCode KSPGetNullSpace(KSP,MatNullSpace*);

EXTERN PetscErrorCode KSPSetPC(KSP,PC);
EXTERN PetscErrorCode KSPGetPC(KSP,PC*);

EXTERN PetscErrorCode KSPSetMonitor(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void*));
EXTERN PetscErrorCode KSPClearMonitor(KSP);
EXTERN PetscErrorCode KSPGetMonitorContext(KSP,void **);
EXTERN PetscErrorCode KSPGetResidualHistory(KSP,PetscReal*[],PetscInt *);
EXTERN PetscErrorCode KSPSetResidualHistory(KSP,PetscReal[],PetscInt,PetscTruth);

/* not sure where to put this */
EXTERN PetscErrorCode PCKSPGetKSP(PC,KSP*);
EXTERN PetscErrorCode PCBJacobiGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
EXTERN PetscErrorCode PCASMGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
EXTERN PetscErrorCode PCFieldSplitGetSubKSP(PC,PetscInt*,KSP*[]);

EXTERN PetscErrorCode KSPBuildSolution(KSP,Vec,Vec *);
EXTERN PetscErrorCode KSPBuildResidual(KSP,Vec,Vec,Vec *);

EXTERN PetscErrorCode KSPRichardsonSetScale(KSP,PetscReal);
EXTERN PetscErrorCode KSPChebychevSetEigenvalues(KSP,PetscReal,PetscReal);
EXTERN PetscErrorCode KSPComputeExtremeSingularValues(KSP,PetscReal*,PetscReal*);
EXTERN PetscErrorCode KSPComputeEigenvalues(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt *);
EXTERN PetscErrorCode KSPComputeEigenvaluesExplicitly(KSP,PetscInt,PetscReal*,PetscReal*);

EXTERN PetscErrorCode KSPGMRESSetRestart(KSP, PetscInt);
EXTERN PetscErrorCode KSPGMRESSetHapTol(KSP,PetscReal);

EXTERN PetscErrorCode KSPGMRESSetPreAllocateVectors(KSP);
EXTERN PetscErrorCode KSPGMRESSetOrthogonalization(KSP,PetscErrorCode (*)(KSP,PetscInt));
EXTERN PetscErrorCode KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,PetscInt);
EXTERN PetscErrorCode KSPGMRESClassicalGramSchmidtOrthogonalization(KSP,PetscInt);

EXTERN PetscErrorCode KSPLGMRESSetAugDim(KSP,PetscInt);
EXTERN PetscErrorCode KSPLGMRESSetConstant(KSP);

/*E
    KSPGMRESCGSRefinementType - How the classical (unmodified) Gram-Schmidt is performed.

   Level: advanced

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSPGMRESModifiedGramSchmidtOrthogonalization()

E*/
typedef enum {KSP_GMRES_CGS_REFINE_NEVER, KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS} KSPGMRESCGSRefinementType;

/*M
    KSP_GMRES_CGS_REFINE_NEVER - Just do the classical (unmodified) Gram-Schmidt process

   Level: advanced

   Note: Possible unstable, but the fastest to compute

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS,
          KSPGMRESModifiedGramSchmidtOrthogonalization()
M*/

/*M
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

/*M
    KSP_GMRES_CGS_REFINE_NEVER - Do two steps of the classical (unmodified) Gram-Schmidt process.

   Level: advanced

   Note: This is roughly twice the cost of KSP_GMRES_CGS_REFINE_NEVER because it performs the process twice
     but it saves the extra norm calculation needed by KSP_GMRES_CGS_REFINE_IFNEEDED.

        You should only use this if you absolutely know that the iterative refinement is needed.

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(),
          KSPGMRESSetCGSRefinementType(), KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS,
          KSPGMRESModifiedGramSchmidtOrthogonalization()
M*/

EXTERN PetscErrorCode KSPGMRESSetCGSRefinementType(KSP,KSPGMRESCGSRefinementType);

EXTERN PetscErrorCode KSPFGMRESModifyPCNoChange(KSP,PetscInt,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode KSPFGMRESModifyPCKSP(KSP,PetscInt,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode KSPFGMRESSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

EXTERN PetscErrorCode KSPQCGSetTrustRegionRadius(KSP,PetscReal);
EXTERN PetscErrorCode KSPQCGGetQuadratic(KSP,PetscReal*);
EXTERN PetscErrorCode KSPQCGGetTrialStepNorm(KSP,PetscReal*);

EXTERN PetscErrorCode KSPSetFromOptions(KSP);
EXTERN PetscErrorCode KSPAddOptionsChecker(PetscErrorCode (*)(KSP));

EXTERN PetscErrorCode KSPSingularValueMonitor(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode KSPDefaultMonitor(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode KSPTrueMonitor(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode KSPDefaultSMonitor(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode KSPVecViewMonitor(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode KSPGMRESKrylovMonitor(KSP,PetscInt,PetscReal,void *);

EXTERN PetscErrorCode KSPUnwindPreconditioner(KSP,Vec,Vec);
EXTERN PetscErrorCode KSPDefaultBuildSolution(KSP,Vec,Vec*);
EXTERN PetscErrorCode KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);

EXTERN PetscErrorCode KSPSetOperators(KSP,Mat,Mat,MatStructure);
EXTERN PetscErrorCode KSPGetOperators(KSP,Mat*,Mat*,MatStructure*);
EXTERN PetscErrorCode KSPSetOptionsPrefix(KSP,const char[]);
EXTERN PetscErrorCode KSPAppendOptionsPrefix(KSP,const char[]);
EXTERN PetscErrorCode KSPGetOptionsPrefix(KSP,char*[]);

EXTERN PetscErrorCode KSPSetDiagonalScale(KSP,PetscTruth);
EXTERN PetscErrorCode KSPGetDiagonalScale(KSP,PetscTruth*);
EXTERN PetscErrorCode KSPSetDiagonalScaleFix(KSP,PetscTruth);
EXTERN PetscErrorCode KSPGetDiagonalScaleFix(KSP,PetscTruth*);

EXTERN PetscErrorCode KSPView(KSP,PetscViewer);

/*E
    KSPNormType - Norm that is passed in the Krylov convergence
       test routines.

   Level: advanced

   Notes: this must match finclude/petscksp.h 

.seealso: KSPSolve(), KSPGetConvergedReason(), KSPSetNormType(),
          KSPSetConvergenceTest()
E*/
typedef enum {KSP_NO_NORM               = 0,
              KSP_PRECONDITIONED_NORM   = 1,
              KSP_UNPRECONDITIONED_NORM = 2,
              KSP_NATURAL_NORM          = 3} KSPNormType;

/*M
    KSP_NO_NORM - Do not compute a norm during the Krylov process. This will 
          possibly save some computation but means the convergence test cannot
          be based on a norm of a residual etc.

   Level: advanced

    Note: Some Krylov methods need to compute a residual norm and then this is ignored

.seealso: KSPNormType, KSPSetNormType(), KSP_PRECONDITIONED_NORM, KSP_UNPRECONDITIONED_NORM, KSP_NATURAL_NORM
M*/

/*M
    KSP_PRECONDITIONED_NORM - Compute the norm of the preconditioned residual and pass that to the 
       convergence test routine.

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSP_NO_NORM, KSP_UNPRECONDITIONED_NORM, KSP_NATURAL_NORM, KSPSetConvergenceTest()
M*/

/*M
    KSP_UNPRECONDITIONED_NORM - Compute the norm of the true residual (b - A*x) and pass that to the 
       convergence test routine.

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSP_NO_NORM, KSP_PRECONDITIONED_NORM, KSP_NATURAL_NORM, KSPSetConvergenceTest()
M*/

/*M
    KSP_NATURAL_NORM - Compute the 'natural norm' of residual sqrt((b - A*x)*B*(b - A*x)) and pass that to the 
       convergence test routine.

   Level: advanced

.seealso: KSPNormType, KSPSetNormType(), KSP_NO_NORM, KSP_PRECONDITIONED_NORM, KSP_UNPRECONDITIONED_NORM, KSPSetConvergenceTest()
M*/

EXTERN PetscErrorCode KSPSetNormType(KSP,KSPNormType);

/*E
    KSPConvergedReason - reason a Krylov method was said to 
         have converged or diverged

   Level: beginner

   Notes: this must match finclude/petscksp.h 

   Developer note: The string versions of these are in 
     src/ksp/ksp/interface/itfunc.c called convergedreasons.
     If these enums are changed you must change those.

.seealso: KSPSolve(), KSPGetConvergedReason(), KSPSetTolerances()
E*/
typedef enum {/* converged */
              KSP_CONVERGED_RTOL               =  2,
              KSP_CONVERGED_ATOL               =  3,
              KSP_CONVERGED_ITS                =  4,
              KSP_CONVERGED_QCG_NEG_CURVE      =  5,
              KSP_CONVERGED_QCG_CONSTRAINED    =  6,
              KSP_CONVERGED_STEP_LENGTH        =  7,
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
     KSP_CONVERGED_ITS - Used by the KSPPREONLY solver after the single iteration of the
           preconditioner is applied.


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

     Notes: This can happen with the PCICC preconditioner, use -pc_icc_shift to force 
  the PCICC preconditioner to generate a positive definite preconditioner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

/*MC
     KSP_CONVERGED_ITERATING - This flag is returned if you call KSPGetConvergedReason()
        while the KSPSolve() is still running.

   Level: beginner

.seealso:  KSPSolve(), KSPGetConvergedReason(), KSPConvergedReason, KSPSetTolerances()

M*/

EXTERN PetscErrorCode KSPSetConvergenceTest(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),void *);
EXTERN PetscErrorCode KSPGetConvergenceContext(KSP,void **);
EXTERN PetscErrorCode KSPDefaultConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
EXTERN PetscErrorCode KSPSkipConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
EXTERN PetscErrorCode KSPGetConvergedReason(KSP,KSPConvergedReason *);

EXTERN PetscErrorCode KSPComputeExplicitOperator(KSP,Mat *);

/*E
    KSPCGType - Determines what type of CG to use

   Level: beginner

.seealso: KSPCGSetType()
E*/
typedef enum {KSP_CG_SYMMETRIC=1,KSP_CG_HERMITIAN=2} KSPCGType;

EXTERN PetscErrorCode KSPCGSetType(KSP,KSPCGType);

EXTERN PetscErrorCode PCPreSolve(PC,KSP);
EXTERN PetscErrorCode PCPostSolve(PC,KSP);

EXTERN PetscErrorCode KSPLGMonitorCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN PetscErrorCode KSPLGMonitor(KSP,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode KSPLGMonitorDestroy(PetscDrawLG);
EXTERN PetscErrorCode KSPLGTrueMonitorCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN PetscErrorCode KSPLGTrueMonitor(KSP,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode KSPLGTrueMonitorDestroy(PetscDrawLG);


PETSC_EXTERN_CXX_END
#endif
