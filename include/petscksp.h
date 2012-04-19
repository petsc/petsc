/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __PETSCKSP_H
#define __PETSCKSP_H
#include "petscpc.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode  KSPInitializePackage(const char[]);

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
#define   KSPDGMRES     "dgmres"
#define   KSPPGMRES     "pgmres"
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
#define KSPGCR        "gcr"
#define KSPSPECEST    "specest"

/* Logging support */
extern PetscClassId  KSP_CLASSID;

extern PetscErrorCode  KSPCreate(MPI_Comm,KSP *);
extern PetscErrorCode  KSPSetType(KSP,const KSPType);
extern PetscErrorCode  KSPSetUp(KSP);
extern PetscErrorCode  KSPSetUpOnBlocks(KSP);
extern PetscErrorCode  KSPSolve(KSP,Vec,Vec);
extern PetscErrorCode  KSPSolveTranspose(KSP,Vec,Vec);
extern PetscErrorCode  KSPReset(KSP);
extern PetscErrorCode  KSPDestroy(KSP*);

extern PetscFList KSPList;
extern PetscBool  KSPRegisterAllCalled;
extern PetscErrorCode  KSPRegisterAll(const char[]);
extern PetscErrorCode  KSPRegisterDestroy(void);
extern PetscErrorCode  KSPRegister(const char[],const char[],const char[],PetscErrorCode (*)(KSP));

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

extern PetscErrorCode  KSPGetType(KSP,const KSPType *);
extern PetscErrorCode  KSPSetPCSide(KSP,PCSide);
extern PetscErrorCode  KSPGetPCSide(KSP,PCSide*);
extern PetscErrorCode  KSPGetTolerances(KSP,PetscReal*,PetscReal*,PetscReal*,PetscInt*);
extern PetscErrorCode  KSPSetTolerances(KSP,PetscReal,PetscReal,PetscReal,PetscInt);
extern PetscErrorCode  KSPSetInitialGuessNonzero(KSP,PetscBool );
extern PetscErrorCode  KSPGetInitialGuessNonzero(KSP,PetscBool  *);
extern PetscErrorCode  KSPSetInitialGuessKnoll(KSP,PetscBool );
extern PetscErrorCode  KSPGetInitialGuessKnoll(KSP,PetscBool *);
extern PetscErrorCode  KSPSetErrorIfNotConverged(KSP,PetscBool );
extern PetscErrorCode  KSPGetErrorIfNotConverged(KSP,PetscBool  *);
extern PetscErrorCode  KSPGetComputeEigenvalues(KSP,PetscBool *);
extern PetscErrorCode  KSPSetComputeEigenvalues(KSP,PetscBool );
extern PetscErrorCode  KSPGetComputeSingularValues(KSP,PetscBool *);
extern PetscErrorCode  KSPSetComputeSingularValues(KSP,PetscBool );
extern PetscErrorCode  KSPGetRhs(KSP,Vec *);
extern PetscErrorCode  KSPGetSolution(KSP,Vec *);
extern PetscErrorCode  KSPGetResidualNorm(KSP,PetscReal*);
extern PetscErrorCode  KSPGetIterationNumber(KSP,PetscInt*);
extern PetscErrorCode  KSPSetNullSpace(KSP,MatNullSpace);
extern PetscErrorCode  KSPGetNullSpace(KSP,MatNullSpace*);
extern PetscErrorCode  KSPGetVecs(KSP,PetscInt,Vec**,PetscInt,Vec**);

extern PetscErrorCode  KSPSetPC(KSP,PC);
extern PetscErrorCode  KSPGetPC(KSP,PC*);

extern PetscErrorCode  KSPMonitor(KSP,PetscInt,PetscReal);
extern PetscErrorCode  KSPMonitorSet(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void**));
extern PetscErrorCode  KSPMonitorCancel(KSP);
extern PetscErrorCode  KSPGetMonitorContext(KSP,void **);
extern PetscErrorCode  KSPGetResidualHistory(KSP,PetscReal*[],PetscInt *);
extern PetscErrorCode  KSPSetResidualHistory(KSP,PetscReal[],PetscInt,PetscBool );

/* not sure where to put this */
extern PetscErrorCode  PCKSPGetKSP(PC,KSP*);
extern PetscErrorCode  PCBJacobiGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
extern PetscErrorCode  PCASMGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
extern PetscErrorCode  PCGASMGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
extern PetscErrorCode  PCFieldSplitGetSubKSP(PC,PetscInt*,KSP*[]);

extern PetscErrorCode  PCGalerkinGetKSP(PC,KSP *);

extern PetscErrorCode  KSPBuildSolution(KSP,Vec,Vec *);
extern PetscErrorCode  KSPBuildResidual(KSP,Vec,Vec,Vec *);

extern PetscErrorCode  KSPRichardsonSetScale(KSP,PetscReal);
extern PetscErrorCode  KSPRichardsonSetSelfScale(KSP,PetscBool );
extern PetscErrorCode  KSPChebychevSetEigenvalues(KSP,PetscReal,PetscReal);
extern PetscErrorCode  KSPChebychevSetEstimateEigenvalues(KSP,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  KSPChebychevSetNewMatrix(KSP);
extern PetscErrorCode  KSPComputeExtremeSingularValues(KSP,PetscReal*,PetscReal*);
extern PetscErrorCode  KSPComputeEigenvalues(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt *);
extern PetscErrorCode  KSPComputeEigenvaluesExplicitly(KSP,PetscInt,PetscReal*,PetscReal*);

extern PetscErrorCode  KSPGMRESSetRestart(KSP, PetscInt);
extern PetscErrorCode  KSPGMRESGetRestart(KSP, PetscInt*);
extern PetscErrorCode  KSPGMRESSetHapTol(KSP,PetscReal);

extern PetscErrorCode  KSPGMRESSetPreAllocateVectors(KSP);
extern PetscErrorCode  KSPGMRESSetOrthogonalization(KSP,PetscErrorCode (*)(KSP,PetscInt));
extern PetscErrorCode  KSPGMRESGetOrthogonalization(KSP,PetscErrorCode (**)(KSP,PetscInt));
extern PetscErrorCode  KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,PetscInt);
extern PetscErrorCode  KSPGMRESClassicalGramSchmidtOrthogonalization(KSP,PetscInt);

extern PetscErrorCode  KSPLGMRESSetAugDim(KSP,PetscInt);
extern PetscErrorCode  KSPLGMRESSetConstant(KSP);

extern PetscErrorCode  KSPGCRSetRestart(KSP,PetscInt);
extern PetscErrorCode  KSPGCRGetRestart(KSP,PetscInt*);
extern PetscErrorCode  KSPGCRSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

/*E
    KSPGMRESCGSRefinementType - How the classical (unmodified) Gram-Schmidt is performed.

   Level: advanced

.seealso: KSPGMRESClassicalGramSchmidtOrthogonalization(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization(), 
          KSPGMRESSetCGSRefinementType(), KSPGMRESGetCGSRefinementType(), KSPGMRESModifiedGramSchmidtOrthogonalization()

E*/
typedef enum {KSP_GMRES_CGS_REFINE_NEVER, KSP_GMRES_CGS_REFINE_IFNEEDED, KSP_GMRES_CGS_REFINE_ALWAYS} KSPGMRESCGSRefinementType;
extern const char *KSPGMRESCGSRefinementTypes[];
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

extern PetscErrorCode  KSPGMRESSetCGSRefinementType(KSP,KSPGMRESCGSRefinementType);
extern PetscErrorCode  KSPGMRESGetCGSRefinementType(KSP,KSPGMRESCGSRefinementType*);

extern PetscErrorCode  KSPFGMRESModifyPCNoChange(KSP,PetscInt,PetscInt,PetscReal,void*);
extern PetscErrorCode  KSPFGMRESModifyPCKSP(KSP,PetscInt,PetscInt,PetscReal,void*);
extern PetscErrorCode  KSPFGMRESSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

extern PetscErrorCode  KSPQCGSetTrustRegionRadius(KSP,PetscReal);
extern PetscErrorCode  KSPQCGGetQuadratic(KSP,PetscReal*);
extern PetscErrorCode  KSPQCGGetTrialStepNorm(KSP,PetscReal*);

extern PetscErrorCode  KSPBCGSLSetXRes(KSP,PetscReal);
extern PetscErrorCode  KSPBCGSLSetPol(KSP,PetscBool );
extern PetscErrorCode  KSPBCGSLSetEll(KSP,PetscInt);

extern PetscErrorCode  KSPSetFromOptions(KSP);
extern PetscErrorCode  KSPAddOptionsChecker(PetscErrorCode (*)(KSP));

extern PetscErrorCode  KSPMonitorSingularValue(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode  KSPMonitorDefault(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode  KSPLSQRMonitorDefault(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode  KSPMonitorRange(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode  KSPMonitorTrueResidualNorm(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode  KSPMonitorDefaultShort(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode  KSPMonitorSolution(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode  KSPMonitorAMS(KSP,PetscInt,PetscReal,void*);
extern PetscErrorCode  KSPMonitorAMSCreate(KSP,const char*,void**);
extern PetscErrorCode  KSPMonitorAMSDestroy(void**);
extern PetscErrorCode  KSPGMRESMonitorKrylov(KSP,PetscInt,PetscReal,void *);

extern PetscErrorCode  KSPUnwindPreconditioner(KSP,Vec,Vec);
extern PetscErrorCode  KSPDefaultBuildSolution(KSP,Vec,Vec*);
extern PetscErrorCode  KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);
extern PetscErrorCode  KSPInitialResidual(KSP,Vec,Vec,Vec,Vec,Vec);

extern PetscErrorCode  KSPSetOperators(KSP,Mat,Mat,MatStructure);
extern PetscErrorCode  KSPGetOperators(KSP,Mat*,Mat*,MatStructure*);
extern PetscErrorCode  KSPGetOperatorsSet(KSP,PetscBool *,PetscBool *);
extern PetscErrorCode  KSPSetOptionsPrefix(KSP,const char[]);
extern PetscErrorCode  KSPAppendOptionsPrefix(KSP,const char[]);
extern PetscErrorCode  KSPGetOptionsPrefix(KSP,const char*[]);

extern PetscErrorCode  KSPSetDiagonalScale(KSP,PetscBool );
extern PetscErrorCode  KSPGetDiagonalScale(KSP,PetscBool *);
extern PetscErrorCode  KSPSetDiagonalScaleFix(KSP,PetscBool );
extern PetscErrorCode  KSPGetDiagonalScaleFix(KSP,PetscBool *);

extern PetscErrorCode  KSPView(KSP,PetscViewer);

extern PetscErrorCode  KSPLSQRSetStandardErrorVec(KSP,Vec);
extern PetscErrorCode  KSPLSQRGetStandardErrorVec(KSP,Vec*);

extern PetscErrorCode  PCRedundantGetKSP(PC,KSP*);
extern PetscErrorCode  PCRedistributeGetKSP(PC,KSP*);

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
extern const char *const*const KSPNormTypes;

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

extern PetscErrorCode  KSPSetNormType(KSP,KSPNormType);
extern PetscErrorCode  KSPGetNormType(KSP,KSPNormType*);
extern PetscErrorCode  KSPSetSupportedNorm(KSP ksp,KSPNormType,PCSide,PetscInt);
extern PetscErrorCode  KSPSetCheckNormIteration(KSP,PetscInt);
extern PetscErrorCode  KSPSetLagNorm(KSP,PetscBool );

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
extern const char *const*KSPConvergedReasons;

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

extern PetscErrorCode  KSPSetConvergenceTest(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),void *,PetscErrorCode (*)(void*));
extern PetscErrorCode  KSPGetConvergenceContext(KSP,void **);
extern PetscErrorCode  KSPDefaultConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
extern PetscErrorCode  KSPConvergedLSQR(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
extern PetscErrorCode  KSPDefaultConvergedDestroy(void *);
extern PetscErrorCode  KSPDefaultConvergedCreate(void **);
extern PetscErrorCode  KSPDefaultConvergedSetUIRNorm(KSP);
extern PetscErrorCode  KSPDefaultConvergedSetUMIRNorm(KSP);
extern PetscErrorCode  KSPSkipConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
extern PetscErrorCode  KSPGetConvergedReason(KSP,KSPConvergedReason *);

extern PetscErrorCode  KSPComputeExplicitOperator(KSP,Mat *);

/*E
    KSPCGType - Determines what type of CG to use

   Level: beginner

.seealso: KSPCGSetType()
E*/
typedef enum {KSP_CG_SYMMETRIC=0,KSP_CG_HERMITIAN=1} KSPCGType;
extern const char *KSPCGTypes[];

extern PetscErrorCode  KSPCGSetType(KSP,KSPCGType);
extern PetscErrorCode  KSPCGUseSingleReduction(KSP,PetscBool );

extern PetscErrorCode  KSPNASHSetRadius(KSP,PetscReal);
extern PetscErrorCode  KSPNASHGetNormD(KSP,PetscReal *);
extern PetscErrorCode  KSPNASHGetObjFcn(KSP,PetscReal *);

extern PetscErrorCode  KSPSTCGSetRadius(KSP,PetscReal);
extern PetscErrorCode  KSPSTCGGetNormD(KSP,PetscReal *);
extern PetscErrorCode  KSPSTCGGetObjFcn(KSP,PetscReal *);

extern PetscErrorCode  KSPGLTRSetRadius(KSP,PetscReal);
extern PetscErrorCode  KSPGLTRGetNormD(KSP,PetscReal *);
extern PetscErrorCode  KSPGLTRGetObjFcn(KSP,PetscReal *);
extern PetscErrorCode  KSPGLTRGetMinEig(KSP,PetscReal *);
extern PetscErrorCode  KSPGLTRGetLambda(KSP,PetscReal *);

extern PetscErrorCode  KSPPythonSetType(KSP,const char[]);

extern PetscErrorCode  PCPreSolve(PC,KSP);
extern PetscErrorCode  PCPostSolve(PC,KSP);

extern PetscErrorCode  KSPMonitorLGCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
extern PetscErrorCode  KSPMonitorLG(KSP,PetscInt,PetscReal,void*);
extern PetscErrorCode  KSPMonitorLGDestroy(PetscDrawLG*);
extern PetscErrorCode  KSPMonitorLGTrueResidualNormCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
extern PetscErrorCode  KSPMonitorLGTrueResidualNorm(KSP,PetscInt,PetscReal,void*);
extern PetscErrorCode  KSPMonitorLGTrueResidualNormDestroy(PetscDrawLG*);
extern PetscErrorCode  KSPMonitorLGRangeCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
extern PetscErrorCode  KSPMonitorLGRange(KSP,PetscInt,PetscReal,void*);
extern PetscErrorCode  KSPMonitorLGRangeDestroy(PetscDrawLG*);

extern PetscErrorCode  PCShellSetPreSolve(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec));
extern PetscErrorCode  PCShellSetPostSolve(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec));

/* see src/ksp/ksp/interface/iguess.c */
typedef struct _p_KSPFischerGuess {PetscInt method,curl,maxl,refcnt;PetscBool  monitor;Mat mat; KSP ksp;}* KSPFischerGuess;

extern PetscErrorCode  KSPFischerGuessCreate(KSP,PetscInt,PetscInt,KSPFischerGuess*);
extern PetscErrorCode  KSPFischerGuessDestroy(KSPFischerGuess*);
extern PetscErrorCode  KSPFischerGuessReset(KSPFischerGuess);
extern PetscErrorCode  KSPFischerGuessUpdate(KSPFischerGuess,Vec);
extern PetscErrorCode  KSPFischerGuessFormGuess(KSPFischerGuess,Vec,Vec);
extern PetscErrorCode  KSPFischerGuessSetFromOptions(KSPFischerGuess);

extern PetscErrorCode  KSPSetUseFischerGuess(KSP,PetscInt,PetscInt);
extern PetscErrorCode  KSPSetFischerGuess(KSP,KSPFischerGuess);
extern PetscErrorCode  KSPGetFischerGuess(KSP,KSPFischerGuess*);

extern PetscErrorCode  MatCreateSchurComplement(Mat,Mat,Mat,Mat,Mat,Mat*);
extern PetscErrorCode  MatSchurComplementGetKSP(Mat,KSP*);
extern PetscErrorCode  MatSchurComplementUpdate(Mat,Mat,Mat,Mat,Mat,Mat,MatStructure);
extern PetscErrorCode  MatSchurComplementGetSubmatrices(Mat,Mat*,Mat*,Mat*,Mat*,Mat*);
extern PetscErrorCode  MatGetSchurComplement(Mat,IS,IS,IS,IS,MatReuse,Mat *,MatReuse,Mat *);

extern PetscErrorCode  MatGetSchurComplement_Basic(Mat mat,IS isrow0,IS iscol0,IS isrow1,IS iscol1,MatReuse mreuse,Mat *newmat,MatReuse preuse,Mat *newpmat);

extern PetscErrorCode  KSPSetDM(KSP,DM);
extern PetscErrorCode  KSPSetDMActive(KSP,PetscBool );
extern PetscErrorCode  KSPGetDM(KSP,DM*);
extern PetscErrorCode  KSPSetApplicationContext(KSP,void*);
extern PetscErrorCode  KSPGetApplicationContext(KSP,void*);
extern PetscErrorCode KSPSetComputeOperators(KSP,PetscErrorCode(*)(KSP,Mat,Mat,MatStructure*,void*),void*);
extern PetscErrorCode KSPSetComputeRHS(KSP,PetscErrorCode(*)(KSP,Vec,void*),void*);
extern PetscErrorCode DMKSPSetComputeOperators(DM,PetscErrorCode(*)(KSP,Mat,Mat,MatStructure*,void*),void*);
extern PetscErrorCode DMKSPGetComputeOperators(DM,PetscErrorCode(**)(KSP,Mat,Mat,MatStructure*,void*),void*);
extern PetscErrorCode DMKSPSetComputeRHS(DM,PetscErrorCode(*)(KSP,Vec,void*),void*);
extern PetscErrorCode DMKSPGetComputeRHS(DM,PetscErrorCode(**)(KSP,Vec,void*),void*);

PETSC_EXTERN_CXX_END
#endif
