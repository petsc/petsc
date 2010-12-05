/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __PETSCKSP_H
#define __PETSCKSP_H
#include "petscpc.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPInitializePackage(const char[]);

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
#define KSPNGMRES     "ngmres"
#define KSPSPECEST    "specest"

/* Logging support */
extern PetscClassId PETSCKSP_DLLEXPORT KSP_CLASSID;

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate(MPI_Comm,KSP *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetType(KSP,const KSPType);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetUp(KSP);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetUpOnBlocks(KSP);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSolve(KSP,Vec,Vec);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSolveTranspose(KSP,Vec,Vec);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPDestroy(KSP);

extern PetscFList KSPList;
extern PetscBool  KSPRegisterAllCalled;
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPRegisterAll(const char[]);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPRegisterDestroy(void);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPRegister(const char[],const char[],const char[],PetscErrorCode (*)(KSP));

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

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetType(KSP,const KSPType *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetPCSide(KSP,PCSide);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetPCSide(KSP,PCSide*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetTolerances(KSP,PetscReal*,PetscReal*,PetscReal*,PetscInt*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetTolerances(KSP,PetscReal,PetscReal,PetscReal,PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetInitialGuessNonzero(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetInitialGuessNonzero(KSP,PetscBool  *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetInitialGuessKnoll(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetInitialGuessKnoll(KSP,PetscBool *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetErrorIfNotConverged(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetErrorIfNotConverged(KSP,PetscBool  *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetComputeEigenvalues(KSP,PetscBool *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetComputeEigenvalues(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetComputeSingularValues(KSP,PetscBool *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetComputeSingularValues(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetRhs(KSP,Vec *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetSolution(KSP,Vec *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetResidualNorm(KSP,PetscReal*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetIterationNumber(KSP,PetscInt*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetNullSpace(KSP,MatNullSpace);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetNullSpace(KSP,MatNullSpace*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetVecs(KSP,PetscInt,Vec**,PetscInt,Vec**);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetPC(KSP,PC);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetPC(KSP,PC*);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorSet(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void*));
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorCancel(KSP);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetMonitorContext(KSP,void **);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetResidualHistory(KSP,PetscReal*[],PetscInt *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetResidualHistory(KSP,PetscReal[],PetscInt,PetscBool );

/* not sure where to put this */
extern PetscErrorCode PETSCKSP_DLLEXPORT PCKSPGetKSP(PC,KSP*);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCBJacobiGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCASMGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCGASMGetSubKSP(PC,PetscInt*,PetscInt*,KSP*[]);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCFieldSplitGetSubKSP(PC,PetscInt*,KSP*[]);

extern PetscErrorCode PETSCKSP_DLLEXPORT PCGalerkinGetKSP(PC,KSP *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPBuildSolution(KSP,Vec,Vec *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPBuildResidual(KSP,Vec,Vec,Vec *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPRichardsonSetScale(KSP,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPRichardsonSetSelfScale(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPChebychevSetEigenvalues(KSP,PetscReal,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPComputeExtremeSingularValues(KSP,PetscReal*,PetscReal*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPComputeEigenvalues(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPComputeEigenvaluesExplicitly(KSP,PetscInt,PetscReal*,PetscReal*);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetRestart(KSP, PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESGetRestart(KSP, PetscInt*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetHapTol(KSP,PetscReal);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetPreAllocateVectors(KSP);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetOrthogonalization(KSP,PetscErrorCode (*)(KSP,PetscInt));
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESGetOrthogonalization(KSP,PetscErrorCode (**)(KSP,PetscInt));
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESClassicalGramSchmidtOrthogonalization(KSP,PetscInt);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPLGMRESSetAugDim(KSP,PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPLGMRESSetConstant(KSP);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGCRSetRestart(KSP,PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGCRGetRestart(KSP,PetscInt*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGCRSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

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

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetCGSRefinementType(KSP,KSPGMRESCGSRefinementType);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESGetCGSRefinementType(KSP,KSPGMRESCGSRefinementType*);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFGMRESModifyPCNoChange(KSP,PetscInt,PetscInt,PetscReal,void*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFGMRESModifyPCKSP(KSP,PetscInt,PetscInt,PetscReal,void*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFGMRESSetModifyPC(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscInt,PetscReal,void*),void*,PetscErrorCode(*)(void*));

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGSetTrustRegionRadius(KSP,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGGetQuadratic(KSP,PetscReal*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPQCGGetTrialStepNorm(KSP,PetscReal*);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetXRes(KSP,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetPol(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPBCGSLSetEll(KSP,PetscInt);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetFromOptions(KSP);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPAddOptionsChecker(PetscErrorCode (*)(KSP));

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorSingularValue(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorDefault(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorDefaultLSQR(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorRange(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorTrueResidualNorm(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorDefaultShort(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorSolution(KSP,PetscInt,PetscReal,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESMonitorKrylov(KSP,PetscInt,PetscReal,void *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPUnwindPreconditioner(KSP,Vec,Vec);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultBuildSolution(KSP,Vec,Vec*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPInitialResidual(KSP,Vec,Vec,Vec,Vec,Vec);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetOperators(KSP,Mat,Mat,MatStructure);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOperators(KSP,Mat*,Mat*,MatStructure*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOperatorsSet(KSP,PetscBool *,PetscBool *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetOptionsPrefix(KSP,const char[]);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPAppendOptionsPrefix(KSP,const char[]);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOptionsPrefix(KSP,const char*[]);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetDiagonalScale(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetDiagonalScale(KSP,PetscBool *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetDiagonalScaleFix(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetDiagonalScaleFix(KSP,PetscBool *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPView(KSP,PetscViewer);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPLSQRSetStandardErrorVec(KSP,Vec);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPLSQRGetStandardErrorVec(KSP,Vec*);

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
typedef enum {KSP_NORM_NONE = 0,KSP_NORM_PRECONDITIONED = 1,KSP_NORM_UNPRECONDITIONED = 2,KSP_NORM_NATURAL = 3} KSPNormType;
extern const char *KSPNormTypes[];
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

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetNormType(KSP,KSPNormType);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetNormType(KSP,KSPNormType*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetCheckNormIteration(KSP,PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetLagNorm(KSP,PetscBool );

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

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetConvergenceTest(KSP,PetscErrorCode (*)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),void *,PetscErrorCode (*)(void*));
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetConvergenceContext(KSP,void **);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPConvergedLSQR(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConvergedDestroy(void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConvergedCreate(void **);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConvergedSetUIRNorm(KSP);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPDefaultConvergedSetUMIRNorm(KSP);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSkipConverged(KSP,PetscInt,PetscReal,KSPConvergedReason*,void *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetConvergedReason(KSP,KSPConvergedReason *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPComputeExplicitOperator(KSP,Mat *);

/*E
    KSPCGType - Determines what type of CG to use

   Level: beginner

.seealso: KSPCGSetType()
E*/
typedef enum {KSP_CG_SYMMETRIC=0,KSP_CG_HERMITIAN=1} KSPCGType;
extern const char *KSPCGTypes[];

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPCGSetType(KSP,KSPCGType);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPCGUseSingleReduction(KSP,PetscBool );

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPNASHSetRadius(KSP,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPNASHGetNormD(KSP,PetscReal *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPNASHGetObjFcn(KSP,PetscReal *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGSetRadius(KSP,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGGetNormD(KSP,PetscReal *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSTCGGetObjFcn(KSP,PetscReal *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRSetRadius(KSP,PetscReal);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetNormD(KSP,PetscReal *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetObjFcn(KSP,PetscReal *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetMinEig(KSP,PetscReal *);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGLTRGetLambda(KSP,PetscReal *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPPythonSetType(KSP,const char[]);

extern PetscErrorCode PETSCKSP_DLLEXPORT PCPreSolve(PC,KSP);
extern PetscErrorCode PETSCKSP_DLLEXPORT PCPostSolve(PC,KSP);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLG(KSP,PetscInt,PetscReal,void*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGDestroy(PetscDrawLG);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGTrueResidualNormCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGTrueResidualNorm(KSP,PetscInt,PetscReal,void*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGTrueResidualNormDestroy(PetscDrawLG);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGRangeCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGRange(KSP,PetscInt,PetscReal,void*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPMonitorLGRangeDestroy(PetscDrawLG);

extern PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetPreSolve(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec));
extern PetscErrorCode PETSCKSP_DLLEXPORT PCShellSetPostSolve(PC,PetscErrorCode (*)(PC,KSP,Vec,Vec));

/* see src/ksp/ksp/interface/iguess.c */
typedef struct _p_KSPFischerGuess {PetscInt method,curl,maxl,refcnt;PetscBool  monitor;Mat mat; KSP ksp;}* KSPFischerGuess;

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessCreate(KSP,PetscInt,PetscInt,KSPFischerGuess*);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessDestroy(KSPFischerGuess);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessReset(KSPFischerGuess);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessUpdate(KSPFischerGuess,Vec);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessFormGuess(KSPFischerGuess,Vec,Vec);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPFischerGuessSetFromOptions(KSPFischerGuess);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetUseFischerGuess(KSP,PetscInt,PetscInt);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetFischerGuess(KSP,KSPFischerGuess);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetFischerGuess(KSP,KSPFischerGuess*);

extern PetscErrorCode PETSCKSP_DLLEXPORT MatCreateSchurComplement(Mat,Mat,Mat,Mat,Mat,Mat*);
extern PetscErrorCode PETSCKSP_DLLEXPORT MatSchurComplementGetKSP(Mat,KSP*);
extern PetscErrorCode PETSCKSP_DLLEXPORT MatSchurComplementUpdate(Mat,Mat,Mat,Mat,Mat,Mat,MatStructure);
extern PetscErrorCode PETSCKSP_DLLEXPORT MatSchurComplementGetSubmatrices(Mat,Mat*,Mat*,Mat*,Mat*,Mat*);
extern PetscErrorCode PETSCKSP_DLLEXPORT MatGetSchurComplement(Mat,IS,IS,IS,IS,MatReuse,Mat *,MatReuse,Mat *);

extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetDM(KSP,DM);
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPSetDMActive(KSP,PetscBool );
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPGetDM(KSP,DM*);

PETSC_EXTERN_CXX_END
#endif
