/* $Id: petscksp.h,v 1.107 2001/08/06 21:16:38 bsmith Exp $ */
/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __PETSCKSP_H
#define __PETSCKSP_H
#include "petscpc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN int KSPInitializePackage(const char[]);

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
extern int KSP_COOKIE;
extern int KSP_GMRESOrthogonalization;
extern int KSP_SetUp, KSP_Solve;

EXTERN int KSPCreate(MPI_Comm,KSP *);
EXTERN int KSPSetType(KSP,const KSPType);
EXTERN int KSPSetUp(KSP);
EXTERN int KSPSetUpOnBlocks(KSP);
EXTERN int KSPSolve(KSP);
EXTERN int KSPSolveTranspose(KSP);
EXTERN int KSPDestroy(KSP);

extern PetscFList KSPList;
EXTERN int KSPRegisterAll(const char[]);
EXTERN int KSPRegisterDestroy(void);

EXTERN int KSPRegister(const char[],const char[],const char[],int(*)(KSP));

/*MC
   KSPRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   int KSPRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(KSP))

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

   Notes: Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, ${BOPT},
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

EXTERN int KSPGetType(KSP,KSPType *);
EXTERN int KSPSetPreconditionerSide(KSP,PCSide);
EXTERN int KSPGetPreconditionerSide(KSP,PCSide*);
EXTERN int KSPGetTolerances(KSP,PetscReal*,PetscReal*,PetscReal*,int*);
EXTERN int KSPSetTolerances(KSP,PetscReal,PetscReal,PetscReal,int);
EXTERN int KSPSetInitialGuessNonzero(KSP,PetscTruth);
EXTERN int KSPGetInitialGuessNonzero(KSP,PetscTruth *);
EXTERN int KSPSetInitialGuessKnoll(KSP,PetscTruth);
EXTERN int KSPGetInitialGuessKnoll(KSP,PetscTruth*);
EXTERN int KSPSetComputeEigenvalues(KSP,PetscTruth);
EXTERN int KSPSetComputeSingularValues(KSP,PetscTruth);
EXTERN int KSPSetRhs(KSP,Vec);
EXTERN int KSPGetRhs(KSP,Vec *);
EXTERN int KSPSetSolution(KSP,Vec);
EXTERN int KSPGetSolution(KSP,Vec *);
EXTERN int KSPGetResidualNorm(KSP,PetscReal*);
EXTERN int KSPGetIterationNumber(KSP,int*);

EXTERN int KSPSetPC(KSP,PC);
EXTERN int KSPGetPC(KSP,PC*);

EXTERN int KSPSetMonitor(KSP,int (*)(KSP,int,PetscReal,void*),void *,int (*)(void*));
EXTERN int KSPClearMonitor(KSP);
EXTERN int KSPGetMonitorContext(KSP,void **);
EXTERN int KSPGetResidualHistory(KSP,PetscReal*[],int *);
EXTERN int KSPSetResidualHistory(KSP,PetscReal[],int,PetscTruth);

/* not sure where to put this */
EXTERN int PCKSPGetKSP(PC,KSP*);
EXTERN int PCBJacobiGetSubKSP(PC,int*,int*,KSP*[]);
EXTERN int PCASMGetSubKSP(PC,int*,int*,KSP*[]);

EXTERN int KSPBuildSolution(KSP,Vec,Vec *);
EXTERN int KSPBuildResidual(KSP,Vec,Vec,Vec *);

EXTERN int KSPRichardsonSetScale(KSP,PetscReal);
EXTERN int KSPChebychevSetEigenvalues(KSP,PetscReal,PetscReal);
EXTERN int KSPComputeExtremeSingularValues(KSP,PetscReal*,PetscReal*);
EXTERN int KSPComputeEigenvalues(KSP,int,PetscReal*,PetscReal*,int *);
EXTERN int KSPComputeEigenvaluesExplicitly(KSP,int,PetscReal*,PetscReal*);

EXTERN int KSPGMRESSetRestart(KSP, int);
EXTERN int KSPGMRESSetHapTol(KSP,PetscReal);

EXTERN int KSPGMRESSetPreAllocateVectors(KSP);
EXTERN int KSPGMRESSetOrthogonalization(KSP,int (*)(KSP,int));
EXTERN int KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,int);
EXTERN int KSPGMRESClassicalGramSchmidtOrthogonalization(KSP,int);

EXTERN int KSPLGMRESSetAugDim(KSP,int);
EXTERN int KSPLGMRESSetConstant(KSP);

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

EXTERN int KSPGMRESSetCGSRefinementType(KSP,KSPGMRESCGSRefinementType);

EXTERN int KSPFGMRESModifyPCNoChange(KSP,int,int,PetscReal,void*);
EXTERN int KSPFGMRESModifyPCKSP(KSP,int,int,PetscReal,void*);
EXTERN int KSPFGMRESSetModifyPC(KSP,int (*)(KSP,int,int,PetscReal,void*),void*,int(*)(void*));

EXTERN int KSPQCGSetTrustRegionRadius(KSP,PetscReal);
EXTERN int KSPQCGGetQuadratic(KSP,PetscReal*);
EXTERN int KSPQCGGetTrialStepNorm(KSP,PetscReal*);

EXTERN int KSPSetFromOptions(KSP);
EXTERN int KSPAddOptionsChecker(int (*)(KSP));

EXTERN int KSPSingularValueMonitor(KSP,int,PetscReal,void *);
EXTERN int KSPDefaultMonitor(KSP,int,PetscReal,void *);
EXTERN int KSPTrueMonitor(KSP,int,PetscReal,void *);
EXTERN int KSPDefaultSMonitor(KSP,int,PetscReal,void *);
EXTERN int KSPVecViewMonitor(KSP,int,PetscReal,void *);
EXTERN int KSPGMRESKrylovMonitor(KSP,int,PetscReal,void *);

EXTERN int KSPUnwindPreconditioner(KSP,Vec,Vec);
EXTERN int KSPDefaultBuildSolution(KSP,Vec,Vec*);
EXTERN int KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);

EXTERN int KSPSetOperators(KSP,Mat,Mat,MatStructure);
EXTERN int KSPSetOptionsPrefix(KSP,const char[]);
EXTERN int KSPAppendOptionsPrefix(KSP,const char[]);
EXTERN int KSPGetOptionsPrefix(KSP,char*[]);

EXTERN int KSPSetDiagonalScale(KSP,PetscTruth);
EXTERN int KSPGetDiagonalScale(KSP,PetscTruth*);
EXTERN int KSPSetDiagonalScaleFix(KSP,PetscTruth);
EXTERN int KSPGetDiagonalScaleFix(KSP,PetscTruth*);

EXTERN int KSPView(KSP,PetscViewer);

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

EXTERN int KSPSetNormType(KSP,KSPNormType);

/*E
    KSPConvergedReason - reason a Krylov method was said to 
         have converged or diverged

   Level: beginner

   Notes: this must match finclude/petscksp.h 

   Developer note: The string versions of these are in 
     src/ksp/ksp/interface/itfunc.c called convergedreasons.
     If these enums are changed you much change those.

.seealso: KSPSolve(), KSPGetConvergedReason()
E*/
typedef enum {/* converged */
              KSP_CONVERGED_RTOL               =  2,
              KSP_CONVERGED_ATOL               =  3,
              KSP_CONVERGED_ITS                =  4,
              KSP_CONVERGED_QCG_NEG_CURVE      =  5,
              KSP_CONVERGED_QCG_CONSTRAINED    =  6,
              KSP_CONVERGED_STEP_LENGTH        =  7,
              /* diverged */
              KSP_DIVERGED_ITS                 = -3,
              KSP_DIVERGED_DTOL                = -4,
              KSP_DIVERGED_BREAKDOWN           = -5,
              KSP_DIVERGED_BREAKDOWN_BICG      = -6,
              KSP_DIVERGED_NONSYMMETRIC        = -7,
              KSP_DIVERGED_INDEFINITE_PC       = -8,
 
              KSP_CONVERGED_ITERATING          =  0} KSPConvergedReason;

EXTERN int KSPSetConvergenceTest(KSP,int (*)(KSP,int,PetscReal,KSPConvergedReason*,void*),void *);
EXTERN int KSPGetConvergenceContext(KSP,void **);
EXTERN int KSPDefaultConverged(KSP,int,PetscReal,KSPConvergedReason*,void *);
EXTERN int KSPSkipConverged(KSP,int,PetscReal,KSPConvergedReason*,void *);
EXTERN int KSPGetConvergedReason(KSP,KSPConvergedReason *);

EXTERN int KSPComputeExplicitOperator(KSP,Mat *);

/*E
    KSPCGType - Determines what type of CG to use

   Level: beginner

.seealso: KSPCGSetType()
E*/
typedef enum {KSP_CG_SYMMETRIC=1,KSP_CG_HERMITIAN=2} KSPCGType;

EXTERN int KSPCGSetType(KSP,KSPCGType);

EXTERN int PCPreSolve(PC,KSP);
EXTERN int PCPostSolve(PC,KSP);

EXTERN int KSPLGMonitorCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN int KSPLGMonitor(KSP,int,PetscReal,void*);
EXTERN int KSPLGMonitorDestroy(PetscDrawLG);
EXTERN int KSPLGTrueMonitorCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN int KSPLGTrueMonitor(KSP,int,PetscReal,void*);
EXTERN int KSPLGTrueMonitorDestroy(PetscDrawLG);

PETSC_EXTERN_CXX_END
#endif
