/* $Id: petscksp.h,v 1.107 2001/08/06 21:16:38 bsmith Exp $ */
/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __PETSCKSP_H
#define __PETSCKSP_H
#include "petscpc.h"

/*S
     KSP - Abstract PETSc object that manages all Krylov methods

   Level: beginner

  Concepts: Krylov methods

.seealso:  KSPCreate(), KSPSetType(), KSPType, SNES, TS, PC, SLES
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
typedef char * KSPType;

/* Logging support */
extern int KSP_COOKIE;
extern int KSP_GMRESOrthogonalization;

EXTERN int KSPCreate(MPI_Comm,KSP *);
EXTERN int KSPSetType(KSP,KSPType);
EXTERN int KSPSetUp(KSP);
EXTERN int KSPSolve(KSP,int *);
EXTERN int KSPSolveTranspose(KSP,int *);
EXTERN int KSPDestroy(KSP);

extern PetscFList KSPList;
EXTERN int KSPRegisterAll(char *);
EXTERN int KSPRegisterDestroy(void);

EXTERN int KSPRegister(char*,char*,char*,int(*)(KSP));

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
EXTERN int KSPGetResidualHistory(KSP,PetscReal **,int *);
EXTERN int KSPSetResidualHistory(KSP,PetscReal *,int,PetscTruth);


EXTERN int KSPBuildSolution(KSP,Vec,Vec *);
EXTERN int KSPBuildResidual(KSP,Vec,Vec,Vec *);

EXTERN int KSPRichardsonSetScale(KSP,PetscReal);
EXTERN int KSPChebychevSetEigenvalues(KSP,PetscReal,PetscReal);
EXTERN int KSPComputeExtremeSingularValues(KSP,PetscReal*,PetscReal*);
EXTERN int KSPComputeEigenvalues(KSP,int,PetscReal*,PetscReal*,int *);
EXTERN int KSPComputeEigenvaluesExplicitly(KSP,int,PetscReal*,PetscReal*);

#define KSPGMRESSetRestart(ksp,r) PetscTryMethod((ksp),KSPGMRESSetRestart_C,(KSP,int),((ksp),(r)))
#define KSPGMRESSetHapTol(ksp,tol) PetscTryMethod((ksp),KSPGMRESSetHapTol_C,(KSP,PetscReal),((ksp),(tol)))

EXTERN int KSPGMRESSetPreAllocateVectors(KSP);
EXTERN int KSPGMRESSetOrthogonalization(KSP,int (*)(KSP,int));
EXTERN int KSPGMRESUnmodifiedGramSchmidtOrthogonalization(KSP,int);
EXTERN int KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,int);
EXTERN int KSPGMRESIROrthogonalization(KSP,int);

EXTERN int KSPFGMRESModifyPCNoChange(KSP,int,int,PetscReal,void*);
EXTERN int KSPFGMRESModifyPCSLES(KSP,int,int,PetscReal,void*);
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

EXTERN int KSPSetOptionsPrefix(KSP,char*);
EXTERN int KSPAppendOptionsPrefix(KSP,char*);
EXTERN int KSPGetOptionsPrefix(KSP,char**);

EXTERN int KSPView(KSP,PetscViewer);

/*E
    KSPNormType - Norm that is passed in the Krylov convergence
       test routines.

   Level: advanced

   Notes: this must match finclude/petscksp.h 

.seealso: SLESSolve(), KSPSolve(), KSPGetConvergedReason(), KSPSetNormType(),
          KSPSetConvergenceTest()
E*/
typedef enum {KSP_NO_NORM               = 0,
              KSP_PRECONDITIONED_NORM   = 1,
              KSP_UNPRECONDITIONED_NORM = 2,
              KSP_NATURAL_NORM          = 3} KSPNormType;
EXTERN int KSPSetNormType(KSP,KSPNormType);

/*E
    KSPConvergedReason - reason a Krylov method was said to 
         have converged or diverged

   Level: beginner

   Notes: this must match finclude/petscksp.h 

.seealso: SLESSolve(), KSPSolve(), KSPGetConvergedReason()
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

EXTERN int KSPLGMonitorCreate(char*,char*,int,int,int,int,PetscDrawLG*);
EXTERN int KSPLGMonitor(KSP,int,PetscReal,void*);
EXTERN int KSPLGMonitorDestroy(PetscDrawLG);
EXTERN int KSPLGTrueMonitorCreate(MPI_Comm,char*,char*,int,int,int,int,PetscDrawLG*);
EXTERN int KSPLGTrueMonitor(KSP,int,PetscReal,void*);
EXTERN int KSPLGTrueMonitorDestroy(PetscDrawLG);

#endif


