/* $Id: petscksp.h,v 1.89 2000/05/04 03:47:37 bsmith Exp balay $ */
/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __PETSCKSP_H
#define __PETSCKSP_H
#include "petscpc.h"

#define KSP_COOKIE  PETSC_COOKIE+8

typedef struct _p_KSP*     KSP;

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
typedef char * KSPType;

extern int KSPCreate(MPI_Comm,KSP *);
extern int KSPSetType(KSP,KSPType);
extern int KSPSetUp(KSP);
extern int KSPSolve(KSP,int *);
extern int KSPSolveTranspose(KSP,int *);
extern int KSPDestroy(KSP);

extern FList KSPList;
extern int KSPRegisterAll(char *);
extern int KSPRegisterDestroy(void);

extern int KSPRegister(char*,char*,char*,int(*)(KSP));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define KSPRegisterDynamic(a,b,c,d) KSPRegister(a,b,c,0)
#else
#define KSPRegisterDynamic(a,b,c,d) KSPRegister(a,b,c,d)
#endif

extern int KSPGetType(KSP,KSPType *);
extern int KSPSetPreconditionerSide(KSP,PCSide);
extern int KSPGetPreconditionerSide(KSP,PCSide*);
extern int KSPGetTolerances(KSP,double*,double*,double*,int*);
extern int KSPSetTolerances(KSP,double,double,double,int);
extern int KSPSetComputeResidual(KSP,PetscTruth);
extern int KSPSetUsePreconditionedResidual(KSP);
extern int KSPSetInitialGuessNonzero(KSP);
extern int KSPGetInitialGuessNonzero(KSP,PetscTruth *);
extern int KSPSetComputeEigenvalues(KSP);
extern int KSPSetComputeSingularValues(KSP);
extern int KSPSetRhs(KSP,Vec);
extern int KSPGetRhs(KSP,Vec *);
extern int KSPSetSolution(KSP,Vec);
extern int KSPGetSolution(KSP,Vec *);
extern int KSPGetResidualNorm(KSP,double*);
extern int KSPGetIterationNumber(KSP,int*);

extern int KSPSetPC(KSP,PC);
extern int KSPGetPC(KSP,PC*);

extern int KSPSetAvoidNorms(KSP);

extern int KSPSetMonitor(KSP,int (*)(KSP,int,double,void*),void *,int (*)(void*));
extern int KSPClearMonitor(KSP);
extern int KSPGetMonitorContext(KSP,void **);
extern int KSPGetResidualHistory(KSP,double **,int *);
extern int KSPSetResidualHistory(KSP,double *,int,PetscTruth);


extern int KSPBuildSolution(KSP,Vec,Vec *);
extern int KSPBuildResidual(KSP,Vec,Vec,Vec *);

extern int KSPRichardsonSetScale(KSP,double);
extern int KSPChebychevSetEigenvalues(KSP,double,double);
extern int KSPComputeExtremeSingularValues(KSP,double*,double*);
extern int KSPComputeEigenvalues(KSP,int,double*,double*,int *);
extern int KSPComputeEigenvaluesExplicitly(KSP,int,double*,double*);

extern int KSPGMRESSetRestart(KSP,int);
extern int KSPGMRESSetPreAllocateVectors(KSP);
extern int KSPGMRESSetOrthogonalization(KSP,int (*)(KSP,int));
extern int KSPGMRESUnmodifiedGramSchmidtOrthogonalization(KSP,int);
extern int KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,int);
extern int KSPGMRESIROrthogonalization(KSP,int);

extern int KSPFGMRESModifyPCNoChange(KSP,int,int,double,void*);
extern int KSPFGMRESModifyPCSLES(KSP,int,int,double,void*);
extern int KSPFGMRESSetModifyPC(KSP,int (*)(KSP,int,int,double,void*),void*,int(*)(void*));

extern int KSPSetFromOptions(KSP);
extern int KSPSetTypeFromOptions(KSP);
extern int KSPAddOptionsChecker(int (*)(KSP));

extern int KSPSingularValueMonitor(KSP,int,double,void *);
extern int KSPDefaultMonitor(KSP,int,double,void *);
extern int KSPTrueMonitor(KSP,int,double,void *);
extern int KSPDefaultSMonitor(KSP,int,double,void *);
extern int KSPVecViewMonitor(KSP,int,double,void *);
extern int KSPGMRESKrylovMonitor(KSP,int,double,void *);


extern int KSPResidual(KSP,Vec,Vec,Vec,Vec,Vec,Vec);
extern int KSPUnwindPreconditioner(KSP,Vec,Vec);
extern int KSPDefaultBuildSolution(KSP,Vec,Vec*);
extern int KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);

extern int KSPPrintHelp(KSP);

extern int KSPSetOptionsPrefix(KSP,char*);
extern int KSPAppendOptionsPrefix(KSP,char*);
extern int KSPGetOptionsPrefix(KSP,char**);

extern int KSPView(KSP,Viewer);

typedef enum {/* converged */
              KSP_CONVERGED_RTOL               =  2,
              KSP_CONVERGED_ATOL               =  3,
              KSP_CONVERGED_ITS                =  4,
              KSP_CONVERGED_QCG_NEGATIVE_CURVE =  5,
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

extern int KSPSetConvergenceTest(KSP,int (*)(KSP,int,double,KSPConvergedReason*,void*),void *);
extern int KSPGetConvergenceContext(KSP,void **);
extern int KSPDefaultConverged(KSP,int,double,KSPConvergedReason*,void *);
extern int KSPSkipConverged(KSP,int,double,KSPConvergedReason*,void *);
extern int KSPGetConvergedReason(KSP,KSPConvergedReason *);

extern int KSPComputeExplicitOperator(KSP,Mat *);

typedef enum {KSP_CG_SYMMETRIC=1,KSP_CG_HERMITIAN=2} KSPCGType;
extern int KSPCGSetType(KSP,KSPCGType);

extern int PCPreSolve(PC,KSP);
extern int PCPostSolve(PC,KSP);

extern int KSPLGMonitorCreate(char*,char*,int,int,int,int,DrawLG*);
extern int KSPLGMonitor(KSP,int,double,void*);
extern int KSPLGMonitorDestroy(DrawLG);
extern int KSPLGTrueMonitorCreate(MPI_Comm,char*,char*,int,int,int,int,DrawLG*);
extern int KSPLGTrueMonitor(KSP,int,double,void*);
extern int KSPLGTrueMonitorDestroy(DrawLG);

#endif


