/* $Id: ksp.h,v 1.82 1999/11/05 14:48:27 bsmith Exp bsmith $ */
/*
   Defines the interface functions for the Krylov subspace accelerators.
*/
#ifndef __KSP_H
#define __KSP_H
#include "petsc.h"
#include "vec.h"
#include "mat.h"
#include "pc.h"

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
typedef char * KSPType;

extern int KSPCreate(MPI_Comm,KSP *);
extern int KSPSetType(KSP,KSPType);
extern int KSPSetUp(KSP);
extern int KSPSolve(KSP,int *);
extern int KSPSolveTrans(KSP,int *);
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

extern int KSPGetType(KSP, KSPType *);
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

extern int KSPSetMonitor(KSP,int (*)(KSP,int,double, void*), void *,int (*)(void*));
extern int KSPClearMonitor(KSP);
extern int KSPGetMonitorContext(KSP,void **);
extern int KSPGetResidualHistory(KSP, double **, int *);
extern int KSPSetResidualHistory(KSP, double *,int,PetscTruth);

extern int KSPSetConvergenceTest(KSP,int (*)(KSP,int,double, void*), void *);
extern int KSPGetConvergenceContext(KSP,void **);

extern int KSPBuildSolution(KSP, Vec,Vec *);
extern int KSPBuildResidual(KSP, Vec, Vec,Vec *);

extern int KSPRichardsonSetScale(KSP , double);
extern int KSPChebychevSetEigenvalues(KSP , double, double);
extern int KSPComputeExtremeSingularValues(KSP, double*,double*);
extern int KSPComputeEigenvalues(KSP,int,double*,double*,int *);
extern int KSPComputeEigenvaluesExplicitly(KSP,int,double*,double*);

extern int KSPGMRESSetRestart(KSP, int);
extern int KSPGMRESSetPreAllocateVectors(KSP);
extern int KSPGMRESSetOrthogonalization(KSP,int (*)(KSP,int));
extern int KSPGMRESUnmodifiedGramSchmidtOrthogonalization(KSP,int);
extern int KSPGMRESModifiedGramSchmidtOrthogonalization(KSP,int);
extern int KSPGMRESIROrthogonalization(KSP,int);
extern int KSPGMRESPrestartSet(KSP,int);


extern int KSPFGMRESSetRestart( KSP, int);
extern int KSPFGMRESSetPreAllocateVectors( KSP );
extern int KSPFGMRESSetOrthogonalization( KSP, int (*)( KSP, int ) );
extern int KSPFGMRESUnmodifiedGramSchmidtOrthogonalization( KSP, int );
extern int KSPFGMRESModifiedGramSchmidtOrthogonalization( KSP, int );
extern int KSPFGMRESIROrthogonalization( KSP, int );
extern int KSPFGMRESPrestartSet( KSP, int );

extern int KSPFGMRESModifyPCNoChange( KSP, int, int, int, int, double );
extern int KSPFGMRESModifyPCGMRESVariableEx( KSP, int, int, int, int, double );
extern int KSPFGMRESModifyPCEx( KSP, int, int, int, int, double );
extern int KSPFGMRESSetModifyPC(KSP,int (*)(KSP,int,int,int,int,double));

extern int KSPSetFromOptions(KSP);
extern int KSPSetTypeFromOptions(KSP);
extern int KSPAddOptionsChecker(int (*)(KSP));

extern int KSPSingularValueMonitor(KSP,int,double, void * );
extern int KSPDefaultMonitor(KSP,int,double, void *);
extern int KSPTrueMonitor(KSP,int,double, void *);
extern int KSPDefaultSMonitor(KSP,int,double, void *);
extern int KSPVecViewMonitor(KSP,int,double,void *);
extern int KSPGMRESKrylovMonitor(KSP,int,double,void *);

extern int KSPDefaultConverged(KSP,int,double, void *);
extern int KSPSkipConverged(KSP,int,double, void *);

extern int KSPResidual(KSP,Vec,Vec,Vec,Vec,Vec,Vec);
extern int KSPUnwindPreconditioner(KSP,Vec,Vec);
extern int KSPDefaultBuildSolution(KSP,Vec,Vec*);
extern int KSPDefaultBuildResidual(KSP,Vec,Vec,Vec *);

extern int KSPPrintHelp(KSP);

extern int KSPSetOptionsPrefix(KSP,char*);
extern int KSPAppendOptionsPrefix(KSP,char*);
extern int KSPGetOptionsPrefix(KSP,char**);

extern int KSPView(KSP,Viewer);

extern int KSPComputeExplicitOperator(KSP,Mat *);

typedef enum {KSP_CG_SYMMETRIC=1, KSP_CG_HERMITIAN=2} KSPCGType;
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


