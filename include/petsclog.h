/* $Id: plog.h,v 1.58 1996/03/01 01:51:38 bsmith Exp bsmith $ */

/*
    Defines high level logging in PETSc.
*/

#if !defined(__PLOG_PACKAGE)
#define __PLOG_PACKAGE
#include "petsc.h"  

/*
  If you add an event here, make sure you add to petsc/bin/petscview.cfg,
  petsc/bin/petscview, petsc/src/sys/src/plog.c, and 
  petsc/src/sys/src/plogmpe.c!!!
*/
#define MAT_Mult                                0
#define MAT_MatrixFreeMult                      1
#define MAT_AssemblyBegin                       2
#define MAT_AssemblyEnd                         3
#define MAT_GetReordering                       4
#define MAT_MultTrans                           5
#define MAT_MultAdd                             6
#define MAT_MultTransAdd                        7
#define MAT_LUFactor                            8
#define MAT_CholeskyFactor                      9
#define MAT_LUFactorSymbolic                    10
#define MAT_ILUFactorSymbolic                   11
#define MAT_CholeskyFactorSymbolic              12
#define MAT_IncompleteCholeskyFactorSymbolic    13
#define MAT_LUFactorNumeric                     14
#define MAT_CholeskyFactorNumeric               15
#define MAT_Relax                               16
#define MAT_Copy                                17
#define MAT_Convert                             18
#define MAT_Scale                               19
#define MAT_ZeroEntries                         20
#define MAT_Solve                               21
#define MAT_SolveAdd                            22
#define MAT_SolveTrans                          23
#define MAT_SolveTransAdd                       24
#define MAT_SetValues                           25
#define MAT_ForwardSolve                        26
#define MAT_BackwardSolve                       27
#define MAT_Load                                28
#define MAT_View                                29
#define MAT_ILUFactor                           30
#define MAT_GetSubMatrix                        31
#define MAT_GetSubMatrices                      32
#define MAT_GetValues                           33
#define MAT_IncreaseOverlap                     34
#define MAT_GetRow                              35

#define VEC_Dot                                 40
#define VEC_Norm                                41
#define VEC_Max                                 42
#define VEC_Min                                 43
#define VEC_TDot                                44
#define VEC_Scale                               45
#define VEC_Copy                                46
#define VEC_Set                                 47
#define VEC_AXPY                                48
#define VEC_AYPX                                49
#define VEC_Swap                                50
#define VEC_WAXPY                               51
#define VEC_AssemblyBegin                       52
#define VEC_AssemblyEnd                         53
#define VEC_MTDot                               54
#define VEC_MDot                                55
#define VEC_MAXPY                               56
#define VEC_PMult                               57
#define VEC_SetValues                           58
#define VEC_Load                                59
#define VEC_View                                60
#define VEC_ScatterBegin                        61
#define VEC_ScatterEnd                          62
#define VEC_SetRandom                           63

#define SLES_Solve                              70
#define SLES_SetUp                              71

#define KSP_GMRESOrthogonalization              72

#define PC_SetUp                                75
#define PC_SetUpOnBlocks                        76
#define PC_Apply                                77
#define PC_ApplySymmLeft                        78
#define PC_ApplySymmRight                       79

#define SNES_Solve                              80
#define SNES_LineSearch                         81
#define SNES_FunctionEval                       82
#define SNES_JacobianEval                       83
#define SNES_MinimizationFunctionEval           84
#define SNES_GradientEval                       85
#define SNES_HessianEval                        86

#define TS_Step                                 90

#define Petsc_Barrier                           100
/* 
   Event numbers PLOG_USER_EVENT_LOW to PLOG_USER_EVENT_HIGH are reserved 
   for applications.  Make sure that src/sys/src/plog.c defines enough
   entries in (*name)[] to go up to PLOG_USER_EVENT_HIGH.
*/
#define PLOG_USER_EVENT_LOW_STATIC              120
#define PLOG_USER_EVENT_HIGH                    200

/* Global flop counter */
extern double _TotalFlops;
#if defined(PETSC_LOG)
#define PLogFlops(n) {_TotalFlops += n;}
#else
#define PLogFlops(n)
#endif 


/*M
   PLogFlops - Adds floating point operations to the global counter.
               You must include "plog.h" to use this function.

   Input Parameter:
.  f - flop counter

   Synopsis:
   PLogFlops(int f)

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

    Example of Usage:
$     int USER_EVENT;
$     PLogEventRegister(&USER_EVENT,"User event","Color");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$     [code segment to monitor]
$     PLogFlops(user_flops)
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.seealso:  PLogEventRegister(), PLogEventBegin(), PLogEventEnd()

.keywords:  Petsc, log, flops, floating point operations
M*/

extern int PLogPrintSummary(MPI_Comm,FILE *);
extern int PLogBegin();
extern int PLogAllBegin();
extern int PLogDump(char*);

#if defined (HAVE_MPE)
#include "mpe.h"
extern int PLogMPEBegin();
extern int PLogMPEDump(char *);
extern int UseMPE,MPEFlags[];
#define MPEBEGIN    1000 
#endif

#if defined(PETSC_LOG)

extern int (*_PLB)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
extern int (*_PLE)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
extern int (*_PHC)(PetscObject);
extern int (*_PHD)(PetscObject);
extern int PLogEventRegister(int*,char*,char*);

/*M   
   PLogEventBegin - Logs the beginning of a user event. 

   Input Parameters:
.  e - integer associated with the event obtained from PLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0

   Synopsis:
   PLogEventBegin(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                  PetscObject o4)

   Notes:
   You should also register each integer event with the command 
   PLogRegisterEvent().  The source code must be compiled with 
   -DPETSC_LOG, which is the default.

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_LOG, and -log, -log_summary, or -log_all are
   specified.  PLogEventBegin() is intended for logging user events
   to supplement this PETSc information.

    Example of Usage:
$     int USER_EVENT;
$     int user_event_flops;
$     PLogEventRegister(&USER_EVENT,"User event","Color");
$     PLogEventBegin(&USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(&USER_EVENT,0,0,0,0);

.seealso: PLogEventRegister(), PLogEventEnd(), PLogFlops()

.keywords: log, event, begin
M*/
#if defined(HAVE_MPE)
#define PLogEventBegin(e,o1,o2,o3,o4) {static int _tacky = 0; \
  { _tacky++; \
   if (_PLB) \
     (*_PLB)(e,_tacky,(PetscObject)o1,(PetscObject)o2,(PetscObject)o3,(PetscObject)o4);\
   if (_tacky == 1 && UseMPE && MPEFlags[e])\
     MPE_Log_event(MPEBEGIN+2*e,0,"");\
  }
#else
#define PLogEventBegin(e,o1,o2,o3,o4) {static int _tacky = 0; \
  { _tacky++; \
   if (_PLB) \
     (*_PLB)(e,_tacky,(PetscObject)o1,(PetscObject)o2,(PetscObject)o3,(PetscObject)o4);\
  }
#endif

/*M   
   PLogEventEnd - Log the end of a user event.

   Input Parameters:
.  e - integer associated with the event obtained with PLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0

   Synopsis:
   PLogEventEnd(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                PetscObject o4)

   Notes:
   You should also register each integer event with the command 
   PLogRegisterEvent(). Source code must be compiled with 
   -DPETSC_LOG, which is the default.

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_LOG, and -log, -log_summary, or -log_all are
   specified.  PLogEventEnd() is intended for logging user events
   to supplement this PETSc information.

    Example of Usage:
$     int USER_EVENT;
$     int user_event_flops;
$     PLogEventRegister(&USER_EVENT,"User event","Color");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.seealso: PLogEventRegister(), PLogEventBegin(), PLogFlops()

.keywords: log, event, end
M*/
#if defined(HAVE_MPE)
#define PLogEventEnd(e,o1,o2,o3,o4) {\
  if (_PLE) \
    (*_PLE)(e,_tacky,(PetscObject)o1,(PetscObject)o2,(PetscObject)o3,(PetscObject)o4);\
  if (_tacky == 1 && UseMPE && MPEFlags[e])\
     MPE_Log_event(MPEBEGIN+2*e+1,0,"");\
  }  _tacky--;}
#else
#define PLogEventEnd(e,o1,o2,o3,o4) {\
  if (_PLE) \
    (*_PLE)(e,_tacky,(PetscObject)o1,(PetscObject)o2,(PetscObject)o3,(PetscObject)o4);\
  } _tacky--;}
#endif


#define PLogObjectParent(p,c)       {PETSCVALIDHEADER((PetscObject)c); \
                                     PETSCVALIDHEADER((PetscObject)p);\
                                     ((PetscObject)(c))->parent = (PetscObject) p;}
#define PLogObjectParents(p,n,d)    {int _i; for ( _i=0; _i<n; _i++ ) \
                                    PLogObjectParent(p,(d)[_i]);}
#define PLogObjectCreate(h)         {if (_PHC) (*_PHC)((PetscObject)h);}
#define PLogObjectDestroy(h)        {if (_PHD) (*_PHD)((PetscObject)h);}
#define PLogObjectMemory(p,m)       {PETSCVALIDHEADER((PetscObject)p);\
                                    ((PetscObject)(p))->mem += (m);}
extern int PLogObjectState(PetscObject,char *,...);
extern int PLogInfo(PetscObject,char*,...);
extern int PLogDestroy();
extern int PLogStagePush(int);
extern int PLogStagePop();
extern int PLogStageRegister(int,char*);

#else

#define PLogObjectCreate(h) 
#define PLogObjectDestroy(h)
#define PLogObjectMemory(p,m)
#define PLogEventBegin(e,o1,o2,o3,o4)
#define PLogEventEnd(e,o1,o2,o3,o4)
#define PLogObjectParent(p,c)
#define PLogObjectParents(p,n,c)
extern int PLogInfo(PetscObject,char*,...);
extern int PLogDestroy();
extern int PLogStagePush(int);
extern int PLogStagePop();
extern int PLogStageRegister(int,char*);
#endif

#endif

