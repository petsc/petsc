/* $Id: plog.h,v 1.21 1995/10/12 04:21:53 bsmith Exp curfman $ */

/*
    Defines high level logging in Petsc.
*/

#if !defined(__PLOG_PACKAGE)
#define __PLOG_PACKAGE
#include "petsc.h"  

/*
  If you add it here, make sure you add to petsc/bin/petscview.cfg
  and src/sys/src/plog.c!!
*/
#define MAT_Mult                                0
#define MAT_AssemblyBegin                       1
#define MAT_AssemblyEnd                         2
#define MAT_GetReordering                       3
#define MAT_MultTrans                           4
#define MAT_MultAdd                             5
#define MAT_MultTransAdd                        6
#define MAT_LUFactor                            7
#define MAT_CholeskyFactor                      8
#define MAT_LUFactorSymbolic                    9
#define MAT_ILUFactorSymbolic                   10
#define MAT_CholeskyFactorSymbolic              11
#define MAT_IncompleteCholeskyFactorSymbolic    12
#define MAT_LUFactorNumeric                     13
#define MAT_CholeskyFactorNumeric               14
#define MAT_Relax                               15
#define MAT_Copy                                16
#define MAT_Convert                             17
#define MAT_Scale                               18
#define MAT_ZeroEntries                         19
#define MAT_Solve                               20
#define MAT_SolveAdd                            21
#define MAT_SolveTrans                          22
#define MAT_SolveTransAdd                       23
#define MAT_SetValues                           24
#define MAT_ForwardSolve                        25
#define MAT_BackwardSolve                       26
#define MAT_Load                                27
#define MAT_View                                28
#define MAT_ILUFactor                           29

#define VEC_Dot                                 30
#define VEC_Norm                                31
#define VEC_ASum                                32
#define VEC_AMax                                33
#define VEC_Max                                 34
#define VEC_Min                                 35
#define VEC_TDot                                36
#define VEC_Scale                               37
#define VEC_Copy                                38
#define VEC_Set                                 39
#define VEC_AXPY                                40
#define VEC_AYPX                                41
#define VEC_Swap                                42
#define VEC_WAXPY                               43
#define VEC_AssemblyBegin                       44
#define VEC_AssemblyEnd                         45
#define VEC_MTDot                               46
#define VEC_MDot                                47
#define VEC_MAXPY                               48
#define VEC_PMult                               49
#define VEC_SetValues                           50
#define VEC_Load                                51
#define VEC_View                                52

#define SLES_Solve                              55
#define PC_SetUp                                56
#define PC_Apply                                57
#define SLES_SetUp                              58

#define SNES_Solve                              60
#define SNES_LineSearch                         61
#define SNES_FunctionEval                       62
#define SNES_JacobianEval                       63
#define SNES_MinimizationFunctionEval           64
#define SNES_GradientEval                       65
#define SNES_HessianEval                        66

#define MAT_GetSubMatrix                        70
#define KSP_GMRESOrthogonalization              71

/* event numbers 80 to 99 are reserved for applications */

/* Global flop counter */
extern double _TotalFlops;
#if defined(PETSC_LOG)
#define PLogFlops(n) {_TotalFlops += n;}
#else
#define PLogFlops(n)
#endif 

/*M
   PLogFlops - Adds floating point operations to the global counter.

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
$     #define USER_EVENT 85
$     PLogEventRegister(USER_EVENT,"User event");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$     [code segment to monitor]
$     PLogFlops(user_flops)
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.seealso:  PLogEventRegister(), PLogEventBegin(), PLogEventEnd()

.keywords:  Petsc, log, flops, floating point operations
M*/

extern int PLogPrint(MPI_Comm,FILE *);
extern int PLogBegin();
extern int PLogAllBegin();
extern int PLogDump(char*);

#if defined(PETSC_LOG)

extern int (*_PLB)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
extern int (*_PLE)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
extern int (*_PHC)(PetscObject);
extern int (*_PHD)(PetscObject);
extern int PLogEventRegister(int,char*);

/*M   
   PLogEventBegin - Logs the beginning of a user event. 

   Input Parameters:
.  e - integer associated with the event (80 < e < 99) 
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
$     #define USER_EVENT 85
$     int user_event_flops;
$     PLogEventRegister(USER_EVENT,"User event");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.seealso: PLogEventRegister(), PLogEventEnd(), PLogFlops()

.keywords: log, event, begin
M*/
#define PLogEventBegin(e,o1,o2,o3,o4) {static int _tacky = 0;\
          { _tacky++;if (_PLB) (*_PLB)(e,_tacky,(PetscObject)o1,\
           (PetscObject)o2,(PetscObject)o3,(PetscObject)o4);};

/*M   
   PLogEventEnd - Log the end of a user event.

   Input Parameters:
.  e - integer associated with the event (80 < e < 99) 
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
$     #define USER_EVENT 85
$     int user_event_flops;
$     PLogEventRegister(USER_EVENT,"User event");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.seealso: PLogEventRegister(), PLogEventBegin(), PLogFlops()

.keywords: log, event, end
M*/
#define PLogEventEnd(e,o1,o2,o3,o4) {if (_PLE) (*_PLE)(e,_tacky,(PetscObject)o1,\
                                    (PetscObject)o2,(PetscObject)o3,(PetscObject)o4);}\
                                    _tacky--;}
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

#else

#define PLogObjectCreate(h) 
#define PLogObjectDestroy(h)
#define PLogObjectMemory(p,m)
#define PLogEventBegin(e,o1,o2,o3,o4)
#define PLogEventEnd(e,o1,o2,o3,o4)
#define PLogObjectParent(p,c)
#define PLogObjectParents(p,n,c)
extern int PLogInfo(PetscObject,char*,...);
#endif

#endif
