/* $Id: petsclog.h,v 1.119 1998/03/31 23:43:11 balay Exp balay $ */

/*
    Defines profile/logging in PETSc.
*/

#if !defined(__PLOG_PACKAGE)
#define __PLOG_PACKAGE
#include "petsc.h"  

/*
  Lists all PETSc events that are logged/profiled.

  If you add an event here, make sure you add it to 
  petsc/bin/petscview.cfg,
  petsc/bin/petscview,
  petsc/src/plog/src/plog.c,  
  petsc/src/plog/src/plogmpe.c, and
  petsc/include/finclude/petsclog.h!!!
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
#define MAT_GetColoring                         31
#define MAT_GetSubMatrices                      32
#define MAT_GetValues                           33
#define MAT_IncreaseOverlap                     34
#define MAT_GetRow                              35
#define MAT_Partitioning                        36

#define VEC_ScatterBarrier                      39
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

#define VEC_NormBarrier                         64
#define VEC_NormComm                            65
#define VEC_DotBarrier                          66
#define VEC_DotComm                             67
#define VEC_MDotBarrier                         68
#define VEC_MDotComm                            69

#define SLES_Solve                              70
#define SLES_SetUp                              71

#define KSP_GMRESOrthogonalization              72

#define PC_ModifySubMatrices                    74
#define PC_SetUp                                75
#define PC_SetUpOnBlocks                        76
#define PC_Apply                                77
#define PC_ApplySymmetricLeft                   78
#define PC_ApplySymmetricRight                  79

#define SNES_Solve                              80
#define SNES_LineSearch                         81
#define SNES_FunctionEval                       82
#define SNES_JacobianEval                       83
#define SNES_MinimizationFunctionEval           84
#define SNES_GradientEval                       85
#define SNES_HessianEval                        86

#define TS_Step                                 90
#define TS_PseudoComputeTimeStep                91

#define Petsc_Barrier                           100

#define EC_SetUp                                105
#define EC_Solve                                106

#define DFVec_RefineVector                      110
#define DFVec_AssembleFullVector                111
#define DFVec_GetComponentVectors               112
#define DFVec_DrawContours                      113

/* 
   Event numbers PLOG_USER_EVENT_LOW to PLOG_USER_EVENT_HIGH are reserved 
   for applications.  Make sure that src/plog/src/plog.c defines enough
   entries in (*name)[] to go up to PLOG_USER_EVENT_HIGH.
*/
#define PLOG_USER_EVENT_LOW_STATIC              120
#define PLOG_USER_EVENT_HIGH                    200

/* Global flop counter */
extern PLogDouble _TotalFlops;

/* General logging of information; different from event logging */
extern int PLogInfo(void*,char*,...);
extern int PLogInfoDeactivateClass(int);
extern int PLogInfoActivateClass(int);
extern int PLogPrintInfo;  /* if 1, indicates PLogInfo() is turned on */

#if defined(USE_PETSC_LOG)  /* --- Logging is turned on --------------------------------*/

/* 
   Flop counting:  We count each arithmetic operation (e.g., addition, multiplication) separately.

   For the complex numbers version, note that
       1 complex addition = 2 flops
       1 complex multiplication = 6 flops,
   where we define 1 flop as that for a double precision scalar.  We roughly approximate
   flop counting for complex numbers by multiplying the total flops by 4; this corresponds
   to the assumption that we're counting mostly additions and multiplications -- and
   roughly the same number of each.  More accurate counting could be done by distinguishing
   among the various arithmetic operations.
 */

#if defined(USE_PETSC_COMPLEX)
#define PLogFlops(n) {_TotalFlops += (4*n);}
#else
#define PLogFlops(n) {_TotalFlops += (n);}
#endif

#if defined (HAVE_MPE)
#include "mpe.h"
#define MPEBEGIN    1000 
extern int PLogMPEBegin(void);
extern int PLogMPEDump(char *);
extern int UseMPE,PLogEventMPEFlags[];
extern int PLogEventMPEActivate(int);
extern int PLogEventMPEDeactivate(int);
#else
#define PLogEventMPEActivate(a) 0
#define PLogEventMPEDeactivate(a) 0
#endif

extern int PLogEventActivate(int);
extern int PLogEventDeactivate(int);

extern int PLogEventActivateClass(int);
extern int PLogEventDeactivateClass(int);

extern int PLogEventFlags[];
extern int (*_PLogPLB)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
extern int (*_PLogPLE)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
extern int (*_PLogPHC)(PetscObject);
extern int (*_PLogPHD)(PetscObject);

#if defined(HAVE_MPE)
#define PLogEventBarrierBegin(e,o1,o2,o3,o4,cm) \
  { \
    if (_PLogPLB && PLogEventFlags[e]) {                           \
      PLogEventBegin((e),o1,o2,o3,o4);                                   \
      if (UseMPE && PLogEventMPEFlags[(e)])\
        MPE_Log_event(MPEBEGIN+2*(e),0,"");\
      MPI_Barrier(cm);                                             \
      PLogEventEnd((e),o1,o2,o3,o4);                                     \
      if (UseMPE && PLogEventMPEFlags[(e)])\
        MPE_Log_event(MPEBEGIN+2*((e)+1),0,"");\
    }                                                                \
    PLogEventBegin(e+1,o1,o2,o3,o4);                                   \
    if (UseMPE && PLogEventMPEFlags[(e)+1])\
      MPE_Log_event(MPEBEGIN+2*((e)+1),0,"");\
  }
#define PLogEventBegin(e,o1,o2,o3,o4)  \
  {  \
   if (_PLogPLB && PLogEventFlags[(e)]) \
     (*_PLogPLB)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));\
   if (UseMPE && PLogEventMPEFlags[(e)])\
     MPE_Log_event(MPEBEGIN+2*(e),0,"");\
  }
#else
#define PLogEventBarrierBegin(e,o1,o2,o3,o4,cm) \
  { \
    if (_PLogPLB && PLogEventFlags[(e)]) {                           \
      PLogEventBegin((e),o1,o2,o3,o4);                                   \
      MPI_Barrier(cm);                                             \
      PLogEventEnd((e),o1,o2,o3,o4);                                     \
    }                                                                \
    PLogEventBegin((e)+1,o1,o2,o3,o4);                                   \
  }
#define PLogEventBegin(e,o1,o2,o3,o4)  \
  {  \
   if (_PLogPLB && PLogEventFlags[(e)]) \
     (*_PLogPLB)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));\
  }
#endif

#if defined(HAVE_MPE)
#define PLogEventBarrierEnd(e,o1,o2,o3,o4,cm) {\
  if (_PLogPLE && PLogEventFlags[(e)+1]) \
    (*_PLogPLE)((e)+1,0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));\
  if (UseMPE && PLogEventMPEFlags[(e)+1])\
     MPE_Log_event(MPEBEGIN+2*((e)+1)+1,0,"");\
  }  
#define PLogEventEnd(e,o1,o2,o3,o4) {\
  if (_PLogPLE && PLogEventFlags[(e)]) \
    (*_PLogPLE)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));\
  if (UseMPE && PLogEventMPEFlags[(e)])\
     MPE_Log_event(MPEBEGIN+2*(e)+1,0,"");\
  }  
#else
#define PLogEventBarrierEnd(e,o1,o2,o3,o4,cm) {\
  if (_PLogPLE && PLogEventFlags[(e)+1]) \
    (*_PLogPLE)((e)+1,0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));\
  } 
#define PLogEventEnd(e,o1,o2,o3,o4) {\
  if (_PLogPLE && PLogEventFlags[(e)]) \
    (*_PLogPLE)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));\
  } 
#endif


#define PLogObjectParent(p,c)       if (c) {PetscValidHeader((PetscObject)(c)); \
                                     PetscValidHeader((PetscObject)(p));\
                                     ((PetscObject)(c))->parent = (PetscObject) (p);}
#define PLogObjectParents(p,n,d)    {int _i; for ( _i=0; _i<n; _i++ ) \
                                    PLogObjectParent(p,(d)[_i]);}
#define PLogObjectCreate(h)         {if (_PLogPHC) (*_PLogPHC)((PetscObject)h);}
#define PLogObjectDestroy(h)        {if (_PLogPHD) (*_PLogPHD)((PetscObject)h);}
#define PLogObjectMemory(p,m)       {PetscValidHeader((PetscObject)p);\
                                    ((PetscObject)(p))->mem += (m);}
extern int  PLogObjectState(PetscObject,char *,...);
extern int  PLogDestroy(void);
extern int  PLogStagePush(int);
extern int  PLogStagePop(void);
extern int  PLogStageRegister(int,char*);
extern int  PLogPrintSummary(MPI_Comm,char *);
extern int  PLogBegin(void);
extern int  PLogTraceBegin(FILE *);
extern int  PLogAllBegin(void);
extern int  PLogSet(int (*)(int,int,PetscObject,PetscObject,PetscObject,PetscObject),
                    int (*)(int,int,PetscObject,PetscObject,PetscObject,PetscObject));
extern int  PLogDump(char*);
extern int  PLogEventRegister(int*,char*,char*);
extern int  PetscGetFlops(PLogDouble*);

extern PLogDouble irecv_ct, isend_ct, wait_ct, wait_any_ct, recv_ct, send_ct;
extern PLogDouble irecv_len, isend_len, recv_len, send_len;
extern PLogDouble wait_all_ct,allreduce_ct,sum_of_waits_ct;
extern int        PETSC_DUMMY,PETSC_DUMMY_SIZE;

/*
     This does not work for MPI-Uni because our src/mpiuni/mpi.h file
   uses macros to defined the MPI operations. 

     It does not work correctly from HP-UX because it processes the 
   macros in a way that sometimes it double counts.

     It does not work with Windows NT because winmpich lacks MPI_Type_size()
*/
#if !defined(PETSC_USING_MPIUNI) && !defined(PARCH_hpux) && !defined (PARCH_nt)
/*
   Logging of MPI activities
*/

#define TypeSize(buff,count,type)                                                \
(                                                                                \
  MPI_Type_size(type,&PETSC_DUMMY_SIZE), buff += ((PLogDouble) ((count)*PETSC_DUMMY_SIZE)) \
)

#define MPI_Irecv( buf, count,  datatype, source, tag, comm, request)        \
(                                                                            \
  PETSC_DUMMY = MPI_Irecv( buf, count,  datatype, source, tag, comm, request),             \
  irecv_ct++, TypeSize(irecv_len,count,datatype),PETSC_DUMMY                            \
)

#define MPI_Isend( buf, count,  datatype, dest, tag, comm, request)          \
(                                                                            \
  PETSC_DUMMY = MPI_Isend( buf, count,  datatype, dest, tag, comm, request),               \
  isend_ct++,   TypeSize(isend_len,count,datatype),PETSC_DUMMY                          \
)

#define MPI_Startall_irecv( count,number,requests)                                     \
(                                                                                      \
  PETSC_DUMMY = MPI_Startall( number, requests),                                                     \
  irecv_ct += (PLogDouble)(number), irecv_len += ((PLogDouble) (count*sizeof(Scalar))),PETSC_DUMMY \
)

#define MPI_Startall_isend( count,number,requests)                                    \
(                                                                                     \
  PETSC_DUMMY = MPI_Startall( number, requests),                                                    \
  isend_ct += (PLogDouble)(number), isend_len += ((PLogDouble) (count*sizeof(Scalar))),PETSC_DUMMY \
)

#define MPI_Start_isend(count,  requests)\
(\
  PETSC_DUMMY = MPI_Start( requests),\
  isend_ct++, isend_len += ((PLogDouble) (count*sizeof(Scalar))),PETSC_DUMMY\
)

#define MPI_Recv( buf, count,  datatype, source, tag, comm, status)           \
(                                                                            \
  PETSC_DUMMY = MPI_Recv( buf, count,  datatype, source, tag, comm, status),                \
  recv_ct++, TypeSize(recv_len,count,datatype), PETSC_DUMMY                              \
)

#define MPI_Send( buf, count,  datatype, dest, tag, comm)                     \
(                                                                             \
  PETSC_DUMMY = MPI_Send( buf, count,  datatype, dest, tag, comm),                          \
  send_ct++,  TypeSize(send_len,count,datatype),PETSC_DUMMY                              \
)

#define MPI_Wait(request, status) \
(                                 \
  wait_ct++, sum_of_waits_ct++,   \
  MPI_Wait(request, status)       \
)

#define MPI_Waitany(a, b, c, d)     \
(                                   \
  wait_any_ct++, sum_of_waits_ct++, \
  MPI_Waitany(a, b, c, d)           \
)

#define MPI_Waitall(count, array_of_requests, array_of_statuses) \
(                                                                \
  wait_all_ct++, sum_of_waits_ct += (PLogDouble) (count),        \
  MPI_Waitall(count, array_of_requests, array_of_statuses)       \
)

#define MPI_Allreduce( sendbuf,  recvbuf, count, datatype, op, comm) \
    (allreduce_ct++,MPI_Allreduce( sendbuf,  recvbuf, count, datatype, op, comm))

#else

#define MPI_Startall_irecv( count,number,requests) \
(                                                  \
  MPI_Startall( number, requests)                 \
)

#define MPI_Startall_isend( count,number,requests) \
(                                                  \
  MPI_Startall( number, requests)                 \
)

#define MPI_Start_isend(count,  requests) \
(                                         \
  MPI_Start( requests)                   \
)

#endif /* ! PETSC_USING_MPIUNI && ! PARCH_hpux */

#else  /* ---Logging is turned off --------------------------------------------*/

#define PLogFlops(n)

/*
     With logging turned off, then MPE has to be turned off
*/
#define MPEBEGIN                  1000 
#define PLogMPEBegin()  
#define PLogMPEDump(a) 
#define PLogEventMPEActivate(a)   0
#define PLogEventMPEDeactivate(a) 0

#define PLogEventActivate(a)   0
#define PLogEventDeactivate(a) 0

#define PLogEventActivateClass(a)   0
#define PLogEventDeactivateClass(a) 0

#define _PLogPLB                        0
#define _PLogPLE                        0
#define _PLogPHC                        0
#define _PLogPHD                        0
#define PetscGetFlops(a)                (*(a) = 0.0,0)
#define PLogEventBegin(e,o1,o2,o3,o4)
#define PLogEventEnd(e,o1,o2,o3,o4)
#define PLogEventBarrierBegin(e,o1,o2,o3,o4,cm)
#define PLogEventBarrierEnd(e,o1,o2,o3,o4,cm)
#define PLogObjectParent(p,c)
#define PLogObjectParents(p,n,c)
#define PLogObjectCreate(h)
#define PLogObjectDestroy(h)
#define PLogObjectMemory(p,m)
#define PLogDestroy()
#define PLogStagePush(a)
#define PLogStagePop()
#define PLogStageRegister(a,b)
#define PLogPrintSummary(comm,file)
#define PLogBegin()
#define PLogTraceBegin(file)            0
#define PLogSet(lb,le)
#define PLogAllBegin()
#define PLogDump(char)              
#define PLogEventRegister(a,b,c)        0
#define PLogMPEBegin()
#define PLogMPEDump(a)
extern int PLogObjectState(PetscObject,char *,...);

/* If USE_PETSC_LOG is NOT defined, these still need to be! */
#define MPI_Startall_irecv( count,number,requests) MPI_Startall( number, requests)

#define MPI_Startall_isend( count,number,requests) MPI_Startall( number, requests)

#define MPI_Start_isend(count,  requests) MPI_Start( requests)

#endif   /* USE_PETSC_LOG */

/*MC
   PLogFlops - Adds floating point operations to the global counter.

   Input Parameter:
.  f - flop counter

   Synopsis:
   void PLogFlops(int f)

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DUSE_PETSC_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

    Example of Usage:
$     int USER_EVENT;
$     PLogEventRegister(&USER_EVENT,"User event","Color:");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_flops)
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.seealso: PLogEventRegister(), PLogEventBegin(), PLogEventEnd(), PetscGetFlops()

.keywords: log, flops, floating point operations
M*/


/*MC
   PLogEventBegin - Logs the beginning of a user event. 

   Input Parameters:
.  e - integer associated with the event obtained from PLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0

   Synopsis:
   void PLogEventBegin(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                  PetscObject o4)

   Notes:
   You should also register each integer event with the command 
   PLogRegisterEvent().  The source code must be compiled with 
   -DUSE_PETSC_LOG, which is the default.

   PETSc automatically logs library events if the code has been
   compiled with -DUSE_PETSC_LOG, and -log, -log_summary, or -log_all are
   specified.  PLogEventBegin() is intended for logging user events
   to supplement this PETSc information.

    Example of Usage:
$     int USER_EVENT;
$     int user_event_flops;
$     PLogEventRegister(&USER_EVENT,"User event","Color:");
$     PLogEventBegin(&USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(&USER_EVENT,0,0,0,0);

.seealso: PLogEventRegister(), PLogEventEnd(), PLogFlops()

.keywords: log, event, begin
M*/

/*MC
   PLogEventEnd - Log the end of a user event.

   Input Parameters:
.  e - integer associated with the event obtained with PLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0

   Synopsis:
   void PLogEventEnd(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                PetscObject o4)

   Notes:
   You should also register each additional integer event with the command 
   PLogRegisterEvent(). Source code must be compiled with 
   -DUSE_PETSC_LOG, which is the default.

   PETSc automatically logs library events if the code has been
   compiled with -DUSE_PETSC_LOG, and -log, -log_summary, or -log_all are
   specified.  PLogEventEnd() is intended for logging user events
   to supplement this PETSc information.

    Example of Usage:
$     int USER_EVENT;
$     int user_event_flops;
$     PLogEventRegister(&USER_EVENT,"User event","Color:");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$        [code segment to monitor]
$        PLogFlops(user_event_flops);
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.seealso: PLogEventRegister(), PLogEventBegin(), PLogFlops()

.keywords: log, event, end
M*/

/*MC
   PLogEventBarrierBegin - Logs the time in a barrier before an event.

   Input Parameters:
.  e - integer associated with the event obtained from PLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0
.  comm - communicator the barrier takes place over

   Synopsis:
   void PLogEventBarrierBegin(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                  PetscObject o4,MPI_Comm comm)

   Notes:
   This is for logging the amount of time spent in a barrier for an event
   that requires synchronization. 

    Example of Usage:
$     PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
$       MPI_Allreduce()
$     PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);

   Additional Notes:
   Synchronization events always come in pairs; for example, VEC_NormBarrier and 
   VEC_NormComm = VEC_NormBarrier + 1

.seealso: PLogEventRegister(), PLogEventEnd(), PLogFlops(), PLogEventBegin(),
          PLogEventBarrierEnd()

.keywords: log, event, begin, barrier
M*/

/*MC
   PLogEventBarrierEnd - Logs the time in a barrier before an event.

   Input Parameters:
.  e - integer associated with the event obtained from PLogEventRegister()
.  o1,o2,o3,o4 - objects associated with the event, or 0
.  comm - communicator the barrier takes place over

   Synopsis:
   void PLogEventBarrierEnd(int e,PetscObject o1,PetscObject o2,PetscObject o3,
                  PetscObject o4,MPI_Comm comm)

   Notes:
   This is for logging the amount of time spent in a barrier for an event
   that requires synchronization. 

    Example of Usage:
$     PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
$       MPI_Allreduce()
$     PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);

   Additional Notes:
   Synchronization events always come in pairs; for example, VEC_NormBarrier and 
   VEC_NormComm = VEC_NormBarrier + 1

.seealso: PLogEventRegister(), PLogEventEnd(), PLogFlops(), PLogEventBegin(),
          PLogEventBarrierBegin()

.keywords: log, event, begin, barrier
M*/

#endif






