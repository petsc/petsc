/* $Id: petsclog.h,v 1.155 2001/09/06 14:51:20 bsmith Exp $ */

/*
    Defines profile/logging in PETSc.
*/

#if !defined(__PetscLog_H)
#define __PetscLog_H
#include "petsc.h"  

/*
  Lists all PETSc events that are logged/profiled.

  If you add an event here, make sure you add it to 
  petsc/src/PetscLog/src/PetscLog.c,  
  petsc/src/PetscLog/src/PetscLogmpe.c, and
  petsc/include/finclude/petsclog.h!!!
*/
#define MAT_Mult                                0
#define MAT_MatrixFreeMult                      1
#define MAT_AssemblyBegin                       2
#define MAT_AssemblyEnd                         3
#define MAT_GetOrdering                         4
#define MAT_MultTranspose                       5
#define MAT_MultAdd                             6
#define MAT_MultTransposeAdd                    7
#define MAT_LUFactor                            8
#define MAT_CholeskyFactor                      9
#define MAT_LUFactorSymbolic                    10
#define MAT_ILUFactorSymbolic                   11
#define MAT_CholeskyFactorSymbolic              12
#define MAT_ICCFactorSymbolic                   13
#define MAT_LUFactorNumeric                     14
#define MAT_CholeskyFactorNumeric               15
#define MAT_Relax                               16
#define MAT_Copy                                17
#define MAT_Convert                             18
#define MAT_Scale                               19
#define MAT_ZeroEntries                         20
#define MAT_Solve                               21
#define MAT_SolveAdd                            22
#define MAT_SolveTranspose                      23
#define MAT_SolveTransposeAdd                   24
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

#define MAT_FDColoringApply                     38
#define MAT_FDColoringCreate                    41

#define VEC_ReduceArithmetic                    37

#define VEC_View                                39

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
#define VEC_MAXPY                               56
#define VEC_PMult                               57
#define VEC_SetValues                           58
#define VEC_Load                                59
#define VEC_ScatterBarrier                      60
#define VEC_ScatterBegin                        61
#define VEC_ScatterEnd                          62
#define VEC_SetRandom                           63

#define VEC_NormBarrier                         64
#define VEC_Norm                                65
#define VEC_DotBarrier                          66
#define VEC_Dot                                 67
#define VEC_MDotBarrier                         68
#define VEC_MDot                                69

#define SLES_Solve                              70
#define SLES_SetUp                              71

#define KSP_GMRESOrthogonalization              72

#define PC_ApplyCoarse                          73
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

#define VEC_ReduceBarrier                       87
#define VEC_ReduceComm                          88

#define TS_Step                                 90
#define TS_PseudoComputeTimeStep                91
#define TS_FunctionEval                         92
#define TS_JacobianEval                         93

#define Petsc_Barrier                           100

#define EC_SetUp                                105
#define EC_Solve                                106

/* 
   Event numbers PETSC_LOG_USER_EVENT_LOW to PETSC_LOG_USER_EVENT_HIGH are reserved 
   for applications.  Make sure that src/PetscLog/src/PetscLog.c defines enough
   entries in (*name)[] to go up to PETSC_LOG_USER_EVENT_HIGH.
*/
#define PETSC_LOG_USER_EVENT_LOW_STATIC              120
#define PETSC_LOG_USER_EVENT_HIGH                    200

/* Global flop counter */
extern PetscLogDouble _TotalFlops;

/* General logging of information; different from event logging */
EXTERN int        PetscLogInfo(void*,const char[],...);
EXTERN int        PetscLogInfoDeactivateClass(int);
EXTERN int        PetscLogInfoActivateClass(int);
extern PetscTruth PetscLogPrintInfo;  /* if true, indicates PetscLogInfo() is turned on */

#if defined(PETSC_USE_LOG)  /* --- Logging is turned on --------------------------------*/

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

#if defined(PETSC_USE_COMPLEX)
#define PetscLogFlops(n) (_TotalFlops += (4*n),0)
#else
#define PetscLogFlops(n) (_TotalFlops += (n),0)
#endif

#if defined (PETSC_HAVE_MPE)
#include "mpe.h"
#define MPEBEGIN    1000 
EXTERN int        PetscLogMPEBegin(void);
EXTERN int        PetscLogMPEDump(const char[]);
extern PetscTruth UseMPE;
extern int        PetscLogEventMPEFlags[];
EXTERN int        PetscLogEventMPEActivate(int);
EXTERN int        PetscLogEventMPEDeactivate(int);
#else
#define PetscLogEventMPEActivate(a) 0
#define PetscLogEventMPEDeactivate(a) 0
#endif

EXTERN int PetscLogEventActivate(int);
EXTERN int PetscLogEventDeactivate(int);

EXTERN int PetscLogEventActivateClass(int);
EXTERN int PetscLogEventDeactivateClass(int);

extern PetscTruth PetscLogEventFlags[];
EXTERN int (*_PetscLogPLB)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
EXTERN int (*_PetscLogPLE)(int,int,PetscObject,PetscObject,PetscObject,PetscObject);
EXTERN int (*_PetscLogPHC)(PetscObject);
EXTERN int (*_PetscLogPHD)(PetscObject);

extern int PetscLogEventDepth[];

#if defined(PETSC_HAVE_MPE)
#define PetscLogEventBarrierBegin(e,o1,o2,o3,o4,cm) \
  0; { int _1_ierr; \
    if (_PetscLogPLB && PetscLogEventFlags[e]) {                         \
      _1_ierr = PetscLogEventBegin((e),o1,o2,o3,o4);CHKERRQ(_1_ierr);  \
      if (UseMPE && PetscLogEventMPEFlags[(e)])                      \
        MPE_Log_event(MPEBEGIN+2*(e),0,"");                      \
      _1_ierr = MPI_Barrier(cm);CHKERRQ(_1_ierr);                  \
      _1_ierr = PetscLogEventEnd((e),o1,o2,o3,o4);CHKERRQ(_1_ierr);    \
      if (UseMPE && PetscLogEventMPEFlags[(e)])                      \
        MPE_Log_event(MPEBEGIN+2*((e)+1),0,"");                  \
    }                                                            \
    _1_ierr = PetscLogEventBegin(e+1,o1,o2,o3,o4);CHKERRQ(_1_ierr);    \
    if (UseMPE && PetscLogEventMPEFlags[(e)+1])                      \
      MPE_Log_event(MPEBEGIN+2*((e)+1),0,"");                    \
  }
#define PetscLogEventBegin(e,o1,o2,o3,o4)  \
  0; {  \
   if (_PetscLogPLB && PetscLogEventFlags[(e)] && !PetscLogEventDepth[e]++) {\
     (*_PetscLogPLB)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));}\
   if (UseMPE && PetscLogEventMPEFlags[(e)])\
     MPE_Log_event(MPEBEGIN+2*(e),0,"");\
  }
#else
#define PetscLogEventBarrierBegin(e,o1,o2,o3,o4,cm) \
  0; { int _2_ierr;\
    if (_PetscLogPLB && PetscLogEventFlags[(e)]) {                         \
      _2_ierr = PetscLogEventBegin((e),o1,o2,o3,o4);CHKERRQ(_2_ierr);    \
      _2_ierr = MPI_Barrier(cm);CHKERRQ(_2_ierr);                    \
      _2_ierr = PetscLogEventEnd((e),o1,o2,o3,o4);CHKERRQ(_2_ierr);      \
    }                                                              \
    _2_ierr = PetscLogEventBegin((e)+1,o1,o2,o3,o4);CHKERRQ(_2_ierr);    \
  }
#define PetscLogEventBegin(e,o1,o2,o3,o4)  \
  0; {  \
   if (_PetscLogPLB && PetscLogEventFlags[(e)] && !PetscLogEventDepth[e]++) {\
     (*_PetscLogPLB)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));}\
  }
#endif

#if defined(PETSC_HAVE_MPE)
#define PetscLogEventBarrierEnd(e,o1,o2,o3,o4,cm) PetscLogEventEnd(e+1,o1,o2,o3,o4)
#define PetscLogEventEnd(e,o1,o2,o3,o4) \
  0; {\
  if (_PetscLogPLE && PetscLogEventFlags[(e)] && !--PetscLogEventDepth[e]) {\
    (*_PetscLogPLE)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));}\
  if (UseMPE && PetscLogEventMPEFlags[(e)])\
     MPE_Log_event(MPEBEGIN+2*(e)+1,0,"");\
  }  
#else
#define PetscLogEventBarrierEnd(e,o1,o2,o3,o4,cm) PetscLogEventEnd(e+1,o1,o2,o3,o4)
#define PetscLogEventEnd(e,o1,o2,o3,o4) \
  0; {\
  if (_PetscLogPLE && PetscLogEventFlags[(e)] && !--PetscLogEventDepth[e]) {\
    (*_PetscLogPLE)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));}\
  } 
#endif

#define PetscLogObjectParent(p,c)       if (c) {PetscValidHeader((PetscObject)(c)); \
                                     PetscValidHeader((PetscObject)(p));\
                                     ((PetscObject)(c))->parent = (PetscObject)(p);\
				     ((PetscObject)(c))->parentid = ((PetscObject)p)->id;}
#define PetscLogObjectParents(p,n,d)    {int _i; for (_i=0; _i<n; _i++) \
                                    PetscLogObjectParent(p,(d)[_i]);}
#define PetscLogObjectCreate(h)         {if (_PetscLogPHC) (*_PetscLogPHC)((PetscObject)h);}
#define PetscLogObjectDestroy(h)        {if (_PetscLogPHD) (*_PetscLogPHD)((PetscObject)h);}
#define PetscLogObjectMemory(p,m)       {PetscValidHeader((PetscObject)p);\
                                    ((PetscObject)(p))->mem += (m);}
EXTERN int  PetscLogObjectState(PetscObject,const char[],...);
EXTERN int  PetscLogDestroy(void);
EXTERN int  PetscLogStagePush(int);
EXTERN int  PetscLogStagePop(void);
EXTERN int  PetscLogStageRegister(int,const char[]);
EXTERN int  PetscLogStagePrint(int,PetscTruth);
EXTERN int  PetscLogPrintSummary(MPI_Comm,const char[]);
EXTERN int  PetscLogBegin(void);
EXTERN int  PetscLogTraceBegin(FILE *);
EXTERN int  PetscLogAllBegin(void);
EXTERN int  PetscLogSet(int (*)(int,int,PetscObject,PetscObject,PetscObject,PetscObject),
                    int (*)(int,int,PetscObject,PetscObject,PetscObject,PetscObject));
EXTERN int  PetscLogDump(const char[]);
EXTERN int  PetscLogEventRegister(int*,const char[],const char[]);
EXTERN int  PetscGetFlops(PetscLogDouble*);

extern PetscLogDouble irecv_ct,isend_ct,wait_ct,wait_any_ct,recv_ct,send_ct;
extern PetscLogDouble irecv_len,isend_len,recv_len,send_len;
extern PetscLogDouble wait_all_ct,allreduce_ct,sum_of_waits_ct;
extern int            PETSC_DUMMY,PETSC_DUMMY_SIZE;

/*
     This does not work for MPI-Uni because our src/mpiuni/mpi.h file
   uses macros to defined the MPI operations. 

     It does not work correctly from HP-UX because it processes the 
   macros in a way that sometimes it double counts, hence 
   PETSC_HAVE_BROKEN_RECURSIVE_MACRO

     It does not work with Windows NT because winmpich lacks MPI_Type_size()
*/
#if !defined(USING_MPIUNI) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO) && !defined (PETSC_HAVE_MPI_MISSING_TYPESIZE)
/*
   Logging of MPI activities
*/

#define TypeSize(buff,count,type)                                                \
(\
  MPI_Type_size(type,&PETSC_DUMMY_SIZE),buff += ((PetscLogDouble) ((count)*PETSC_DUMMY_SIZE)) \
)

#define MPI_Irecv(buf,count, datatype,source,tag,comm,request)        \
(\
  PETSC_DUMMY = MPI_Irecv(buf,count, datatype,source,tag,comm,request),            \
  irecv_ct++,TypeSize(irecv_len,count,datatype),PETSC_DUMMY                            \
)

#define MPI_Isend(buf,count, datatype,dest,tag,comm,request)          \
(\
  PETSC_DUMMY = MPI_Isend(buf,count, datatype,dest,tag,comm,request),              \
  isend_ct++,  TypeSize(isend_len,count,datatype),PETSC_DUMMY                          \
)

#define MPI_Startall_irecv(count,number,requests)                                     \
(\
  PETSC_DUMMY = MPI_Startall(number,requests),                                                    \
  irecv_ct += (PetscLogDouble)(number),irecv_len += ((PetscLogDouble) ((count)*sizeof(PetscScalar))),PETSC_DUMMY \
)

#define MPI_Startall_isend(count,number,requests)                                    \
(\
  PETSC_DUMMY = MPI_Startall(number,requests),                                                   \
  isend_ct += (PetscLogDouble)(number),isend_len += ((PetscLogDouble) ((count)*sizeof(PetscScalar))),PETSC_DUMMY \
)

#define MPI_Start_isend(count, requests)\
(\
  PETSC_DUMMY = MPI_Start(requests),\
  isend_ct++,isend_len += ((PetscLogDouble) ((count)*sizeof(PetscScalar))),PETSC_DUMMY\
)

#define MPI_Recv(buf,count, datatype,source,tag,comm,status)           \
(\
  PETSC_DUMMY = MPI_Recv(buf,count, datatype,source,tag,comm,status),               \
  recv_ct++,TypeSize(recv_len,count,datatype),PETSC_DUMMY                              \
)

#define MPI_Send(buf,count, datatype,dest,tag,comm)                     \
(\
  PETSC_DUMMY = MPI_Send(buf,count, datatype,dest,tag,comm),                         \
  send_ct++, TypeSize(send_len,count,datatype),PETSC_DUMMY                              \
)

#define MPI_Wait(request,status) \
(\
  wait_ct++,sum_of_waits_ct++,  \
  MPI_Wait(request,status)       \
)

#define MPI_Waitany(a,b,c,d)     \
(\
  wait_any_ct++,sum_of_waits_ct++,\
  MPI_Waitany(a,b,c,d)           \
)

#define MPI_Waitall(count,array_of_requests,array_of_statuses) \
(\
  wait_all_ct++,sum_of_waits_ct += (PetscLogDouble) (count),       \
  MPI_Waitall(count,array_of_requests,array_of_statuses)       \
)

#define MPI_Allreduce(sendbuf, recvbuf,count,datatype,op,comm) \
(\
  allreduce_ct++,MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm)\
)

#else

#define MPI_Startall_irecv(count,number,requests) \
(\
  MPI_Startall(number,requests)                 \
)

#define MPI_Startall_isend(count,number,requests) \
(\
  MPI_Startall(number,requests)                 \
)

#define MPI_Start_isend(count, requests) \
(\
  MPI_Start(requests)                   \
)

#endif /* !USING_MPIUNI && ! PETSC_HAVE_BROKEN_RECURSIVE_MACRO */

#else  /* ---Logging is turned off --------------------------------------------*/

#define PetscLogFlops(n) 0

/*
     With logging turned off, then MPE has to be turned off
*/
#define MPEBEGIN                  1000 
#define PetscLogMPEBegin()            0
#define PetscLogMPEDump(a)            0
#define PetscLogEventMPEActivate(a)   0
#define PetscLogEventMPEDeactivate(a) 0

#define PetscLogEventActivate(a)   0
#define PetscLogEventDeactivate(a) 0

#define PetscLogEventActivateClass(a)   0
#define PetscLogEventDeactivateClass(a) 0

#define _PetscLogPLB                        0
#define _PetscLogPLE                        0
#define _PetscLogPHC                        0
#define _PetscLogPHD                        0
#define PetscGetFlops(a)                (*(a) = 0.0,0)
#define PetscLogEventBegin(e,o1,o2,o3,o4)   0
#define PetscLogEventEnd(e,o1,o2,o3,o4)     0
#define PetscLogEventBarrierBegin(e,o1,o2,o3,o4,cm) 0
#define PetscLogEventBarrierEnd(e,o1,o2,o3,o4,cm)   0
#define PetscLogObjectParent(p,c)
#define PetscLogObjectParents(p,n,c)
#define PetscLogObjectCreate(h)
#define PetscLogObjectDestroy(h)
#define PetscLogObjectMemory(p,m)
#define PetscLogDestroy()                   0
#define PetscLogStagePush(a)                0
#define PetscLogStagePop()                  0
#define PetscLogStageRegister(a,b)          0
#define PetscLogStagePrint(a,flg)           0
#define PetscLogPrintSummary(comm,file)     0
#define PetscLogBegin()                     0
#define PetscLogTraceBegin(file)            0
#define PetscLogSet(lb,le)                  0
#define PetscLogAllBegin()                  0
#define PetscLogDump(c)                     0
#define PetscLogEventRegister(a,b,c)        0
EXTERN int PetscLogObjectState(PetscObject,const char[],...);

/* If PETSC_USE_LOG is NOT defined, these still need to be! */
#define MPI_Startall_irecv(count,number,requests) MPI_Startall(number,requests)

#define MPI_Startall_isend(count,number,requests) MPI_Startall(number,requests)

#define MPI_Start_isend(count,requests) MPI_Start(requests)

#endif   /* PETSC_USE_LOG */

extern PetscTruth PetscPreLoadingUsed;       /* true if we are or have done preloading */
extern PetscTruth PetscPreLoadingOn;         /* true if we are currently in a preloading calculation */

#define PreLoadBegin(flag,name) {PetscTruth PreLoading = flag; int PreLoadMax,PreLoadIt,_3_ierr;\
                                 _3_ierr = PetscOptionsGetLogical(PETSC_NULL,"-preload",&PreLoading,PETSC_NULL);CHKERRQ(_3_ierr);\
                                 PreLoadMax = (int)(PreLoading);PetscPreLoadingUsed = PreLoading ? PETSC_TRUE : PetscPreLoadingUsed;\
                                 for (PreLoadIt=0; PreLoadIt<=PreLoadMax; PreLoadIt++) {\
                                   PetscPreLoadingOn = PreLoading;\
                                   _3_ierr = PetscBarrier(PETSC_NULL);CHKERRQ(_3_ierr);\
                                   _3_ierr = PetscLogStagePush(PETSC_DETERMINE);CHKERRQ(_3_ierr);\
                                   _3_ierr = PetscLogStageRegister(PETSC_DETERMINE,name);CHKERRQ(_3_ierr);\
                                   _3_ierr = PetscLogStagePrint(PETSC_DETERMINE,(PetscTruth)(!PreLoadMax || PreLoadIt));
#define PreLoadEnd()               _3_ierr = PetscLogStagePop();CHKERRQ(_3_ierr);PreLoading = PETSC_FALSE;}}
#define PreLoadStage(name)         _3_ierr = PetscLogStagePop();CHKERRQ(_3_ierr);\
                                   _3_ierr = PetscLogStagePush(PETSC_DETERMINE);CHKERRQ(_3_ierr);\
                                   _3_ierr = PetscLogStageRegister(PETSC_DETERMINE,name);CHKERRQ(_3_ierr);\
                                   _3_ierr = PetscLogStagePrint(PETSC_DETERMINE,(PetscTruth)(!PreLoadMax || PreLoadIt));
#endif






