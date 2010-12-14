/*
    Defines profile/logging in PETSc.
*/

#if !defined(__PetscLog_H)
#define __PetscLog_H
#include "petscsys.h"  
PETSC_EXTERN_CXX_BEGIN

/*MC
    PetscLogEvent - id used to identify PETSc or user events which timed portions (blocks of executable)
     code.

    Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(), PetscLogStage
M*/
typedef int PetscLogEvent;

/*MC
    PetscLogStage - id used to identify user stages (phases, sections) of runs - for logging

    Level: intermediate

.seealso: PetscLogStageRegister(), PetscLogStageBegin(), PetscLogStageEnd(), PetscLogEvent
M*/
typedef int PetscLogStage;

#define PETSC_EVENT  1311311
extern PetscLogEvent PETSC_LARGEST_EVENT;

/* Global flop counter */
extern PetscLogDouble  _TotalFlops;
extern PetscLogDouble petsc_tmp_flops;

/* General logging of information; different from event logging */
extern PetscErrorCode  PetscInfo_Private(const char[],void*,const char[],...);
#if defined(PETSC_USE_INFO)
#define PetscInfo(A,S)                       PetscInfo_Private(PETSC_FUNCTION_NAME,A,S)
#define PetscInfo1(A,S,a1)                   PetscInfo_Private(PETSC_FUNCTION_NAME,A,S,a1)
#define PetscInfo2(A,S,a1,a2)                PetscInfo_Private(PETSC_FUNCTION_NAME,A,S,a1,a2)
#define PetscInfo3(A,S,a1,a2,a3)             PetscInfo_Private(PETSC_FUNCTION_NAME,A,S,a1,a2,a3)
#define PetscInfo4(A,S,a1,a2,a3,a4)          PetscInfo_Private(PETSC_FUNCTION_NAME,A,S,a1,a2,a3,a4)
#define PetscInfo5(A,S,a1,a2,a3,a4,a5)       PetscInfo_Private(PETSC_FUNCTION_NAME,A,S,a1,a2,a3,a4,a5)
#define PetscInfo6(A,S,a1,a2,a3,a4,a5,a6)    PetscInfo_Private(PETSC_FUNCTION_NAME,A,S,a1,a2,a3,a4,a5,a6)
#define PetscInfo7(A,S,a1,a2,a3,a4,a5,a6,a7) PetscInfo_Private(PETSC_FUNCTION_NAME,A,S,a1,a2,a3,a4,a5,a6,a7)
#else 
#define PetscInfo(A,S)                       0
#define PetscInfo1(A,S,a1)                   0
#define PetscInfo2(A,S,a1,a2)                0
#define PetscInfo3(A,S,a1,a2,a3)             0
#define PetscInfo4(A,S,a1,a2,a3,a4)          0
#define PetscInfo5(A,S,a1,a2,a3,a4,a5)       0
#define PetscInfo6(A,S,a1,a2,a3,a4,a5,a6)    0
#define PetscInfo7(A,S,a1,a2,a3,a4,a5,a6,a7) 0
#endif
extern PetscErrorCode  PetscInfoDeactivateClass(PetscClassId);
extern PetscErrorCode  PetscInfoActivateClass(PetscClassId);
extern PetscBool       PetscLogPrintInfo;  /* if true, indicates PetscInfo() is turned on */

/* We must make the following structures available to access the event
     activation flags in the PetscLogEventBegin/End() macros. These are not part of the PETSc public
     API and are not intended to be used by other parts of PETSc or by users.
  
     The code that manipulates these structures is in src/sys/plog/utils.
*/
typedef struct _n_IntStack *IntStack;

/*
    ClassRegInfo, ClassPerfInfo - Each class has two data structures associated with it. The first has 
       static information about it, the second collects statistics on how many objects of the class are created,
       how much memory they use, etc.

    ClassRegLog, ClassPerfLog - arrays of the ClassRegInfo and ClassPerfInfo for all classes.
*/
typedef struct  {
  char           *name;   /* The class name */
  PetscClassId   classid; /* The integer identifying this class */
} ClassRegInfo;

typedef struct {
  PetscClassId   id;           /* The integer identifying this class */
  int            creations;    /* The number of objects of this class created */
  int            destructions; /* The number of objects of this class destroyed */
  PetscLogDouble mem;          /* The total memory allocated by objects of this class */
  PetscLogDouble descMem;      /* The total memory allocated by descendents of these objects */
} ClassPerfInfo;

typedef struct _n_ClassRegLog *ClassRegLog;
struct _n_ClassRegLog {
  int            numClasses; /* The number of classes registered */
  int            maxClasses; /* The maximum number of classes */
  ClassRegInfo * classInfo;  /* The structure for class information (classids are monotonicly increasing) */
};

typedef struct _n_ClassPerfLog *ClassPerfLog;
struct _n_ClassPerfLog {
  int            numClasses; /* The number of logging classes */
  int            maxClasses; /* The maximum number of classes */
  ClassPerfInfo *classInfo;  /* The structure for class information (classids are monotonicly increasing) */
};
/* -----------------------------------------------------------------------------------------------------*/
/*
    EventRegInfo, EventPerfInfo - Each event has two data structures associated with it. The first has 
       static information about it, the second collects statistics on how many times the event is used, how 
       much time it takes, etc.

    EventRegLog, EventPerfLog - an array of all EventRegInfo and EventPerfInfo for all events. There is one
      of these for each stage.

*/
typedef struct {
  char         *name;         /* The name of this event */
  PetscClassId classid;       /* The class the event is associated with */
#if defined (PETSC_HAVE_MPE)
  int          mpe_id_begin; /* MPE IDs that define the event */
  int          mpe_id_end;
#endif
} EventRegInfo;

typedef struct {
  int            id;            /* The integer identifying this event */
  PetscBool      active;        /* The flag to activate logging */
  PetscBool      visible;       /* The flag to print info in summary */
  int            depth;         /* The nesting depth of the event call */
  int            count;         /* The number of times this event was executed */
  PetscLogDouble flops;         /* The flops used in this event */
  PetscLogDouble time;          /* The time taken for this event */
  PetscLogDouble numMessages;   /* The number of messages in this event */
  PetscLogDouble messageLength; /* The total message lengths in this event */
  PetscLogDouble numReductions; /* The number of reductions in this event */
} EventPerfInfo;

typedef struct _n_EventRegLog *EventRegLog;
struct _n_EventRegLog {
  int           numEvents; /* The number of registered events */
  int           maxEvents; /* The maximum number of events */
  EventRegInfo *eventInfo; /* The registration information for each event */
};

typedef struct _n_EventPerfLog *EventPerfLog;
struct _n_EventPerfLog {
  int            numEvents; /* The number of logging events */
  int            maxEvents; /* The maximum number of events */
  EventPerfInfo *eventInfo; /* The performance information for each event */
};
/* ------------------------------------------------------------------------------------------------------------*/
/*
   StageInfo - Contains all the information about a particular stage.

   StageLog - An array of StageInfo for each registered stage. There is a single one of these in the code.
*/
typedef struct _StageInfo {
  char         *name;     /* The stage name */
  PetscBool     used;     /* The stage was pushed on this processor */
  EventPerfInfo perfInfo; /* The stage performance information */
  EventPerfLog  eventLog; /* The event information for this stage */
  ClassPerfLog  classLog; /* The class information for this stage */
} StageInfo;

typedef struct _n_StageLog *StageLog;
extern  StageLog _stageLog;
struct _n_StageLog {
  int         numStages; /* The number of registered stages */
  int         maxStages; /* The maximum number of stages */
  IntStack    stack;     /* The stack for active stages */
  int         curStage;  /* The current stage (only used in macros so we don't call StackTop) */
  StageInfo  *stageInfo; /* The information for each stage */
  EventRegLog eventLog;  /* The registered events */
  ClassRegLog classLog;  /* The registered classes */
};

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
#define PETSC_FLOPS_PER_OP 4.0
#else
#define PETSC_FLOPS_PER_OP 1.0
#endif

#if defined(PETSC_USE_DEBUG)
#define PetscLogFlops(n) (petsc_tmp_flops = (PETSC_FLOPS_PER_OP*((PetscLogDouble)n)), ((petsc_tmp_flops < 0) ? PETSC_ERR_FLOP_COUNT : (_TotalFlops += petsc_tmp_flops,0)))
#define PetscLogFlopsNoError(n) (_TotalFlops += PETSC_FLOPS_PER_OP*((PetscLogDouble)n))
#else
#define PetscLogFlops(n) (_TotalFlops += PETSC_FLOPS_PER_OP*((PetscLogDouble)n),0)
#define PetscLogFlopsNoError(n) (_TotalFlops += PETSC_FLOPS_PER_OP*((PetscLogDouble)n))
#endif

#if defined (PETSC_HAVE_MPE)
#include "mpe.h"
extern PetscErrorCode         PetscLogMPEBegin(void);
extern PetscErrorCode         PetscLogMPEDump(const char[]);
extern PetscBool  UseMPE;
#define PETSC_LOG_EVENT_MPE_BEGIN(e) \
  ((UseMPE && _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) ? \
   MPE_Log_event(_stageLog->eventLog->eventInfo[e].mpe_id_begin,0,NULL) : 0)

#define PETSC_LOG_EVENT_MPE_END(e) \
  ((UseMPE && _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) ? \
   MPE_Log_event(_stageLog->eventLog->eventInfo[e].mpe_id_end,0,NULL) : 0)

#else 
#define PETSC_LOG_EVENT_MPE_BEGIN(e) 0 
#define PETSC_LOG_EVENT_MPE_END(e)   0
#endif

extern  PetscErrorCode (*_PetscLogPLB)(PetscLogEvent,int,PetscObject,PetscObject,PetscObject,PetscObject);
extern  PetscErrorCode (*_PetscLogPLE)(PetscLogEvent,int,PetscObject,PetscObject,PetscObject,PetscObject);
extern  PetscErrorCode (*_PetscLogPHC)(PetscObject);
extern  PetscErrorCode (*_PetscLogPHD)(PetscObject);

#define PetscLogObjectParent(p,c) \
  (c && p && (((PetscObject)(c))->parent = (PetscObject)(p),((PetscObject)(c))->parentid = ((PetscObject)p)->id,0))

#define PetscLogObjectParents(p,n,d)  0;{int _i; for (_i=0; _i<n; _i++) {ierr = PetscLogObjectParent(p,(d)[_i]);CHKERRQ(ierr);}}
#define PetscLogObjectCreate(h)      ((_PetscLogPHC) ? (*_PetscLogPHC)((PetscObject)h) : 0)
#define PetscLogObjectDestroy(h)     ((_PetscLogPHD) ? (*_PetscLogPHD)((PetscObject)h) : 0)
#define PetscLogObjectMemory(p,m)    (((PetscObject)(p))->mem += (m),0)
/* Initialization functions */
extern PetscErrorCode  PetscLogBegin(void);
extern PetscErrorCode  PetscLogAllBegin(void);
extern PetscErrorCode  PetscLogTraceBegin(FILE *);
extern PetscErrorCode  PetscLogActions(PetscBool);
extern PetscErrorCode  PetscLogObjects(PetscBool);
/* General functions */
extern PetscErrorCode  PetscLogGetRGBColor(const char*[]);
extern PetscErrorCode  PetscLogDestroy(void);
extern PetscErrorCode  PetscLogSet(PetscErrorCode (*)(int, int, PetscObject, PetscObject, PetscObject, PetscObject),
                   PetscErrorCode (*)(int, int, PetscObject, PetscObject, PetscObject, PetscObject));
extern PetscErrorCode  PetscLogObjectState(PetscObject, const char[], ...);
/* Output functions */
extern PetscErrorCode  PetscLogView(PetscViewer);
extern PetscErrorCode  PetscLogViewPython(PetscViewer);
extern PetscErrorCode  PetscLogPrintDetailed(MPI_Comm, const char[]);
extern PetscErrorCode  PetscLogDump(const char[]);

extern PetscErrorCode  PetscGetFlops(PetscLogDouble *);

extern PetscErrorCode  PetscLogStageRegister(const char[],PetscLogStage*);
extern PetscErrorCode  PetscLogStagePush(PetscLogStage);
extern PetscErrorCode  PetscLogStagePop(void);
extern PetscErrorCode  PetscLogStageSetActive(PetscLogStage, PetscBool );
extern PetscErrorCode  PetscLogStageGetActive(PetscLogStage, PetscBool  *);
extern PetscErrorCode  PetscLogStageSetVisible(PetscLogStage, PetscBool );
extern PetscErrorCode  PetscLogStageGetVisible(PetscLogStage, PetscBool  *);
extern PetscErrorCode  PetscLogStageGetId(const char [], PetscLogStage *);
/* Event functions */
extern PetscErrorCode  PetscLogEventRegister(const char[], PetscClassId,PetscLogEvent*);
extern PetscErrorCode  PetscLogEventActivate(PetscLogEvent);
extern PetscErrorCode  PetscLogEventDeactivate(PetscLogEvent);
extern PetscErrorCode  PetscLogEventSetActiveAll(PetscLogEvent, PetscBool );
extern PetscErrorCode  PetscLogEventActivateClass(PetscClassId);
extern PetscErrorCode  PetscLogEventDeactivateClass(PetscClassId);


/* Global counters */
extern  PetscLogDouble irecv_ct;
extern  PetscLogDouble isend_ct;
extern  PetscLogDouble recv_ct;
extern  PetscLogDouble send_ct;
extern  PetscLogDouble irecv_len;
extern  PetscLogDouble isend_len;
extern  PetscLogDouble recv_len;
extern  PetscLogDouble send_len;
extern  PetscLogDouble allreduce_ct;
extern  PetscLogDouble gather_ct;
extern  PetscLogDouble scatter_ct;
extern  PetscLogDouble wait_ct;
extern  PetscLogDouble wait_any_ct;
extern  PetscLogDouble wait_all_ct;
extern  PetscLogDouble sum_of_waits_ct;

#define PetscLogEventBarrierBegin(e,o1,o2,o3,o4,cm) \
  (((_PetscLogPLB && _stageLog->stageInfo[_stageLog->curStage].perfInfo.active &&  _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) ? \
    (PetscLogEventBegin((e),o1,o2,o3,o4) || MPI_Barrier(cm) || PetscLogEventEnd((e),o1,o2,o3,o4)) : 0 ) || \
   PetscLogEventBegin((e)+1,o1,o2,o3,o4))

#define PetscLogEventBegin(e,o1,o2,o3,o4) \
  (((_PetscLogPLB && _stageLog->stageInfo[_stageLog->curStage].perfInfo.active && _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) ? \
    (*_PetscLogPLB)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4)) : 0 ) || \
  PETSC_LOG_EVENT_MPE_BEGIN(e))

#define PetscLogEventBarrierEnd(e,o1,o2,o3,o4,cm) PetscLogEventEnd(e+1,o1,o2,o3,o4)

#define PetscLogEventEnd(e,o1,o2,o3,o4) \
  (((_PetscLogPLE && _stageLog->stageInfo[_stageLog->curStage].perfInfo.active && _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) ? \
    (*_PetscLogPLE)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4)) : 0 ) || \
  PETSC_LOG_EVENT_MPE_END(e))

extern PetscErrorCode PetscLogEventGetFlops(PetscLogEvent, PetscLogDouble*);
extern PetscErrorCode PetscLogEventZeroFlops(PetscLogEvent);

/*
     These are used internally in the PETSc routines to keep a count of MPI messages and 
   their sizes.

     This does not work for MPI-Uni because our include/mpiuni/mpi.h file
   uses macros to defined the MPI operations. 

     It does not work correctly from HP-UX because it processes the 
   macros in a way that sometimes it double counts, hence 
   PETSC_HAVE_BROKEN_RECURSIVE_MACRO

     It does not work with Windows because winmpich lacks MPI_Type_size()
*/
#if !defined(__MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO) && !defined (PETSC_HAVE_MPI_MISSING_TYPESIZE)
/*
   Logging of MPI activities
*/
PETSC_STATIC_INLINE PetscErrorCode TypeSize(PetscLogDouble *buff,PetscMPIInt count,MPI_Datatype type) 
{
  PetscMPIInt mysize; return  (MPI_Type_size(type,&mysize) || ((*buff += (PetscLogDouble) (count*mysize)),0));
}

#define MPI_Irecv(buf,count,datatype,source,tag,comm,request) \
 ((irecv_ct++,0) || TypeSize(&irecv_len,count,datatype) || MPI_Irecv(buf,count,datatype,source,tag,comm,request))

#define MPI_Isend(buf,count,datatype,dest,tag,comm,request) \
 ((isend_ct++,0) || TypeSize(&isend_len,count,datatype) || MPI_Isend(buf,count,datatype,dest,tag,comm,request))

#define MPI_Startall_irecv(count,number,requests) \
 ((irecv_ct += (PetscLogDouble)(number),0) || TypeSize(&irecv_len,count,MPIU_SCALAR) || MPI_Startall(number,requests))

#define MPI_Startall_isend(count,number,requests) \
 ((isend_ct += (PetscLogDouble)(number),0) || TypeSize(&isend_len,count,MPIU_SCALAR) || MPI_Startall(number,requests))

#define MPI_Start_isend(count,requests) \
 ((isend_ct++,0) || TypeSize(&isend_len,count,MPIU_SCALAR) || MPI_Start(requests))

#define MPI_Recv(buf,count,datatype,source,tag,comm,status) \
 ((recv_ct++,0) || TypeSize(&recv_len,count,datatype) || MPI_Recv(buf,count,datatype,source,tag,comm,status))

#define MPI_Send(buf,count,datatype,dest,tag,comm) \
 ((send_ct++,0) || TypeSize(&send_len,count,datatype) || MPI_Send(buf,count,datatype,dest,tag,comm))

#define MPI_Wait(request,status) \
 ((wait_ct++,sum_of_waits_ct++,0) || MPI_Wait(request,status))
  
#define MPI_Waitany(a,b,c,d) \
 ((wait_any_ct++,sum_of_waits_ct++,0) || MPI_Waitany(a,b,c,d))

#define MPI_Waitall(count,array_of_requests,array_of_statuses) \
 ((wait_all_ct++,sum_of_waits_ct += (PetscLogDouble) (count),0) || MPI_Waitall(count,array_of_requests,array_of_statuses))

#define MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm) \
 ((allreduce_ct++,0) || MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm))

#define MPI_Allgather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,comm) \
 ((gather_ct++,0) || MPI_Allgather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,comm))

#define MPI_Allgatherv(sendbuf,sendcount,sendtype,recvbuf,recvcount,displs,recvtype,comm) \
 ((gather_ct++,0) || MPI_Allgatherv(sendbuf,sendcount,sendtype,recvbuf,recvcount,displs,recvtype,comm))

#define MPI_Gather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,root,comm) \
 ((gather_ct++,0) || TypeSize(&send_len,sendcount,sendtype) || MPI_Gather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,root,comm))

#define MPI_Gatherv(sendbuf,sendcount,sendtype,recvbuf,recvcount,displs,recvtype,root,comm) \
 ((gather_ct++,0) || TypeSize(&send_len,sendcount,sendtype) || MPI_Gatherv(sendbuf,sendcount,sendtype,recvbuf,recvcount,displs,recvtype,root,comm))

#define MPI_Scatter(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,root,comm) \
  ((scatter_ct++,0) || TypeSize(&recv_len,recvcount,recvtype) || MPI_Scatter(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,root,comm))

#define MPI_Scatterv(sendbuf,sendcount,displs,sendtype,recvbuf,recvcount,recvtype,root,comm) \
  ((scatter_ct++,0) || TypeSize(&recv_len,recvcount,recvtype) || MPI_Scatterv(sendbuf,sendcount,displs,sendtype,recvbuf,recvcount,recvtype,root,comm))

#else

#define MPI_Startall_irecv(count,number,requests) \
 (MPI_Startall(number,requests))

#define MPI_Startall_isend(count,number,requests) \
 (MPI_Startall(number,requests))

#define MPI_Start_isend(count,requests) \
 (MPI_Start(requests))

#endif /* !__MPIUNI_H && ! PETSC_HAVE_BROKEN_RECURSIVE_MACRO */

#else  /* ---Logging is turned off --------------------------------------------*/

#define PetscLogFlops(n) 0
#define PetscLogFlopsNoError(n)

/*
     With logging turned off, then MPE has to be turned off
*/
#define PetscLogMPEBegin()         0
#define PetscLogMPEDump(a)         0

#define PetscLogEventActivate(a)   0
#define PetscLogEventDeactivate(a) 0

#define PetscLogEventActivateClass(a)   0
#define PetscLogEventDeactivateClass(a) 0
#define PetscLogEventSetActiveAll(a,b)  0

#define _PetscLogPLB                        0
#define _PetscLogPLE                        0
#define _PetscLogPHC                        0
#define _PetscLogPHD                        0
#define PetscGetFlops(a)                (*(a) = 0.0,0)
#define PetscLogEventBegin(e,o1,o2,o3,o4)   0
#define PetscLogEventEnd(e,o1,o2,o3,o4)     0
#define PetscLogEventBarrierBegin(e,o1,o2,o3,o4,cm) 0
#define PetscLogEventBarrierEnd(e,o1,o2,o3,o4,cm)   0
#define PetscLogObjectParent(p,c)           0
#define PetscLogObjectParents(p,n,c)        0
#define PetscLogObjectCreate(h)             0
#define PetscLogObjectDestroy(h)            0
#define PetscLogObjectMemory(p,m)           0
#define PetscLogDestroy()                   0
#define PetscLogStagePush(a)                0
#define PetscLogStagePop()                  0
#define PetscLogStageRegister(a,b)          0
#define PetscLogStagePrint(a,flg)           0
#define PetscLogView(viewer)                0
#define PetscLogViewPython(viewer)          0
#define PetscLogPrintDetailed(comm,file)    0
#define PetscLogBegin()                     0
#define PetscLogTraceBegin(file)            0
#define PetscLogSet(lb,le)                  0
#define PetscLogAllBegin()                  0
#define PetscLogDump(c)                     0
#define PetscLogEventRegister(a,b,c)        0
#define PetscLogObjects(a)                  0
#define PetscLogActions(a)                  0
extern PetscErrorCode  PetscLogObjectState(PetscObject,const char[],...);

/* If PETSC_USE_LOG is NOT defined, these still need to be! */
#define MPI_Startall_irecv(count,number,requests) MPI_Startall(number,requests)
#define MPI_Startall_isend(count,number,requests) MPI_Startall(number,requests)
#define MPI_Start_isend(count,requests) MPI_Start(requests)
#define PetscLogStageGetId(a,b)                      (*(b)=0,0)
#define PetscLogStageSetActive(a,b)                  0
#define PetscLogStageGetActive(a,b)                  0
#define PetscLogStageGetVisible(a,b)                 0
#define PetscLogStageSetVisible(a,b)                 0

#endif   /* PETSC_USE_LOG */

/* Special support for C++ */
#include "petsclog.hh"

#define PreLoadBegin(flag,name) \
{\
  PetscBool      PreLoading = flag;\
  int            PreLoadMax,PreLoadIt;\
  PetscLogStage  _stageNum;\
  PetscErrorCode _3_ierr;	\
  _3_ierr = PetscOptionsGetBool(PETSC_NULL,"-preload",&PreLoading,PETSC_NULL);CHKERRQ(_3_ierr);\
  PreLoadMax = (int)(PreLoading);\
  PetscPreLoadingUsed = PreLoading ? PETSC_TRUE : PetscPreLoadingUsed;\
  for (PreLoadIt=0; PreLoadIt<=PreLoadMax; PreLoadIt++) {\
    PetscPreLoadingOn = PreLoading;\
    _3_ierr = PetscBarrier(PETSC_NULL);CHKERRQ(_3_ierr);\
    if (PreLoadIt>0) {\
      _3_ierr = PetscLogStageGetId(name,&_stageNum);CHKERRQ(_3_ierr);\
    } else {\
      _3_ierr = PetscLogStageRegister(name,&_stageNum);CHKERRQ(_3_ierr); \
    }\
    _3_ierr = PetscLogStageSetActive(_stageNum,(PetscBool)(!PreLoadMax || PreLoadIt));\
    _3_ierr = PetscLogStagePush(_stageNum);CHKERRQ(_3_ierr);

#define PreLoadEnd() \
    _3_ierr = PetscLogStagePop();CHKERRQ(_3_ierr);\
    PreLoading = PETSC_FALSE;\
  }\
}

#define PreLoadStage(name) \
  _3_ierr = PetscLogStagePop();CHKERRQ(_3_ierr);\
  if (PreLoadIt>0) {\
    _3_ierr = PetscLogStageGetId(name,&_stageNum);CHKERRQ(_3_ierr);\
  } else {\
    _3_ierr = PetscLogStageRegister(name,&_stageNum);CHKERRQ(_3_ierr);	\
  }\
  _3_ierr = PetscLogStageSetActive(_stageNum,(PetscBool)(!PreLoadMax || PreLoadIt));\
  _3_ierr = PetscLogStagePush(_stageNum);CHKERRQ(_3_ierr);

PETSC_EXTERN_CXX_END
#endif
