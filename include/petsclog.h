/*
    Defines profile/logging in PETSc.
*/

#if !defined(__PetscLog_H)
#define __PetscLog_H
#include "petsc.h"  
PETSC_EXTERN_CXX_BEGIN
/*
  Each PETSc object class has it's own cookie (internal integer in the 
  data structure used for error checking). These are all defined by an offset 
  from the lowest one, PETSC_COOKIE.
*/
#define PETSC_COOKIE 1211211
extern PetscCookie PETSC_LARGEST_COOKIE;
#define PETSC_EVENT  1311311
extern PetscEvent PETSC_LARGEST_EVENT;

/* Events for the Petsc standard library */
extern PetscEvent PETSC_Barrier;

/* Global flop counter */
extern PetscLogDouble _TotalFlops;

/* General logging of information; different from event logging */
EXTERN PetscErrorCode        PetscLogInfo(void*,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN PetscErrorCode        PetscLogInfoDeactivateClass(PetscCookie);
EXTERN PetscErrorCode        PetscLogInfoActivateClass(PetscCookie);
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
EXTERN PetscErrorCode        PetscLogMPEBegin(void);
EXTERN PetscErrorCode        PetscLogMPEDump(const char[]);
extern PetscTruth UseMPE;
#define PETSC_LOG_EVENT_MPE_BEGIN(e) \
  if(UseMPE && _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) \
    MPE_Log_event(_stageLog->eventLog->eventInfo[e].mpe_id_begin,0,(char*)"");

#define PETSC_LOG_EVENT_MPE_END(e) \
  if(UseMPE && _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) \
    MPE_Log_event(_stageLog->eventLog->eventInfo[e].mpe_id_end,0,(char*)"");

#else 
#define PETSC_LOG_EVENT_MPE_BEGIN(e)
#define PETSC_LOG_EVENT_MPE_END(e)
#endif

EXTERN PetscErrorCode (*_PetscLogPLB)(PetscEvent,int,PetscObject,PetscObject,PetscObject,PetscObject);
EXTERN PetscErrorCode (*_PetscLogPLE)(PetscEvent,int,PetscObject,PetscObject,PetscObject,PetscObject);
EXTERN PetscErrorCode (*_PetscLogPHC)(PetscObject);
EXTERN PetscErrorCode (*_PetscLogPHD)(PetscObject);

#define PetscLogObjectParent(p,c) \
  if (c) {\
    PetscValidHeader((PetscObject)(c),2);\
    PetscValidHeader((PetscObject)(p),1);\
    ((PetscObject)(c))->parent = (PetscObject)(p);\
    ((PetscObject)(c))->parentid = ((PetscObject)p)->id;\
  }
#define PetscLogObjectParents(p,n,d) {int _i; for (_i=0; _i<n; _i++) PetscLogObjectParent(p,(d)[_i]);}
#define PetscLogObjectCreate(h)      {if (_PetscLogPHC) (*_PetscLogPHC)((PetscObject)h);}
#define PetscLogObjectDestroy(h)     {if (_PetscLogPHD) (*_PetscLogPHD)((PetscObject)h);}
#define PetscLogObjectMemory(p,m)    {PetscValidHeader((PetscObject)p,1);((PetscObject)(p))->mem += (m);}
/* Initialization functions */
EXTERN PetscErrorCode PetscLogBegin(void);
EXTERN PetscErrorCode PetscLogAllBegin(void);
EXTERN PetscErrorCode PetscLogTraceBegin(FILE *);
EXTERN PetscErrorCode PetscLogActions(PetscTruth);
EXTERN PetscErrorCode PetscLogObjects(PetscTruth);
/* General functions */
EXTERN PetscErrorCode PetscLogGetRGBColor(const char*[]);
EXTERN PetscErrorCode PetscLogDestroy(void);
EXTERN PetscErrorCode PetscLogSet(PetscErrorCode (*)(int, int, PetscObject, PetscObject, PetscObject, PetscObject),
                   PetscErrorCode (*)(int, int, PetscObject, PetscObject, PetscObject, PetscObject));
EXTERN PetscErrorCode PetscLogObjectState(PetscObject, const char[], ...)  PETSC_PRINTF_FORMAT_CHECK(2,3);
/* Output functions */
EXTERN PetscErrorCode PetscLogPrintSummary(MPI_Comm, const char[]);
EXTERN PetscErrorCode PetscLogDump(const char[]);
/* Counter functions */
EXTERN PetscErrorCode PetscGetFlops(PetscLogDouble *);
/* Stage functions */
EXTERN PetscErrorCode PetscLogStageRegister(int*, const char[]);
EXTERN PetscErrorCode PetscLogStagePush(int);
EXTERN PetscErrorCode PetscLogStagePop(void);
EXTERN PetscErrorCode PetscLogStageSetActive(int, PetscTruth);
EXTERN PetscErrorCode PetscLogStageGetActive(int, PetscTruth *);
EXTERN PetscErrorCode PetscLogStageSetVisible(int, PetscTruth);
EXTERN PetscErrorCode PetscLogStageGetVisible(int, PetscTruth *);
EXTERN PetscErrorCode PetscLogStageGetId(const char [], int *);
/* Event functions */
EXTERN PetscErrorCode PetscLogEventRegister(PetscEvent*, const char[], PetscCookie);
EXTERN PetscErrorCode PetscLogEventActivate(PetscEvent);
EXTERN PetscErrorCode PetscLogEventDeactivate(PetscEvent);
EXTERN PetscErrorCode PetscLogEventSetActiveAll(PetscEvent, PetscTruth);
EXTERN PetscErrorCode PetscLogEventActivateClass(PetscCookie);
EXTERN PetscErrorCode PetscLogEventDeactivateClass(PetscCookie);
/* Class functions */
EXTERN PetscErrorCode PetscLogClassRegister(PetscCookie*, const char []);

/* Global counters */
extern PetscLogDouble irecv_ct,  isend_ct,  recv_ct,  send_ct;
extern PetscLogDouble irecv_len, isend_len, recv_len, send_len;
extern PetscLogDouble allreduce_ct;
extern PetscLogDouble wait_ct, wait_any_ct, wait_all_ct, sum_of_waits_ct;
extern int            PETSC_DUMMY, PETSC_DUMMY_SIZE, PETSC_DUMMY_COUNT;

/* We must make these structures available if we are to access the event
   activation flags in the PetscLogEventBegin/End() macros. If we forced a
   function call each time, we could leave these structures in plog.h
*/
/* Default log */
typedef struct _StageLog *StageLog;
extern StageLog _stageLog;

/* A simple stack (should replace) */
typedef struct _IntStack *IntStack;

/* The structures for logging performance */
typedef struct _EventPerfInfo {
  int            id;            /* The integer identifying this section */
  PetscTruth     active;        /* The flag to activate logging */
  PetscTruth     visible;       /* The flag to print info in summary */
  int            depth;         /* The nesting depth of the event call */
  int            count;         /* The number of times this section was executed */
  PetscLogDouble flops;         /* The flops used in this section */
  PetscLogDouble time;          /* The time taken for this section */
  PetscLogDouble numMessages;   /* The number of messages in this section */
  PetscLogDouble messageLength; /* The total message lengths in this section */
  PetscLogDouble numReductions; /* The number of reductions in this section */
} EventPerfInfo;

typedef struct _ClassPerfInfo {
  int            id;           /* The integer identifying this class */
  int            creations;    /* The number of objects of this class created */
  int            destructions; /* The number of objects of this class destroyed */
  PetscLogDouble mem;          /* The total memory allocated by objects of this class */
  PetscLogDouble descMem;      /* The total memory allocated by descendents of these objects */
} ClassPerfInfo;

/* The structures for logging registration */
typedef struct _ClassRegInfo {
  char            *name;   /* The class name */
  PetscCookie cookie; /* The integer identifying this class */
} ClassRegInfo;

typedef struct _EventRegInfo {
  char            *name;   /* The name of this event */
  PetscCookie cookie; /* The class id for this event (should maybe give class ID instead) */
#if defined (PETSC_HAVE_MPE)
  int             mpe_id_begin; /* MPE IDs that define the event */
  int             mpe_id_end;
#endif
} EventRegInfo;

typedef struct _EventRegLog *EventRegLog;
struct _EventRegLog {
  int           numEvents; /* The number of registered events */
  int           maxEvents; /* The maximum number of events */
  EventRegInfo *eventInfo; /* The registration information for each event */
};

typedef struct _EventPerfLog *EventPerfLog;
struct _EventPerfLog {
  int            numEvents; /* The number of logging events */
  int            maxEvents; /* The maximum number of events */
  EventPerfInfo *eventInfo; /* The performance information for each event */
};

/* The structure for logging class information */
typedef struct _ClassRegLog *ClassRegLog;
struct _ClassRegLog {
  int           numClasses; /* The number of classes registered */
  int           maxClasses; /* The maximum number of classes */
  ClassRegInfo *classInfo;  /* The structure for class information (cookies are monotonicly increasing) */
};

typedef struct _ClassPerfLog *ClassPerfLog;
struct _ClassPerfLog {
  int            numClasses; /* The number of logging classes */
  int            maxClasses; /* The maximum number of classes */
  ClassPerfInfo *classInfo;  /* The structure for class information (cookies are monotonicly increasing) */
};

/* The structures for logging in stages */
typedef struct _StageInfo {
  char         *name;     /* The stage name */
  PetscTruth    used;     /* The stage was pushed on this processor */
  EventPerfInfo perfInfo; /* The stage performance information */
  EventPerfLog  eventLog; /* The event information for this stage */
  ClassPerfLog  classLog; /* The class information for this stage */
} StageInfo;

struct _StageLog {
  /* Size information */
  int         numStages; /* The number of registered stages */
  int         maxStages; /* The maximum number of stages */
  /* Runtime information */
  IntStack    stack;     /* The stack for active stages */
  int         curStage;  /* The current stage (only used in macros so we don't call StackTop) */
  /* Stage specific information */
  StageInfo  *stageInfo; /* The information for each stage */
  EventRegLog eventLog;  /* The registered events */
  ClassRegLog classLog;  /* The registered classes */
};

#define PetscLogEventBarrierBegin(e,o1,o2,o3,o4,cm) 0; \
{\
  int _2_ierr;\
  if (_PetscLogPLB && \
      _stageLog->stageInfo[_stageLog->curStage].perfInfo.active && \
      _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) {\
    _2_ierr = PetscLogEventBegin((e),o1,o2,o3,o4);CHKERRQ(_2_ierr);\
    _2_ierr = MPI_Barrier(cm);CHKERRQ(_2_ierr);\
    _2_ierr = PetscLogEventEnd((e),o1,o2,o3,o4);CHKERRQ(_2_ierr);\
  }\
  _2_ierr = PetscLogEventBegin((e)+1,o1,o2,o3,o4);CHKERRQ(_2_ierr);\
}

#define PetscLogEventBegin(e,o1,o2,o3,o4) 0; \
{\
  if (_PetscLogPLB && \
      _stageLog->stageInfo[_stageLog->curStage].perfInfo.active && \
      _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) {\
    (*_PetscLogPLB)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));\
  }\
  PETSC_LOG_EVENT_MPE_BEGIN(e); \
}

#define PetscLogEventBarrierEnd(e,o1,o2,o3,o4,cm) PetscLogEventEnd(e+1,o1,o2,o3,o4)

#define PetscLogEventEnd(e,o1,o2,o3,o4) 0; \
{\
  if (_PetscLogPLE && \
      _stageLog->stageInfo[_stageLog->curStage].perfInfo.active && \
      _stageLog->stageInfo[_stageLog->curStage].eventLog->eventInfo[e].active) {\
    (*_PetscLogPLE)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4));\
  }\
  PETSC_LOG_EVENT_MPE_END(e); \
} 

/* Creation and destruction functions */
EXTERN PetscErrorCode StageLogCreate(StageLog *);
EXTERN PetscErrorCode StageLogDestroy(StageLog);
/* Registration functions */
EXTERN PetscErrorCode StageLogRegister(StageLog, const char [], int *);
/* Runtime functions */
EXTERN PetscErrorCode PetscLogGetStageLog(StageLog *);
EXTERN PetscErrorCode StageLogPush(StageLog, int);
EXTERN PetscErrorCode StageLogPop(StageLog);
EXTERN PetscErrorCode StageLogGetCurrent(StageLog, int *);
EXTERN PetscErrorCode StageLogSetActive(StageLog, int, PetscTruth);
EXTERN PetscErrorCode StageLogGetActive(StageLog, int, PetscTruth *);
EXTERN PetscErrorCode StageLogSetVisible(StageLog, int, PetscTruth);
EXTERN PetscErrorCode StageLogGetVisible(StageLog, int, PetscTruth *);
EXTERN PetscErrorCode StageLogGetStage(StageLog, const char [], int *);
EXTERN PetscErrorCode StageLogGetClassRegLog(StageLog, ClassRegLog *);
EXTERN PetscErrorCode StageLogGetEventRegLog(StageLog, EventRegLog *);
EXTERN PetscErrorCode StageLogGetClassPerfLog(StageLog, int, ClassPerfLog *);
EXTERN PetscErrorCode StageLogGetEventPerfLog(StageLog, int, EventPerfLog *);

/*
     This does not work for MPI-Uni because our include/mpiuni/mpi.h file
   uses macros to defined the MPI operations. 

     It does not work correctly from HP-UX because it processes the 
   macros in a way that sometimes it double counts, hence 
   PETSC_HAVE_BROKEN_RECURSIVE_MACRO

     It does not work with Windows because winmpich lacks MPI_Type_size()
*/
#if !defined(_petsc_mpi_uni) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO) && !defined (PETSC_HAVE_MPI_MISSING_TYPESIZE)
/*
   Logging of MPI activities
*/

#define TypeSize(buff,count,type) \
(\
  MPI_Type_size(type,&PETSC_DUMMY_SIZE),buff += ((PetscLogDouble) ((count)*PETSC_DUMMY_SIZE))\
)

#define MPI_Irecv(buf,count, datatype,source,tag,comm,request) \
(\
  PETSC_DUMMY_COUNT = count,PETSC_DUMMY = MPI_Irecv(buf,PETSC_DUMMY_COUNT, datatype,source,tag,comm,request),\
  irecv_ct++,TypeSize(irecv_len,PETSC_DUMMY_COUNT,datatype),PETSC_DUMMY\
)

#define MPI_Isend(buf,count, datatype,dest,tag,comm,request) \
(\
  PETSC_DUMMY_COUNT = count,PETSC_DUMMY = MPI_Isend(buf,PETSC_DUMMY_COUNT, datatype,dest,tag,comm,request),\
  isend_ct++,  TypeSize(isend_len,PETSC_DUMMY_COUNT,datatype),PETSC_DUMMY\
)

#define MPI_Startall_irecv(count,number,requests) \
(\
  PETSC_DUMMY = MPI_Startall(number,requests),\
  irecv_ct += (PetscLogDouble)(number),irecv_len += ((PetscLogDouble) ((count)*sizeof(PetscScalar))),PETSC_DUMMY\
)

#define MPI_Startall_isend(count,number,requests) \
(\
  PETSC_DUMMY = MPI_Startall(number,requests),\
  isend_ct += (PetscLogDouble)(number),isend_len += ((PetscLogDouble) ((count)*sizeof(PetscScalar))),PETSC_DUMMY\
)

#define MPI_Start_isend(count, requests) \
(\
  PETSC_DUMMY = MPI_Start(requests),\
  isend_ct++,isend_len += ((PetscLogDouble) ((count)*sizeof(PetscScalar))),PETSC_DUMMY\
)

#define MPI_Recv(buf,count, datatype,source,tag,comm,status) \
(\
  PETSC_DUMMY_COUNT = count,PETSC_DUMMY = MPI_Recv(buf,PETSC_DUMMY_COUNT, datatype,source,tag,comm,status),\
  recv_ct++,TypeSize(recv_len,PETSC_DUMMY_COUNT,datatype),PETSC_DUMMY\
)

#define MPI_Send(buf,count, datatype,dest,tag,comm) \
(\
  PETSC_DUMMY_COUNT = count,PETSC_DUMMY = MPI_Send(buf,PETSC_DUMMY_COUNT, datatype,dest,tag,comm),\
  send_ct++, TypeSize(send_len,PETSC_DUMMY_COUNT,datatype),PETSC_DUMMY\
)

#define MPI_Wait(request,status) \
(\
  wait_ct++,sum_of_waits_ct++,\
  MPI_Wait(request,status)\
)

#define MPI_Waitany(a,b,c,d) \
(\
  wait_any_ct++,sum_of_waits_ct++,\
  MPI_Waitany(a,b,c,d)\
)

#define MPI_Waitall(count,array_of_requests,array_of_statuses) \
(\
  PETSC_DUMMY_COUNT= count,wait_all_ct++,sum_of_waits_ct += (PetscLogDouble) (PETSC_DUMMY_COUNT),\
  MPI_Waitall(PETSC_DUMMY_COUNT,array_of_requests,array_of_statuses)\
)

#define MPI_Allreduce(sendbuf, recvbuf,count,datatype,op,comm) \
(\
  allreduce_ct++,MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm)\
)

#else

#define MPI_Startall_irecv(count,number,requests) \
(\
  MPI_Startall(number,requests)\
)

#define MPI_Startall_isend(count,number,requests) \
(\
  MPI_Startall(number,requests)\
)

#define MPI_Start_isend(count, requests) \
(\
  MPI_Start(requests)\
)

#endif /* !_petsc_mpi_uni && ! PETSC_HAVE_BROKEN_RECURSIVE_MACRO */

#else  /* ---Logging is turned off --------------------------------------------*/

#define PetscLogFlops(n) 0

/*
     With logging turned off, then MPE has to be turned off
*/
#define PetscLogMPEBegin()         0
#define PetscLogMPEDump(a)         0

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
#define PetscLogObjects(a)                  0
#define PetscLogActions(a)                  0
EXTERN PetscErrorCode PetscLogObjectState(PetscObject,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);

/* If PETSC_USE_LOG is NOT defined, these still need to be! */
#define MPI_Startall_irecv(count,number,requests) MPI_Startall(number,requests)
#define MPI_Startall_isend(count,number,requests) MPI_Startall(number,requests)
#define MPI_Start_isend(count,requests) MPI_Start(requests)

/* Creation and destruction functions */
#define StageLogCreate(stageLog)                     0
#define StageLogDestroy(stageLog)                    0
/* Registration functions */
#define StageLogRegister(stageLog, name, stage)      0
/* Runtime functions */
#define PetscLogGetStageLog(stageLog)                0
#define StageLogPush(stageLog, stage)                0
#define StageLogPop(stageLog)                        0
#define StageLogGetCurrent(stageLog, stage)          0
#define StageLogSetActive(stageLog, stage, active)   0
#define StageLogGetActive(stageLog, stage, active)   0
#define StageLogSetVisible(stageLog, stage, visible) 0
#define StageLogGetVisible(stageLog, stage, visible) 0
#define StageLogGetStage(stageLog, name, stage)      0

#endif   /* PETSC_USE_LOG */

extern PetscTruth PetscPreLoadingUsed;       /* true if we are or have done preloading */
extern PetscTruth PetscPreLoadingOn;         /* true if we are currently in a preloading calculation */

#define PreLoadBegin(flag,name) \
{\
  PetscTruth PreLoading = flag;\
  int        PreLoadMax,PreLoadIt,_stageNum,_3_ierr;\
  _3_ierr = PetscOptionsGetLogical(PETSC_NULL,"-preload",&PreLoading,PETSC_NULL);CHKERRQ(_3_ierr);\
  PreLoadMax = (int)(PreLoading);\
  PetscPreLoadingUsed = PreLoading ? PETSC_TRUE : PetscPreLoadingUsed;\
  for (PreLoadIt=0; PreLoadIt<=PreLoadMax; PreLoadIt++) {\
    PetscPreLoadingOn = PreLoading;\
    _3_ierr = PetscBarrier(PETSC_NULL);CHKERRQ(_3_ierr);\
    if (PreLoadIt>0) {\
      _3_ierr = PetscLogStageGetId(name,&_stageNum);CHKERRQ(_3_ierr);\
    } else {\
      _3_ierr = PetscLogStageRegister(&_stageNum,name);CHKERRQ(_3_ierr);\
    }\
    _3_ierr = PetscLogStageSetActive(_stageNum,(PetscTruth)(!PreLoadMax || PreLoadIt));\
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
    _3_ierr = PetscLogStageRegister(&_stageNum,name);CHKERRQ(_3_ierr);\
  }\
  _3_ierr = PetscLogStageSetActive(_stageNum,(PetscTruth)(!PreLoadMax || PreLoadIt));\
  _3_ierr = PetscLogStagePush(_stageNum);CHKERRQ(_3_ierr);

PETSC_EXTERN_CXX_END
#endif
