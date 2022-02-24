/*
    Defines profile/logging in PETSc.
*/

#if !defined(PETSCLOG_H)
#define PETSCLOG_H
#include <petscsys.h>
#include <petsctime.h>

/* General logging of information; different from event logging */
PETSC_EXTERN PetscErrorCode PetscInfo_Private(const char[],PetscObject,const char[],...) PETSC_ATTRIBUTE_FORMAT(3,4);
#if defined(PETSC_USE_INFO)
#define PetscInfo(A,...) PetscInfo_Private(PETSC_FUNCTION_NAME,((PetscObject)A),__VA_ARGS__)
#else
#define PetscInfo(A,...) 0
#endif

#define PetscInfo1(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo2(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo3(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo4(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo5(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo6(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo7(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo8(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)
#define PetscInfo9(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use PetscInfo() (since version 3.17)\"") PetscInfo(__VA_ARGS__)

/*E
    PetscInfoCommFlag - Describes the method by which to filter PetscInfo() by communicator size

    Used as an input for PetscInfoSetFilterCommSelf()

$   PETSC_INFO_COMM_ALL - Default uninitialized value. PetscInfo() will not filter based on communicator size (i.e. will
print for all communicators)
$   PETSC_INFO_COMM_NO_SELF - PetscInfo() will NOT print for communicators with size = 1 (i.e. *_COMM_SELF)
$   PETSC_INFO_COMM_ONLY_SELF - PetscInfo will ONLY print for communicators with size = 1

    Level: intermediate

.seealso: PetscInfo(), PetscInfoSetFromOptions(), PetscInfoSetFilterCommSelf()
E*/
typedef enum {
  PETSC_INFO_COMM_ALL = -1,
  PETSC_INFO_COMM_NO_SELF = 0,
  PETSC_INFO_COMM_ONLY_SELF = 1
} PetscInfoCommFlag;

PETSC_EXTERN const char * const PetscInfoCommFlags[];
PETSC_EXTERN PetscErrorCode PetscInfoDeactivateClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscInfoActivateClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscInfoEnabled(PetscClassId, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscInfoAllow(PetscBool);
PETSC_EXTERN PetscErrorCode PetscInfoSetFile(const char[],const char[]);
PETSC_EXTERN PetscErrorCode PetscInfoGetFile(char **,FILE **);
PETSC_EXTERN PetscErrorCode PetscInfoSetClasses(PetscBool,PetscInt,const char *const *);
PETSC_EXTERN PetscErrorCode PetscInfoGetClass(const char *, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscInfoGetInfo(PetscBool *,PetscBool *,PetscBool *,PetscBool *,PetscInfoCommFlag *);
PETSC_EXTERN PetscErrorCode PetscInfoProcessClass(const char[],PetscInt,PetscClassId[]);
PETSC_EXTERN PetscErrorCode PetscInfoSetFilterCommSelf(PetscInfoCommFlag);
PETSC_EXTERN PetscErrorCode PetscInfoSetFromOptions(PetscOptions);
PETSC_EXTERN PetscErrorCode PetscInfoDestroy(void);
PETSC_EXTERN PetscBool      PetscLogPrintInfo;  /* if true, indicates PetscInfo() is turned on */

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

.seealso: PetscLogStageRegister(), PetscLogStagePush(), PetscLogStagePop(), PetscLogEvent
M*/
typedef int PetscLogStage;

#define PETSC_EVENT  1311311
PETSC_EXTERN PetscLogEvent PETSC_LARGEST_EVENT;

/* Global flop counter */
PETSC_EXTERN PetscLogDouble petsc_TotalFlops;
PETSC_EXTERN PetscLogDouble petsc_tmp_flops;

/* We must make the following structures available to access the event
     activation flags in the PetscLogEventBegin/End() macros. These are not part of the PETSc public
     API and are not intended to be used by other parts of PETSc or by users.

     The code that manipulates these structures is in src/sys/logging/utils.
*/
typedef struct _n_PetscIntStack *PetscIntStack;

/* -----------------------------------------------------------------------------------------------------*/
/*
    PetscClassRegInfo, PetscClassPerfInfo - Each class has two data structures associated with it. The first has
       static information about it, the second collects statistics on how many objects of the class are created,
       how much memory they use, etc.

    PetscClassRegLog, PetscClassPerfLog - arrays of the PetscClassRegInfo and PetscClassPerfInfo for all classes.
*/
typedef struct  {
  char           *name;   /* The class name */
  PetscClassId   classid; /* The integer identifying this class */
} PetscClassRegInfo;

typedef struct {
  PetscClassId   id;           /* The integer identifying this class */
  int            creations;    /* The number of objects of this class created */
  int            destructions; /* The number of objects of this class destroyed */
  PetscLogDouble mem;          /* The total memory allocated by objects of this class */
  PetscLogDouble descMem;      /* The total memory allocated by descendents of these objects */
} PetscClassPerfInfo;

typedef struct _n_PetscClassRegLog *PetscClassRegLog;
struct _n_PetscClassRegLog {
  int               numClasses; /* The number of classes registered */
  int               maxClasses; /* The maximum number of classes */
  PetscClassRegInfo *classInfo; /* The structure for class information (classids are monotonicly increasing) */
};

typedef struct _n_PetscClassPerfLog *PetscClassPerfLog;
struct _n_PetscClassPerfLog {
  int                numClasses; /* The number of logging classes */
  int                maxClasses; /* The maximum number of classes */
  PetscClassPerfInfo *classInfo; /* The structure for class information (classids are monotonicly increasing) */
};
/* -----------------------------------------------------------------------------------------------------*/
/*
    PetscEventRegInfo, PetscEventPerfInfo - Each event has two data structures associated with it. The first has
       static information about it, the second collects statistics on how many times the event is used, how
       much time it takes, etc.

    PetscEventRegLog, PetscEventPerfLog - an array of all PetscEventRegInfo and PetscEventPerfInfo for all events. There is one
      of these for each stage.

*/
typedef struct {
  char         *name;         /* The name of this event */
  PetscClassId classid;       /* The class the event is associated with */
  PetscBool    collective;    /* Flag this event as collective */
#if defined (PETSC_HAVE_MPE)
  int          mpe_id_begin;  /* MPE IDs that define the event */
  int          mpe_id_end;
#endif
} PetscEventRegInfo;

typedef struct {
  int            id;            /* The integer identifying this event */
  PetscBool      active;        /* The flag to activate logging */
  PetscBool      visible;       /* The flag to print info in summary */
  int            depth;         /* The nesting depth of the event call */
  int            count;         /* The number of times this event was executed */
  PetscLogDouble flops, flops2, flopsTmp; /* The flops and flops^2 used in this event */
  PetscLogDouble time, time2, timeTmp;    /* The time and time^2 taken for this event */
  PetscLogDouble syncTime;                /* The synchronization barrier time */
  PetscLogDouble dof[8];        /* The number of degrees of freedom associated with this event */
  PetscLogDouble errors[8];     /* The errors (user-defined) associated with this event */
  PetscLogDouble numMessages;   /* The number of messages in this event */
  PetscLogDouble messageLength; /* The total message lengths in this event */
  PetscLogDouble numReductions; /* The number of reductions in this event */
  PetscLogDouble memIncrease;   /* How much the resident memory has increased in this event */
  PetscLogDouble mallocIncrease;/* How much the maximum malloced space has increased in this event */
  PetscLogDouble mallocSpace;   /* How much the space was malloced and kept during this event */
  PetscLogDouble mallocIncreaseEvent;  /* Maximum of the high water mark with in event minus memory available at the end of the event */
  #if defined(PETSC_HAVE_DEVICE)
  PetscLogDouble CpuToGpuCount; /* The total number of CPU to GPU copies */
  PetscLogDouble GpuToCpuCount; /* The total number of GPU to CPU copies */
  PetscLogDouble CpuToGpuSize;  /* The total size of CPU to GPU copies */
  PetscLogDouble GpuToCpuSize;  /* The total size of GPU to CPU copies */
  PetscLogDouble GpuFlops;      /* The flops done on a GPU in this event */
  PetscLogDouble GpuTime;       /* The time spent on a GPU in this event */
  #endif
} PetscEventPerfInfo;

typedef struct _n_PetscEventRegLog *PetscEventRegLog;
struct _n_PetscEventRegLog {
  int               numEvents;  /* The number of registered events */
  int               maxEvents;  /* The maximum number of events */
  PetscEventRegInfo *eventInfo; /* The registration information for each event */
};

typedef struct _n_PetscEventPerfLog *PetscEventPerfLog;
struct _n_PetscEventPerfLog {
  int                numEvents;  /* The number of logging events */
  int                maxEvents;  /* The maximum number of events */
  PetscEventPerfInfo *eventInfo; /* The performance information for each event */
};
/* ------------------------------------------------------------------------------------------------------------*/
/*
   PetscStageInfo - Contains all the information about a particular stage.

   PetscStageLog - An array of PetscStageInfo for each registered stage. There is a single one of these in the code.
*/
typedef struct _PetscStageInfo {
  char               *name;     /* The stage name */
  PetscBool          used;      /* The stage was pushed on this processor */
  PetscEventPerfInfo perfInfo;  /* The stage performance information */
  PetscEventPerfLog  eventLog;  /* The event information for this stage */
  PetscClassPerfLog  classLog;  /* The class information for this stage */
} PetscStageInfo;

typedef struct _n_PetscStageLog *PetscStageLog;
struct _n_PetscStageLog {
  int              numStages;   /* The number of registered stages */
  int              maxStages;   /* The maximum number of stages */
  PetscIntStack    stack;       /* The stack for active stages */
  int              curStage;    /* The current stage (only used in macros so we don't call PetscIntStackTop) */
  PetscStageInfo   *stageInfo;  /* The information for each stage */
  PetscEventRegLog eventLog;    /* The registered events */
  PetscClassRegLog classLog;    /* The registered classes */
};
/* -----------------------------------------------------------------------------------------------------*/

PETSC_EXTERN PetscErrorCode PetscLogObjectParent(PetscObject,PetscObject);
PETSC_EXTERN PetscErrorCode PetscLogObjectMemory(PetscObject,PetscLogDouble);

#if defined(PETSC_USE_LOG)  /* --- Logging is turned on --------------------------------*/
PETSC_EXTERN PetscStageLog petsc_stageLog;
PETSC_EXTERN PetscErrorCode PetscLogGetStageLog(PetscStageLog*);
PETSC_EXTERN PetscErrorCode PetscStageLogGetCurrent(PetscStageLog,int*);
PETSC_EXTERN PetscErrorCode PetscStageLogGetEventPerfLog(PetscStageLog,int,PetscEventPerfLog*);

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

/*@C
       PetscLogFlops - Log how many flops are performed in a calculation

   Input Parameter:
.   flops - the number of flops

   Notes:
     To limit the chance of integer overflow when multiplying by a constant, represent the constant as a double,
     not an integer. Use PetscLogFlops(4.0*n) not PetscLogFlops(4*n)

   Level: intermediate

.seealso: PetscLogView(), PetscLogGpuFlops()
@*/

static inline PetscErrorCode PetscLogFlops(PetscLogDouble n)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot log negative flops");
#endif
  petsc_TotalFlops += PETSC_FLOPS_PER_OP*n;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscGetFlops(PetscLogDouble *);

#if defined (PETSC_HAVE_MPE)
PETSC_EXTERN PetscErrorCode PetscLogMPEBegin(void);
PETSC_EXTERN PetscErrorCode PetscLogMPEDump(const char[]);
#endif

PETSC_EXTERN PetscErrorCode (*PetscLogPLB)(PetscLogEvent,int,PetscObject,PetscObject,PetscObject,PetscObject);
PETSC_EXTERN PetscErrorCode (*PetscLogPLE)(PetscLogEvent,int,PetscObject,PetscObject,PetscObject,PetscObject);
PETSC_EXTERN PetscErrorCode (*PetscLogPHC)(PetscObject);
PETSC_EXTERN PetscErrorCode (*PetscLogPHD)(PetscObject);

#define PetscLogObjectParents(p,n,d)  0;do{for (int _i=0; _i<(n); ++_i) CHKERRQ(PetscLogObjectParent((PetscObject)(p),(PetscObject)(d)[_i]));}while (0)
#define PetscLogObjectCreate(h)      ((PetscLogPHC) ? (*PetscLogPHC)((PetscObject)(h)) : 0)
#define PetscLogObjectDestroy(h)     ((PetscLogPHD) ? (*PetscLogPHD)((PetscObject)(h)) : 0)
PETSC_EXTERN PetscErrorCode PetscLogObjectState(PetscObject, const char[], ...) PETSC_ATTRIBUTE_FORMAT(2,3);

/* Initialization functions */
PETSC_EXTERN PetscErrorCode PetscLogDefaultBegin(void);
PETSC_EXTERN PetscErrorCode PetscLogAllBegin(void);
PETSC_EXTERN PetscErrorCode PetscLogNestedBegin(void);
PETSC_EXTERN PetscErrorCode PetscLogTraceBegin(FILE *);
PETSC_EXTERN PetscErrorCode PetscLogActions(PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogObjects(PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogSetThreshold(PetscLogDouble,PetscLogDouble*);
PETSC_EXTERN PetscErrorCode PetscLogSet(PetscErrorCode (*)(int, int, PetscObject, PetscObject, PetscObject, PetscObject),
                                        PetscErrorCode (*)(int, int, PetscObject, PetscObject, PetscObject, PetscObject));

/* Output functions */
PETSC_EXTERN PetscErrorCode PetscLogView(PetscViewer);
PETSC_EXTERN PetscErrorCode PetscLogViewFromOptions(void);
PETSC_EXTERN PetscErrorCode PetscLogDump(const char[]);

/* Status checking functions */
PETSC_EXTERN PetscErrorCode PetscLogIsActive(PetscBool*);

/* Stage functions */
PETSC_EXTERN PetscErrorCode PetscLogStageRegister(const char[],PetscLogStage*);
PETSC_EXTERN PetscErrorCode PetscLogStagePush(PetscLogStage);
PETSC_EXTERN PetscErrorCode PetscLogStagePop(void);
PETSC_EXTERN PetscErrorCode PetscLogStageSetActive(PetscLogStage,PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogStageGetActive(PetscLogStage,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscLogStageSetVisible(PetscLogStage,PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogStageGetVisible(PetscLogStage,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscLogStageGetId(const char[],PetscLogStage*);

/* Event functions */
PETSC_EXTERN PetscErrorCode PetscLogEventRegister(const char[],PetscClassId,PetscLogEvent*);
PETSC_EXTERN PetscErrorCode PetscLogEventSetCollective(PetscLogEvent,PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogEventIncludeClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscLogEventExcludeClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscLogEventActivate(PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscLogEventDeactivate(PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscLogEventDeactivatePush(PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscLogEventDeactivatePop(PetscLogEvent);
PETSC_EXTERN PetscErrorCode PetscLogEventSetActiveAll(PetscLogEvent,PetscBool);
PETSC_EXTERN PetscErrorCode PetscLogEventActivateClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscLogEventDeactivateClass(PetscClassId);
PETSC_EXTERN PetscErrorCode PetscLogEventGetId(const char[],PetscLogEvent*);
PETSC_EXTERN PetscErrorCode PetscLogEventGetPerfInfo(int,PetscLogEvent,PetscEventPerfInfo*);
PETSC_EXTERN PetscErrorCode PetscLogEventSetDof(PetscLogEvent, PetscInt, PetscLogDouble);
PETSC_EXTERN PetscErrorCode PetscLogEventSetError(PetscLogEvent, PetscInt, PetscLogDouble);

/* Global counters */
PETSC_EXTERN PetscLogDouble petsc_irecv_ct;
PETSC_EXTERN PetscLogDouble petsc_isend_ct;
PETSC_EXTERN PetscLogDouble petsc_recv_ct;
PETSC_EXTERN PetscLogDouble petsc_send_ct;
PETSC_EXTERN PetscLogDouble petsc_irecv_len;
PETSC_EXTERN PetscLogDouble petsc_isend_len;
PETSC_EXTERN PetscLogDouble petsc_recv_len;
PETSC_EXTERN PetscLogDouble petsc_send_len;
PETSC_EXTERN PetscLogDouble petsc_allreduce_ct;
PETSC_EXTERN PetscLogDouble petsc_gather_ct;
PETSC_EXTERN PetscLogDouble petsc_scatter_ct;
PETSC_EXTERN PetscLogDouble petsc_wait_ct;
PETSC_EXTERN PetscLogDouble petsc_wait_any_ct;
PETSC_EXTERN PetscLogDouble petsc_wait_all_ct;
PETSC_EXTERN PetscLogDouble petsc_sum_of_waits_ct;

PETSC_EXTERN PetscBool      PetscLogMemory;

PETSC_EXTERN PetscBool PetscLogSyncOn;  /* true if logging synchronization is enabled */
PETSC_EXTERN PetscErrorCode PetscLogEventSynchronize(PetscLogEvent, MPI_Comm);

#define PetscLogEventSync(e,comm) \
  (((PetscLogPLB && petsc_stageLog->stageInfo[petsc_stageLog->curStage].perfInfo.active && petsc_stageLog->stageInfo[petsc_stageLog->curStage].eventLog->eventInfo[e].active) ? \
    PetscLogEventSynchronize((e),(comm)) : 0))

#define PetscLogEventBegin(e,o1,o2,o3,o4) \
  (((PetscLogPLB && petsc_stageLog->stageInfo[petsc_stageLog->curStage].perfInfo.active && petsc_stageLog->stageInfo[petsc_stageLog->curStage].eventLog->eventInfo[e].active) ? \
    (*PetscLogPLB)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4)) : 0))

#define PetscLogEventEnd(e,o1,o2,o3,o4) \
  (((PetscLogPLE && petsc_stageLog->stageInfo[petsc_stageLog->curStage].perfInfo.active && petsc_stageLog->stageInfo[petsc_stageLog->curStage].eventLog->eventInfo[e].active) ? \
    (*PetscLogPLE)((e),0,(PetscObject)(o1),(PetscObject)(o2),(PetscObject)(o3),(PetscObject)(o4)) : 0))

PETSC_EXTERN PetscErrorCode PetscLogEventGetFlops(PetscLogEvent,PetscLogDouble*);
PETSC_EXTERN PetscErrorCode PetscLogEventZeroFlops(PetscLogEvent);

/*
     These are used internally in the PETSc routines to keep a count of MPI messages and
   their sizes.

     This does not work for MPI-Uni because our include/petsc/mpiuni/mpi.h file
   uses macros to defined the MPI operations.

     It does not work correctly from HP-UX because it processes the
   macros in a way that sometimes it double counts, hence
   PETSC_HAVE_BROKEN_RECURSIVE_MACRO

     It does not work with Windows because winmpich lacks MPI_Type_size()
*/
#if !defined(MPIUNI_H) && !defined(PETSC_HAVE_BROKEN_RECURSIVE_MACRO) && !defined (PETSC_HAVE_MPI_MISSING_TYPESIZE)
/*
   Logging of MPI activities
*/
static inline PetscErrorCode PetscMPITypeSize(PetscInt count,MPI_Datatype type,PetscLogDouble *length)
{
  PetscMPIInt typesize;

  if (type == MPI_DATATYPE_NULL) return 0;
  CHKERRMPI(MPI_Type_size(type,&typesize));
  *length += (PetscLogDouble) (count*typesize);
  return 0;
}

static inline PetscErrorCode PetscMPITypeSizeComm(MPI_Comm comm,const PetscMPIInt *counts,MPI_Datatype type,PetscLogDouble *length)
{
  PetscMPIInt    typesize,size,p;

  if (type == MPI_DATATYPE_NULL) return 0;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Type_size(type,&typesize));
  for (p=0; p<size; ++p) *length += (PetscLogDouble)(counts[p]*typesize);
  return 0;
}

static inline PetscErrorCode PetscMPITypeSizeCount(PetscInt n,const PetscMPIInt *counts,MPI_Datatype type,PetscLogDouble *length)
{
  PetscMPIInt typesize,p;

  if (type == MPI_DATATYPE_NULL) return 0;
  CHKERRMPI(MPI_Type_size(type,&typesize));
  for (p=0; p<n; ++p) *length += (PetscLogDouble)(counts[p]*typesize);
  return 0;
}

/*
    Returns 1 if the communicator is parallel else zero
*/
static inline int PetscMPIParallelComm(MPI_Comm comm)
{
  PetscMPIInt size; MPI_Comm_size(comm,&size); return size > 1;
}

#define MPI_Irecv(buf,count,datatype,source,tag,comm,request) \
  ((petsc_irecv_ct++,0) || PetscMPITypeSize((count),(datatype),&(petsc_irecv_len)) || MPI_Irecv((buf),(count),(datatype),(source),(tag),(comm),(request)))

#define MPI_Irecv_c(buf,count,datatype,source,tag,comm,request) \
  ((petsc_irecv_ct++,0) || PetscMPITypeSize((count),(datatype),&(petsc_irecv_len)) || MPI_Irecv_c((buf),(count),(datatype),(source),(tag),(comm),(request)))

#define MPI_Isend(buf,count,datatype,dest,tag,comm,request) \
  ((petsc_isend_ct++,0) || PetscMPITypeSize((count),(datatype),&(petsc_isend_len)) || MPI_Isend((buf),(count),(datatype),(dest),(tag),(comm),(request)))

#define MPI_Isend_c(buf,count,datatype,dest,tag,comm,request) \
  ((petsc_isend_ct++,0) || PetscMPITypeSize((count),(datatype),&(petsc_isend_len)) || MPI_Isend_c((buf),(count),(datatype),(dest),(tag),(comm),(request)))

#define MPI_Startall_irecv(count,datatype,number,requests) \
  ((petsc_irecv_ct += (PetscLogDouble)(number),0) || PetscMPITypeSize((count),(datatype),&(petsc_irecv_len)) || ((number) && MPI_Startall((number),(requests))))

#define MPI_Startall_isend(count,datatype,number,requests) \
  ((petsc_isend_ct += (PetscLogDouble)(number),0) || PetscMPITypeSize((count),(datatype),&(petsc_isend_len)) || ((number) && MPI_Startall((number),(requests))))

#define MPI_Start_isend(count,datatype,requests) \
  ((petsc_isend_ct++,0) || PetscMPITypeSize((count),(datatype),(&petsc_isend_len)) || MPI_Start((requests)))

#define MPI_Recv(buf,count,datatype,source,tag,comm,status) \
  ((petsc_recv_ct++,0) || PetscMPITypeSize((count),(datatype),(&petsc_recv_len)) || MPI_Recv((buf),(count),(datatype),(source),(tag),(comm),(status)))

#define MPI_Recv_c(buf,count,datatype,source,tag,comm,status) \
  ((petsc_recv_ct++,0) || PetscMPITypeSize((count),(datatype),(&petsc_recv_len)) || MPI_Recv_c((buf),(count),(datatype),(source),(tag),(comm),(status)))

#define MPI_Send(buf,count,datatype,dest,tag,comm) \
  ((petsc_send_ct++,0) || PetscMPITypeSize((count),(datatype),(&petsc_send_len)) || MPI_Send((buf),(count),(datatype),(dest),(tag),(comm)))

#define MPI_Send_c(buf,count,datatype,dest,tag,comm) \
  ((petsc_send_ct++,0) || PetscMPITypeSize((count),(datatype),(&petsc_send_len)) || MPI_Send_c((buf),(count),(datatype),(dest),(tag),(comm)))

#define MPI_Wait(request,status) \
  ((petsc_wait_ct++,petsc_sum_of_waits_ct++,0) || MPI_Wait((request),(status)))

#define MPI_Waitany(a,b,c,d) \
  ((petsc_wait_any_ct++,petsc_sum_of_waits_ct++,0) || MPI_Waitany((a),(b),(c),(d)))

#define MPI_Waitall(count,array_of_requests,array_of_statuses) \
  ((petsc_wait_all_ct++,petsc_sum_of_waits_ct += (PetscLogDouble) (count),0) || MPI_Waitall((count),(array_of_requests),(array_of_statuses)))

#define MPI_Allreduce(sendbuf,recvbuf,count,datatype,op,comm) \
  (petsc_allreduce_ct += PetscMPIParallelComm((comm)),MPI_Allreduce((sendbuf),(recvbuf),(count),(datatype),(op),(comm)))

#define MPI_Bcast(buffer,count,datatype,root,comm) \
  ((petsc_allreduce_ct += PetscMPIParallelComm((comm)),0) || MPI_Bcast((buffer),(count),(datatype),(root),(comm)))

#define MPI_Reduce_scatter_block(sendbuf,recvbuf,recvcount,datatype,op,comm) \
  ((petsc_allreduce_ct += PetscMPIParallelComm((comm)),0) || MPI_Reduce_scatter_block((sendbuf),(recvbuf),(recvcount),(datatype),(op),(comm)))

#define MPI_Alltoall(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,comm) \
  ((petsc_allreduce_ct += PetscMPIParallelComm((comm)),0) || PetscMPITypeSize((sendcount),(sendtype),(&petsc_send_len)) || MPI_Alltoall((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(recvtype),(comm)))

#define MPI_Alltoallv(sendbuf,sendcnts,sdispls,sendtype,recvbuf,recvcnts,rdispls,recvtype,comm) \
  ((petsc_allreduce_ct += PetscMPIParallelComm((comm)),0) || PetscMPITypeSizeComm((comm),(sendcnts),(sendtype),(&petsc_send_len)) || MPI_Alltoallv((sendbuf),(sendcnts),(sdispls),(sendtype),(recvbuf),(recvcnts),(rdispls),(recvtype),(comm)))

#define MPI_Allgather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,comm) \
  ((petsc_gather_ct += PetscMPIParallelComm((comm)),0) || MPI_Allgather((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(recvtype),(comm)))

#define MPI_Allgatherv(sendbuf,sendcount,sendtype,recvbuf,recvcount,displs,recvtype,comm) \
  ((petsc_gather_ct += PetscMPIParallelComm((comm)),0) || MPI_Allgatherv((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(displs),(recvtype),(comm)))

#define MPI_Gather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,root,comm) \
  ((petsc_gather_ct++,0) || PetscMPITypeSize((sendcount),(sendtype),(&petsc_send_len)) || MPI_Gather((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(recvtype),(root),(comm)))

#define MPI_Gatherv(sendbuf,sendcount,sendtype,recvbuf,recvcount,displs,recvtype,root,comm) \
  ((petsc_gather_ct++,0) || PetscMPITypeSize((sendcount),(sendtype),(&petsc_send_len)) || MPI_Gatherv((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(displs),(recvtype),(root),(comm)))

#define MPI_Scatter(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,root,comm) \
  ((petsc_scatter_ct++,0) || PetscMPITypeSize((recvcount),(recvtype),(&petsc_recv_len)) || MPI_Scatter((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(recvtype),(root),(comm)))

#define MPI_Scatterv(sendbuf,sendcount,displs,sendtype,recvbuf,recvcount,recvtype,root,comm) \
  ((petsc_scatter_ct++,0) || PetscMPITypeSize((recvcount),(recvtype),(&petsc_recv_len)) || MPI_Scatterv((sendbuf),(sendcount),(displs),(sendtype),(recvbuf),(recvcount),(recvtype),(root),(comm)))

#define MPI_Ialltoall(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,comm,request) \
  ((petsc_allreduce_ct += PetscMPIParallelComm((comm)),0) || PetscMPITypeSize((sendcount),(sendtype),(&petsc_send_len)) || MPI_Ialltoall((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(recvtype),(comm),(request)))

#define MPI_Ialltoallv(sendbuf,sendcnts,sdispls,sendtype,recvbuf,recvcnts,rdispls,recvtype,comm,request) \
  ((petsc_allreduce_ct += PetscMPIParallelComm((comm)),0) || PetscMPITypeSizeComm((comm),(sendcnts),(sendtype),(&petsc_send_len)) || MPI_Ialltoallv((sendbuf),(sendcnts),(sdispls),(sendtype),(recvbuf),(recvcnts),(rdispls),(recvtype),(comm),(request)))

#define MPI_Iallgather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,comm,request) \
  ((petsc_gather_ct += PetscMPIParallelComm((comm)),0) || MPI_Iallgather((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(recvtype),(comm),(request)))

#define MPI_Iallgatherv(sendbuf,sendcount,sendtype,recvbuf,recvcount,displs,recvtype,comm,request) \
  ((petsc_gather_ct += PetscMPIParallelComm((comm)),0) || MPI_Iallgatherv((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(displs),(recvtype),(comm),(request)))

#define MPI_Igather(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,root,comm,request) \
  ((petsc_gather_ct++,0) || PetscMPITypeSize((sendcount),(sendtype),(&petsc_send_len)) || MPI_Igather((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(recvtype),(root),(comm),(request)))

#define MPI_Igatherv(sendbuf,sendcount,sendtype,recvbuf,recvcount,displs,recvtype,root,comm,request) \
  ((petsc_gather_ct++,0) || PetscMPITypeSize((sendcount),(sendtype),(&petsc_send_len)) || MPI_Igatherv((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(displs),(recvtype),(root),(comm),(request)))

#define MPI_Iscatter(sendbuf,sendcount,sendtype,recvbuf,recvcount,recvtype,root,comm,request) \
  ((petsc_scatter_ct++,0) || PetscMPITypeSize((recvcount),(recvtype),(&petsc_recv_len)) || MPI_Iscatter((sendbuf),(sendcount),(sendtype),(recvbuf),(recvcount),(recvtype),(root),(comm),(request)))

#define MPI_Iscatterv(sendbuf,sendcount,displs,sendtype,recvbuf,recvcount,recvtype,root,comm,request) \
  ((petsc_scatter_ct++,0) || PetscMPITypeSize((recvcount),(recvtype),(&petsc_recv_len)) || MPI_Iscatterv((sendbuf),(sendcount),(displs),(sendtype),(recvbuf),(recvcount),(recvtype),(root),(comm),(request)))

#else

#define MPI_Startall_irecv(count,datatype,number,requests) \
  ((number) && MPI_Startall((number),(requests)))

#define MPI_Startall_isend(count,datatype,number,requests) \
  ((number) && MPI_Startall((number),(requests)))

#define MPI_Start_isend(count,datatype,requests) \
  (MPI_Start((requests)))

#endif /* !MPIUNI_H && ! PETSC_HAVE_BROKEN_RECURSIVE_MACRO */

#else  /* ---Logging is turned off --------------------------------------------*/

#define PetscLogMemory                     PETSC_FALSE

#define PetscLogFlops(n)                   0
#define PetscGetFlops(a)                   (*(a) = 0.0,0)

#define PetscLogStageRegister(a,b)         0
#define PetscLogStagePush(a)               0
#define PetscLogStagePop()                 0
#define PetscLogStageSetActive(a,b)        0
#define PetscLogStageGetActive(a,b)        0
#define PetscLogStageGetVisible(a,b)       0
#define PetscLogStageSetVisible(a,b)       0
#define PetscLogStageGetId(a,b)            (*(b)=0,0)

#define PetscLogEventRegister(a,b,c)       0
#define PetscLogEventSetCollective(a,b)    0
#define PetscLogEventIncludeClass(a)       0
#define PetscLogEventExcludeClass(a)       0
#define PetscLogEventActivate(a)           0
#define PetscLogEventDeactivate(a)         0
#define PetscLogEventDeactivatePush(a)     0
#define PetscLogEventDeactivatePop(a)      0
#define PetscLogEventActivateClass(a)      0
#define PetscLogEventDeactivateClass(a)    0
#define PetscLogEventSetActiveAll(a,b)     0
#define PetscLogEventGetId(a,b)            (*(b)=0,0)
#define PetscLogEventGetPerfInfo(a,b,c)    0
#define PetscLogEventSetDof(a,b,c)         0
#define PetscLogEventSetError(a,b,c)       0

#define PetscLogPLB                        0
#define PetscLogPLE                        0
#define PetscLogPHC                        0
#define PetscLogPHD                        0

#define PetscLogObjectParents(p,n,c)       0
#define PetscLogObjectCreate(h)            0
#define PetscLogObjectDestroy(h)           0
PETSC_EXTERN PetscErrorCode PetscLogObjectState(PetscObject,const char[],...) PETSC_ATTRIBUTE_FORMAT(2,3);

#define PetscLogDefaultBegin()             0
#define PetscLogAllBegin()                 0
#define PetscLogNestedBegin()              0
#define PetscLogTraceBegin(file)           0
#define PetscLogActions(a)                 0
#define PetscLogObjects(a)                 0
#define PetscLogSetThreshold(a,b)          0
#define PetscLogSet(lb,le)                 0
#define PetscLogIsActive(flag)             (*(flag) = PETSC_FALSE,0)

#define PetscLogView(viewer)               0
#define PetscLogViewFromOptions()          0
#define PetscLogDump(c)                    0

#define PetscLogEventSync(e,comm)          0
#define PetscLogEventBegin(e,o1,o2,o3,o4)  0
#define PetscLogEventEnd(e,o1,o2,o3,o4)    0

/* If PETSC_USE_LOG is NOT defined, these still need to be! */
#define MPI_Startall_irecv(count,datatype,number,requests) ((number) && MPI_Startall(number,requests))
#define MPI_Startall_isend(count,datatype,number,requests) ((number) && MPI_Startall(number,requests))
#define MPI_Start_isend(count,datatype,requests)           MPI_Start(requests)

#endif   /* PETSC_USE_LOG */

#if defined (PETSC_USE_LOG) && defined(PETSC_HAVE_DEVICE)

/* Global GPU counters */
PETSC_EXTERN PetscLogDouble petsc_ctog_ct;
PETSC_EXTERN PetscLogDouble petsc_gtoc_ct;
PETSC_EXTERN PetscLogDouble petsc_ctog_sz;
PETSC_EXTERN PetscLogDouble petsc_gtoc_sz;
PETSC_EXTERN PetscLogDouble petsc_ctog_ct_scalar;
PETSC_EXTERN PetscLogDouble petsc_gtoc_ct_scalar;
PETSC_EXTERN PetscLogDouble petsc_ctog_sz_scalar;
PETSC_EXTERN PetscLogDouble petsc_gtoc_sz_scalar;
PETSC_EXTERN PetscLogDouble petsc_gflops;
PETSC_EXTERN PetscLogDouble petsc_gtime;

static inline PetscErrorCode PetscLogCpuToGpu(PetscLogDouble size)
{
  PetscFunctionBegin;
  petsc_ctog_ct += 1;
  petsc_ctog_sz += size;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscLogGpuToCpu(PetscLogDouble size)
{
  PetscFunctionBegin;
  petsc_gtoc_ct += 1;
  petsc_gtoc_sz += size;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscLogCpuToGpuScalar(PetscLogDouble size)
{
  PetscFunctionBegin;
  petsc_ctog_ct_scalar += 1;
  petsc_ctog_sz_scalar += size;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscLogGpuToCpuScalar(PetscLogDouble size)
{
  PetscFunctionBegin;
  petsc_gtoc_ct_scalar += 1;
  petsc_gtoc_sz_scalar += size;
  PetscFunctionReturn(0);
}

/*@C
       PetscLogGpuFlops - Log how many flops are performed in a calculation on the device

   Input Parameter:
.   flops - the number of flops

   Notes:
     To limit the chance of integer overflow when multiplying by a constant, represent the constant as a double,
     not an integer. Use PetscLogFlops(4.0*n) not PetscLogFlops(4*n)

   Level: intermediate

.seealso: PetscLogView(), PetscLogFlops(), PetscLogGpuTimeBegin(), PetscLogGpuTimeEnd()
@*/
static inline PetscErrorCode PetscLogGpuFlops(PetscLogDouble n)
{
  PetscFunctionBegin;
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Cannot log negative flops");
  petsc_TotalFlops += PETSC_FLOPS_PER_OP*n;
  petsc_gflops += PETSC_FLOPS_PER_OP*n;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PetscLogGpuTimeAdd(PetscLogDouble t)
{
  PetscFunctionBegin;
  petsc_gtime += t;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscLogGpuTimeBegin(void);
PETSC_EXTERN PetscErrorCode PetscLogGpuTimeEnd(void);

#else

#define PetscLogCpuToGpu(a)                0
#define PetscLogGpuToCpu(a)                0
#define PetscLogCpuToGpuScalar(a)          0
#define PetscLogGpuToCpuScalar(a)          0
#define PetscLogGpuFlops(a)                0
#define PetscLogGpuTimeAdd(a)              0
#define PetscLogGpuTimeBegin()             0
#define PetscLogGpuTimeEnd()               0

#endif /* PETSC_USE_LOG && PETSC_HAVE_DEVICE */

#define PetscPreLoadBegin(flag,name) \
do {\
  PetscBool      PetscPreLoading = flag;\
  int            PetscPreLoadMax,PetscPreLoadIt;\
  PetscLogStage  _stageNum;\
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-preload",&PetscPreLoading,NULL));     \
  PetscPreLoadMax = (int)(PetscPreLoading);\
  PetscPreLoadingUsed = PetscPreLoading ? PETSC_TRUE : PetscPreLoadingUsed;\
  for (PetscPreLoadIt=0; PetscPreLoadIt<=PetscPreLoadMax; PetscPreLoadIt++) {\
    PetscPreLoadingOn = PetscPreLoading;\
    CHKERRQ(PetscBarrier(NULL));\
    if (PetscPreLoadIt>0) CHKERRQ(PetscLogStageGetId(name,&_stageNum));\
    else CHKERRQ(PetscLogStageRegister(name,&_stageNum));\
    CHKERRQ(PetscLogStageSetActive(_stageNum,(PetscBool)(!PetscPreLoadMax || PetscPreLoadIt)));\
    CHKERRQ(PetscLogStagePush(_stageNum));

#define PetscPreLoadEnd() \
    CHKERRQ(PetscLogStagePop());\
    PetscPreLoading = PETSC_FALSE;\
  }\
} while (0)

#define PetscPreLoadStage(name) do {                                                           \
    CHKERRQ(PetscLogStagePop());                                                               \
    if (PetscPreLoadIt>0)   CHKERRQ(PetscLogStageGetId(name,&_stageNum));                      \
    else CHKERRQ(PetscLogStageRegister(name,&_stageNum));                                      \
    CHKERRQ(PetscLogStageSetActive(_stageNum,(PetscBool)(!PetscPreLoadMax || PetscPreLoadIt))); \
    CHKERRQ(PetscLogStagePush(_stageNum));                                                     \
  } while (0)

/* some vars for logging */
PETSC_EXTERN PetscBool PetscPreLoadingUsed;       /* true if we are or have done preloading */
PETSC_EXTERN PetscBool PetscPreLoadingOn;         /* true if we are currently in a preloading calculation */

#endif
