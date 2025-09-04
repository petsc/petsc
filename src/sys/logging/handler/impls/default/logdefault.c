#include <petscviewer.h>
#include <petscdevice.h>
#include <petsc/private/logimpl.h> /*I "petscsys.h" I*/
#include <petsc/private/loghandlerimpl.h>
#include <petsc/private/deviceimpl.h>
#include <petscconfiginfo.h>
#include <petscmachineinfo.h>
#include "logdefault.h"

static PetscErrorCode PetscEventPerfInfoInit(PetscEventPerfInfo *eventInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(eventInfo, sizeof(*eventInfo)));
  eventInfo->visible   = PETSC_TRUE;
  eventInfo->id        = -1;
  eventInfo->dof[0]    = -1.0;
  eventInfo->dof[1]    = -1.0;
  eventInfo->dof[2]    = -1.0;
  eventInfo->dof[3]    = -1.0;
  eventInfo->dof[4]    = -1.0;
  eventInfo->dof[5]    = -1.0;
  eventInfo->dof[6]    = -1.0;
  eventInfo->dof[7]    = -1.0;
  eventInfo->errors[0] = -1.0;
  eventInfo->errors[1] = -1.0;
  eventInfo->errors[2] = -1.0;
  eventInfo->errors[3] = -1.0;
  eventInfo->errors[4] = -1.0;
  eventInfo->errors[5] = -1.0;
  eventInfo->errors[6] = -1.0;
  eventInfo->errors[7] = -1.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventPerfInfoTic_Internal(PetscEventPerfInfo *eventInfo, PetscLogDouble time, PetscBool logMemory, int event, PetscBool resume)
{
  PetscFunctionBegin;
  if (resume) {
    eventInfo->timeTmp -= time;
    eventInfo->flopsTmp -= petsc_TotalFlops_th;
  } else {
    eventInfo->timeTmp  = -time;
    eventInfo->flopsTmp = -petsc_TotalFlops_th;
  }
  eventInfo->numMessages -= petsc_irecv_ct_th + petsc_isend_ct_th + petsc_recv_ct_th + petsc_send_ct_th;
  eventInfo->messageLength -= petsc_irecv_len_th + petsc_isend_len_th + petsc_recv_len_th + petsc_send_len_th;
  eventInfo->numReductions -= petsc_allreduce_ct_th + petsc_gather_ct_th + petsc_scatter_ct_th;
#if defined(PETSC_HAVE_DEVICE)
  eventInfo->CpuToGpuCount -= petsc_ctog_ct_th;
  eventInfo->GpuToCpuCount -= petsc_gtoc_ct_th;
  eventInfo->CpuToGpuSize -= petsc_ctog_sz_th;
  eventInfo->GpuToCpuSize -= petsc_gtoc_sz_th;
  eventInfo->GpuFlops -= petsc_gflops_th;
  eventInfo->GpuTime -= petsc_gtime;
#endif
  if (logMemory) {
    PetscLogDouble usage;
    PetscCall(PetscMemoryGetCurrentUsage(&usage));
    eventInfo->memIncrease -= usage;
    PetscCall(PetscMallocGetCurrentUsage(&usage));
    eventInfo->mallocSpace -= usage;
    PetscCall(PetscMallocGetMaximumUsage(&usage));
    eventInfo->mallocIncrease -= usage;
    PetscCall(PetscMallocPushMaximumUsage(event));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventPerfInfoTic(PetscEventPerfInfo *eventInfo, PetscLogDouble time, PetscBool logMemory, int event)
{
  PetscFunctionBegin;
  PetscCall(PetscEventPerfInfoTic_Internal(eventInfo, time, logMemory, event, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventPerfInfoResume(PetscEventPerfInfo *eventInfo, PetscLogDouble time, PetscBool logMemory, int event)
{
  PetscFunctionBegin;
  PetscCall(PetscEventPerfInfoTic_Internal(eventInfo, time, logMemory, event, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventPerfInfoToc_Internal(PetscEventPerfInfo *eventInfo, PetscLogDouble time, PetscBool logMemory, int event, PetscBool pause)
{
  PetscFunctionBegin;
  eventInfo->timeTmp += time;
  eventInfo->flopsTmp += petsc_TotalFlops_th;
  if (!pause) {
    eventInfo->time += eventInfo->timeTmp;
    eventInfo->time2 += eventInfo->timeTmp * eventInfo->timeTmp;
    eventInfo->flops += eventInfo->flopsTmp;
    eventInfo->flops2 += eventInfo->flopsTmp * eventInfo->flopsTmp;
  }
  eventInfo->numMessages += petsc_irecv_ct_th + petsc_isend_ct_th + petsc_recv_ct_th + petsc_send_ct_th;
  eventInfo->messageLength += petsc_irecv_len_th + petsc_isend_len_th + petsc_recv_len + petsc_send_len_th;
  eventInfo->numReductions += petsc_allreduce_ct_th + petsc_gather_ct_th + petsc_scatter_ct_th;
#if defined(PETSC_HAVE_DEVICE)
  eventInfo->CpuToGpuCount += petsc_ctog_ct_th;
  eventInfo->GpuToCpuCount += petsc_gtoc_ct_th;
  eventInfo->CpuToGpuSize += petsc_ctog_sz_th;
  eventInfo->GpuToCpuSize += petsc_gtoc_sz_th;
  eventInfo->GpuFlops += petsc_gflops_th;
  eventInfo->GpuTime += petsc_gtime;
#endif
  if (logMemory) {
    PetscLogDouble usage, musage;
    PetscCall(PetscMemoryGetCurrentUsage(&usage)); /* the comments below match the column labels printed in PetscLogView_Default() */
    eventInfo->memIncrease += usage;               /* RMI */
    PetscCall(PetscMallocGetCurrentUsage(&usage));
    eventInfo->mallocSpace += usage; /* Malloc */
    PetscCall(PetscMallocPopMaximumUsage(event, &musage));
    eventInfo->mallocIncreaseEvent = PetscMax(musage - usage, eventInfo->mallocIncreaseEvent); /* EMalloc */
    PetscCall(PetscMallocGetMaximumUsage(&usage));
    eventInfo->mallocIncrease += usage; /* MMalloc */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventPerfInfoToc(PetscEventPerfInfo *eventInfo, PetscLogDouble time, PetscBool logMemory, int event)
{
  PetscFunctionBegin;
  PetscCall(PetscEventPerfInfoToc_Internal(eventInfo, time, logMemory, event, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventPerfInfoPause(PetscEventPerfInfo *eventInfo, PetscLogDouble time, PetscBool logMemory, int event)
{
  PetscFunctionBegin;
  PetscCall(PetscEventPerfInfoToc_Internal(eventInfo, time, logMemory, event, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventPerfInfoAdd_Internal(const PetscEventPerfInfo *eventInfo, PetscEventPerfInfo *outInfo)
{
  PetscFunctionBegin;
  outInfo->count += eventInfo->count;
  outInfo->time += eventInfo->time;
  outInfo->time2 += eventInfo->time2;
  outInfo->flops += eventInfo->flops;
  outInfo->flops2 += eventInfo->flops2;
  outInfo->numMessages += eventInfo->numMessages;
  outInfo->messageLength += eventInfo->messageLength;
  outInfo->numReductions += eventInfo->numReductions;
#if defined(PETSC_HAVE_DEVICE)
  outInfo->CpuToGpuCount += eventInfo->CpuToGpuCount;
  outInfo->GpuToCpuCount += eventInfo->GpuToCpuCount;
  outInfo->CpuToGpuSize += eventInfo->CpuToGpuSize;
  outInfo->GpuToCpuSize += eventInfo->GpuToCpuSize;
  outInfo->GpuFlops += eventInfo->GpuFlops;
  outInfo->GpuTime += eventInfo->GpuTime;
#endif
  outInfo->memIncrease += eventInfo->memIncrease;
  outInfo->mallocSpace += eventInfo->mallocSpace;
  outInfo->mallocIncreaseEvent += eventInfo->mallocIncreaseEvent;
  outInfo->mallocIncrease += eventInfo->mallocIncrease;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_LOG_RESIZABLE_ARRAY(EventPerfArray, PetscEventPerfInfo, PetscLogEvent, PetscEventPerfInfoInit, NULL, NULL)

/* --- PetscClassPerf --- */

typedef struct {
  int            creations;    /* The number of objects of this class created */
  int            destructions; /* The number of objects of this class destroyed */
  PetscLogDouble mem;          /* The total memory allocated by objects of this class; this is completely wrong and should possibly be removed */
  PetscLogDouble descMem;      /* The total memory allocated by descendents of these objects; this is completely wrong and should possibly be removed */
} PetscClassPerf;

static PetscErrorCode PetscClassPerfInit(PetscClassPerf *classInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(classInfo, sizeof(*classInfo)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_LOG_RESIZABLE_ARRAY(ClassPerfArray, PetscClassPerf, PetscLogClass, PetscClassPerfInit, NULL, NULL)

/* --- PetscStagePerf --- */

typedef struct _PetscStagePerf {
  PetscBool              used;     /* The stage was pushed on this processor */
  PetscEventPerfInfo     perfInfo; /* The stage performance information */
  PetscLogEventPerfArray eventLog; /* The event information for this stage */
  PetscLogClassPerfArray classLog; /* The class information for this stage */
} PetscStagePerf;

static PetscErrorCode PetscStageInfoInit(PetscStagePerf *stageInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscMemzero(stageInfo, sizeof(*stageInfo)));
  PetscCall(PetscLogEventPerfArrayCreate(128, &stageInfo->eventLog));
  PetscCall(PetscLogClassPerfArrayCreate(128, &stageInfo->classLog));
  PetscCall(PetscEventPerfInfoInit(&stageInfo->perfInfo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscStageInfoReset(PetscStagePerf *stageInfo)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventPerfArrayDestroy(&stageInfo->eventLog));
  PetscCall(PetscLogClassPerfArrayDestroy(&stageInfo->classLog));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_LOG_RESIZABLE_ARRAY(StageInfoArray, PetscStagePerf, PetscLogStage, PetscStageInfoInit, PetscStageInfoReset, NULL)

/* --- Action --- */

/* The structure for action logging */
typedef enum {
  PETSC_LOG_ACTION_CREATE,
  PETSC_LOG_ACTION_DESTROY,
  PETSC_LOG_ACTION_BEGIN,
  PETSC_LOG_ACTION_END,
} PetscLogActionType;

typedef struct _Action {
  PetscLogActionType action;        /* The type of execution */
  PetscLogEvent      event;         /* The event number */
  PetscClassId       classid;       /* The event class id */
  PetscLogDouble     time;          /* The time of occurrence */
  PetscLogDouble     flops;         /* The cumulative flops */
  PetscLogDouble     mem;           /* The current memory usage */
  PetscLogDouble     maxmem;        /* The maximum memory usage */
  PetscObjectId      id1, id2, id3; /* The ids of associated objects */
} Action;

PETSC_LOG_RESIZABLE_ARRAY(ActionArray, Action, PetscLogEvent, NULL, NULL, NULL)

/* --- Object --- */

/* The structure for object logging */
typedef struct _Object {
  PetscObject    obj;      /* The associated PetscObject */
  int            parent;   /* The parent id */
  PetscLogDouble mem;      /* The memory associated with the object */
  char           name[64]; /* The object name */
  char           info[64]; /* The information string */
} Object;

PETSC_LOG_RESIZABLE_ARRAY(ObjectArray, Object, PetscObject, NULL, NULL, NULL)

/* Map from (threadid,stage,event) to perfInfo data struct */
#include <petsc/private/hashmapijk.h>

PETSC_HASH_MAP(HMapEvent, PetscHashIJKKey, PetscEventPerfInfo *, PetscHashIJKKeyHash, PetscHashIJKKeyEqual, NULL)

typedef struct _n_PetscLogHandler_Default *PetscLogHandler_Default;
struct _n_PetscLogHandler_Default {
  PetscLogStageInfoArray stages;
  PetscSpinlock          lock;
  PetscLogActionArray    petsc_actions;
  PetscLogObjectArray    petsc_objects;
  PetscBool              petsc_logActions;
  PetscBool              petsc_logObjects;
  int                    petsc_numObjectsCreated;
  int                    petsc_numObjectsDestroyed;
  PetscHMapEvent         eventInfoMap_th;
  int                    pause_depth;
  PetscBool              use_threadsafe;
};

/* --- PetscLogHandler_Default --- */

static PetscErrorCode PetscLogHandlerContextCreate_Default(PetscLogHandler_Default *def_p)
{
  PetscLogHandler_Default def;

  PetscFunctionBegin;
  PetscCall(PetscNew(def_p));
  def = *def_p;
  PetscCall(PetscLogStageInfoArrayCreate(8, &def->stages));
  PetscCall(PetscLogActionArrayCreate(64, &def->petsc_actions));
  PetscCall(PetscLogObjectArrayCreate(64, &def->petsc_objects));
  PetscCall(PetscSpinlockCreate(&def->lock));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-log_include_actions", &def->petsc_logActions, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-log_include_objects", &def->petsc_logObjects, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-log_handler_default_use_threadsafe_events", &def->use_threadsafe, NULL));
  if (PetscDefined(HAVE_THREADSAFETY) || def->use_threadsafe) PetscCall(PetscHMapEventCreate(&def->eventInfoMap_th));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDestroy_Default(PetscLogHandler h)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)h->data;

  PetscFunctionBegin;
  PetscCall(PetscLogStageInfoArrayDestroy(&def->stages));
  PetscCall(PetscLogActionArrayDestroy(&def->petsc_actions));
  PetscCall(PetscLogObjectArrayDestroy(&def->petsc_objects));
  PetscCall(PetscSpinlockDestroy(&def->lock));
  if (def->eventInfoMap_th) {
    PetscEventPerfInfo **array;
    PetscInt             n, off = 0;

    PetscCall(PetscHMapEventGetSize(def->eventInfoMap_th, &n));
    PetscCall(PetscMalloc1(n, &array));
    PetscCall(PetscHMapEventGetVals(def->eventInfoMap_th, &off, array));
    for (PetscInt i = 0; i < n; i++) PetscCall(PetscFree(array[i]));
    PetscCall(PetscFree(array));
    PetscCall(PetscHMapEventDestroy(&def->eventInfoMap_th));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerGetEventPerfInfo_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerGetStagePerfInfo_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerSetLogActions_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerSetLogObjects_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerLogObjectState_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerGetNumObjects_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerEventDeactivatePush_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerEventDeactivatePop_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerEventsPause_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerEventsResume_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerDump_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerStageSetVisible_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)h, "PetscLogHandlerStageGetVisible_C", NULL));
  PetscCall(PetscFree(def));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDefaultGetStageInfo(PetscLogHandler handler, PetscLogStage stage, PetscStagePerf **stage_info_p)
{
  PetscStagePerf         *stage_info = NULL;
  PetscLogHandler_Default def        = (PetscLogHandler_Default)handler->data;

  PetscFunctionBegin;
  PetscCall(PetscLogStageInfoArrayResize(def->stages, stage + 1));
  PetscCall(PetscLogStageInfoArrayGetRef(def->stages, stage, &stage_info));
  *stage_info_p = stage_info;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerGetEventPerfInfo_Default(PetscLogHandler handler, PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo **event_info_p)
{
  PetscEventPerfInfo    *event_info = NULL;
  PetscStagePerf        *stage_info = NULL;
  PetscLogEventPerfArray event_log;

  PetscFunctionBegin;
  if (stage < 0) PetscCall(PetscLogStateGetCurrentStage(handler->state, &stage));
  PetscCall(PetscLogHandlerDefaultGetStageInfo(handler, stage, &stage_info));
  event_log = stage_info->eventLog;
  PetscCall(PetscLogEventPerfArrayResize(event_log, event + 1));
  PetscCall(PetscLogEventPerfArrayGetRef(event_log, event, &event_info));
  event_info->id = event;
  *event_info_p  = event_info;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerGetStagePerfInfo_Default(PetscLogHandler handler, PetscLogStage stage, PetscEventPerfInfo **stage_info_p)
{
  PetscStagePerf *stage_perf_info = NULL;

  PetscFunctionBegin;
  if (stage < 0) PetscCall(PetscLogStateGetCurrentStage(handler->state, &stage));
  PetscCall(PetscLogHandlerDefaultGetStageInfo(handler, stage, &stage_perf_info));
  *stage_info_p = &stage_perf_info->perfInfo;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDefaultGetClassPerf(PetscLogHandler handler, PetscLogStage stage, PetscLogClass clss, PetscClassPerf **class_info)
{
  PetscLogClassPerfArray class_log;
  PetscStagePerf        *stage_info;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerDefaultGetStageInfo(handler, stage, &stage_info));
  class_log = stage_info->classLog;
  PetscCall(PetscLogClassPerfArrayResize(class_log, clss + 1));
  PetscCall(PetscLogClassPerfArrayGetRef(class_log, clss, class_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerObjectCreate_Default(PetscLogHandler h, PetscObject obj)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)h->data;
  PetscLogState           state;
  PetscLogStage           stage;
  PetscClassPerf         *classInfo;
  int                     oclass = 0;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscSpinlockLock(&def->lock));
  /* Record stage info */
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscLogStateGetClassFromClassId(state, obj->classid, &oclass));
  PetscCall(PetscLogHandlerDefaultGetClassPerf(h, stage, oclass, &classInfo));
  classInfo->creations++;
  /* Record the creation action */
  if (def->petsc_logActions) {
    Action new_action;

    PetscCall(PetscTime(&new_action.time));
    new_action.time -= petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_CREATE;
    new_action.event   = -1;
    new_action.classid = obj->classid;
    new_action.id1     = obj->id;
    new_action.id2     = -1;
    new_action.id3     = -1;
    new_action.flops   = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogActionArrayPush(def->petsc_actions, new_action));
  }
  /* We don't just use obj->id to count all objects that are created
     because PetscLogHandlers are objects and PetscLogObjectDestroy() will not
     be called for them: the number of objects created and destroyed as counted
     here and below would have an imbalance */
  def->petsc_numObjectsCreated++;
  /* Record the object */
  if (def->petsc_logObjects) {
    Object   new_object;
    PetscInt objid;

    new_object.parent = -1;
    new_object.obj    = obj;
    new_object.mem    = 0;

    PetscCall(PetscMemzero(new_object.name, sizeof(new_object.name)));
    PetscCall(PetscMemzero(new_object.info, sizeof(new_object.info)));
    PetscAssert(obj->id >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Object ids from PetscObjectNewId_Internal() start at 1");
    PetscCall(PetscIntCast(obj->id, &objid));
    PetscCall(PetscLogObjectArrayResize(def->petsc_objects, objid));
    PetscCall(PetscLogObjectArraySet(def->petsc_objects, objid - 1, new_object));
  }
  PetscCall(PetscSpinlockUnlock(&def->lock));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerObjectDestroy_Default(PetscLogHandler h, PetscObject obj)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)h->data;
  PetscLogState           state;
  PetscLogStage           stage;
  PetscClassPerf         *classInfo;
  int                     oclass = 0;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  /* Record stage info */
  PetscCall(PetscSpinlockLock(&def->lock));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  if (stage >= 0) {
    /* stage < 0 can happen if the log summary is output before some things are destroyed */
    PetscCall(PetscLogStateGetClassFromClassId(state, obj->classid, &oclass));
    PetscCall(PetscLogHandlerDefaultGetClassPerf(h, stage, oclass, &classInfo));
    classInfo->destructions++;
  }
  /* Cannot Credit all ancestors with your memory because they may have already been destroyed*/
  def->petsc_numObjectsDestroyed++;
  /* Dynamically enlarge logging structures */
  /* Record the destruction action */
  if (def->petsc_logActions) {
    Action new_action;

    PetscCall(PetscTime(&new_action.time));
    new_action.time -= petsc_BaseTime;
    new_action.event   = -1;
    new_action.action  = PETSC_LOG_ACTION_DESTROY;
    new_action.classid = obj->classid;
    new_action.id1     = obj->id;
    new_action.id2     = -1;
    new_action.id3     = -1;
    new_action.flops   = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogActionArrayPush(def->petsc_actions, new_action));
  }
  if (def->petsc_logObjects) {
    Object  *obj_entry = NULL;
    PetscInt objid;

    PetscAssert(obj->id >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Object ids from PetscObjectNewId_Internal() start at 1");
    PetscCall(PetscIntCast(obj->id - 1, &objid));
    PetscCall(PetscLogObjectArrayGetRef(def->petsc_objects, objid, &obj_entry));
    if (obj->name) PetscCall(PetscStrncpy(obj_entry->name, obj->name, 64));
    obj_entry->obj = NULL;
  }
  PetscCall(PetscSpinlockUnlock(&def->lock));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventSync_Default(PetscLogHandler h, PetscLogEvent event, MPI_Comm comm)
{
  PetscLogState       state;
  PetscLogEventInfo   event_info;
  PetscEventPerfInfo *event_perf_info;
  int                 stage;
  PetscLogDouble      time = 0.0;

  PetscFunctionBegin;
  if (!PetscLogSyncOn || comm == MPI_COMM_NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogHandlerGetState(h, &state));
  PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
  if (!event_info.collective) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  PetscCall(PetscLogHandlerGetEventPerfInfo_Default(h, stage, event, &event_perf_info));
  if (event_perf_info->depth > 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscTimeSubtract(&time));
  PetscCallMPI(MPI_Barrier(comm));
  PetscCall(PetscTimeAdd(&time));
  event_perf_info->syncTime += time;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogGetStageEventPerfInfo_threaded(PetscLogHandler_Default def, PetscLogStage stage, PetscLogEvent event, PetscEventPerfInfo **eventInfo)
{
  PetscEventPerfInfo *leventInfo = NULL;
  PetscHashIJKKey     key;

  PetscFunctionBegin;
#if PetscDefined(HAVE_THREADSAFETY)
  key.i = PetscLogGetTid();
#else
  key.i = 0;
#endif
  key.j = stage;
  key.k = event;
  PetscCall(PetscSpinlockLock(&def->lock));
  PetscCall(PetscHMapEventGet(def->eventInfoMap_th, key, &leventInfo));
  if (!leventInfo) {
    PetscCall(PetscNew(&leventInfo));
    leventInfo->id = event;
    PetscCall(PetscHMapEventSet(def->eventInfoMap_th, key, leventInfo));
  }
  PetscCall(PetscSpinlockUnlock(&def->lock));
  *eventInfo = leventInfo;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventBegin_Default(PetscLogHandler h, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Default def             = (PetscLogHandler_Default)h->data;
  PetscEventPerfInfo     *event_perf_info = NULL;
  PetscLogEventInfo       event_info;
  PetscLogDouble          time;
  PetscLogState           state;
  PetscLogStage           stage;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
    if (PetscUnlikely(o1)) PetscValidHeader(o1, 3);
    if (PetscUnlikely(o2)) PetscValidHeader(o2, 4);
    if (PetscUnlikely(o3)) PetscValidHeader(o3, 5);
    if (PetscUnlikely(o4)) PetscValidHeader(o4, 6);
    if (event_info.collective && o1) {
      PetscInt64 b1[2], b2[2];

      b1[0] = -o1->cidx;
      b1[1] = o1->cidx;
      PetscCallMPI(MPIU_Allreduce(b1, b2, 2, MPIU_INT64, MPI_MAX, PetscObjectComm(o1)));
      PetscCheck(-b2[0] == b2[1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Collective event %s not called collectively %" PetscInt64_FMT " != %" PetscInt64_FMT, event_info.name, -b2[0], b2[1]);
    }
  }
  /* Synchronization */
  PetscCall(PetscLogHandlerEventSync_Default(h, event, PetscObjectComm(o1)));
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  if (def->pause_depth > 0) stage = 0; // in pause-mode, all events run on the main stage
  if (PetscDefined(HAVE_THREADSAFETY) || def->use_threadsafe) {
    PetscCall(PetscLogGetStageEventPerfInfo_threaded(def, stage, event, &event_perf_info));
    if (event_perf_info->depth == 0) PetscCall(PetscEventPerfInfoInit(event_perf_info));
  } else {
    PetscCall(PetscLogHandlerGetEventPerfInfo_Default(h, stage, event, &event_perf_info));
  }
  PetscCheck(event_perf_info->depth >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Trying to begin a paused event, this is not allowed");
  event_perf_info->depth++;
  /* Check for double counting */
  if (event_perf_info->depth > 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
  /* Log the performance info */
  event_perf_info->count++;
  PetscCall(PetscTime(&time));
  PetscCall(PetscEventPerfInfoTic(event_perf_info, time, PetscLogMemory, event));
  if (def->petsc_logActions) {
    PetscLogDouble curTime;
    Action         new_action;

    PetscCall(PetscTime(&curTime));
    new_action.time    = curTime - petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_BEGIN;
    new_action.event   = event;
    new_action.classid = event_info.classid;
    new_action.id1     = o1 ? o1->id : -1;
    new_action.id2     = o2 ? o2->id : -1;
    new_action.id3     = o3 ? o3->id : -1;
    new_action.flops   = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogActionArrayPush(def->petsc_actions, new_action));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventEnd_Default(PetscLogHandler h, PetscLogEvent event, PetscObject o1, PetscObject o2, PetscObject o3, PetscObject o4)
{
  PetscLogHandler_Default def             = (PetscLogHandler_Default)h->data;
  PetscEventPerfInfo     *event_perf_info = NULL;
  PetscLogDouble          time;
  PetscLogState           state;
  int                     stage;
  PetscLogEventInfo       event_info;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(h, &state));
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
    if (PetscUnlikely(o1)) PetscValidHeader(o1, 3);
    if (PetscUnlikely(o2)) PetscValidHeader(o2, 4);
    if (PetscUnlikely(o3)) PetscValidHeader(o3, 5);
    if (PetscUnlikely(o4)) PetscValidHeader(o4, 6);
    if (event_info.collective && o1) {
      PetscInt64 b1[2], b2[2];

      b1[0] = -o1->cidx;
      b1[1] = o1->cidx;
      PetscCallMPI(MPIU_Allreduce(b1, b2, 2, MPIU_INT64, MPI_MAX, PetscObjectComm(o1)));
      PetscCheck(-b2[0] == b2[1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Collective event %s not called collectively %" PetscInt64_FMT " != %" PetscInt64_FMT, event_info.name, -b2[0], b2[1]);
    }
  }
  if (def->petsc_logActions) {
    PetscLogDouble curTime;
    Action         new_action;

    PetscCall(PetscLogStateEventGetInfo(state, event, &event_info));
    PetscCall(PetscTime(&curTime));
    new_action.time    = curTime - petsc_BaseTime;
    new_action.action  = PETSC_LOG_ACTION_END;
    new_action.event   = event;
    new_action.classid = event_info.classid;
    new_action.id1     = o1 ? o1->id : -1;
    new_action.id2     = o2 ? o2->id : -2;
    new_action.id3     = o3 ? o3->id : -3;
    new_action.flops   = petsc_TotalFlops;
    PetscCall(PetscMallocGetCurrentUsage(&new_action.mem));
    PetscCall(PetscMallocGetMaximumUsage(&new_action.maxmem));
    PetscCall(PetscLogActionArrayPush(def->petsc_actions, new_action));
  }
  PetscCall(PetscLogStateGetCurrentStage(state, &stage));
  if (def->pause_depth > 0) stage = 0; // all events run on the main stage in pause-mode
  if (PetscDefined(HAVE_THREADSAFETY) || def->use_threadsafe) {
    PetscCall(PetscLogGetStageEventPerfInfo_threaded(def, stage, event, &event_perf_info));
  } else {
    PetscCall(PetscLogHandlerGetEventPerfInfo_Default(h, stage, event, &event_perf_info));
  }
  PetscCheck(event_perf_info->depth > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Trying to end paused event, not allowed");
  event_perf_info->depth--;
  /* Check for double counting */
  if (event_perf_info->depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  else PetscCheck(event_perf_info->depth == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Logging event had unbalanced begin/end pairs");

  /* Log performance info */
  PetscCall(PetscTime(&time));
  PetscCall(PetscEventPerfInfoToc(event_perf_info, time, PetscLogMemory, event));
  if (PetscDefined(HAVE_THREADSAFETY) || def->use_threadsafe) {
    PetscEventPerfInfo *event_perf_info_global;
    PetscCall(PetscSpinlockLock(&def->lock));
    PetscCall(PetscLogHandlerGetEventPerfInfo_Default(h, stage, event, &event_perf_info_global));
    PetscCall(PetscEventPerfInfoAdd_Internal(event_perf_info, event_perf_info_global));
    PetscCall(PetscSpinlockUnlock(&def->lock));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventDeactivatePush_Default(PetscLogHandler h, PetscLogStage stage, PetscLogEvent event)
{
  PetscEventPerfInfo *event_perf_info;

  PetscFunctionBegin;
  if (stage < 0) PetscCall(PetscLogStateGetCurrentStage(h->state, &stage));
  PetscCall(PetscLogHandlerGetEventPerfInfo_Default(h, stage, event, &event_perf_info));
  event_perf_info->depth++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventDeactivatePop_Default(PetscLogHandler h, PetscLogStage stage, PetscLogEvent event)
{
  PetscEventPerfInfo *event_perf_info;

  PetscFunctionBegin;
  if (stage < 0) PetscCall(PetscLogStateGetCurrentStage(h->state, &stage));
  PetscCall(PetscLogHandlerGetEventPerfInfo_Default(h, stage, event, &event_perf_info));
  event_perf_info->depth--;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventsPause_Default(PetscLogHandler h)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)h->data;
  PetscLogDouble          time;
  PetscInt                num_stages;

  PetscFunctionBegin;
  if (def->pause_depth++ > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStageInfoArrayGetSize(def->stages, &num_stages, NULL));
  PetscCall(PetscTime(&time));
  /* Pause stages in reverse of the order they were pushed */
  for (PetscInt stage = num_stages - 1; stage >= 0; stage--) {
    PetscStagePerf *stage_info = NULL;
    PetscInt        num_events;

    PetscCall(PetscLogStageInfoArrayGetRef(def->stages, stage, &stage_info));
    PetscCall(PetscLogEventPerfArrayGetSize(stage_info->eventLog, &num_events, NULL));
    /* Pause events in reverse of the order they were pushed */
    for (PetscInt event = num_events - 1; event >= 0; event--) {
      PetscEventPerfInfo *event_info = NULL;
      PetscCall(PetscLogEventPerfArrayGetRef(stage_info->eventLog, event, &event_info));
      if (event_info->depth > 0) {
        event_info->depth *= -1;
        PetscCall(PetscEventPerfInfoPause(event_info, time, PetscLogMemory, event));
      }
    }
    if (stage > 0 && stage_info->perfInfo.depth > 0) {
      stage_info->perfInfo.depth *= -1;
      PetscCall(PetscEventPerfInfoPause(&stage_info->perfInfo, time, PetscLogMemory, -(stage + 2)));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerEventsResume_Default(PetscLogHandler h)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)h->data;
  PetscLogDouble          time;
  PetscInt                num_stages;

  PetscFunctionBegin;
  if (--def->pause_depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogStageInfoArrayGetSize(def->stages, &num_stages, NULL));
  PetscCall(PetscTime(&time));
  /* Unpause stages in the same order they were pushed */
  for (PetscInt stage = 0; stage < num_stages; stage++) {
    PetscStagePerf *stage_info = NULL;
    PetscInt        num_events;

    PetscCall(PetscLogStageInfoArrayGetRef(def->stages, stage, &stage_info));
    PetscCall(PetscLogEventPerfArrayGetSize(stage_info->eventLog, &num_events, NULL));
    /* Unpause events in the same order they were pushed */
    for (PetscInt event = 0; event < num_events; event++) {
      PetscEventPerfInfo *event_info = NULL;
      PetscCall(PetscLogEventPerfArrayGetRef(stage_info->eventLog, event, &event_info));
      if (event_info->depth < 0) {
        event_info->depth *= -1;
        PetscCall(PetscEventPerfInfoResume(event_info, time, PetscLogMemory, event));
      }
    }
    if (stage > 0 && stage_info->perfInfo.depth < 0) {
      stage_info->perfInfo.depth *= -1;
      PetscCall(PetscEventPerfInfoResume(&stage_info->perfInfo, time, PetscLogMemory, -(stage + 2)));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePush_Default(PetscLogHandler h, PetscLogStage new_stage)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)h->data;
  PetscLogDouble          time;
  PetscLogState           state;
  PetscLogStage           current_stage;
  PetscStagePerf         *new_stage_info;

  PetscFunctionBegin;
  if (def->pause_depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogHandlerGetState(h, &state));
  current_stage = state->current_stage;
  PetscCall(PetscLogHandlerDefaultGetStageInfo(h, new_stage, &new_stage_info));
  PetscCall(PetscTime(&time));

  /* Record flops/time of previous stage */
  if (current_stage >= 0) {
    if (PetscBTLookup(state->active, current_stage)) {
      PetscStagePerf *current_stage_info;
      PetscCall(PetscLogHandlerDefaultGetStageInfo(h, current_stage, &current_stage_info));
      PetscCall(PetscEventPerfInfoToc(&current_stage_info->perfInfo, time, PetscLogMemory, -(current_stage + 2)));
    }
  }
  new_stage_info->used = PETSC_TRUE;
  new_stage_info->perfInfo.count++;
  new_stage_info->perfInfo.depth++;
  /* Subtract current quantities so that we obtain the difference when we pop */
  if (PetscBTLookup(state->active, new_stage)) PetscCall(PetscEventPerfInfoTic(&new_stage_info->perfInfo, time, PetscLogMemory, -(new_stage + 2)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStagePop_Default(PetscLogHandler h, PetscLogStage old_stage)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)h->data;
  PetscLogStage           current_stage;
  PetscStagePerf         *old_stage_info;
  PetscLogState           state;
  PetscLogDouble          time;

  PetscFunctionBegin;
  if (def->pause_depth > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogHandlerGetState(h, &state));
  current_stage = state->current_stage;
  PetscCall(PetscLogHandlerDefaultGetStageInfo(h, old_stage, &old_stage_info));
  PetscCall(PetscTime(&time));
  old_stage_info->perfInfo.depth--;
  if (PetscBTLookup(state->active, old_stage)) PetscCall(PetscEventPerfInfoToc(&old_stage_info->perfInfo, time, PetscLogMemory, -(old_stage + 2)));
  if (current_stage >= 0) {
    if (PetscBTLookup(state->active, current_stage)) {
      PetscStagePerf *current_stage_info;
      PetscCall(PetscLogHandlerDefaultGetStageInfo(h, current_stage, &current_stage_info));
      PetscCall(PetscEventPerfInfoTic(&current_stage_info->perfInfo, time, PetscLogMemory, -(current_stage + 2)));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStageSetVisible_Default(PetscLogHandler h, PetscLogStage stage, PetscBool is_visible)
{
  PetscStagePerf *stage_info;

  PetscFunctionBegin;
  if (stage < 0) PetscCall(PetscLogStateGetCurrentStage(h->state, &stage));
  PetscCall(PetscLogHandlerDefaultGetStageInfo(h, stage, &stage_info));
  stage_info->perfInfo.visible = is_visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerStageGetVisible_Default(PetscLogHandler h, PetscLogStage stage, PetscBool *is_visible)
{
  PetscStagePerf *stage_info;

  PetscFunctionBegin;
  if (stage < 0) PetscCall(PetscLogStateGetCurrentStage(h->state, &stage));
  PetscCall(PetscLogHandlerDefaultGetStageInfo(h, stage, &stage_info));
  *is_visible = stage_info->perfInfo.visible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerSetLogActions_Default(PetscLogHandler handler, PetscBool flag)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)handler->data;

  PetscFunctionBegin;
  def->petsc_logActions = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerSetLogObjects_Default(PetscLogHandler handler, PetscBool flag)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)handler->data;

  PetscFunctionBegin;
  def->petsc_logObjects = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerLogObjectState_Default(PetscLogHandler handler, PetscObject obj, const char format[], va_list Argp)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)handler->data;
  size_t                  fullLength;

  PetscFunctionBegin;
  if (def->petsc_logObjects) {
    Object  *obj_entry = NULL;
    PetscInt objid;

    PetscCall(PetscIntCast(obj->id - 1, &objid));
    PetscCall(PetscLogObjectArrayGetRef(def->petsc_objects, objid, &obj_entry));
    PetscCall(PetscVSNPrintf(obj_entry->info, 64, format, &fullLength, Argp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerGetNumObjects_Default(PetscLogHandler handler, PetscInt *num_objects)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)handler->data;

  PetscFunctionBegin;
  PetscCall(PetscLogObjectArrayGetSize(def->petsc_objects, num_objects, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerDump_Default(PetscLogHandler handler, const char sname[])
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)handler->data;
  FILE                   *fd;
  char                    file[PETSC_MAX_PATH_LEN], fname[PETSC_MAX_PATH_LEN];
  PetscLogDouble          flops, _TotalTime;
  PetscMPIInt             rank;
  int                     curStage;
  PetscLogState           state;
  PetscInt                num_events;
  PetscLogEvent           event;

  PetscFunctionBegin;
  /* Calculate the total elapsed time */
  PetscCall(PetscTime(&_TotalTime));
  _TotalTime -= petsc_BaseTime;
  /* Open log file */
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)handler), &rank));
  PetscCall(PetscSNPrintf(file, PETSC_STATIC_ARRAY_LENGTH(file), "%s.%d", sname && sname[0] ? sname : "Log", rank));
  PetscCall(PetscFixFilename(file, fname));
  PetscCall(PetscFOpen(PETSC_COMM_SELF, fname, "w", &fd));
  PetscCheck(!(rank == 0) || !(!fd), PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open file: %s", fname);
  /* Output totals */
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Total Flop %14e %16.8e\n", petsc_TotalFlops, _TotalTime));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Clock Resolution %g\n", 0.0));
  /* Output actions */
  if (def->petsc_logActions) {
    PetscInt num_actions;
    PetscCall(PetscLogActionArrayGetSize(def->petsc_actions, &num_actions, NULL));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Actions accomplished %" PetscInt_FMT "\n", num_actions));
    for (int a = 0; a < num_actions; a++) {
      Action *action;

      PetscCall(PetscLogActionArrayGetRef(def->petsc_actions, a, &action));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "%g %d %d %d  %" PetscInt64_FMT " %" PetscInt64_FMT " %" PetscInt64_FMT " %g %g %g\n", action->time, action->action, action->event, action->classid, action->id1, action->id2, action->id3, action->flops,
                             action->mem, action->maxmem));
    }
  }
  /* Output objects */
  if (def->petsc_logObjects) {
    PetscInt num_objects;

    PetscCall(PetscLogObjectArrayGetSize(def->petsc_objects, &num_objects, NULL));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Objects created %d destroyed %d\n", def->petsc_numObjectsCreated, def->petsc_numObjectsDestroyed));
    for (int o = 0; o < num_objects; o++) {
      Object *object = NULL;

      PetscCall(PetscLogObjectArrayGetRef(def->petsc_objects, o, &object));
      if (object->parent != -1) continue; // object with this id wasn't logged, probably a PetscLogHandler
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Parent ID: %d Memory: %d\n", object->parent, (int)object->mem));
      if (!object->name[0]) {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "No Name\n"));
      } else {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Name: %s\n", object->name));
      }
      if (!object->info[0]) {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "No Info\n"));
      } else {
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Info: %s\n", object->info));
      }
    }
  }
  /* Output events */
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "Event log:\n"));
  PetscCall(PetscLogHandlerGetState(handler, &state));
  PetscCall(PetscLogStateGetNumEvents(state, &num_events));
  PetscCall(PetscLogStateGetCurrentStage(state, &curStage));
  for (event = 0; event < num_events; event++) {
    PetscEventPerfInfo *event_info;

    PetscCall(PetscLogHandlerGetEventPerfInfo_Default(handler, curStage, event, &event_info));
    if (event_info->time != 0.0) flops = event_info->flops / event_info->time;
    else flops = 0.0;
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "%d %16d %16g %16g %16g\n", event, event_info->count, event_info->flops, event_info->time, flops));
  }
  PetscCall(PetscFClose(PETSC_COMM_SELF, fd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscLogView_Detailed - Each process prints the times for its own events

*/
static PetscErrorCode PetscLogHandlerView_Default_Detailed(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)handler->data;
  PetscLogDouble          locTotalTime, numRed, maxMem;
  PetscInt                numStages, numEvents;
  MPI_Comm                comm = PetscObjectComm((PetscObject)viewer);
  PetscMPIInt             rank, size;
  PetscLogGlobalNames     global_stages, global_events;
  PetscLogState           state;
  PetscEventPerfInfo      zero_info;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(handler, &state));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Must preserve reduction count before we go on */
  numRed = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;
  /* Get the total elapsed time */
  PetscCall(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;
  PetscCall(PetscViewerASCIIPrintf(viewer, "size = %d\n", size));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalTimes = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalMessages = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalMessageLens = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalReductions = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalFlop = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalObjects = {}\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "LocalMemory = {}\n"));
  PetscCall(PetscLogRegistryCreateGlobalStageNames(comm, state->registry, &global_stages));
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, state->registry, &global_events));
  PetscCall(PetscLogGlobalNamesGetSize(global_stages, NULL, &numStages));
  PetscCall(PetscLogGlobalNamesGetSize(global_events, NULL, &numEvents));
  PetscCall(PetscMemzero(&zero_info, sizeof(zero_info)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Stages = {}\n"));
  for (PetscInt stage = 0; stage < numStages; stage++) {
    PetscInt    stage_id;
    const char *stage_name;

    PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_stages, stage, &stage_id));
    PetscCall(PetscLogGlobalNamesGlobalGetName(global_stages, stage, &stage_name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stages[\"%s\"] = {}\n", stage_name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Stages[\"%s\"][\"summary\"] = {}\n", stage_name));
    for (PetscInt event = 0; event < numEvents; event++) {
      PetscEventPerfInfo *eventInfo = &zero_info;
      PetscBool           is_zero   = PETSC_FALSE;
      PetscInt            event_id;
      const char         *event_name;

      PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_events, event, &event_id));
      PetscCall(PetscLogGlobalNamesGlobalGetName(global_events, event, &event_name));
      if (event_id >= 0 && stage_id >= 0) PetscCall(PetscLogHandlerGetEventPerfInfo_Default(handler, stage_id, event_id, &eventInfo));
      is_zero = eventInfo->count == 0 ? PETSC_TRUE : PETSC_FALSE;
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &is_zero, 1, MPI_C_BOOL, MPI_LAND, comm));
      if (!is_zero) PetscCall(PetscViewerASCIIPrintf(viewer, "Stages[\"%s\"][\"%s\"] = {}\n", stage_name, event_name));
    }
  }
  PetscCall(PetscMallocGetMaximumUsage(&maxMem));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalTimes[%d] = %g\n", rank, locTotalTime));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalMessages[%d] = %g\n", rank, petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalMessageLens[%d] = %g\n", rank, petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalReductions[%d] = %g\n", rank, numRed));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalFlop[%d] = %g\n", rank, petsc_TotalFlops));
  {
    PetscInt num_objects;

    PetscCall(PetscLogObjectArrayGetSize(def->petsc_objects, &num_objects, NULL));
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalObjects[%d] = %" PetscInt_FMT "\n", rank, num_objects));
  }
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "LocalMemory[%d] = %g\n", rank, maxMem));
  PetscCall(PetscViewerFlush(viewer));
  for (PetscInt stage = 0; stage < numStages; stage++) {
    PetscEventPerfInfo *stage_perf_info = &zero_info;
    PetscInt            stage_id;
    const char         *stage_name;

    PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_stages, stage, &stage_id));
    PetscCall(PetscLogGlobalNamesGlobalGetName(global_stages, stage, &stage_name));
    if (stage_id >= 0) {
      PetscStagePerf *stage_info;
      PetscCall(PetscLogHandlerDefaultGetStageInfo(handler, stage_id, &stage_info));
      stage_perf_info = &stage_info->perfInfo;
    }
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Stages[\"%s\"][\"summary\"][%d] = {\"time\" : %g, \"numMessages\" : %g, \"messageLength\" : %g, \"numReductions\" : %g, \"flop\" : %g}\n", stage_name, rank, stage_perf_info->time,
                                                 stage_perf_info->numMessages, stage_perf_info->messageLength, stage_perf_info->numReductions, stage_perf_info->flops));
    for (PetscInt event = 0; event < numEvents; event++) {
      PetscEventPerfInfo *eventInfo = &zero_info;
      PetscBool           is_zero   = PETSC_FALSE;
      PetscInt            event_id;
      const char         *event_name;

      PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_events, event, &event_id));
      PetscCall(PetscLogGlobalNamesGlobalGetName(global_events, event, &event_name));
      if (event_id >= 0 && stage_id >= 0) PetscCall(PetscLogHandlerGetEventPerfInfo_Default(handler, stage_id, event_id, &eventInfo));
      is_zero = eventInfo->count == 0 ? PETSC_TRUE : PETSC_FALSE;
      PetscCall(PetscMemcmp(eventInfo, &zero_info, sizeof(zero_info), &is_zero));
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &is_zero, 1, MPI_C_BOOL, MPI_LAND, comm));
      if (!is_zero) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Stages[\"%s\"][\"%s\"][%d] = {\"count\" : %d, \"time\" : %g, \"syncTime\" : %g, \"numMessages\" : %g, \"messageLength\" : %g, \"numReductions\" : %g, \"flop\" : %g", stage_name, event_name, rank,
                                                     eventInfo->count, eventInfo->time, eventInfo->syncTime, eventInfo->numMessages, eventInfo->messageLength, eventInfo->numReductions, eventInfo->flops));
        if (eventInfo->dof[0] >= 0.) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ", \"dof\" : ["));
          for (PetscInt d = 0; d < 8; ++d) {
            if (d > 0) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ", "));
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%g", eventInfo->dof[d]));
          }
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "]"));
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ", \"error\" : ["));
          for (PetscInt e = 0; e < 8; ++e) {
            if (e > 0) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ", "));
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%g", eventInfo->errors[e]));
          }
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "]"));
        }
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "}\n"));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));
  PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscLogView_CSV - Each process prints the times for its own events in Comma-Separated Value Format
*/
static PetscErrorCode PetscLogHandlerView_Default_CSV(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogDouble      locTotalTime, maxMem;
  PetscInt            numStages, numEvents, stage, event;
  MPI_Comm            comm = PetscObjectComm((PetscObject)viewer);
  PetscMPIInt         rank, size;
  PetscLogGlobalNames global_stages, global_events;
  PetscLogState       state;
  PetscEventPerfInfo  zero_info;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(handler, &state));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Must preserve reduction count before we go on */
  /* Get the total elapsed time */
  PetscCall(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;
  PetscCall(PetscMallocGetMaximumUsage(&maxMem));
  PetscCall(PetscLogRegistryCreateGlobalStageNames(comm, state->registry, &global_stages));
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, state->registry, &global_events));
  PetscCall(PetscLogGlobalNamesGetSize(global_stages, NULL, &numStages));
  PetscCall(PetscLogGlobalNamesGetSize(global_events, NULL, &numEvents));
  PetscCall(PetscMemzero(&zero_info, sizeof(zero_info)));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Stage Name,Event Name,Rank,Count,Time,Num Messages,Message Length,Num Reductions,FLOP,dof0,dof1,dof2,dof3,dof4,dof5,dof6,dof7,e0,e1,e2,e3,e4,e5,e6,e7,%d\n", size));
  PetscCall(PetscViewerFlush(viewer));
  for (stage = 0; stage < numStages; stage++) {
    PetscEventPerfInfo *stage_perf_info;
    PetscInt            stage_id;
    const char         *stage_name;

    PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_stages, stage, &stage_id));
    PetscCall(PetscLogGlobalNamesGlobalGetName(global_stages, stage, &stage_name));
    stage_perf_info = &zero_info;
    if (stage_id >= 0) {
      PetscStagePerf *stage_info;
      PetscCall(PetscLogHandlerDefaultGetStageInfo(handler, stage_id, &stage_info));
      stage_perf_info = &stage_info->perfInfo;
    }
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%s,summary,%d,1,%g,%g,%g,%g,%g\n", stage_name, rank, stage_perf_info->time, stage_perf_info->numMessages, stage_perf_info->messageLength, stage_perf_info->numReductions, stage_perf_info->flops));
    for (event = 0; event < numEvents; event++) {
      PetscEventPerfInfo *eventInfo = &zero_info;
      PetscBool           is_zero   = PETSC_FALSE;
      PetscInt            event_id;
      const char         *event_name;

      PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_events, event, &event_id));
      PetscCall(PetscLogGlobalNamesGlobalGetName(global_events, event, &event_name));
      if (event_id >= 0 && stage_id >= 0) PetscCall(PetscLogHandlerGetEventPerfInfo_Default(handler, stage_id, event_id, &eventInfo));
      PetscCall(PetscMemcmp(eventInfo, &zero_info, sizeof(zero_info), &is_zero));
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &is_zero, 1, MPI_C_BOOL, MPI_LAND, comm));
      if (!is_zero) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%s,%s,%d,%d,%g,%g,%g,%g,%g", stage_name, event_name, rank, eventInfo->count, eventInfo->time, eventInfo->numMessages, eventInfo->messageLength, eventInfo->numReductions, eventInfo->flops));
        if (eventInfo->dof[0] >= 0.) {
          for (PetscInt d = 0; d < 8; ++d) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ",%g", eventInfo->dof[d]));
          for (PetscInt e = 0; e < 8; ++e) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ",%g", eventInfo->errors[e]));
        }
      }
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogViewWarnSync(PetscViewer viewer)
{
  PetscFunctionBegin;
  if (!PetscLogSyncOn) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      ##########################################################\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                       WARNING!!!                       #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   This program was run with logging synchronization.   #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   This option provides more meaningful imbalance       #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   figures at the expense of slowing things down and    #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   providing a distorted view of the overall runtime.   #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      ##########################################################\n\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogViewWarnDebugging(PetscViewer viewer)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      ##########################################################\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      #                       WARNING!!!                       #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      #   This code was compiled with a debugging option.      #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      #   To get timing results run ./configure                #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      #   using --with-debugging=no, the performance will      #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      #   be generally two or three times faster.              #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "      ##########################################################\n\n\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogViewWarnNoGpuAwareMpi(PetscViewer viewer)
{
#if defined(PETSC_HAVE_DEVICE)
  PetscMPIInt size;
  PetscBool   deviceInitialized = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
  for (int i = PETSC_DEVICE_HOST + 1; i < PETSC_DEVICE_MAX; ++i) {
    const PetscDeviceType dtype = PetscDeviceTypeCast(i);
    if (PetscDeviceInitialized(dtype)) { /* a non-host device was initialized */
      deviceInitialized = PETSC_TRUE;
      break;
    }
  }
  /* the last condition says PETSc is configured with device but it is a pure CPU run, so don't print misleading warnings */
  if (use_gpu_aware_mpi || size == 1 || !deviceInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      ##########################################################\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                       WARNING!!!                       #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   This code was compiled with GPU support and you've   #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   created PETSc/GPU objects, but you intentionally     #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   used -use_gpu_aware_mpi 0, requiring PETSc to copy   #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   additional data between the GPU and CPU. To obtain   #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   meaningful timing results on multi-rank runs, use    #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   GPU-aware MPI instead.                               #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      ##########################################################\n\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  return PETSC_SUCCESS;
#endif
}

static PetscErrorCode PetscLogViewWarnGpuTime(PetscViewer viewer)
{
#if defined(PETSC_HAVE_DEVICE) && !defined(PETSC_HAVE_KOKKOS_WITHOUT_GPU)

  PetscFunctionBegin;
  if (!PetscLogGpuTimeFlag || petsc_gflops == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      ##########################################################\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                       WARNING!!!                       #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   This code was run with -log_view_gpu_time            #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   This provides accurate timing within the GPU kernels #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   but can slow down the entire computation by a        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   measurable amount. For fastest runs we recommend     #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #   not using this option.                               #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      #                                                        #\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      ##########################################################\n\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  return PETSC_SUCCESS;
#endif
}

PETSC_INTERN int    PetscGlobalArgc;
PETSC_INTERN char **PetscGlobalArgs;

static PetscErrorCode PetscLogHandlerView_Default_Info(PetscLogHandler handler, PetscViewer viewer)
{
  PetscLogHandler_Default def = (PetscLogHandler_Default)handler->data;
  char                    arch[128], hostname[128], username[128], pname[PETSC_MAX_PATH_LEN], date[128];
  PetscLogDouble          locTotalTime, TotalTime, TotalFlops;
  PetscLogDouble          numMessages, messageLength, avgMessLen, numReductions;
  PetscLogDouble          stageTime, flops, flopr, mem, mess, messLen, red;
  PetscLogDouble          fracTime, fracFlops, fracMessages, fracLength, fracReductions, fracMess, fracMessLen, fracRed;
  PetscLogDouble          fracStageTime, fracStageFlops, fracStageMess, fracStageMessLen, fracStageRed;
  PetscLogDouble          min, max, tot, ratio, avg, x, y;
  PetscLogDouble          minf, maxf, totf, ratf, mint, maxt, tott, ratt, ratC, totm, totml, totr, mal, malmax, emalmax;
#if defined(PETSC_HAVE_DEVICE)
  PetscLogEvent  KSP_Solve, SNES_Solve, TS_Step, TAO_Solve; /* These need to be fixed to be some events registered with certain objects */
  PetscLogDouble cct, gct, csz, gsz, gmaxt, gflops, gflopr, fracgflops;
#endif
  PetscMPIInt   minC, maxC;
  PetscMPIInt   size, rank;
  PetscBool    *localStageUsed, *stageUsed;
  PetscBool    *localStageVisible, *stageVisible;
  PetscInt      numStages, numEvents;
  int           stage, oclass;
  PetscLogEvent event;
  char          version[256];
  MPI_Comm      comm;
#if defined(PETSC_HAVE_DEVICE) && !defined(PETSC_HAVE_KOKKOS_WITHOUT_GPU)
  PetscInt64 nas = 0x7FF0000000000002;
#endif
  PetscLogGlobalNames global_stages, global_events;
  PetscEventPerfInfo  zero_info;
  PetscLogState       state;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetState(handler, &state));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Get the total elapsed time */
  PetscCall(PetscTime(&locTotalTime));
  locTotalTime -= petsc_BaseTime;

  PetscCall(PetscViewerASCIIPrintf(viewer, "****************************************************************************************************************************************************************\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "***                                WIDEN YOUR WINDOW TO 160 CHARACTERS.  Use 'enscript -r -fCourier9' to print this document                                 ***\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "****************************************************************************************************************************************************************\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n------------------------------------------------------------------ PETSc Performance Summary: ------------------------------------------------------------------\n\n"));
  PetscCall(PetscLogViewWarnSync(viewer));
  PetscCall(PetscLogViewWarnDebugging(viewer));
  PetscCall(PetscLogViewWarnNoGpuAwareMpi(viewer));
  PetscCall(PetscLogViewWarnGpuTime(viewer));
  PetscCall(PetscGetArchType(arch, sizeof(arch)));
  PetscCall(PetscGetHostName(hostname, sizeof(hostname)));
  PetscCall(PetscGetUserName(username, sizeof(username)));
  PetscCall(PetscGetProgramName(pname, sizeof(pname)));
  PetscCall(PetscGetDate(date, sizeof(date)));
  PetscCall(PetscGetVersion(version, sizeof(version)));

#if defined(PETSC_HAVE_CUPM)
  const char *cupm = PetscDefined(HAVE_CUDA) ? "CUDA" : "HIP";
  if (PetscDeviceCUPMRuntimeArch)
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s on a %s named %s with %d process%s and %s architecture %d, by %s on %s\n", pname, arch, hostname, size, size > 1 ? "es" : "", cupm, PetscDeviceCUPMRuntimeArch, username, date));
  else
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s on a %s named %s with %d process%s, by %s on %s\n", pname, arch, hostname, size, size > 1 ? "es" : "", username, date));

#if defined(PETSC_HAVE_OPENMP)
  PetscCall(PetscViewerASCIIPrintf(viewer, "Using %" PetscInt_FMT " OpenMP threads\n", PetscNumOMPThreads));
#endif
  PetscCall(PetscViewerASCIIPrintf(viewer, "Using %s\n", version));

  /* Must preserve reduction count before we go on */
  red = petsc_allreduce_ct + petsc_gather_ct + petsc_scatter_ct;

  /* Calculate summary information */
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n                         Max       Max/Min     Avg       Total\n"));
  /*   Time */
  PetscCallMPI(MPIU_Allreduce(&locTotalTime, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&locTotalTime, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&locTotalTime, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Time (sec):           %5.3e   %7.3f   %5.3e\n", max, ratio, avg));
  TotalTime = tot;
  /*   Objects */
  {
    PetscInt num_objects;

    PetscCall(PetscLogObjectArrayGetSize(def->petsc_objects, &num_objects, NULL));
    avg = (PetscLogDouble)num_objects;
  }
  PetscCallMPI(MPIU_Allreduce(&avg, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&avg, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&avg, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Objects:              %5.3e   %7.3f   %5.3e\n", max, ratio, avg));
  /*   Flops */
  PetscCallMPI(MPIU_Allreduce(&petsc_TotalFlops, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&petsc_TotalFlops, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&petsc_TotalFlops, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Flops:                %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  TotalFlops = tot;
  /*   Flops/sec -- Must talk to Barry here */
  if (locTotalTime != 0.0) flops = petsc_TotalFlops / locTotalTime;
  else flops = 0.0;
  PetscCallMPI(MPIU_Allreduce(&flops, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&flops, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&flops, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Flops/sec:            %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  /*   Memory */
  PetscCall(PetscMallocGetMaximumUsage(&mem));
  if (mem > 0.0) {
    PetscCallMPI(MPIU_Allreduce(&mem, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
    PetscCallMPI(MPIU_Allreduce(&mem, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
    PetscCallMPI(MPIU_Allreduce(&mem, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    avg = tot / ((PetscLogDouble)size);
    if (min != 0.0) ratio = max / min;
    else ratio = 0.0;
    PetscCall(PetscViewerASCIIPrintf(viewer, "Memory (bytes):       %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  }
  /*   Messages */
  mess = 0.5 * (petsc_irecv_ct + petsc_isend_ct + petsc_recv_ct + petsc_send_ct);
  PetscCallMPI(MPIU_Allreduce(&mess, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&mess, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&mess, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  avg = tot / ((PetscLogDouble)size);
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscViewerASCIIPrintf(viewer, "MPI Msg Count:        %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  numMessages = tot;
  /*   Message Lengths */
  mess = 0.5 * (petsc_irecv_len + petsc_isend_len + petsc_recv_len + petsc_send_len);
  PetscCallMPI(MPIU_Allreduce(&mess, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&mess, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&mess, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  if (numMessages != 0) avg = tot / numMessages;
  else avg = 0.0;
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscViewerASCIIPrintf(viewer, "MPI Msg Len (bytes):  %5.3e   %7.3f   %5.3e  %5.3e\n", max, ratio, avg, tot));
  messageLength = tot;
  /*   Reductions */
  PetscCallMPI(MPIU_Allreduce(&red, &min, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(&red, &max, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(&red, &tot, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
  if (min != 0.0) ratio = max / min;
  else ratio = 0.0;
  PetscCall(PetscViewerASCIIPrintf(viewer, "MPI Reductions:       %5.3e   %7.3f\n", max, ratio));
  numReductions = red; /* wrong because uses count from process zero */
  PetscCall(PetscViewerASCIIPrintf(viewer, "\nFlop counting convention: 1 flop = 1 real number operation of type (multiply/divide/add/subtract)\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "                            e.g., VecAXPY() for real vectors of length N --> 2N flops\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "                            and VecAXPY() for complex vectors of length N --> 8N flops\n"));

  PetscCall(PetscLogRegistryCreateGlobalStageNames(comm, state->registry, &global_stages));
  PetscCall(PetscLogRegistryCreateGlobalEventNames(comm, state->registry, &global_events));
  PetscCall(PetscLogGlobalNamesGetSize(global_stages, NULL, &numStages));
  PetscCall(PetscLogGlobalNamesGetSize(global_events, NULL, &numEvents));
  PetscCall(PetscMemzero(&zero_info, sizeof(zero_info)));
  PetscCall(PetscMalloc1(numStages, &localStageUsed));
  PetscCall(PetscMalloc1(numStages, &stageUsed));
  PetscCall(PetscMalloc1(numStages, &localStageVisible));
  PetscCall(PetscMalloc1(numStages, &stageVisible));
  if (numStages > 0) {
    for (stage = 0; stage < numStages; stage++) {
      PetscInt stage_id;

      PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_stages, stage, &stage_id));
      if (stage_id >= 0) {
        PetscStagePerf *stage_info;

        PetscCall(PetscLogHandlerDefaultGetStageInfo(handler, stage, &stage_info));
        localStageUsed[stage]    = stage_info->used;
        localStageVisible[stage] = stage_info->perfInfo.visible;
      } else {
        localStageUsed[stage]    = PETSC_FALSE;
        localStageVisible[stage] = PETSC_TRUE;
      }
    }
    PetscCallMPI(MPIU_Allreduce(localStageUsed, stageUsed, numStages, MPI_C_BOOL, MPI_LOR, comm));
    PetscCallMPI(MPIU_Allreduce(localStageVisible, stageVisible, numStages, MPI_C_BOOL, MPI_LAND, comm));
    for (stage = 0; stage < numStages; stage++) {
      if (stageUsed[stage] && stageVisible[stage]) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "\nSummary of Stages:   ----- Time ------  ----- Flop ------  --- Messages ---  -- Message Lengths --  -- Reductions --\n"));
        PetscCall(PetscViewerASCIIPrintf(viewer, "                        Avg     %%Total     Avg     %%Total    Count   %%Total     Avg         %%Total    Count   %%Total\n"));
        break;
      }
    }
    for (stage = 0; stage < numStages; stage++) {
      PetscInt            stage_id;
      PetscEventPerfInfo *stage_info;
      const char         *stage_name;

      if (!(stageUsed[stage] && stageVisible[stage])) continue;
      PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_stages, stage, &stage_id));
      PetscCall(PetscLogGlobalNamesGlobalGetName(global_stages, stage, &stage_name));
      stage_info = &zero_info;
      if (localStageUsed[stage]) {
        PetscStagePerf *stage_perf_info;

        PetscCall(PetscLogHandlerDefaultGetStageInfo(handler, stage, &stage_perf_info));
        stage_info = &stage_perf_info->perfInfo;
      }
      PetscCallMPI(MPIU_Allreduce(&stage_info->time, &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCallMPI(MPIU_Allreduce(&stage_info->flops, &flops, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCallMPI(MPIU_Allreduce(&stage_info->numMessages, &mess, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCallMPI(MPIU_Allreduce(&stage_info->messageLength, &messLen, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      PetscCallMPI(MPIU_Allreduce(&stage_info->numReductions, &red, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
      mess *= 0.5;
      messLen *= 0.5;
      red /= size;
      if (TotalTime != 0.0) fracTime = stageTime / TotalTime;
      else fracTime = 0.0;
      if (TotalFlops != 0.0) fracFlops = flops / TotalFlops;
      else fracFlops = 0.0;
      /* Talk to Barry if (stageTime     != 0.0) flops          = (size*flops)/stageTime; else flops          = 0.0; */
      if (numMessages != 0.0) fracMessages = mess / numMessages;
      else fracMessages = 0.0;
      if (mess != 0.0) avgMessLen = messLen / mess;
      else avgMessLen = 0.0;
      if (messageLength != 0.0) fracLength = messLen / messageLength;
      else fracLength = 0.0;
      if (numReductions != 0.0) fracReductions = red / numReductions;
      else fracReductions = 0.0;
      PetscCall(PetscViewerASCIIPrintf(viewer, "%2d: %15s: %6.4e %5.1f%%  %6.4e %5.1f%%  %5.3e %5.1f%%  %5.3e      %5.1f%%  %5.3e %5.1f%%\n", stage, stage_name, stageTime / size, 100.0 * fracTime, flops, 100.0 * fracFlops, mess, 100.0 * fracMessages, avgMessLen, 100.0 * fracLength, red, 100.0 * fracReductions));
    }
  }

  PetscCall(PetscViewerASCIIPrintf(viewer, "\n------------------------------------------------------------------------------------------------------------------------\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "See the 'Profiling' chapter of the users' manual for details on interpreting output.\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Phase summary info:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   Count: number of times phase was executed\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   Time and Flop: Max - maximum over all processes\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "                  Ratio - ratio of maximum to minimum over all processes\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   Mess: number of messages sent\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   AvgLen: average message length (bytes)\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   Reduct: number of global reductions\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   Global: entire computation\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   Stage: stages of a computation. Set stages with PetscLogStagePush() and PetscLogStagePop().\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      %%T - percent time in this phase         %%F - percent flop in this phase\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      %%M - percent messages in this phase     %%L - percent message lengths in this phase\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "      %%R - percent reductions in this phase\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   Total Mflop/s: 1e-6 * (sum of flop over all processes)/(max time over all processes)\n"));
  if (PetscLogMemory) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "   Memory usage is summed over all MPI processes, it is given in mega-bytes\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "   Malloc Mbytes: Memory allocated and kept during event (sum over all calls to event). May be negative\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "   EMalloc Mbytes: extra memory allocated during event and then freed (maximum over all calls to events). Never negative\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "   MMalloc Mbytes: Increase in high water mark of allocated memory (sum over all calls to event). Never negative\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "   RMI Mbytes: Increase in resident memory (sum over all calls to event)\n"));
  }
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscViewerASCIIPrintf(viewer, "   GPU Mflop/s: 1e-6 * (sum of flop on GPU over all processes)/(max GPU time over all processes)\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   CpuToGpu Count: total number of CPU to GPU copies per processor\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   CpuToGpu Size (Mbytes): 1e-6 * (total size of CPU to GPU copies per processor)\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   GpuToCpu Count: total number of GPU to CPU copies per processor\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   GpuToCpu Size (Mbytes): 1e-6 * (total size of GPU to CPU copies per processor)\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "   GPU %%F: percent flops on GPU in this event\n"));
#endif
  PetscCall(PetscViewerASCIIPrintf(viewer, "------------------------------------------------------------------------------------------------------------------------\n"));

  PetscCall(PetscLogViewWarnDebugging(viewer));

  /* Report events */
  PetscCall(PetscViewerASCIIPrintf(viewer, "Event                Count      Time (sec)     Flop                              --- Global ---  --- Stage ----  Total"));
  if (PetscLogMemory) PetscCall(PetscViewerASCIIPrintf(viewer, "  Malloc EMalloc MMalloc RMI"));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscViewerASCIIPrintf(viewer, "   GPU    - CpuToGpu -   - GpuToCpu - GPU"));
#endif
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "                   Max Ratio  Max     Ratio   Max  Ratio  Mess   AvgLen  Reduct  %%T %%F %%M %%L %%R  %%T %%F %%M %%L %%R Mflop/s"));
  if (PetscLogMemory) PetscCall(PetscViewerASCIIPrintf(viewer, " Mbytes Mbytes Mbytes Mbytes"));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscViewerASCIIPrintf(viewer, " Mflop/s Count   Size   Count   Size  %%F"));
#endif
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "------------------------------------------------------------------------------------------------------------------------"));
  if (PetscLogMemory) PetscCall(PetscViewerASCIIPrintf(viewer, "-----------------------------"));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscViewerASCIIPrintf(viewer, "---------------------------------------"));
#endif
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));

#if defined(PETSC_HAVE_DEVICE)
  /* this indirect way of accessing these values is needed when PETSc is build with multiple libraries since the symbols are not in libpetscsys */
  PetscCall(PetscLogStateGetEventFromName(state, "TaoSolve", &TAO_Solve));
  PetscCall(PetscLogStateGetEventFromName(state, "TSStep", &TS_Step));
  PetscCall(PetscLogStateGetEventFromName(state, "SNESSolve", &SNES_Solve));
  PetscCall(PetscLogStateGetEventFromName(state, "KSPSolve", &KSP_Solve));
#endif

  for (stage = 0; stage < numStages; stage++) {
    PetscInt            stage_id;
    PetscEventPerfInfo *stage_info;
    const char         *stage_name;

    if (!(stageVisible[stage] && stageUsed[stage])) continue;
    PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_stages, stage, &stage_id));
    PetscCall(PetscLogGlobalNamesGlobalGetName(global_stages, stage, &stage_name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n--- Event Stage %d: %s\n\n", stage, stage_name));
    stage_info = &zero_info;
    if (localStageUsed[stage]) {
      PetscStagePerf *stage_perf_info;

      PetscCall(PetscLogHandlerDefaultGetStageInfo(handler, stage_id, &stage_perf_info));
      stage_info = &stage_perf_info->perfInfo;
    }
    PetscCallMPI(MPIU_Allreduce(&stage_info->time, &stageTime, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscCallMPI(MPIU_Allreduce(&stage_info->flops, &flops, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscCallMPI(MPIU_Allreduce(&stage_info->numMessages, &mess, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscCallMPI(MPIU_Allreduce(&stage_info->messageLength, &messLen, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    PetscCallMPI(MPIU_Allreduce(&stage_info->numReductions, &red, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
    mess *= 0.5;
    messLen *= 0.5;
    red /= size;

    for (event = 0; event < numEvents; event++) {
      PetscInt            event_id;
      PetscEventPerfInfo *event_info = &zero_info;
      PetscBool           is_zero    = PETSC_FALSE;
      const char         *event_name;

      PetscCall(PetscLogGlobalNamesGlobalGetLocal(global_events, event, &event_id));
      PetscCall(PetscLogGlobalNamesGlobalGetName(global_events, event, &event_name));
      if (event_id >= 0 && stage_id >= 0) PetscCall(PetscLogHandlerGetEventPerfInfo_Default(handler, stage_id, event_id, &event_info));
      PetscCall(PetscMemcmp(event_info, &zero_info, sizeof(zero_info), &is_zero));
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &is_zero, 1, MPI_C_BOOL, MPI_LAND, comm));
      if (!is_zero) {
        flopr = event_info->flops;
        PetscCallMPI(MPIU_Allreduce(&flopr, &minf, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
        PetscCallMPI(MPIU_Allreduce(&flopr, &maxf, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->flops, &totf, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->time, &mint, 1, MPIU_PETSCLOGDOUBLE, MPI_MIN, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->time, &maxt, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->time, &tott, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->numMessages, &totm, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->messageLength, &totml, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->numReductions, &totr, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->count, &minC, 1, MPI_INT, MPI_MIN, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->count, &maxC, 1, MPI_INT, MPI_MAX, comm));
        if (PetscLogMemory) {
          PetscCallMPI(MPIU_Allreduce(&event_info->memIncrease, &mem, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
          PetscCallMPI(MPIU_Allreduce(&event_info->mallocSpace, &mal, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
          PetscCallMPI(MPIU_Allreduce(&event_info->mallocIncrease, &malmax, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
          PetscCallMPI(MPIU_Allreduce(&event_info->mallocIncreaseEvent, &emalmax, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        }
#if defined(PETSC_HAVE_DEVICE)
        PetscCallMPI(MPIU_Allreduce(&event_info->CpuToGpuCount, &cct, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->GpuToCpuCount, &gct, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->CpuToGpuSize, &csz, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->GpuToCpuSize, &gsz, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->GpuFlops, &gflops, 1, MPIU_PETSCLOGDOUBLE, MPI_SUM, comm));
        PetscCallMPI(MPIU_Allreduce(&event_info->GpuTime, &gmaxt, 1, MPIU_PETSCLOGDOUBLE, MPI_MAX, comm));
#endif
        if (mint < 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "WARNING!!! Minimum time %g over all processes for %s is negative! This happens\n on some machines whose times cannot handle too rapid calls.!\n artificially changing minimum to zero.\n", mint, event_name));
          mint = 0;
        }
        PetscCheck(minf >= 0.0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Minimum flop %g over all processes for %s is negative! Not possible!", minf, event_name);
#if defined(PETSC_HAVE_DEVICE) && !defined(PETSC_HAVE_KOKKOS_WITHOUT_GPU)
        /* Put NaN into the time for all events that may not be time accurately since they may happen asynchronously on the GPU */
        if (!PetscLogGpuTimeFlag && petsc_gflops > 0) {
          memcpy(&gmaxt, &nas, sizeof(PetscLogDouble));
          if (event_id != SNES_Solve && event_id != KSP_Solve && event_id != TS_Step && event_id != TAO_Solve) {
            memcpy(&mint, &nas, sizeof(PetscLogDouble));
            memcpy(&maxt, &nas, sizeof(PetscLogDouble));
          }
        }
#endif
        totm *= 0.5;
        totml *= 0.5;
        totr /= size;

        if (maxC != 0) {
          if (minC != 0) ratC = ((PetscLogDouble)maxC) / minC;
          else ratC = 0.0;
          if (mint != 0.0) ratt = maxt / mint;
          else ratt = 0.0;
          if (minf != 0.0) ratf = maxf / minf;
          else ratf = 0.0;
          if (TotalTime != 0.0) fracTime = tott / TotalTime;
          else fracTime = 0.0;
          if (TotalFlops != 0.0) fracFlops = totf / TotalFlops;
          else fracFlops = 0.0;
          if (stageTime != 0.0) fracStageTime = tott / stageTime;
          else fracStageTime = 0.0;
          if (flops != 0.0) fracStageFlops = totf / flops;
          else fracStageFlops = 0.0;
          if (numMessages != 0.0) fracMess = totm / numMessages;
          else fracMess = 0.0;
          if (messageLength != 0.0) fracMessLen = totml / messageLength;
          else fracMessLen = 0.0;
          if (numReductions != 0.0) fracRed = totr / numReductions;
          else fracRed = 0.0;
          if (mess != 0.0) fracStageMess = totm / mess;
          else fracStageMess = 0.0;
          if (messLen != 0.0) fracStageMessLen = totml / messLen;
          else fracStageMessLen = 0.0;
          if (red != 0.0) fracStageRed = totr / red;
          else fracStageRed = 0.0;
          if (totm != 0.0) totml /= totm;
          else totml = 0.0;
          if (maxt != 0.0) flopr = totf / maxt;
          else flopr = 0.0;
          if (fracStageTime > 1.0 || fracStageFlops > 1.0 || fracStageMess > 1.0 || fracStageMessLen > 1.0 || fracStageRed > 1.0) {
            if (PetscIsNanReal(maxt))
              PetscCall(PetscViewerASCIIPrintf(viewer, "%-16s %7d %3.1f  n/a     n/a   %3.2e %3.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f Multiple stages n/a", event_name, maxC, ratC, maxf, ratf, totm, totml, totr, 100.0 * fracTime, 100.0 * fracFlops, 100.0 * fracMess, 100.0 * fracMessLen, 100.0 * fracRed));
            else
              PetscCall(PetscViewerASCIIPrintf(viewer, "%-16s %7d %3.1f %5.4e %3.1f %3.2e %3.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f Multiple stages %5.0f", event_name, maxC, ratC, maxt, ratt, maxf, ratf, totm, totml, totr, 100.0 * fracTime, 100.0 * fracFlops, 100.0 * fracMess, 100.0 * fracMessLen, 100.0 * fracRed, PetscAbs(flopr) / 1.0e6));
          } else {
            if (PetscIsNanReal(maxt)) { // when maxt, ratt, flopr are NaN (i.e., run with GPUs but without -log_view_gpu_time), replace the confusing "nan" with "n/a"
              PetscCall(PetscViewerASCIIPrintf(viewer, "%-16s %7d %3.1f  n/a     n/a   %3.2e %3.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f %3.0f %2.0f %2.0f %2.0f %2.0f  n/a", event_name, maxC, ratC, maxf, ratf, totm, totml, totr, 100.0 * fracTime, 100.0 * fracFlops, 100.0 * fracMess, 100.0 * fracMessLen, 100.0 * fracRed, 100.0 * fracStageTime, 100.0 * fracStageFlops, 100.0 * fracStageMess, 100.0 * fracStageMessLen, 100.0 * fracStageRed));
            } else {
              PetscCall(PetscViewerASCIIPrintf(viewer, "%-16s %7d %3.1f %5.4e %3.1f %3.2e %3.1f %2.1e %2.1e %2.1e %2.0f %2.0f %2.0f %2.0f %2.0f %3.0f %2.0f %2.0f %2.0f %2.0f %5.0f", event_name, maxC, ratC, maxt, ratt, maxf, ratf, totm, totml, totr, 100.0 * fracTime, 100.0 * fracFlops, 100.0 * fracMess, 100.0 * fracMessLen, 100.0 * fracRed, 100.0 * fracStageTime, 100.0 * fracStageFlops, 100.0 * fracStageMess, 100.0 * fracStageMessLen, 100.0 * fracStageRed, PetscAbs(flopr) / 1.0e6));
            }
          }
          if (PetscLogMemory) PetscCall(PetscViewerASCIIPrintf(viewer, " %5.0f   %5.0f   %5.0f   %5.0f", mal / 1.0e6, emalmax / 1.0e6, malmax / 1.0e6, mem / 1.0e6));
#if defined(PETSC_HAVE_DEVICE)
          if (totf != 0.0) fracgflops = gflops / totf;
          else fracgflops = 0.0;
          if (gmaxt != 0.0) gflopr = gflops / gmaxt;
          else gflopr = 0.0;
          if (PetscIsNanReal(gflopr)) {
            PetscCall(PetscViewerASCIIPrintf(viewer, "    n/a    %4.0f %3.2e %4.0f %3.2e % 2.0f", cct / size, csz / (1.0e6 * size), gct / size, gsz / (1.0e6 * size), 100.0 * fracgflops));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer, "   %5.0f   %4.0f %3.2e %4.0f %3.2e % 2.0f", PetscAbs(gflopr) / 1.0e6, cct / size, csz / (1.0e6 * size), gct / size, gsz / (1.0e6 * size), 100.0 * fracgflops));
          }
#endif
          PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
        }
      }
    }
  }

  /* Memory usage and object creation */
  PetscCall(PetscViewerASCIIPrintf(viewer, "------------------------------------------------------------------------------------------------------------------------"));
  if (PetscLogMemory) PetscCall(PetscViewerASCIIPrintf(viewer, "-----------------------------"));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(PetscViewerASCIIPrintf(viewer, "---------------------------------------"));
#endif
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));

  /* Right now, only stages on the first processor are reported here, meaning only objects associated with
     the global communicator, or MPI_COMM_SELF for proc 1. We really should report global stats and then
     stats for stages local to processor sets.
  */
  /* We should figure out the longest object name here (now 20 characters) */
  PetscCall(PetscViewerASCIIPrintf(viewer, "Object Type          Creations   Destructions. Reports information only for process 0.\n"));
  for (stage = 0; stage < numStages; stage++) {
    const char *stage_name;

    PetscCall(PetscLogGlobalNamesGlobalGetName(global_stages, stage, &stage_name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n--- Event Stage %d: %s\n\n", stage, stage_name));
    if (localStageUsed[stage]) {
      PetscInt num_classes;

      PetscCall(PetscLogStateGetNumClasses(state, &num_classes));
      for (oclass = 0; oclass < num_classes; oclass++) {
        PetscClassPerf *class_perf_info;

        PetscCall(PetscLogHandlerDefaultGetClassPerf(handler, stage, oclass, &class_perf_info));
        if (class_perf_info->creations > 0 || class_perf_info->destructions > 0) {
          PetscLogClassInfo class_reg_info;
          PetscBool         flg = PETSC_FALSE;

          PetscCall(PetscLogStateClassGetInfo(state, oclass, &class_reg_info));
          if (stage == 0 && oclass == num_classes - 1) {
            if (PetscGlobalArgc == 0 && PetscGlobalArgs == NULL) {
              PetscCall(PetscStrcmp(class_reg_info.name, "Viewer", &flg));
              if (flg && class_perf_info->creations == PetscLogNumViewersCreated && class_perf_info->destructions == PetscLogNumViewersDestroyed) continue;
            }
          }
          PetscCall(PetscViewerASCIIPrintf(viewer, "%20s %5d          %5d\n", class_reg_info.name, class_perf_info->creations, class_perf_info->destructions));
        }
      }
    }
  }

  PetscCall(PetscFree(localStageUsed));
  PetscCall(PetscFree(stageUsed));
  PetscCall(PetscFree(localStageVisible));
  PetscCall(PetscFree(stageVisible));
  PetscCall(PetscLogGlobalNamesDestroy(&global_stages));
  PetscCall(PetscLogGlobalNamesDestroy(&global_events));

  /* Information unrelated to this particular run */
  PetscCall(PetscViewerASCIIPrintf(viewer, "========================================================================================================================\n"));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&x));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscTime(&y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Average time to get PetscTime(): %g\n", (y - x) / 10.0));
  /* MPI information */
  if (size > 1) {
    MPI_Status  status;
    PetscMPIInt tag;
    MPI_Comm    newcomm;

    PetscCallMPI(MPI_Barrier(comm));
    PetscCall(PetscTime(&x));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCallMPI(MPI_Barrier(comm));
    PetscCall(PetscTime(&y));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Average time for MPI_Barrier(): %g\n", (y - x) / 5.0));
    PetscCall(PetscCommDuplicate(comm, &newcomm, &tag));
    PetscCallMPI(MPI_Barrier(comm));
    if (rank) {
      PetscCallMPI(MPI_Recv(NULL, 0, MPI_INT, rank - 1, tag, newcomm, &status));
      PetscCallMPI(MPI_Send(NULL, 0, MPI_INT, (rank + 1) % size, tag, newcomm));
    } else {
      PetscCall(PetscTime(&x));
      PetscCallMPI(MPI_Send(NULL, 0, MPI_INT, 1, tag, newcomm));
      PetscCallMPI(MPI_Recv(NULL, 0, MPI_INT, size - 1, tag, newcomm, &status));
      PetscCall(PetscTime(&y));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Average time for zero size MPI_Send(): %g\n", (y - x) / size));
    }
    PetscCall(PetscCommDestroy(&newcomm));
  }
  PetscCall(PetscOptionsView(NULL, viewer));

  /* Machine and compile information */
  if (PetscDefined(USE_FORTRAN_KERNELS)) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Compiled with FORTRAN kernels\n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Compiled without FORTRAN kernels\n"));
  }
  if (PetscDefined(USE_64BIT_INDICES)) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Compiled with 64-bit PetscInt\n"));
  } else if (PetscDefined(USE___FLOAT128)) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Compiled with 32-bit PetscInt\n"));
  }
  if (PetscDefined(USE_REAL_SINGLE)) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Compiled with single precision PetscScalar and PetscReal\n"));
  } else if (PetscDefined(USE___FLOAT128)) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Compiled with 128 bit precision PetscScalar and PetscReal\n"));
  }
  if (PetscDefined(USE_REAL_MAT_SINGLE)) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Compiled with single precision matrices\n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Compiled with full precision matrices (default)\n"));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "sizeof(short) %d sizeof(int) %d sizeof(long) %d sizeof(void*) %d sizeof(PetscScalar) %d sizeof(PetscInt) %d\n", (int)sizeof(short), (int)sizeof(int), (int)sizeof(long), (int)sizeof(void *), (int)sizeof(PetscScalar), (int)sizeof(PetscInt)));

  PetscCall(PetscViewerASCIIPrintf(viewer, "Configure options: %s", petscconfigureoptions));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s", petscmachineinfo));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s", petsccompilerinfo));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s", petsccompilerflagsinfo));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s", petsclinkerinfo));

  /* Cleanup */
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscLogViewWarnNoGpuAwareMpi(viewer));
  PetscCall(PetscLogViewWarnDebugging(viewer));
  PetscCall(PetscFPTrapPop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscLogHandlerView_Default(PetscLogHandler handler, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_ASCII_INFO) {
    PetscCall(PetscLogHandlerView_Default_Info(handler, viewer));
  } else if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscCall(PetscLogHandlerView_Default_Detailed(handler, viewer));
  } else if (format == PETSC_VIEWER_ASCII_CSV) {
    PetscCall(PetscLogHandlerView_Default_CSV(handler, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCLOGHANDLERDEFAULT - PETSCLOGHANDLERDEFAULT = "default" -  A `PetscLogHandler` that collects data for PETSc
  default profiling log viewers (`PetscLogView()` and `PetscLogDump()`).  A log handler of this type is
  created and started (`PetscLogHandlerStart()`) by `PetscLogDefaultBegin()`.

  Options Database Keys:
+ -log_include_actions - include a growing list of actions (event beginnings and endings, object creations and destructions) in `PetscLogDump()` (`PetscLogActions()`).
- -log_include_objects - include a growing list of object creations and destructions in `PetscLogDump()` (`PetscLogObjects()`).

  Level: developer

.seealso: [](ch_profiling), `PetscLogHandler`
M*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Default(PetscLogHandler handler)
{
  PetscFunctionBegin;
  PetscCall(PetscLogHandlerContextCreate_Default((PetscLogHandler_Default *)&handler->data));
  handler->ops->destroy       = PetscLogHandlerDestroy_Default;
  handler->ops->eventbegin    = PetscLogHandlerEventBegin_Default;
  handler->ops->eventend      = PetscLogHandlerEventEnd_Default;
  handler->ops->eventsync     = PetscLogHandlerEventSync_Default;
  handler->ops->objectcreate  = PetscLogHandlerObjectCreate_Default;
  handler->ops->objectdestroy = PetscLogHandlerObjectDestroy_Default;
  handler->ops->stagepush     = PetscLogHandlerStagePush_Default;
  handler->ops->stagepop      = PetscLogHandlerStagePop_Default;
  handler->ops->view          = PetscLogHandlerView_Default;
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerGetEventPerfInfo_C", PetscLogHandlerGetEventPerfInfo_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerGetStagePerfInfo_C", PetscLogHandlerGetStagePerfInfo_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerSetLogActions_C", PetscLogHandlerSetLogActions_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerSetLogObjects_C", PetscLogHandlerSetLogObjects_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerLogObjectState_C", PetscLogHandlerLogObjectState_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerGetNumObjects_C", PetscLogHandlerGetNumObjects_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerEventDeactivatePush_C", PetscLogHandlerEventDeactivatePush_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerEventDeactivatePop_C", PetscLogHandlerEventDeactivatePop_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerEventsPause_C", PetscLogHandlerEventsPause_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerEventsResume_C", PetscLogHandlerEventsResume_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerDump_C", PetscLogHandlerDump_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerStageSetVisible_C", PetscLogHandlerStageSetVisible_Default));
  PetscCall(PetscObjectComposeFunction((PetscObject)handler, "PetscLogHandlerStageGetVisible_C", PetscLogHandlerStageGetVisible_Default));
  PetscFunctionReturn(PETSC_SUCCESS);
}
