
#include <petsc/private/logimpl.h> /*I "petsclog.h" I*/

#define PETSC_LOG_RESIZABLE_ARRAY_HAS_NAME(Container, Entry, Key, Equal) \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Destructor(Entry *entry) \
  { \
    PetscFunctionBegin; \
    PetscCall(PetscFree(entry->name)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  PETSC_LOG_RESIZABLE_ARRAY(Container, Entry, Key, NULL, PetscLog##Container##Destructor, Equal)

#define PETSC_LOG_RESIZABLE_ARRAY_KEY_BY_NAME(Container, Entry) \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Equal(Entry *entry, const char *name, PetscBool *is_equal) \
  { \
    PetscFunctionBegin; \
    PetscCall(PetscStrcmp(entry->name, name, is_equal)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  PETSC_LOG_RESIZABLE_ARRAY_HAS_NAME(Container, Entry, const char *, PetscLog##Container##Equal)

static PetscErrorCode PetscLogClassArrayEqual(PetscLogClassInfo *class_info, PetscLogClassInfo *key, PetscBool *is_equal)
{
  PetscFunctionBegin;
  if (key->name) {
    PetscCall(PetscStrcmp(class_info->name, key->name, is_equal));
  } else {
    *is_equal = (class_info->classid == key->classid) ? PETSC_TRUE : PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_LOG_RESIZABLE_ARRAY_KEY_BY_NAME(EventArray, PetscLogEventInfo)
PETSC_LOG_RESIZABLE_ARRAY_KEY_BY_NAME(StageArray, PetscLogStageInfo)
PETSC_LOG_RESIZABLE_ARRAY_HAS_NAME(ClassArray, PetscLogClassInfo, PetscLogClassInfo *, PetscLogClassArrayEqual)

struct _n_PetscLogRegistry {
  PetscLogEventArray events;
  PetscLogClassArray classes;
  PetscLogStageArray stages;
};

PETSC_INTERN PetscErrorCode PetscLogRegistryCreate(PetscLogRegistry *registry_p)
{
  PetscLogRegistry registry;

  PetscFunctionBegin;
  PetscCall(PetscNew(registry_p));
  registry = *registry_p;
  PetscCall(PetscLogEventArrayCreate(128, &registry->events));
  PetscCall(PetscLogStageArrayCreate(8, &registry->stages));
  PetscCall(PetscLogClassArrayCreate(128, &registry->classes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryDestroy(PetscLogRegistry registry)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventArrayDestroy(&registry->events));
  PetscCall(PetscLogClassArrayDestroy(&registry->classes));
  PetscCall(PetscLogStageArrayDestroy(&registry->stages));
  PetscCall(PetscFree(registry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumEvents(PetscLogRegistry registry, PetscInt *num_events, PetscInt *max_events)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventArrayGetSize(registry->events, num_events, max_events));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumStages(PetscLogRegistry registry, PetscInt *num_stages, PetscInt *max_stages)
{
  PetscFunctionBegin;
  PetscCall(PetscLogStageArrayGetSize(registry->stages, num_stages, max_stages));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumClasses(PetscLogRegistry registry, PetscInt *num_classes, PetscInt *max_classes)
{
  PetscFunctionBegin;
  PetscCall(PetscLogClassArrayGetSize(registry->classes, num_classes, max_classes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryStageRegister(PetscLogRegistry registry, const char sname[], int *stage)
{
  int               idx;
  PetscLogStageInfo stage_info;

  PetscFunctionBegin;
  PetscCall(PetscLogStageArrayFind(registry->stages, sname, &idx));
  PetscCheck(idx == -1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "An event named %s is already registered", sname);
  *stage = registry->stages->num_entries;
  PetscCall(PetscStrallocpy(sname, &stage_info.name));
  PetscCall(PetscLogStageArrayPush(registry->stages, stage_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryEventRegister(PetscLogRegistry registry, const char name[], PetscClassId classid, PetscLogEvent *event)
{
  PetscLogEventInfo new_info;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetEventFromName(registry, name, event));
  if (*event >= 0) PetscFunctionReturn(PETSC_SUCCESS);
  *event              = registry->events->num_entries;
  new_info.classid    = classid;
  new_info.collective = PETSC_TRUE;
  PetscCall(PetscStrallocpy(name, &new_info.name));
  PetscCall(PetscLogEventArrayPush(registry->events, new_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryClassRegister(PetscLogRegistry registry, const char name[], PetscClassId classid, PetscLogClass *clss)
{
  PetscLogClassInfo new_info;

  PetscFunctionBegin;
  PetscCall(PetscLogRegistryGetClassFromClassId(registry, classid, clss));
  if (*clss >= 0) PetscFunctionReturn(PETSC_SUCCESS);
  *clss            = registry->classes->num_entries;
  new_info.classid = classid;
  PetscCall(PetscStrallocpy(name, &new_info.name));
  PetscCall(PetscLogClassArrayPush(registry->classes, new_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryGetEventFromName(PetscLogRegistry registry, const char name[], PetscLogEvent *event)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventArrayFind(registry->events, name, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryGetStageFromName(PetscLogRegistry registry, const char name[], PetscLogStage *stage)
{
  PetscFunctionBegin;
  PetscCall(PetscLogStageArrayFind(registry->stages, name, stage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryGetClassFromClassId(PetscLogRegistry registry, PetscClassId classid, PetscLogStage *clss)
{
  PetscLogClassInfo key;

  PetscFunctionBegin;
  key.name    = NULL;
  key.classid = classid;
  PetscCall(PetscLogClassArrayFind(registry->classes, &key, clss));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryGetClassFromName(PetscLogRegistry registry, const char name[], PetscLogStage *clss)
{
  PetscLogClassInfo key;

  PetscFunctionBegin;
  key.name    = (char *)name;
  key.classid = -1;
  PetscCall(PetscLogClassArrayFind(registry->classes, &key, clss));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryEventGetInfo(PetscLogRegistry registry, PetscLogEvent event, PetscLogEventInfo *event_info)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventArrayGet(registry->events, event, event_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryStageGetInfo(PetscLogRegistry registry, PetscLogStage stage, PetscLogStageInfo *stage_info)
{
  PetscFunctionBegin;
  PetscCall(PetscLogStageArrayGet(registry->stages, stage, stage_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryClassGetInfo(PetscLogRegistry registry, PetscLogClass clss, PetscLogClassInfo *class_info)
{
  PetscFunctionBegin;
  PetscCall(PetscLogClassArrayGet(registry->classes, clss, class_info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryEventSetCollective(PetscLogRegistry registry, PetscLogEvent event, PetscBool collective)
{
  PetscLogEventInfo *event_info;

  PetscFunctionBegin;
  PetscCall(PetscLogEventArrayGetRef(registry->events, event, &event_info));
  event_info->collective = collective;
  PetscFunctionReturn(PETSC_SUCCESS);
}
