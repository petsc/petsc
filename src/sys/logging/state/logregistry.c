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

/* Given a list of strings on each process, create a global numbering.  Order
 them by their order on the first process, then the remaining by their order
 on the second process, etc.  The expectation is that most processes have the
 same names in the same order so it shouldn't take too many rounds to figure
 out */

struct _n_PetscLogGlobalNames {
  MPI_Comm     comm;
  PetscInt     count_global;
  PetscInt     count_local;
  const char **names;
  PetscInt    *global_to_local;
  PetscInt    *local_to_global;
};

static PetscErrorCode PetscLogGlobalNamesCreate_Internal(MPI_Comm comm, PetscInt num_names_local, const char **names, PetscInt *num_names_global_p, PetscInt **global_index_to_local_index_p, PetscInt **local_index_to_global_index_p, const char ***global_names_p)
{
  PetscMPIInt size, rank;
  PetscInt    num_names_global          = 0;
  PetscInt    num_names_local_remaining = num_names_local;
  PetscBool  *local_name_seen;
  PetscInt   *global_index_to_local_index = NULL;
  PetscInt   *local_index_to_global_index = NULL;
  PetscInt    max_name_len                = 0;
  char       *str_buffer;
  char      **global_names = NULL;
  PetscMPIInt p;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    PetscCall(PetscMalloc1(num_names_local, &global_index_to_local_index));
    PetscCall(PetscMalloc1(num_names_local, &local_index_to_global_index));
    PetscCall(PetscMalloc1(num_names_local, &global_names));
    for (PetscInt i = 0; i < num_names_local; i++) {
      global_index_to_local_index[i] = i;
      local_index_to_global_index[i] = i;
      PetscCall(PetscStrallocpy(names[i], &global_names[i]));
    }
    *num_names_global_p            = num_names_local;
    *global_index_to_local_index_p = global_index_to_local_index;
    *local_index_to_global_index_p = local_index_to_global_index;
    *global_names_p                = (const char **)global_names;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscCalloc1(num_names_local, &local_name_seen));
  PetscCall(PetscMalloc1(num_names_local, &local_index_to_global_index));

  for (PetscInt i = 0; i < num_names_local; i++) {
    size_t i_len;
    PetscCall(PetscStrlen(names[i], &i_len));
    max_name_len = PetscMax(max_name_len, (PetscInt)i_len);
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &max_name_len, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscCalloc1(max_name_len + 1, &str_buffer));

  p = 0;
  while (p < size) {
    PetscInt my_loc, next_loc;
    PetscInt num_to_add;

    my_loc = num_names_local_remaining > 0 ? rank : PETSC_MPI_INT_MAX;
    PetscCallMPI(MPIU_Allreduce(&my_loc, &next_loc, 1, MPIU_INT, MPI_MIN, comm));
    if (next_loc == PETSC_MPI_INT_MAX) break;
    PetscAssert(next_loc >= p, comm, PETSC_ERR_PLIB, "Failed invariant, expected increasing next process");
    p          = next_loc;
    num_to_add = (rank == p) ? num_names_local_remaining : -1;
    PetscCallMPI(MPI_Bcast(&num_to_add, 1, MPIU_INT, p, comm));
    {
      PetscInt  new_num_names_global = num_names_global + num_to_add;
      PetscInt *new_global_index_to_local_index;
      char    **new_global_names;

      PetscCall(PetscMalloc1(new_num_names_global, &new_global_index_to_local_index));
      PetscCall(PetscArraycpy(new_global_index_to_local_index, global_index_to_local_index, num_names_global));
      for (PetscInt i = num_names_global; i < new_num_names_global; i++) new_global_index_to_local_index[i] = -1;
      PetscCall(PetscFree(global_index_to_local_index));
      global_index_to_local_index = new_global_index_to_local_index;

      PetscCall(PetscCalloc1(new_num_names_global, &new_global_names));
      PetscCall(PetscArraycpy(new_global_names, global_names, num_names_global));
      PetscCall(PetscFree(global_names));
      global_names = new_global_names;
    }

    if (rank == p) {
      for (PetscInt s = 0; s < num_names_local; s++) {
        if (local_name_seen[s]) continue;
        local_name_seen[s] = PETSC_TRUE;
        PetscCall(PetscArrayzero(str_buffer, max_name_len + 1));
        PetscCall(PetscStrallocpy(names[s], &global_names[num_names_global]));
        PetscCall(PetscStrncpy(str_buffer, names[s], max_name_len + 1));
        PetscCallMPI(MPI_Bcast(str_buffer, max_name_len + 1, MPI_CHAR, p, comm));
        local_index_to_global_index[s]                  = num_names_global;
        global_index_to_local_index[num_names_global++] = s;
        num_names_local_remaining--;
      }
    } else {
      for (PetscInt i = 0; i < num_to_add; i++) {
        PetscInt s;
        PetscCallMPI(MPI_Bcast(str_buffer, max_name_len + 1, MPI_CHAR, p, comm));
        PetscCall(PetscStrallocpy(str_buffer, &global_names[num_names_global]));
        for (s = 0; s < num_names_local; s++) {
          PetscBool same;

          if (local_name_seen[s]) continue;
          PetscCall(PetscStrncmp(names[s], str_buffer, max_name_len + 1, &same));
          if (same) {
            local_name_seen[s]                            = PETSC_TRUE;
            global_index_to_local_index[num_names_global] = s;
            local_index_to_global_index[s]                = num_names_global;
            num_names_local_remaining--;
            break;
          }
        }
        if (s == num_names_local) global_index_to_local_index[num_names_global] = -1; // this name is not present on this process
        num_names_global++;
      }
    }
  }

  PetscCall(PetscFree(str_buffer));
  PetscCall(PetscFree(local_name_seen));
  *num_names_global_p            = num_names_global;
  *global_index_to_local_index_p = global_index_to_local_index;
  *local_index_to_global_index_p = local_index_to_global_index;
  *global_names_p                = (const char **)global_names;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesCreate(MPI_Comm comm, PetscInt num_names_local, const char **local_names, PetscLogGlobalNames *global_names_p)
{
  PetscLogGlobalNames global_names;

  PetscFunctionBegin;
  PetscCall(PetscNew(&global_names));
  PetscCall(PetscLogGlobalNamesCreate_Internal(comm, num_names_local, local_names, &global_names->count_global, &global_names->global_to_local, &global_names->local_to_global, &global_names->names));
  global_names->count_local = num_names_local;
  *global_names_p           = global_names;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesDestroy(PetscLogGlobalNames *global_names_p)
{
  PetscLogGlobalNames global_names;

  PetscFunctionBegin;
  global_names    = *global_names_p;
  *global_names_p = NULL;
  PetscCall(PetscFree(global_names->global_to_local));
  PetscCall(PetscFree(global_names->local_to_global));
  for (PetscInt i = 0; i < global_names->count_global; i++) PetscCall(PetscFree(global_names->names[i]));
  PetscCall(PetscFree(global_names->names));
  PetscCall(PetscFree(global_names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGlobalGetName(PetscLogGlobalNames global_names, PetscInt idx, const char **name)
{
  PetscFunctionBegin;
  PetscCheck(idx >= 0 && idx < global_names->count_global, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Idx %" PetscInt_FMT " not in range [0,%" PetscInt_FMT ")", idx, global_names->count_global);
  *name = global_names->names[idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGlobalGetLocal(PetscLogGlobalNames global_names, PetscInt idx, PetscInt *local_idx)
{
  PetscFunctionBegin;
  PetscCheck(idx >= 0 && idx < global_names->count_global, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Idx %" PetscInt_FMT " not in range [0,%" PetscInt_FMT ")", idx, global_names->count_global);
  *local_idx = global_names->global_to_local[idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesLocalGetGlobal(PetscLogGlobalNames global_names, PetscInt local_idx, PetscInt *idx)
{
  PetscFunctionBegin;
  PetscCheck(local_idx >= 0 && local_idx < global_names->count_local, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Idx %" PetscInt_FMT " not in range [0,%" PetscInt_FMT ")", local_idx, global_names->count_local);
  *idx = global_names->local_to_global[local_idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGetSize(PetscLogGlobalNames global_names, PetscInt *local_size, PetscInt *global_size)
{
  PetscFunctionBegin;
  if (local_size) *local_size = global_names->count_local;
  if (global_size) *global_size = global_names->count_global;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryCreateGlobalStageNames(MPI_Comm comm, PetscLogRegistry registry, PetscLogGlobalNames *global_names_p)
{
  PetscInt     num_stages_local;
  const char **names;

  PetscFunctionBegin;
  PetscCall(PetscLogStageArrayGetSize(registry->stages, &num_stages_local, NULL));
  PetscCall(PetscMalloc1(num_stages_local, &names));
  for (PetscInt i = 0; i < num_stages_local; i++) {
    PetscLogStageInfo stage_info = {NULL};
    PetscCall(PetscLogRegistryStageGetInfo(registry, i, &stage_info));
    names[i] = stage_info.name;
  }
  PetscCall(PetscLogGlobalNamesCreate(comm, num_stages_local, names, global_names_p));
  PetscCall(PetscFree(names));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscLogRegistryCreateGlobalEventNames(MPI_Comm comm, PetscLogRegistry registry, PetscLogGlobalNames *global_names_p)
{
  PetscInt     num_events_local;
  const char **names;

  PetscFunctionBegin;
  PetscCall(PetscLogEventArrayGetSize(registry->events, &num_events_local, NULL));
  PetscCall(PetscMalloc1(num_events_local, &names));
  for (PetscInt i = 0; i < num_events_local; i++) {
    PetscLogEventInfo event_info = {NULL, 0, PETSC_FALSE};

    PetscCall(PetscLogRegistryEventGetInfo(registry, i, &event_info));
    names[i] = event_info.name;
  }
  PetscCall(PetscLogGlobalNamesCreate(comm, num_events_local, names, global_names_p));
  PetscCall(PetscFree(names));
  PetscFunctionReturn(PETSC_SUCCESS);
}
