#pragma once
/* all of the logging files have problems with automatic integer casting so checking is turned off for them here */
#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic ignored "-Wconversion"
#endif

#include <petsc/private/petscimpl.h>

#include <petsc/private/logimpldeprecated.h>

/* --- Macros for resizable arrays that show up frequently in the implementation of logging --- */

#define PETSC_LOG_RESIZABLE_ARRAY(Container, Entry, Key, Constructor, Destructor, Equal) \
  typedef struct _n_PetscLog##Container    *PetscLog##Container; \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Create(int, PetscLog##Container *); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Destroy(PetscLog##Container *); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Recapacity(PetscLog##Container, int); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Resize(PetscLog##Container, int); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Push(PetscLog##Container, Entry); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Find(PetscLog##Container, Key, int *); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##GetSize(PetscLog##Container, PetscInt *, PetscInt *); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Get(PetscLog##Container, PetscInt, Entry *); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##GetRef(PetscLog##Container, PetscInt, Entry **); \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Set(PetscLog##Container, PetscInt, Entry); \
  struct _n_PetscLog##Container { \
    int    num_entries; \
    int    max_entries; \
    Entry *array; \
  }; \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Create(int max_init, PetscLog##Container *a_p) \
  { \
    PetscLog##Container a; \
    PetscErrorCode (*constructor)(Entry *) = Constructor; \
    PetscFunctionBegin; \
    PetscCall(PetscNew(a_p)); \
    a              = *a_p; \
    a->num_entries = 0; \
    a->max_entries = max_init; \
    if (constructor) { \
      PetscCall(PetscMalloc1(max_init, &a->array)); \
    } else { \
      PetscCall(PetscCalloc1(max_init, &a->array)); \
    } \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Destroy(PetscLog##Container *a_p) \
  { \
    PetscLog##Container a; \
    PetscErrorCode (*destructor)(Entry *) = Destructor; \
    PetscFunctionBegin; \
    a    = *a_p; \
    *a_p = NULL; \
    if (a == NULL) PetscFunctionReturn(PETSC_SUCCESS); \
    if (destructor) { \
      for (int i = 0; i < a->num_entries; i++) PetscCall((*destructor)(&a->array[i])); \
    } \
    PetscCall(PetscFree(a->array)); \
    PetscCall(PetscFree(a)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Recapacity(PetscLog##Container a, int new_size) \
  { \
    PetscErrorCode (*constructor)(Entry *) = Constructor; \
    PetscFunctionBegin; \
    if (new_size > a->max_entries) { \
      int    new_max_entries = 2; \
      int    rem_size        = PetscMax(0, new_size - 1); \
      Entry *new_array; \
      while (rem_size >>= 1) new_max_entries *= 2; \
      if (constructor) { \
        PetscCall(PetscMalloc1(new_max_entries, &new_array)); \
      } else { \
        PetscCall(PetscCalloc1(new_max_entries, &new_array)); \
      } \
      PetscCall(PetscArraycpy(new_array, a->array, a->num_entries)); \
      PetscCall(PetscFree(a->array)); \
      a->array       = new_array; \
      a->max_entries = new_max_entries; \
    } \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Resize(PetscLog##Container a, int new_size) \
  { \
    PetscErrorCode (*constructor)(Entry *) = Constructor; \
    PetscFunctionBegin; \
    PetscCall(PetscLog##Container##Recapacity(a, new_size)); \
    if (constructor) \
      for (int i = a->num_entries; i < new_size; i++) PetscCall((*constructor)(&a->array[i])); \
    a->num_entries = PetscMax(a->num_entries, new_size); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Push(PetscLog##Container a, Entry new_entry) \
  { \
    PetscFunctionBegin; \
    PetscCall(PetscLog##Container##Recapacity(a, a->num_entries + 1)); \
    a->array[a->num_entries++] = new_entry; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Find(PetscLog##Container a, Key key, int *idx_p) \
  { \
    PetscErrorCode (*equal)(Entry *, Key, PetscBool *) = Equal; \
    PetscFunctionBegin; \
    *idx_p = -1; \
    if (equal) { \
      for (int i = 0; i < a->num_entries; i++) { \
        PetscBool is_equal; \
        PetscCall((*equal)(&a->array[i], key, &is_equal)); \
        if (is_equal) { \
          *idx_p = i; \
          break; \
        } \
      } \
    } \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##GetSize(PetscLog##Container a, PetscInt *num_entries, PetscInt *max_entries) \
  { \
    PetscFunctionBegin; \
    if (num_entries) *num_entries = a->num_entries; \
    if (max_entries) *max_entries = a->max_entries; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Get(PetscLog##Container a, PetscInt i, Entry *entry) \
  { \
    PetscFunctionBegin; \
    PetscCheck(i >= 0 && i < a->num_entries, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in range [0,%d)", i, a->num_entries); \
    *entry = a->array[i]; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##GetRef(PetscLog##Container a, PetscInt i, Entry **entry) \
  { \
    PetscFunctionBegin; \
    PetscCheck(i >= 0 && i < a->num_entries, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in range [0,%d)", i, a->num_entries); \
    *entry = &a->array[i]; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static inline PETSC_UNUSED PetscErrorCode PetscLog##Container##Set(PetscLog##Container a, PetscInt i, Entry entry) \
  { \
    PetscFunctionBegin; \
    PetscCheck(i >= 0 && i < a->num_entries, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Index %" PetscInt_FMT " is not in range [0,%d)", i, a->num_entries); \
    a->array[i] = entry; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

/* --- the registry: information about registered things ---

   Log handler instances should not change the registry: it is shared
   data that should be useful to more than one type of logging

 */

PETSC_INTERN PetscErrorCode PetscLogRegistryCreate(PetscLogRegistry *);
PETSC_INTERN PetscErrorCode PetscLogRegistryDestroy(PetscLogRegistry);
PETSC_INTERN PetscErrorCode PetscLogRegistryStageRegister(PetscLogRegistry, const char[], PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryEventRegister(PetscLogRegistry, const char[], PetscClassId, PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryClassRegister(PetscLogRegistry, const char[], PetscClassId, PetscLogClass *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetEventFromName(PetscLogRegistry, const char[], PetscLogEvent *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetStageFromName(PetscLogRegistry, const char[], PetscLogStage *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetClassFromClassId(PetscLogRegistry, PetscClassId, PetscLogClass *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetClassFromName(PetscLogRegistry, const char[], PetscLogClass *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumEvents(PetscLogRegistry, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumStages(PetscLogRegistry, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogRegistryGetNumClasses(PetscLogRegistry, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogRegistryEventGetInfo(PetscLogRegistry, PetscLogEvent, PetscLogEventInfo *);
PETSC_INTERN PetscErrorCode PetscLogRegistryStageGetInfo(PetscLogRegistry, PetscLogStage, PetscLogStageInfo *);
PETSC_INTERN PetscErrorCode PetscLogRegistryClassGetInfo(PetscLogRegistry, PetscLogClass, PetscLogClassInfo *);
PETSC_INTERN PetscErrorCode PetscLogRegistryEventSetCollective(PetscLogRegistry, PetscLogEvent, PetscBool);

/* --- globally synchronized registry information --- */

typedef struct _n_PetscLogGlobalNames *PetscLogGlobalNames;

PETSC_INTERN PetscErrorCode PetscLogGlobalNamesCreate(MPI_Comm, PetscInt, const char **, PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesDestroy(PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGetSize(PetscLogGlobalNames, PetscInt *, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGlobalGetName(PetscLogGlobalNames, PetscInt, const char **);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesGlobalGetLocal(PetscLogGlobalNames, PetscInt, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogGlobalNamesLocalGetGlobal(PetscLogGlobalNames, PetscInt, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogRegistryCreateGlobalStageNames(MPI_Comm, PetscLogRegistry, PetscLogGlobalNames *);
PETSC_INTERN PetscErrorCode PetscLogRegistryCreateGlobalEventNames(MPI_Comm, PetscLogRegistry, PetscLogGlobalNames *);

/* A simple stack */
struct _n_PetscIntStack {
  int  top;   /* The top of the stack */
  int  max;   /* The maximum stack size */
  int *stack; /* The storage */
};

/* Thread-safety internals */

/* SpinLock for shared Log variables */
PETSC_INTERN PetscSpinlock PetscLogSpinLock;

#if defined(PETSC_HAVE_THREADSAFETY)
  #if defined(__cplusplus)
    #define PETSC_TLS thread_local
  #else
    #define PETSC_TLS _Thread_local
  #endif
  #define PETSC_INTERN_TLS extern PETSC_TLS PETSC_VISIBILITY_INTERNAL

/* Access PETSc internal thread id */
PETSC_INTERN PetscInt PetscLogGetTid(void);
#else
  #define PETSC_TLS
  #define PETSC_INTERN_TLS PETSC_INTERN
#endif

PETSC_EXTERN PetscBool PetscLogGpuTimeFlag;
PETSC_INTERN PetscInt  PetscLogNumViewersCreated;
PETSC_INTERN PetscInt  PetscLogNumViewersDestroyed;

#if PetscDefined(USE_LOG)
PETSC_INTERN PetscErrorCode PetscLogTypeBegin(PetscLogHandlerType type);
#else
  #define PetscLogTypeBegin(t) ((void)(t), PETSC_SUCCESS)
#endif

#define PETSC_LOG_VIEW_FROM_OPTIONS_MAX 4
