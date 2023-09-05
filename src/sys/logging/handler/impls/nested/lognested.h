#pragma once

#include <petsc/private/loghandlerimpl.h>
#include <petsc/private/logimpl.h>
#include <../src/sys/logging/handler/impls/default/logdefault.h>
#include <petsc/private/hashmap.h>

typedef int NestedId;

static inline PETSC_UNUSED NestedId NestedIdFromEvent(PetscLogEvent e)
{
  return e;
}

static inline PETSC_UNUSED PetscLogEvent NestedIdToEvent(NestedId e)
{
  return e;
}

static inline PETSC_UNUSED NestedId NestedIdFromStage(PetscLogStage s)
{
  return -(s + 2);
}

static inline PETSC_UNUSED PetscLogStage NestedIdToStage(NestedId s)
{
  return -(s + 2);
}

typedef struct _n_NestedIdPair NestedIdPair;
struct _n_NestedIdPair {
  PetscLogEvent root;
  NestedId      leaf;
};

#define NestedIdPairHash(key)     PetscHashCombine(PetscHash_UInt32((PetscHash32_t)((key).root)), PetscHash_UInt32((PetscHash32_t)((key).leaf)))
#define NestedIdPairEqual(k1, k2) (((k1).root == (k2).root) && ((k1).leaf == (k2).leaf))

PETSC_HASH_MAP(NestedHash, NestedIdPair, PetscLogEvent, NestedIdPairHash, NestedIdPairEqual, -1)

typedef struct _n_PetscLogHandler_Nested *PetscLogHandler_Nested;
struct _n_PetscLogHandler_Nested {
  PetscLogState   state;
  PetscLogHandler handler;
  PetscNestedHash pair_map;
  PetscIntStack   nested_stack; // stack of nested ids
  PetscIntStack   orig_stack;   // stack of the original ids
  PetscClassId    nested_stage_id;
  PetscLogDouble  threshold;
};

typedef struct {
  const char *name;
  PetscInt    id;
  PetscInt    parent;
  PetscInt    num_descendants;
} PetscNestedEventNode;

typedef struct {
  MPI_Comm              comm;
  PetscLogGlobalNames   global_events;
  PetscNestedEventNode *nodes;
  PetscEventPerfInfo   *perf;
} PetscNestedEventTree;

typedef enum {
  PETSC_LOG_NESTED_XML,
  PETSC_LOG_NESTED_FLAMEGRAPH
} PetscLogNestedType;

PETSC_INTERN PetscErrorCode PetscLogHandlerView_Nested_XML(PetscLogHandler_Nested, PetscNestedEventTree *, PetscViewer);
PETSC_INTERN PetscErrorCode PetscLogHandlerView_Nested_Flamegraph(PetscLogHandler_Nested, PetscNestedEventTree *, PetscViewer);
